#ifndef TENSORRT_HPP
#define TENSORRT_HPP

#include "common.h"
#include "logger.hpp"
#include "network.hpp"

using namespace nvinfer1;

class TenserRTLogger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override
    {
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
};

class TensorRTComponent {
public:
    TenserRTLogger* tensorrt_logger;
    AlphaZeroResNetImpl* net;

    IBuilder* builder;
    IBuilderConfig* config;
    ICudaEngine* engine = nullptr;
    IOptimizationProfile* profile;

    void* buffers[3];
    std::mutex stream_mutex; 
    cudaStream_t stream;
    IExecutionContext* context = nullptr;

    Timer timer_different_stream;
    Timer timer_transfer; 
    Timer timer_kernel;

    TensorRTComponent(AlphaZeroResNetImpl* _net){
        net = _net; 
        tensorrt_logger = new TenserRTLogger();
        CUDA_CHECK(cudaStreamCreate(&stream));
        CUDA_CHECK(cudaMalloc(&buffers[0], TENSORRT_MAX_BATCH_SIZE * PLANES_SIZE * BOARD_SIZE * BOARD_SIZE * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&buffers[1], TENSORRT_MAX_BATCH_SIZE * ACTION_SIZE * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&buffers[2], TENSORRT_MAX_BATCH_SIZE * sizeof(float)));

        builder = createInferBuilder(*tensorrt_logger);
        config = builder->createBuilderConfig();
        config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1 << 28);
        config->setFlag(TENSORRT_PREDICT_DATA_TYPE);
        config->setFlag(BuilderFlag::kINT8);
        
        profile = builder->createOptimizationProfile();
        
        profile->setDimensions("input", OptProfileSelector::kMIN, Dims4(1, PLANES_SIZE, BOARD_SIZE, BOARD_SIZE));
        profile->setDimensions("input", OptProfileSelector::kOPT, Dims4(PREDICT_BUFFER_SIZE, PLANES_SIZE, BOARD_SIZE, BOARD_SIZE));
        profile->setDimensions("input", OptProfileSelector::kMAX, Dims4(TENSORRT_MAX_BATCH_SIZE, PLANES_SIZE, BOARD_SIZE, BOARD_SIZE));
        config->addOptimizationProfile(profile);

        update_engine();
    }
    
    ~TensorRTComponent() {
        CUDA_CHECK(cudaStreamDestroy(stream));
        CUDA_CHECK(cudaFree(buffers[0]));
        CUDA_CHECK(cudaFree(buffers[1]));
        CUDA_CHECK(cudaFree(buffers[2]));

        if(context != nullptr) {
            delete context;
        }
    
        delete config;

        if(engine != nullptr) {
            delete engine;
        }
        
        delete builder;
        delete tensorrt_logger;
    }


    std::map<std::string, Weights> loadWeights() {
        if(USE_GPU) 
            net->to(torch::kCPU);
        std::map<std::string, Weights> weightMap;
        torch::OrderedDict<std::string, torch::Tensor> named_parameters = net->named_parameters();
        torch::OrderedDict<std::string, torch::Tensor> named_buffers = net->named_buffers();
        
        for (auto& pair : named_parameters) {
            Weights wt{DataType::kFLOAT, nullptr, 0};
            int size;
            std::string name = pair.key();
            torch::Tensor tensor = pair.value().contiguous().view(-1);
            wt.type = DataType::kFLOAT;
            size = tensor.numel();
            wt.count = size;
            float* data = tensor.data_ptr<float>();
            float* val = (float*)(malloc(sizeof(float) * size));
            for(int i = 0; i < size; i++) val[i] = data[i];
            wt.values = val;
            weightMap[name] = wt;
            // printf("name: %s, size: %d\n", name.c_str(), size);
        }
    
        for (auto& pair : named_buffers) {
            Weights wt{DataType::kFLOAT, nullptr, 0};
            int size;
            std::string name = pair.key();
            if (name.find("running_mean") == std::string::npos && name.find("running_var") == std::string::npos) continue;
            torch::Tensor tensor = pair.value().contiguous().view(-1);
            wt.type = DataType::kFLOAT;
            size = tensor.numel();
            wt.count = size;
            float* data = tensor.data_ptr<float>();
            float* val = (float*)(malloc(sizeof(float) * size));
            for(int i = 0; i < size; i++) val[i] = data[i];
            wt.values = val;
            weightMap[name] = wt;
            // printf("name: %s, size: %d\n", name.c_str(), size);
        }
        
        if(USE_GPU) net->to(torch::kCUDA);
        return weightMap;
    }

    ICudaEngine* createEngine(IBuilder* builder, IBuilderConfig* config)
    {
        float input_scale_value = 1.0f / 127.0f;
        // float weight_scale_value = 0.8419753909111023 / 127.0f;
        float weight_scale_value = 1.0f / 127.0f;
        float output_scale_value = weight_scale_value * input_scale_value;

        INetworkDefinition* network = builder->createNetworkV2(0U);
    
        std::map<std::string, Weights> weightMap = loadWeights(); //need to free
        Weights emptywts{DataType::kFLOAT, nullptr, 0};
    
        ITensor* data = network->addInput("input", DataType::kFLOAT, Dims4{-1, PLANES_SIZE, BOARD_SIZE, BOARD_SIZE});
        assert(data);
        
        // auto conv_1_input_scale = network->addConstant(Dims{1, {1}}, Weights{DataType::kFLOAT, &input_scale_value, 1});
        // auto conv_1_input_q = network->addQuantize(*data, *conv_1_input_scale->getOutput(0), DataType::kINT8);
        // conv_1_input_q->setAxis(0);
        // auto conv_1_input_dq = network->addDequantize(*conv_1_input_q->getOutput(0), *conv_1_input_scale->getOutput(0), DataType::kHALF);
        // conv_1_input_dq->setAxis(0);
        
        // auto conv_1_weight_scale = network->addConstant(Dims{1, {1}}, Weights{DataType::kFLOAT, &weight_scale_value, 1});
        // auto conv_1_weight_float = network->addConstant(Dims4{32, 4, 3, 3}, weightMap["conv1.weight"]);
        // auto conv_1_weight_q = network->addQuantize(*conv_1_weight_float->getOutput(0), *conv_1_weight_scale->getOutput(0), DataType::kINT8);
        // conv_1_weight_q->setAxis(0);
        // auto conv_1_weight_dq = network->addDequantize(*conv_1_weight_q->getOutput(0), *conv_1_weight_scale->getOutput(0), DataType::kHALF);
        // conv_1_weight_dq->setAxis(0);

        // auto conv_1_bias_scale = network->addConstant(Dims{1, {1}}, Weights{DataType::kFLOAT, &output_scale_value, 1});
        // auto conv_1_bias_float = network->addConstant(Dims{1, {32}}, weightMap["conv1.bias"]);
        // auto conv_1_bias_q = network->addQuantize(*conv_1_bias_float->getOutput(0), *conv_1_bias_scale->getOutput(0), DataType::kINT8);
        // conv_1_bias_q->setAxis(0);
        // auto conv_1_bias_dq = network->addDequantize(*conv_1_bias_q->getOutput(0), *conv_1_bias_scale->getOutput(0), DataType::kHALF);
        // conv_1_bias_dq->setAxis(0);

        // IConvolutionLayer* conv1 = network->addConvolutionNd(*conv_1_input_dq->getOutput(0), 32, DimsHW{3, 3}, emptywts, emptywts);
        // assert(conv1);
        // conv1->setPaddingNd(DimsHW{1, 1});
        // conv1->setInput(1, *conv_1_weight_dq->getOutput(0));
        // conv1->setInput(2, *conv_1_bias_dq->getOutput(0));

        // auto conv1_output_scale = network->addConstant(Dims{1, {1}}, Weights{DataType::kFLOAT, &output_scale_value, 1});
        // auto conv1_output_q = network->addQuantize(*conv1->getOutput(0), *conv1_output_scale->getOutput(0), DataType::kINT8);
        // conv1_output_q->setAxis(0);
        // auto conv1_output_dq = network->addDequantize(*conv1_output_q->getOutput(0), *conv1_output_scale->getOutput(0), DataType::kFLOAT);
        // conv1_output_dq->setAxis(0);

        IConvolutionLayer* conv1 = network->addConvolutionNd(*data, 32, DimsHW{3, 3}, weightMap["conv1.weight"], weightMap["conv1.bias"]);
        assert(conv1);
        conv1->setPaddingNd(DimsHW{1, 1});

        IActivationLayer* relu1 = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU);
        assert(relu1);

        IConvolutionLayer* conv2 = network->addConvolutionNd(*relu1->getOutput(0), 64, DimsHW{3, 3}, weightMap["conv2.weight"], weightMap["conv2.bias"]);
        assert(conv2);
        conv2->setPaddingNd(DimsHW{1, 1});

        IActivationLayer* relu2 = network->addActivation(*conv2->getOutput(0), ActivationType::kRELU);
        assert(relu2);

        IConvolutionLayer* conv3 = network->addConvolutionNd(*relu2->getOutput(0), 128, DimsHW{3, 3}, weightMap["conv3.weight"], weightMap["conv3.bias"]);
        assert(conv3);
        conv3->setPaddingNd(DimsHW{1, 1});
    
        IActivationLayer* relu3 = network->addActivation(*conv3->getOutput(0), ActivationType::kRELU);
        assert(relu3);
    
        // Policy head
        IConvolutionLayer* policy_conv = network->addConvolutionNd(*relu3->getOutput(0), 4, DimsHW{1, 1}, weightMap["policy_head.0.weight"], weightMap["policy_head.0.bias"]);
        assert(policy_conv);
        policy_conv->setStrideNd(DimsHW{1, 1});
        policy_conv->setPaddingNd(DimsHW{0, 0});
    
        IActivationLayer* policy_relu = network->addActivation(*policy_conv->getOutput(0), ActivationType::kRELU);
        assert(policy_relu);
    
        IShuffleLayer* policy_flatten = network->addShuffle(*policy_relu->getOutput(0));
        policy_flatten->setReshapeDimensions(Dims2{-1, 4 * ACTION_SIZE});
    
        auto policy_weights = network->addConstant(Dims2(ACTION_SIZE, 4 * ACTION_SIZE), weightMap["policy_head.3.weight"]);
        auto policy_bias = network->addConstant(Dims2(1, ACTION_SIZE), weightMap["policy_head.3.bias"]);

        auto policy_mm = network->addMatrixMultiply(*policy_flatten->getOutput(0), MatrixOperation::kNONE, *policy_weights->getOutput(0), MatrixOperation::kTRANSPOSE);
        auto policy_fc = network->addElementWise(*policy_mm->getOutput(0), *policy_bias->getOutput(0), ElementWiseOperation::kSUM);
        assert(policy_fc);
    
        ISoftMaxLayer* policy_softmax = network->addSoftMax(*policy_fc->getOutput(0));
        policy_softmax->setAxes(1 << 1); 
        
        assert(policy_softmax);
    
        // Value head
        IConvolutionLayer* value_conv = network->addConvolutionNd(*relu3->getOutput(0), 2, DimsHW{1, 1}, weightMap["value_head.0.weight"], weightMap["value_head.0.bias"]);
        assert(value_conv);
        value_conv->setStrideNd(DimsHW{1, 1});
        value_conv->setPaddingNd(DimsHW{0, 0});
    
        IActivationLayer* value_relu1 = network->addActivation(*value_conv->getOutput(0), ActivationType::kRELU);
        assert(value_relu1);
    
        IShuffleLayer* value_flatten = network->addShuffle(*value_relu1->getOutput(0));
        value_flatten->setReshapeDimensions(Dims2{-1, 2 * ACTION_SIZE});
    
        auto value_fc1_weights = network->addConstant(Dims2(ACTION_SIZE, 2 * ACTION_SIZE), weightMap["value_head.3.weight"]);
        auto value_fc1_bias = network->addConstant(Dims2(ACTION_SIZE, 1), weightMap["value_head.3.bias"]);
        
        auto value_mm1 = network->addMatrixMultiply(*value_fc1_weights->getOutput(0), MatrixOperation::kNONE, *value_flatten->getOutput(0), MatrixOperation::kTRANSPOSE);
        auto value_fc1 = network->addElementWise(*value_mm1->getOutput(0), *value_fc1_bias->getOutput(0), ElementWiseOperation::kSUM);
        assert(value_fc1);
    
        IActivationLayer* value_relu2 = network->addActivation(*value_fc1->getOutput(0), ActivationType::kRELU);
        assert(value_relu2);
    
        auto value_fc2_weights = network->addConstant(Dims2(1, ACTION_SIZE), weightMap["value_head.5.weight"]);
        auto value_fc2_bias = network->addConstant(Dims2(1, 1), weightMap["value_head.5.bias"]);
    
        auto value_mm2 = network->addMatrixMultiply(*value_fc2_weights->getOutput(0), MatrixOperation::kNONE, *value_relu2->getOutput(0), MatrixOperation::kNONE);
        auto value_fc2 = network->addElementWise(*value_mm2->getOutput(0), *value_fc2_bias->getOutput(0), ElementWiseOperation::kSUM);
        assert(value_fc2);
    
        IActivationLayer* value_tanh = network->addActivation(*value_fc2->getOutput(0), ActivationType::kTANH);
        assert(value_tanh);
    
        policy_softmax->getOutput(0)->setName("policy");
        value_tanh->getOutput(0)->setName("value");
        network->markOutput(*policy_softmax->getOutput(0));
        network->markOutput(*value_tanh->getOutput(0));

        // auto conv2_weight_scale = network->addConstant(Dims{1, {1}}, Weights{DataType::kFLOAT, &weight_scale_value, 1});
        // auto conv2_weight_float = network->addConstant(Dims4{64, 32, 3, 3}, weightMap["conv2.weight"]);
        // auto conv2_weight_q = network->addQuantize(*conv2_weight_float->getOutput(0), *conv2_weight_scale->getOutput(0), DataType::kINT8);
        // conv2_weight_q->setAxis(0);
        // auto conv2_weight_dq = network->addDequantize(*conv2_weight_q->getOutput(0), *conv2_weight_scale->getOutput(0), DataType::kHALF);
        // conv2_weight_dq->setAxis(0);

        // auto conv2_bias_scale = network->addConstant(Dims{1, {1}}, Weights{DataType::kFLOAT, &output_scale_value, 1});
        // auto conv2_bias_float = network->addConstant(Dims{1, {64}}, weightMap["conv2.bias"]);
        // auto conv2_bias_q = network->addQuantize(*conv2_bias_float->getOutput(0), *conv2_bias_scale->getOutput(0), DataType::kINT8);
        // conv2_bias_q->setAxis(0);
        // auto conv2_bias_dq = network->addDequantize(*conv2_bias_q->getOutput(0), *conv2_bias_scale->getOutput(0), DataType::kHALF);
        // conv2_bias_dq->setAxis(0);

        // IConvolutionLayer* conv2 = network->addConvolutionNd(*relu1_output_dq->getOutput(0), 64, DimsHW{3, 3}, emptywts, emptywts);
        // assert(conv2);
        // conv2->setPaddingNd(DimsHW{1, 1});
        // conv2->setInput(1, *conv2_weight_dq->getOutput(0));
        // conv2->setInput(2, *conv2_bias_dq->getOutput(0));

        // IActivationLayer* relu2 = network->addActivation(*conv2->getOutput(0), ActivationType::kRELU);
        // assert(relu2);

        // auto relu2_output_scale = network->addConstant(Dims{1, {1}}, Weights{DataType::kFLOAT, &output_scale_value, 1});
        // auto relu2_output_q = network->addQuantize(*relu2->getOutput(0), *relu2_output_scale->getOutput(0), DataType::kINT8);
        // relu2_output_q->setAxis(0);
        // auto relu2_output_dq = network->addDequantize(*relu2_output_q->getOutput(0), *relu2_output_scale->getOutput(0), DataType::kHALF);
        // relu2_output_dq->setAxis(0);

        // auto conv3_weight_scale = network->addConstant(Dims{1, {1}}, Weights{DataType::kFLOAT, &weight_scale_value, 1});
        // auto conv3_weight_float = network->addConstant(Dims4{128, 64, 3, 3}, weightMap["conv3.weight"]);
        // auto conv3_weight_q = network->addQuantize(*conv3_weight_float->getOutput(0), *conv3_weight_scale->getOutput(0), DataType::kINT8);
        // conv3_weight_q->setAxis(0);
        // auto conv3_weight_dq = network->addDequantize(*conv3_weight_q->getOutput(0), *conv3_weight_scale->getOutput(0), DataType::kHALF);
        // conv3_weight_dq->setAxis(0);
        // auto conv3_bias_scale = network->addConstant(Dims{1, {1}}, Weights{DataType::kFLOAT, &output_scale_value, 1});
        // auto conv3_bias_float = network->addConstant(Dims{1, {128}}, weightMap["conv3.bias"]);
        // auto conv3_bias_q = network->addQuantize(*conv3_bias_float->getOutput(0), *conv3_bias_scale->getOutput(0), DataType::kINT8);
        // conv3_bias_q->setAxis(0);
        // auto conv3_bias_dq = network->addDequantize(*conv3_bias_q->getOutput(0), *conv3_bias_scale->getOutput(0), DataType::kHALF);
        // conv3_bias_dq->setAxis(0);

        // IConvolutionLayer* conv3 = network->addConvolutionNd(*relu2_output_dq->getOutput(0), 128, DimsHW{3, 3}, emptywts, emptywts);
        // assert(conv3);
        // conv3->setPaddingNd(DimsHW{1, 1});
        // conv3->setInput(1, *conv3_weight_dq->getOutput(0));
        // conv3->setInput(2, *conv3_bias_dq->getOutput(0));
    
        // IActivationLayer* relu3 = network->addActivation(*conv3->getOutput(0), ActivationType::kRELU);
        // assert(relu3);

        // auto relu3_output_scale = network->addConstant(Dims{1, {1}}, Weights{DataType::kFLOAT, &output_scale_value, 1});
        // auto relu3_output_q = network->addQuantize(*relu3->getOutput(0), *relu3_output_scale->getOutput(0), DataType::kINT8);
        // relu3_output_q->setAxis(0);
        // auto relu3_output_dq = network->addDequantize(*relu3_output_q->getOutput(0), *relu3_output_scale->getOutput(0), DataType::kHALF);
        // relu3_output_dq->setAxis(0);
        
        // // Policy head
        // auto policy_conv_weight_scale = network->addConstant(Dims{1, {1}}, Weights{DataType::kFLOAT, &weight_scale_value, 1});
        // auto policy_conv_weight_float = network->addConstant(Dims4{4, 128, 1, 1}, weightMap["policy_head.0.weight"]);
        // auto policy_conv_weight_q = network->addQuantize(*policy_conv_weight_float->getOutput(0), *policy_conv_weight_scale->getOutput(0), DataType::kINT8);
        // policy_conv_weight_q->setAxis(0);
        // auto policy_conv_weight_dq = network->addDequantize(*policy_conv_weight_q->getOutput(0), *policy_conv_weight_scale->getOutput(0), DataType::kHALF);
        // policy_conv_weight_dq->setAxis(0);
        // auto policy_conv_bias_scale = network->addConstant(Dims{1, {1}}, Weights{DataType::kFLOAT, &output_scale_value, 1});
        // auto policy_conv_bias_float = network->addConstant(Dims{1, {4}}, weightMap["policy_head.0.bias"]);
        // auto policy_conv_bias_q = network->addQuantize(*policy_conv_bias_float->getOutput(0), *policy_conv_bias_scale->getOutput(0), DataType::kINT8);
        // policy_conv_bias_q->setAxis(0);
        // auto policy_conv_bias_dq = network->addDequantize(*policy_conv_bias_q->getOutput(0), *policy_conv_bias_scale->getOutput(0), DataType::kHALF);
        // policy_conv_bias_dq->setAxis(0);

        // IConvolutionLayer* policy_conv = network->addConvolutionNd(*relu3_output_dq->getOutput(0), 4, DimsHW{1, 1}, emptywts, emptywts);
        // assert(policy_conv);
        // policy_conv->setStrideNd(DimsHW{1, 1});
        // policy_conv->setPaddingNd(DimsHW{0, 0});
        // policy_conv->setInput(1, *policy_conv_weight_dq->getOutput(0));
        // policy_conv->setInput(2, *policy_conv_bias_dq->getOutput(0));
    
        // IActivationLayer* policy_relu = network->addActivation(*policy_conv->getOutput(0), ActivationType::kRELU);
        // assert(policy_relu);
    
        // IShuffleLayer* policy_flatten = network->addShuffle(*policy_relu->getOutput(0));
        // policy_flatten->setReshapeDimensions(Dims2{-1, 4 * ACTION_SIZE});

        // auto policy_flatten_scale = network->addConstant(Dims{1, {1}}, Weights{DataType::kFLOAT, &output_scale_value, 1});
        // auto policy_flatten_q = network->addQuantize(*policy_flatten->getOutput(0), *policy_flatten_scale->getOutput(0), DataType::kINT8);
        // policy_flatten_q->setAxis(0);
        // auto policy_flatten_dq = network->addDequantize(*policy_flatten_q->getOutput(0), *policy_flatten_scale->getOutput(0), DataType::kHALF);
        // policy_flatten_dq->setAxis(0);

        // // auto policy_weights = network->addConstant(Dims2(ACTION_SIZE, 4 * ACTION_SIZE), weightMap["policy_head.3.weight"]);
        // // auto policy_bias = network->addConstant(Dims2(1, ACTION_SIZE), weightMap["policy_head.3.bias"]);

        // auto policy_weights_scale = network->addConstant(Dims{1, {1}}, Weights{DataType::kFLOAT, &weight_scale_value, 1});
        // auto policy_weights_float = network->addConstant(Dims2(ACTION_SIZE, 4 * ACTION_SIZE), weightMap["policy_head.3.weight"]);
        // auto policy_weights_q = network->addQuantize(*policy_weights_float->getOutput(0), *policy_weights_scale->getOutput(0), DataType::kINT8);
        // policy_weights_q->setAxis(0);
        // auto policy_weights_dq = network->addDequantize(*policy_weights_q->getOutput(0), *policy_weights_scale->getOutput(0), DataType::kHALF);
        // policy_weights_dq->setAxis(0);
        // auto policy_bias_scale = network->addConstant(Dims{1, {1}}, Weights{DataType::kFLOAT, &output_scale_value, 1});
        // auto policy_bias_float = network->addConstant(Dims{2, {1, ACTION_SIZE}}, weightMap["policy_head.3.bias"]);
        // auto policy_bias_q = network->addQuantize(*policy_bias_float->getOutput(0), *policy_bias_scale->getOutput(0), DataType::kINT8);
        // policy_bias_q->setAxis(0);
        // auto policy_bias_dq = network->addDequantize(*policy_bias_q->getOutput(0), *policy_bias_scale->getOutput(0), DataType::kHALF);
        // policy_bias_dq->setAxis(0);

        // auto policy_mm = network->addMatrixMultiply(*policy_flatten_dq->getOutput(0), MatrixOperation::kNONE, *policy_weights_dq->getOutput(0), MatrixOperation::kTRANSPOSE);
        // auto policy_fc = network->addElementWise(*policy_mm->getOutput(0), *policy_bias_dq->getOutput(0), ElementWiseOperation::kSUM);
        // assert(policy_fc);

        // auto policy_fc_scale = network->addConstant(Dims{1, {1}}, Weights{DataType::kFLOAT, &output_scale_value, 1});
        // auto policy_fc_q = network->addQuantize(*policy_fc->getOutput(0), *policy_fc_scale->getOutput(0), DataType::kINT8);
        // policy_fc_q->setAxis(0);
        // auto policy_fc_dq = network->addDequantize(*policy_fc_q->getOutput(0), *policy_fc_scale->getOutput(0), DataType::kHALF);
        // policy_fc_dq->setAxis(0);
    
        // ISoftMaxLayer* policy_softmax = network->addSoftMax(*policy_fc_dq->getOutput(0));
        // policy_softmax->setAxes(1 << 1); 
        
        // assert(policy_softmax);
    
        // // Value head
        // auto value_conv_weight_scale = network->addConstant(Dims{1, {1}}, Weights{DataType::kFLOAT, &weight_scale_value, 1});
        // auto value_conv_weight_float = network->addConstant(Dims4{2, 128, 1, 1}, weightMap["value_head.0.weight"]);
        // auto value_conv_weight_q = network->addQuantize(*value_conv_weight_float->getOutput(0), *value_conv_weight_scale->getOutput(0), DataType::kINT8);
        // value_conv_weight_q->setAxis(0);
        // auto value_conv_weight_dq = network->addDequantize(*value_conv_weight_q->getOutput(0), *value_conv_weight_scale->getOutput(0), DataType::kHALF);
        // value_conv_weight_dq->setAxis(0);
        // auto value_conv_bias_scale = network->addConstant(Dims{1, {1}}, Weights{DataType::kFLOAT, &output_scale_value, 1});
        // auto value_conv_bias_float = network->addConstant(Dims{1, {2}}, weightMap["value_head.0.bias"]);
        // auto value_conv_bias_q = network->addQuantize(*value_conv_bias_float->getOutput(0), *value_conv_bias_scale->getOutput(0), DataType::kINT8);
        // value_conv_bias_q->setAxis(0);
        // auto value_conv_bias_dq = network->addDequantize(*value_conv_bias_q->getOutput(0), *value_conv_bias_scale->getOutput(0), DataType::kHALF);
        // value_conv_bias_dq->setAxis(0);

        // IConvolutionLayer* value_conv = network->addConvolutionNd(*relu3_output_dq->getOutput(0), 2, DimsHW{1, 1}, emptywts, emptywts);
        // assert(value_conv);
        // value_conv->setStrideNd(DimsHW{1, 1});
        // value_conv->setPaddingNd(DimsHW{0, 0});
        // value_conv->setInput(1, *value_conv_weight_dq->getOutput(0));
        // value_conv->setInput(2, *value_conv_bias_dq->getOutput(0));
    
        // IActivationLayer* value_relu1 = network->addActivation(*value_conv->getOutput(0), ActivationType::kRELU);
        // assert(value_relu1);

        // IShuffleLayer* value_flatten = network->addShuffle(*value_relu1->getOutput(0));
        // value_flatten->setReshapeDimensions(Dims2{-1, 2 * ACTION_SIZE});

        // auto value_flatten_scale = network->addConstant(Dims{1, {1}}, Weights{DataType::kFLOAT, &output_scale_value, 1});
        // auto value_flatten_q = network->addQuantize(*value_flatten->getOutput(0), *value_flatten_scale->getOutput(0), DataType::kINT8);
        // value_flatten_q->setAxis(0);
        // auto value_flatten_dq = network->addDequantize(*value_flatten_q->getOutput(0), *value_flatten_scale->getOutput(0), DataType::kHALF);
        // value_flatten_dq->setAxis(0);

        // auto value_fc1_weights_scale = network->addConstant(Dims{1, {1}}, Weights{DataType::kFLOAT, &weight_scale_value, 1});
        // auto value_fc1_weights_float = network->addConstant(Dims2(ACTION_SIZE, 2 * ACTION_SIZE), weightMap["value_head.3.weight"]);
        // auto value_fc1_weights_q = network->addQuantize(*value_fc1_weights_float->getOutput(0), *value_fc1_weights_scale->getOutput(0), DataType::kINT8);
        // value_fc1_weights_q->setAxis(0);
        // auto value_fc1_weights_dq = network->addDequantize(*value_fc1_weights_q->getOutput(0), *value_fc1_weights_scale->getOutput(0), DataType::kHALF);
        // value_fc1_weights_dq->setAxis(0);
        // auto value_fc1_bias_scale = network->addConstant(Dims{1, {1}}, Weights{DataType::kFLOAT, &output_scale_value, 1});
        // auto value_fc1_bias_float = network->addConstant(Dims2(ACTION_SIZE, 1), weightMap["value_head.3.bias"]);
        // auto value_fc1_bias_q = network->addQuantize(*value_fc1_bias_float->getOutput(0), *value_fc1_bias_scale->getOutput(0), DataType::kINT8);
        // value_fc1_bias_q->setAxis(0);
        // auto value_fc1_bias_dq = network->addDequantize(*value_fc1_bias_q->getOutput(0), *value_fc1_bias_scale->getOutput(0), DataType::kHALF);
        // value_fc1_bias_dq->setAxis(0);
        
        // auto value_mm1 = network->addMatrixMultiply(*value_fc1_weights_dq->getOutput(0), MatrixOperation::kNONE, *value_flatten_dq->getOutput(0), MatrixOperation::kTRANSPOSE);
        // auto value_fc1 = network->addElementWise(*value_mm1->getOutput(0), *value_fc1_bias_dq->getOutput(0), ElementWiseOperation::kSUM);
        // assert(value_fc1);
    
        // IActivationLayer* value_relu2 = network->addActivation(*value_fc1->getOutput(0), ActivationType::kRELU);
        // assert(value_relu2);

        // auto value_relu2_output_scale = network->addConstant(Dims{1, {1}}, Weights{DataType::kFLOAT, &output_scale_value, 1});
        // auto value_relu2_output_q = network->addQuantize(*value_relu2->getOutput(0), *value_relu2_output_scale->getOutput(0), DataType::kINT8);
        // value_relu2_output_q->setAxis(0);
        // auto value_relu2_output_dq = network->addDequantize(*value_relu2_output_q->getOutput(0), *value_relu2_output_scale->getOutput(0), DataType::kHALF);
        // value_relu2_output_dq->setAxis(0);

        // // auto value_fc2_weights_scale = network->addConstant(Dims{1, {1}}, Weights{DataType::kFLOAT, &weight_scale_value, 1});
        // // auto value_fc2_weights_float = network->addConstant(Dims2(1, ACTION_SIZE), weightMap["value_head.5.weight"]);
        // // auto value_fc2_weights_q = network->addQuantize(*value_fc2_weights_float->getOutput(0), *value_fc2_weights_scale->getOutput(0), DataType::kINT8);
        // // value_fc2_weights_q->setAxis(0);
        // // auto value_fc2_weights_dq = network->addDequantize(*value_fc2_weights_q->getOutput(0), *value_fc2_weights_scale->getOutput(0), DataType::kHALF);
        // // value_fc2_weights_dq->setAxis(0);
        // // auto value_fc2_bias_scale = network->addConstant(Dims{1, {1}}, Weights{DataType::kFLOAT, &output_scale_value, 1});
        // // auto value_fc2_bias_float = network->addConstant(Dims2(1, 1), weightMap["value_head.5.bias"]);
        // // auto value_fc2_bias_q = network->addQuantize(*value_fc2_bias_float->getOutput(0), *value_fc2_bias_scale->getOutput(0), DataType::kINT8);
        // // value_fc2_bias_q->setAxis(0);
        // // auto value_fc2_bias_dq = network->addDequantize(*value_fc2_bias_q->getOutput(0), *value_fc2_bias_scale->getOutput(0), DataType::kHALF);
        // // value_fc2_bias_dq->setAxis(0);
        // auto value_fc2_weights = network->addConstant(Dims2(1, ACTION_SIZE), weightMap["value_head.5.weight"]);
        // auto value_fc2_bias = network->addConstant(Dims2(1, 1), weightMap["value_head.5.bias"]);
    
        // auto value_mm2 = network->addMatrixMultiply(*value_fc2_weights->getOutput(0), MatrixOperation::kNONE, *value_relu2_output_dq->getOutput(0), MatrixOperation::kNONE);
        // auto value_fc2 = network->addElementWise(*value_mm2->getOutput(0), *value_fc2_bias->getOutput(0), ElementWiseOperation::kSUM);
        // assert(value_fc2);

        // // auto value_fc2_scale = network->addConstant(Dims{1, {1}}, Weights{DataType::kFLOAT, &output_scale_value, 1});
        // // auto value_fc2_q = network->addQuantize(*value_fc2->getOutput(0), *value_fc2_scale->getOutput(0), DataType::kINT8);
        // // value_fc2_q->setAxis(0);
        // // auto value_fc2_dq = network->addDequantize(*value_fc2_q->getOutput(0), *value_fc2_scale->getOutput(0), DataType::kHALF);
        // // value_fc2_dq->setAxis(0);
    
        // IActivationLayer* value_tanh = network->addActivation(*value_fc2->getOutput(0), ActivationType::kTANH);
        // assert(value_tanh);
    
        // policy_softmax->getOutput(0)->setName("policy");
        // value_tanh->getOutput(0)->setName("value");
        // network->markOutput(*policy_softmax->getOutput(0));
        // network->markOutput(*value_tanh->getOutput(0));
    
        // Build engine

        ICudaEngine* _engine = builder->buildEngineWithConfig(*network, *config);
    
        delete network;
        for (auto& mem : weightMap)
        {
            free((void*) (mem.second.values));
        }
    
        return _engine;
    }

    static void save_engine(ICudaEngine* engine) {
        if(engine == nullptr) {
            std::cerr << "[tensorRTComponent] engine is null" << std::endl;
            exit(0);
        }

        IHostMemory* modelStream{nullptr};
        modelStream = engine->serialize();
        assert(modelStream != nullptr);
        std::ofstream p("alphazero.engine", std::ios::binary);
        if (!p)
        {
            std::cerr << "[tensorRTComponent] could not open plan output file" << std::endl;
            exit(0);
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        p.close();
        delete modelStream;
    }

    void update_engine() {

        if(context != nullptr) {
            delete context;
            context = nullptr;
        }
        
        if(engine != nullptr) {
            delete engine;
            engine = nullptr;
        }

        engine = createEngine(builder, config);
        assert(engine != nullptr);

        context = engine->createExecutionContext();
        assert(context != nullptr);
        context->setTensorAddress("input", buffers[0]);
        context->setTensorAddress("policy", buffers[1]);
        context->setTensorAddress("value", buffers[2]);
    }

    void predict(int batch_size, float* input, float* pi, float* val)
    {
        {
            std::lock_guard<std::mutex> lock(stream_mutex);

            context->setOptimizationProfileAsync(0, stream);
            context->setInputShape("input", Dims4(batch_size, PLANES_SIZE, BOARD_SIZE, BOARD_SIZE));
            CUDA_CHECK(cudaMemcpyAsync(buffers[0], input, batch_size * PLANES_SIZE * BOARD_SIZE * BOARD_SIZE * sizeof(float), cudaMemcpyHostToDevice, stream));
            context->setTensorAddress("input", buffers[0]);
            context->setTensorAddress("policy", buffers[1]);  
            context->setTensorAddress("value", buffers[2]);
            bool success = context->enqueueV3(stream);
            if (!success) {
                printf("Error: enqueueV3 failed with batch_size %d\n", batch_size);
                return;
            }

            CUDA_CHECK(cudaMemcpyAsync(pi, buffers[1], batch_size * ACTION_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(val, buffers[2], batch_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
            cudaStreamSynchronize(stream);
        }
    }


    void predict_without_transfer(
        int batch_size, 
        void* input, 
        void* pi, 
        void* val, 
        cudaStream_t& transfer_stream
    )
    {
        {
            std::lock_guard<std::mutex> lock(stream_mutex);
            cudaStreamSynchronize(transfer_stream);
            context->setOptimizationProfileAsync(0, stream);
            context->setInputShape("input", Dims4(batch_size, PLANES_SIZE, BOARD_SIZE, BOARD_SIZE));
            // CUDA_CHECK(cudaMemcpyAsync(buffers[0], input, batch_size * PLANES_SIZE * BOARD_SIZE * BOARD_SIZE * sizeof(float), cudaMemcpyHostToDevice, stream));
            context->setTensorAddress("input", input);
            context->setTensorAddress("policy", pi);  
            context->setTensorAddress("value", val);
            bool success = context->enqueueV3(stream);
            if (!success) {
                printf("Error: enqueueV3 failed with batch_size %d\n", batch_size);
                // Dims dims = context->getTensorShape("input");
                // printf("Input shape: %d %d %d %d\n", dims.d[0], dims.d[1], dims.d[2], dims.d[3]);
                // fflush(stdout);
                return;
            }

            // CUDA_CHECK(cudaMemcpyAsync(pi, buffers[1], batch_size * ACTION_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
            // CUDA_CHECK(cudaMemcpyAsync(val, buffers[2], batch_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
            cudaStreamSynchronize(stream);
        }
    }
};

#endif