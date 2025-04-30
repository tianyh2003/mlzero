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
    Logger* logger = new Logger(LOG_RANK, "tensorrt.log");
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
        
        profile = builder->createOptimizationProfile();
        
        profile->setDimensions("input", OptProfileSelector::kMIN, Dims4(1, PLANES_SIZE, BOARD_SIZE, BOARD_SIZE));
        profile->setDimensions("input", OptProfileSelector::kOPT, Dims4(PREDICT_BUFFER_SIZE, PLANES_SIZE, BOARD_SIZE, BOARD_SIZE));
        profile->setDimensions("input", OptProfileSelector::kMAX, Dims4(TENSORRT_MAX_BATCH_SIZE, PLANES_SIZE, BOARD_SIZE, BOARD_SIZE));
        config->addOptimizationProfile(profile);

        update_engine();
        
        logger->log(5, "[tensorRTComponent] update engine finished");
    }
    
    ~TensorRTComponent() {

        delete logger;

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
        }
        
        if(USE_GPU) 
            net->to(torch::kCUDA);
        logger->log(5, "[tensorRTComponent] load weight finished");
        return weightMap;
    }

    IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps) {
        float *gamma = (float*)weightMap[lname + ".weight"].values;
        float *beta = (float*)weightMap[lname + ".bias"].values;
        float *mean = (float*)weightMap[lname + ".running_mean"].values;
        float *var = (float*)weightMap[lname + ".running_var"].values;
        int len = weightMap[lname + ".running_var"].count;
    
        float *scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
        for (int i = 0; i < len; i++) {
            scval[i] = gamma[i] / sqrt(var[i] + eps);
        }
        Weights scale{DataType::kFLOAT, scval, len};
        
        float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
        for (int i = 0; i < len; i++) {
            shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
        }
        Weights shift{DataType::kFLOAT, shval, len};
    
        float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
        for (int i = 0; i < len; i++) {
            pval[i] = 1.0;
        }
        Weights power{DataType::kFLOAT, pval, len};
    
        weightMap[lname + ".scale"] = scale;
        weightMap[lname + ".shift"] = shift;
        weightMap[lname + ".power"] = power;
        IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
        assert(scale_1);
        return scale_1;
    }

    IActivationLayer* basicBlock(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int inch, int outch, int stride, std::string lname) {
        Weights emptywts{DataType::kFLOAT, nullptr, 0};
    
        IConvolutionLayer* conv1 = network->addConvolutionNd(input, NUM_CHANNELS, DimsHW{3, 3}, weightMap[lname + "conv1.weight"], emptywts);
        assert(conv1);
        conv1->setStrideNd(DimsHW{1, 1});
        conv1->setPaddingNd(DimsHW{1, 1});
    
        IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "bn1", 1e-5);
    
        IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
        assert(relu1);
    
        IConvolutionLayer* conv2 = network->addConvolutionNd(*relu1->getOutput(0), NUM_CHANNELS, DimsHW{3, 3}, weightMap[lname + "conv2.weight"], emptywts);
        assert(conv2);
        conv2->setStrideNd(DimsHW{1, 1});
        conv2->setPaddingNd(DimsHW{1, 1});
    
        IScaleLayer* bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + "bn2", 1e-5);
    
        IElementWiseLayer* ew1;
        ew1 = network->addElementWise(input, *bn2->getOutput(0), ElementWiseOperation::kSUM);
    
        IActivationLayer* relu2 = network->addActivation(*ew1->getOutput(0), ActivationType::kRELU);
        assert(relu2);
        return relu2;
    }

    ICudaEngine* createEngine(IBuilder* builder, IBuilderConfig* config)
    {
        if(NUM_RES_BLOCKS != 3) {
            logger->log(0, "[tensorRTComponent] ERROR: NUM_RES_BLOCKS must be 3");
            exit(1);
        }

        DataType dt = DataType::kFLOAT;

        INetworkDefinition* network = builder->createNetworkV2(0U);
    
        std::map<std::string, Weights> weightMap = loadWeights(); //need to free
        Weights emptywts{dt, nullptr, 0};
    
        ITensor* data = network->addInput("input", dt, Dims4{-1, PLANES_SIZE, BOARD_SIZE, BOARD_SIZE});
        assert(data);
    
        IConvolutionLayer* conv1 = network->addConvolutionNd(*data, NUM_CHANNELS, DimsHW{3, 3}, weightMap["conv1.weight"], emptywts);
        assert(conv1);
        conv1->setStrideNd(DimsHW{1, 1});
        conv1->setPaddingNd(DimsHW{1, 1});
    
        IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), "bn1", 1e-5);
    
        IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
        assert(relu1);
    
        IActivationLayer* relu2 = basicBlock(network, weightMap, *relu1->getOutput(0), NUM_CHANNELS, NUM_CHANNELS, 1, "layer1.0.");
    
        IActivationLayer* relu3 = basicBlock(network, weightMap, *relu2->getOutput(0), NUM_CHANNELS, NUM_CHANNELS, 1, "layer1.1.");
    
        IActivationLayer* relu4 = basicBlock(network, weightMap, *relu3->getOutput(0), NUM_CHANNELS, NUM_CHANNELS, 1, "layer1.2.");
    
        // Policy head
        IConvolutionLayer* policy_conv = network->addConvolutionNd(*relu4->getOutput(0), 4, DimsHW{1, 1}, weightMap["policy_head.0.weight"], emptywts);
        assert(policy_conv);
        policy_conv->setStrideNd(DimsHW{1, 1});
        policy_conv->setPaddingNd(DimsHW{0, 0});
    
        IScaleLayer* policy_bn = addBatchNorm2d(network, weightMap, *policy_conv->getOutput(0), "policy_head.1", 1e-5);
        IActivationLayer* policy_relu = network->addActivation(*policy_bn->getOutput(0), ActivationType::kRELU);
        assert(policy_relu);
    
        IShuffleLayer* policy_flatten = network->addShuffle(*policy_relu->getOutput(0));
        policy_flatten->setReshapeDimensions(Dims2{-1, 4 * ACTION_SIZE});
    
        auto policy_weights = network->addConstant(Dims2(ACTION_SIZE, 4 * ACTION_SIZE), weightMap["policy_head.4.weight"]);
        auto policy_bias = network->addConstant(Dims2(1, ACTION_SIZE), weightMap["policy_head.4.bias"]);

        auto policy_mm = network->addMatrixMultiply(*policy_flatten->getOutput(0), MatrixOperation::kNONE, *policy_weights->getOutput(0), MatrixOperation::kTRANSPOSE);
        auto policy_fc = network->addElementWise(*policy_mm->getOutput(0), *policy_bias->getOutput(0), ElementWiseOperation::kSUM);
        assert(policy_fc);
    
        ISoftMaxLayer* policy_softmax = network->addSoftMax(*policy_fc->getOutput(0));
        policy_softmax->setAxes(1 << 1); 
        
        assert(policy_softmax);
    
        // Value head
        IConvolutionLayer* value_conv = network->addConvolutionNd(*relu4->getOutput(0), 2, DimsHW{1, 1}, weightMap["value_head.0.weight"], emptywts);
        assert(value_conv);
        value_conv->setStrideNd(DimsHW{1, 1});
        value_conv->setPaddingNd(DimsHW{0, 0});
    
        IScaleLayer* value_bn = addBatchNorm2d(network, weightMap, *value_conv->getOutput(0), "value_head.1", 1e-5);
        IActivationLayer* value_relu1 = network->addActivation(*value_bn->getOutput(0), ActivationType::kRELU);
        assert(value_relu1);
    
        IShuffleLayer* value_flatten = network->addShuffle(*value_relu1->getOutput(0));
        value_flatten->setReshapeDimensions(Dims2{-1, 2 * ACTION_SIZE});
    
        auto value_fc1_weights = network->addConstant(Dims2(ACTION_SIZE, 2 * ACTION_SIZE), weightMap["value_head.4.weight"]);
        auto value_fc1_bias = network->addConstant(Dims2(ACTION_SIZE, 1), weightMap["value_head.4.bias"]);
        
        auto value_mm1 = network->addMatrixMultiply(*value_fc1_weights->getOutput(0), MatrixOperation::kNONE, *value_flatten->getOutput(0), MatrixOperation::kTRANSPOSE);
        auto value_fc1 = network->addElementWise(*value_mm1->getOutput(0), *value_fc1_bias->getOutput(0), ElementWiseOperation::kSUM);
        assert(value_fc1);
    
        IActivationLayer* value_relu2 = network->addActivation(*value_fc1->getOutput(0), ActivationType::kRELU);
        assert(value_relu2);
    
        auto value_fc2_weights = network->addConstant(Dims2(1, ACTION_SIZE), weightMap["value_head.6.weight"]);
        auto value_fc2_bias = network->addConstant(Dims2(1, 1), weightMap["value_head.6.bias"]);
    
        auto value_mm2 = network->addMatrixMultiply(*value_fc2_weights->getOutput(0), MatrixOperation::kNONE, *value_relu2->getOutput(0), MatrixOperation::kNONE);
        auto value_fc2 = network->addElementWise(*value_mm2->getOutput(0), *value_fc2_bias->getOutput(0), ElementWiseOperation::kSUM);
        assert(value_fc2);
    
        IActivationLayer* value_tanh = network->addActivation(*value_fc2->getOutput(0), ActivationType::kTANH);
        assert(value_tanh);
    
        policy_softmax->getOutput(0)->setName("policy");
        value_tanh->getOutput(0)->setName("value");
        network->markOutput(*policy_softmax->getOutput(0));
        network->markOutput(*value_tanh->getOutput(0));
    
        // Build engine

        ICudaEngine* _engine = builder->buildEngineWithConfig(*network, *config);

        logger->log(5, "[tensorRTComponent] build engine finished");
    
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

    void check_mem() {
        FILE* fp = fopen("/proc/self/status", "r");
        char line[128];
        while (fgets(line, 128, fp) != NULL)
        {
            if (strncmp(line, "VmRSS:", 6) == 0)
            {
                logger->log(5, "[tensorRTComponent] 当前进程占用内存大小为:" + std::string(line + 6));
                logger->flush();
                break;
            }
        }
        fclose(fp);
    }

    void update_engine() {
        logger->log(5, "[tensorRTComponent] start update engine");

        if(context != nullptr) {
            delete context;
            context = nullptr;
        }
        
        if(engine != nullptr) {
            delete engine;
            engine = nullptr;
        }

        // logger->log(5, "[tensorRTComponent] end destroy old");
        // check_mem();

        // logger->log(5, "[tensorRTComponent] start create engine");
        // check_mem();

        engine = createEngine(builder, config);
        assert(engine != nullptr);

        // logger->log(5, "[tensorRTComponent] end create engine");
        // check_mem();

        // logger->log(5, "[tensorRTComponent] start create context");
        // check_mem();

        context = engine->createExecutionContext();
        assert(context != nullptr);
        context->setTensorAddress("input", buffers[0]);
        context->setTensorAddress("policy", buffers[1]);
        context->setTensorAddress("value", buffers[2]);

        // logger->log(5, "[tensorRTComponent] end create context");
        // check_mem();

        // std::thread thread_to_save_engine(save_engine, engine); 
        // thread_to_save_engine.detach();
        // save_engine();
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
                // Dims dims = context->getTensorShape("input");
                // printf("Input shape: %d %d %d %d\n", dims.d[0], dims.d[1], dims.d[2], dims.d[3]);
                // fflush(stdout);
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