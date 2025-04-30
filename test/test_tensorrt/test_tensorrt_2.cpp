
#include "../network_wrapper.hpp"
#include "../common.h"
#include "../logger.hpp"
#include "NvInfer.h"
#include "cuda_runtime_api.h"

using namespace nvinfer1;

const nvinfer1::BuilderFlag TENSORRT_PREDICT_DATA_TYPE = nvinfer1::BuilderFlag::kTF32;

#define CUDA_CHECK(call)                                   \
do                                                    \
{                                                     \
    const cudaError_t error_code = call;              \
    if (error_code != cudaSuccess)                    \
    {                                                 \
        printf("CUDA Error:\n");                      \
        printf("    File:       %s\n", __FILE__);     \
        printf("    Line:       %d\n", __LINE__);     \
        printf("    Error code: %d\n", error_code);   \
        printf("    Error text: %s\n",                \
            cudaGetErrorString(error_code));          \
        exit(1);                                      \
    }                                                 \
} while (0)

class TenserRTLogger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override
    {
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
} logger;

class TensorRTComponent {
public:
    TenserRTLogger* tensorrt_logger;
    NetworkWrapper* network_wrapper;
    ICudaEngine* engine = nullptr;

    Timer predict_timer = Timer();
    Timer transfer_timer = Timer();

    void* buffers[5];
    cudaStream_t stream;
    IExecutionContext *context = nullptr;

    TensorRTComponent(
        NetworkWrapper* network_wrapper
    ) : network_wrapper(network_wrapper) {
        tensorrt_logger = new TenserRTLogger();
        CUDA_CHECK(cudaStreamCreate(&stream));
        CUDA_CHECK(cudaMalloc(&buffers[0], PREDICT_BUFFER_SIZE * PLANES_SIZE * BOARD_SIZE * BOARD_SIZE * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&buffers[1], PREDICT_BUFFER_SIZE * ACTION_SIZE * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&buffers[2], PREDICT_BUFFER_SIZE * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&buffers[3], NUM_FILTERS * PREDICT_BUFFER_SIZE * ACTION_SIZE * sizeof(float)));
        update_engine();
        printf("update engine finished\n");
        fflush(stdout);
    }
    
    ~TensorRTComponent() {
        delete tensorrt_logger;
        if(context != nullptr) {
            delete context;
        }
        if(engine != nullptr) {
            delete engine;
        }
    }

    std::map<std::string, Weights> loadWeights() {
        AlphaZeroResNetImpl* net = network_wrapper->get_net();
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
            
            std::cout << "name: " << name << std::endl;
            std::cout << "size: " << pair.value().sizes() << std::endl;

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

            std::cout << "name: " << name << std::endl;
            std::cout << "size: " << pair.value().sizes() << std::endl;

        }
        net->to(torch::kCUDA);
        network_wrapper->logger->log(5, "[tensorRTComponent] load weight finished");
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
    
        IConvolutionLayer* conv1 = network->addConvolutionNd(input, NUM_FILTERS, DimsHW{3, 3}, weightMap[lname + "conv1.weight"], emptywts);
        assert(conv1);
        conv1->setStrideNd(DimsHW{1, 1});
        conv1->setPaddingNd(DimsHW{1, 1});
    
        IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "bn1", 1e-5);
    
        IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
        assert(relu1);
    
        IConvolutionLayer* conv2 = network->addConvolutionNd(*relu1->getOutput(0), NUM_FILTERS, DimsHW{3, 3}, weightMap[lname + "conv2.weight"], emptywts);
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
        if(NUM_RES_BLOCKS != 7) {
            network_wrapper->logger->log(0, "[tensorRTComponent] NUM_RES_BLOCKS must be 7");
            exit(1);
        }

        DataType dt = DataType::kFLOAT;

        INetworkDefinition* network = builder->createNetworkV2(0U);
    
        std::map<std::string, Weights> weightMap = loadWeights();
        Weights emptywts{dt, nullptr, 0};
    
        ITensor* data = network->addInput("input", dt, Dims4{PREDICT_BUFFER_SIZE, PLANES_SIZE, BOARD_SIZE, BOARD_SIZE});
        assert(data);
    
        IConvolutionLayer* conv1 = network->addConvolutionNd(*data, NUM_FILTERS, DimsHW{3, 3}, weightMap["conv1.weight"], emptywts);
        assert(conv1);
        conv1->setStrideNd(DimsHW{1, 1});
        conv1->setPaddingNd(DimsHW{1, 1});
    
        IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), "bn1", 1e-5);
    
        IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
        assert(relu1);
    
        IActivationLayer* relu2 = basicBlock(network, weightMap, *relu1->getOutput(0), NUM_FILTERS, NUM_FILTERS, 1, "layer1.0.");
    
        IActivationLayer* relu3 = basicBlock(network, weightMap, *relu2->getOutput(0), NUM_FILTERS, NUM_FILTERS, 1, "layer1.1.");
    
        IActivationLayer* relu4 = basicBlock(network, weightMap, *relu3->getOutput(0), NUM_FILTERS, NUM_FILTERS, 1, "layer1.2.");
    
        IActivationLayer* relu5 = basicBlock(network, weightMap, *relu4->getOutput(0), NUM_FILTERS, NUM_FILTERS, 1, "layer1.3.");
    
        IActivationLayer* relu6 = basicBlock(network, weightMap, *relu5->getOutput(0), NUM_FILTERS, NUM_FILTERS, 1, "layer1.4.");
    
        IActivationLayer* relu7 = basicBlock(network, weightMap, *relu6->getOutput(0), NUM_FILTERS, NUM_FILTERS, 1, "layer1.5.");
    
        IActivationLayer* relu8 = basicBlock(network, weightMap, *relu7->getOutput(0), NUM_FILTERS, NUM_FILTERS, 1, "layer1.6.");
    
        // Policy head
        // IConvolutionLayer* policy_conv = network->addConvolutionNd(*relu8->getOutput(0), 2, DimsHW{1, 1}, weightMap["policy_head.0.weight"], emptywts);
        IConvolutionLayer* policy_conv = network->addConvolutionNd(*relu8->getOutput(0), 2, DimsHW{1, 1}, weightMap["policy_head_1.0.weight"], emptywts);
        assert(policy_conv);
        policy_conv->setStrideNd(DimsHW{1, 1});
        policy_conv->setPaddingNd(DimsHW{0, 0});
    
        IScaleLayer* policy_bn = addBatchNorm2d(network, weightMap, *policy_conv->getOutput(0), "policy_head_1.1", 1e-5);
        IActivationLayer* policy_relu = network->addActivation(*policy_bn->getOutput(0), ActivationType::kRELU);
        assert(policy_relu);
    
        IShuffleLayer* policy_flatten = network->addShuffle(*policy_relu->getOutput(0));
        policy_flatten->setReshapeDimensions(Dims2{PREDICT_BUFFER_SIZE, 2 * ACTION_SIZE});
    
        auto policy_weights = network->addConstant(Dims2(ACTION_SIZE, 2 * ACTION_SIZE), weightMap["policy_head_3.0.weight"]);
        auto policy_bias = network->addConstant(Dims2(1, ACTION_SIZE), weightMap["policy_head_3.0.bias"]);

        auto policy_mm = network->addMatrixMultiply(*policy_flatten->getOutput(0), MatrixOperation::kNONE, *policy_weights->getOutput(0), MatrixOperation::kTRANSPOSE);
        auto policy_fc = network->addElementWise(*policy_mm->getOutput(0), *policy_bias->getOutput(0), ElementWiseOperation::kSUM);
        assert(policy_fc);

        // auto policy_mm = network->addMatrixMultiply(*policy_weights->getOutput(0), MatrixOperation::kNONE, *policy_flatten->getOutput(0), MatrixOperation::kTRANSPOSE);
        // auto policy_fc = network->addElementWise(*policy_mm->getOutput(0), *policy_bias->getOutput(0), ElementWiseOperation::kSUM);
        // assert(policy_fc);
    
        ISoftMaxLayer* policy_softmax = network->addSoftMax(*policy_fc->getOutput(0));
        policy_softmax->setAxes(1 << 1); 
        assert(policy_softmax);
    
        // Value head
        IConvolutionLayer* value_conv = network->addConvolutionNd(*relu8->getOutput(0), 1, DimsHW{1, 1}, weightMap["value_head.0.weight"], emptywts);
        assert(value_conv);
        value_conv->setStrideNd(DimsHW{1, 1});
        value_conv->setPaddingNd(DimsHW{0, 0});
    
        IScaleLayer* value_bn = addBatchNorm2d(network, weightMap, *value_conv->getOutput(0), "value_head.1", 1e-5);
        IActivationLayer* value_relu1 = network->addActivation(*value_bn->getOutput(0), ActivationType::kRELU);
        assert(value_relu1);
    
        IShuffleLayer* value_flatten = network->addShuffle(*value_relu1->getOutput(0));
        value_flatten->setReshapeDimensions(Dims2{-1, ACTION_SIZE});
    
        auto value_fc1_weights = network->addConstant(Dims2(NUM_FC_UNITS, ACTION_SIZE), weightMap["value_head.4.weight"]);
        auto value_fc1_bias = network->addConstant(Dims2(NUM_FC_UNITS, 1), weightMap["value_head.4.bias"]);
        
        auto value_mm1 = network->addMatrixMultiply(*value_fc1_weights->getOutput(0), MatrixOperation::kNONE, *value_flatten->getOutput(0), MatrixOperation::kTRANSPOSE);
        auto value_fc1 = network->addElementWise(*value_mm1->getOutput(0), *value_fc1_bias->getOutput(0), ElementWiseOperation::kSUM);
        assert(value_fc1);
    
        IActivationLayer* value_relu2 = network->addActivation(*value_fc1->getOutput(0), ActivationType::kRELU);
        assert(value_relu2);
    
        auto value_fc2_weights = network->addConstant(Dims2(1, NUM_FC_UNITS), weightMap["value_head.6.weight"]);
        auto value_fc2_bias = network->addConstant(Dims2(1, 1), weightMap["value_head.6.bias"]);
    
        auto value_mm2 = network->addMatrixMultiply(*value_fc2_weights->getOutput(0), MatrixOperation::kNONE, *value_relu2->getOutput(0), MatrixOperation::kNONE);
        auto value_fc2 = network->addElementWise(*value_mm2->getOutput(0), *value_fc2_bias->getOutput(0), ElementWiseOperation::kSUM);
        assert(value_fc2);
    
        IActivationLayer* value_tanh = network->addActivation(*value_fc2->getOutput(0), ActivationType::kTANH);
        assert(value_tanh);
    
        policy_softmax->getOutput(0)->setName("policy");
        value_tanh->getOutput(0)->setName("value");
        // relu8->getOutput(0)->setName("test_x");
        policy_fc->getOutput(0)->setName("test_x");
        network->markOutput(*policy_softmax->getOutput(0));
        network->markOutput(*value_tanh->getOutput(0));
        // network->markOutput(*relu8->getOutput(0));
        network->markOutput(*policy_fc->getOutput(0));
    
        // Build engine
        config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1 << 28);
        config->setFlag(TENSORRT_PREDICT_DATA_TYPE);
        ICudaEngine* _engine = builder->buildEngineWithConfig(*network, *config);

        network_wrapper->logger->log(5, "[tensorRTComponent] build engine finished");
    
        delete network;
        for (auto& mem : weightMap)
        {
            free((void*) (mem.second.values));
        }
    
        return _engine;
    }

    void save_engine() {
        if(engine == nullptr) {
            network_wrapper->logger->log(0, "[tensorRTComponent] engine is null");
            exit(0);
        }

        IHostMemory* modelStream{nullptr};
        modelStream = engine->serialize();
        assert(modelStream != nullptr);
        std::ofstream p("alphazero.engine", std::ios::binary);
        if (!p)
        {
            network_wrapper->logger->log(0, "[tensorRTComponent] could not open plan output file");
            exit(0);
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        p.close();
        delete modelStream;
        
    }

    void update_engine() {
        IBuilder* builder = createInferBuilder(*tensorrt_logger);
        IBuilderConfig* config = builder->createBuilderConfig();
        if(engine != nullptr) {
            delete engine;
            engine = nullptr;
        }
        if(context != nullptr) {
            delete context;
            context = nullptr;
        }
        engine = createEngine(builder, config);
        assert(engine != nullptr);
        context = engine->createExecutionContext();
        assert(context != nullptr);

        context->setTensorAddress("input", buffers[0]);
        context->setTensorAddress("policy", buffers[1]);
        context->setTensorAddress("value", buffers[2]);
        context->setTensorAddress("test_x", buffers[3]);

        save_engine();

        delete config;
        delete builder;
    }

    void predict(float* input, float* pi, float* val)
    {
        // IExecutionContext *context = engine->createExecutionContext();
        // assert(context != nullptr);

        // void* buffers[3];

        // CUDA_CHECK(cudaMalloc(&buffers[0], PREDICT_BUFFER_SIZE * PLANES_SIZE * BOARD_SIZE * BOARD_SIZE * sizeof(float)));
        // CUDA_CHECK(cudaMalloc(&buffers[1], PREDICT_BUFFER_SIZE * ACTION_SIZE * sizeof(float)));
        // CUDA_CHECK(cudaMalloc(&buffers[2], PREDICT_BUFFER_SIZE * sizeof(float)));

        // Create stream
        // cudaStream_t stream;
        // CUDA_CHECK(cudaStreamCreate(&stream));

        transfer_timer.start();
        // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
        CUDA_CHECK(cudaMemcpyAsync(buffers[0], input, PREDICT_BUFFER_SIZE * PLANES_SIZE * BOARD_SIZE * BOARD_SIZE * sizeof(float), cudaMemcpyHostToDevice, stream));

        cudaStreamSynchronize(stream);
        transfer_timer.end();

        // context->setTensorAddress("input", buffers[0]);
        // context->setTensorAddress("policy", buffers[1]);
        // context->setTensorAddress("value", buffers[2]);

        predict_timer.start();
        context->enqueueV3(stream);
        cudaStreamSynchronize(stream);
        predict_timer.end();
        // printf("predict time: %.3f ms\n", predict_timer.get_duration_us() / 1000.0);

        transfer_timer.start();
        CUDA_CHECK(cudaMemcpyAsync(pi, buffers[1], PREDICT_BUFFER_SIZE * ACTION_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(val, buffers[2], PREDICT_BUFFER_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));

        float test_x_cpu[10];
        CUDA_CHECK(cudaMemcpyAsync(test_x_cpu, buffers[3], 10 * sizeof(float), cudaMemcpyDeviceToHost, stream));
        
        cudaStreamSynchronize(stream);
        transfer_timer.end();

        for(int i = 0; i < 10; i++) {
            printf("test_x_cpu[%d]: %f\n", i, test_x_cpu[i]);
        }

        // Release stream and buffers
        // cudaStreamDestroy(stream);
        // CUDA_CHECK(cudaFree(buffers[0]));
        // CUDA_CHECK(cudaFree(buffers[1]));
        // CUDA_CHECK(cudaFree(buffers[2]));
        // delete context;

        // printf("print pi\n");
        // for(int i = 0; i < PREDICT_BUFFER_SIZE; i++) {
        //     printf("Predicted batch %d\n", i);
        //     printf("pi: \n"); 
        //     for(int i = 0; i < BOARD_SIZE; i++) {
        //         for(int j = 0; j < BOARD_SIZE; j++) {
        //             printf("%.2f ", pi[i * BOARD_SIZE + j]);
        //         }
        //         printf("\n");
        //     }
        //     printf("val: %.2f\n\n", val[i]);
        // }
    }
};

void print_dims(const nvinfer1::Dims& dim)
{
	for (int nIdxShape = 0; nIdxShape < dim.nbDims; ++nIdxShape)
	{
		
		printf("dim %d=%d\n", nIdxShape, dim.d[nIdxShape]);
		
	}
}

int main() {


    Logger* logger = new Logger(5, "test.log");
    NetworkWrapper* network_wrapper = new NetworkWrapper(logger, PREDICT_BUFFER_SIZE);
    // network_wrapper->load_weight("/home/work/file/mlzero/ckpt/latest.ckpt"); 
    network_wrapper->save_weights();
    network_wrapper->load_weight("/home/work/file/mlzero/test/ckpt/latest.ckpt"); 
    printf("load weight finished\n");
    fflush(stdout);

    TensorRTComponent tensorrt_component(network_wrapper);

    Timer tensorrt_timer = Timer();
    
    nvinfer1::Dims dim = tensorrt_component.context->getTensorShape("input");
	print_dims(dim);

    for(int i = 0; i < 1; i++) {
        tensorrt_timer.start();
        float data[PREDICT_BUFFER_SIZE * PLANES_SIZE * BOARD_SIZE * BOARD_SIZE];
        srand(time(NULL)); // Seed the random generator
        for (int i = 0; i < PREDICT_BUFFER_SIZE * PLANES_SIZE * BOARD_SIZE * BOARD_SIZE; i++) {
            data[i] = static_cast<float>(rand() % 2); // Generate either 0 or 1
        }
        float pi[PREDICT_BUFFER_SIZE * ACTION_SIZE];
        float val[PREDICT_BUFFER_SIZE * 1];
        tensorrt_component.predict(data, pi, val);
        tensorrt_timer.end();

        for(int i = 0; i < PREDICT_BUFFER_SIZE; i++) {
            printf("tensorrt Predicted batch %d\n", i);
            printf("pi: \n"); 
            for(int j = 0; j < BOARD_SIZE; j++) {
                for(int k = 0; k < BOARD_SIZE; k++) {
                    printf("%.3f ", pi[i * ACTION_SIZE + j * BOARD_SIZE + k]);
                }
                printf("\n");
            }
            printf("val: %.3f\n\n", val[i]);
        }       
        // printf("Predicted time: %.3f ms\n", tensorrt_timer.get_duration_us() / 1000.0);

        // Generate random binary data (0 or 1) for validation
        float data_ref[PREDICT_BUFFER_SIZE * PLANES_SIZE * BOARD_SIZE * BOARD_SIZE];
        for (int i = 0; i < PREDICT_BUFFER_SIZE * PLANES_SIZE * BOARD_SIZE * BOARD_SIZE; i++) {
            data_ref[i] = data[i];
        }
        float pi_ref[PREDICT_BUFFER_SIZE * ACTION_SIZE];
        float val_ref[PREDICT_BUFFER_SIZE * 1];
        network_wrapper->predict_single(PREDICT_BUFFER_SIZE, data_ref, pi_ref, val_ref);

        for(int i = 0; i < PREDICT_BUFFER_SIZE; i++) {
            printf("libtorch Predicted batch %d\n", i);
            printf("pi: \n"); 
            for(int j = 0; j < BOARD_SIZE; j++) {
                for(int k = 0; k < BOARD_SIZE; k++) {
                    printf("%.3f ", pi_ref[i * ACTION_SIZE + j * BOARD_SIZE + k]);
                }
                printf("\n");
            }
            printf("val: %.3f\n\n", val_ref[i]);
        }       


        for(int i = 0; i < PREDICT_BUFFER_SIZE; i++) {
            for(int j = 0; j < ACTION_SIZE; j++) {
                if(fabs(pi[i * ACTION_SIZE + j] - pi_ref[i * ACTION_SIZE + j]) > 0.01) {
                    printf("pi not equal i: %d j: %d, pi: %lf, pi_ref: %lf\n", i, j, pi[i * ACTION_SIZE + j], pi_ref[i * ACTION_SIZE + j]);
                    exit(1);
                }
            }
            if(fabs(val[i] - val_ref[i]) > 0.01) {
                printf("val not equal i: %d\n", i);
                exit(1);
            }
        }
    }
    
    printf("tensorrt predict: %.3f ms\n", tensorrt_component.predict_timer.get_all_duration_us() / 1000.0);
    printf("tensorrt transfer: %.3f ms\n", tensorrt_component.transfer_timer.get_all_duration_us() / 1000.0);
    printf("tensorrt: %.3f ms\n", tensorrt_timer.get_all_duration_us() / 1000.0);
    fflush(stdout);

    Timer libtorch_timer = Timer();
    
    for(int i = 0; i < 1; i++) {
        libtorch_timer.start();
        float data[PREDICT_BUFFER_SIZE * PLANES_SIZE * BOARD_SIZE * BOARD_SIZE];
        for (int i = 0; i < PREDICT_BUFFER_SIZE * PLANES_SIZE * BOARD_SIZE * BOARD_SIZE; i++)
            data[i] = 1.0;
        float pi[PREDICT_BUFFER_SIZE * ACTION_SIZE];
        float val[PREDICT_BUFFER_SIZE * 1];
        network_wrapper->predict_single(PREDICT_BUFFER_SIZE, data, pi, val);
        libtorch_timer.end();

        // printf("Predicted time: %.3f ms\n", libtorch_timer.get_duration_us() / 1000.0);

    }
    
    printf("libtorch predict: %.3f ms\n", network_wrapper->predict_single_predict_timer.get_all_duration_us() / 1000.0);
    printf("libtorch transfer: %.3f ms\n", network_wrapper->predict_single_transfer_timer.get_all_duration_us() / 1000.0);
    printf("libtorch: %.3f ms\n", libtorch_timer.get_all_duration_us() / 1000.0);
    fflush(stdout);

    delete logger;
    delete network_wrapper;
}