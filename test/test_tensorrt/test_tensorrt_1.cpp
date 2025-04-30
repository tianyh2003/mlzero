
#include "../network_wrapper.hpp"
#include "../common.h"
#include "../logger.hpp"

using namespace nvinfer1;

// const nvinfer1::BuilderFlag TENSORRT_PREDICT_DATA_TYPE = nvinfer1::BuilderFlag::kTF32;

int MAX_CUDA_STREAMS = 2

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

    std::mutex thread_mtx;
    int stream_idx = 0;
    void* buffers[MAX_CUDA_STREAMS][3];
    cudaStream_t streams[MAX_CUDA_STREAMS];
    IExecutionContext* context[MAX_CUDA_STREAMS] = {nullptr};
    std::mutex stream_mtx[MAX_CUDA_STREAMS];

    TensorRTComponent(
        NetworkWrapper* network_wrapper
    ) : network_wrapper(network_wrapper) {
        tensorrt_logger = new TenserRTLogger();
        for(int i = 0; i < MAX_CUDA_STREAMS; i++) {
            CUDA_CHECK(cudaStreamCreate(&streams[i]));
            CUDA_CHECK(cudaMalloc(&buffers[i][0], PREDICT_BUFFER_SIZE * PLANES_SIZE * BOARD_SIZE * BOARD_SIZE * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&buffers[i][1], PREDICT_BUFFER_SIZE * ACTION_SIZE * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&buffers[i][2], PREDICT_BUFFER_SIZE * sizeof(float)));
        }
        
        update_engine();
        
        printf("update engine finished\n");
        fflush(stdout);
    }
    
    ~TensorRTComponent() {
        delete tensorrt_logger;
        for(int i = 0; i < MAX_CUDA_STREAMS; i++) {
            CUDA_CHECK(cudaStreamDestroy(streams[i]));
            CUDA_CHECK(cudaFree(buffers[i][0]));
            CUDA_CHECK(cudaFree(buffers[i][1]));
            CUDA_CHECK(cudaFree(buffers[i][2]));
            if(context[i] != nullptr) {
                delete context[i];
            }
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
        IConvolutionLayer* policy_conv = network->addConvolutionNd(*relu8->getOutput(0), 2, DimsHW{1, 1}, weightMap["policy_head.0.weight"], emptywts);
        assert(policy_conv);
        policy_conv->setStrideNd(DimsHW{1, 1});
        policy_conv->setPaddingNd(DimsHW{0, 0});
    
        IScaleLayer* policy_bn = addBatchNorm2d(network, weightMap, *policy_conv->getOutput(0), "policy_head.1", 1e-5);
        IActivationLayer* policy_relu = network->addActivation(*policy_bn->getOutput(0), ActivationType::kRELU);
        assert(policy_relu);
    
        IShuffleLayer* policy_flatten = network->addShuffle(*policy_relu->getOutput(0));
        policy_flatten->setReshapeDimensions(Dims2{PREDICT_BUFFER_SIZE, 2 * ACTION_SIZE});
    
        auto policy_weights = network->addConstant(Dims2(ACTION_SIZE, 2 * ACTION_SIZE), weightMap["policy_head.4.weight"]);
        auto policy_bias = network->addConstant(Dims2(1, ACTION_SIZE), weightMap["policy_head.4.bias"]);

        auto policy_mm = network->addMatrixMultiply(*policy_flatten->getOutput(0), MatrixOperation::kNONE, *policy_weights->getOutput(0), MatrixOperation::kTRANSPOSE);
        auto policy_fc = network->addElementWise(*policy_mm->getOutput(0), *policy_bias->getOutput(0), ElementWiseOperation::kSUM);
        assert(policy_fc);
    
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
        network->markOutput(*policy_softmax->getOutput(0));
        network->markOutput(*value_tanh->getOutput(0));
    
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

    static void save_engine(ICudaEngine* engine, NetworkWrapper* network_wrapper) {
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

        for(int i = 0; i < MAX_CUDA_STREAMS; i++) {
            if(context[i] != nullptr) {
                delete context[i];
                context[i] = nullptr;
            }
        }

        engine = createEngine(builder, config);
        assert(engine != nullptr);

        for(int i = 0; i < MAX_CUDA_STREAMS; i++) {
            context[i] = engine->createExecutionContext();
            assert(context[i] != nullptr);
            context[i]->setTensorAddress("input", buffers[i][0]);
            context[i]->setTensorAddress("policy", buffers[i][1]);
            context[i]->setTensorAddress("value", buffers[i][2]);
        }

        std::thread thread_to_save_engine(save_engine, engine, network_wrapper); 
        thread_to_save_engine.detach();
        // save_engine();

        delete config;
        delete builder;
    }

    void predict(float* input, float* pi, float* val)
    {
        int idx = 0; 

        {
            std::lock_guard<std::mutex> lock(thread_mtx);
            idx = stream_idx;
            stream_idx = (stream_idx + 1) % MAX_CUDA_STREAMS;
        }

        {
            std::lock_guard<std::mutex> lock(stream_mtx[idx]);
            CUDA_CHECK(cudaMemcpyAsync(buffers[idx][0], input, PREDICT_BUFFER_SIZE * PLANES_SIZE * BOARD_SIZE * BOARD_SIZE * sizeof(float), cudaMemcpyHostToDevice, streams[idx]));
            context[idx]->enqueueV3(streams[idx]);
            cudaStreamSynchronize(streams[idx]);
            CUDA_CHECK(cudaMemcpyAsync(pi, buffers[idx][1], PREDICT_BUFFER_SIZE * ACTION_SIZE * sizeof(float), cudaMemcpyDeviceToHost, streams[idx]));
            CUDA_CHECK(cudaMemcpyAsync(val, buffers[idx][2], PREDICT_BUFFER_SIZE * sizeof(float), cudaMemcpyDeviceToHost, streams[idx]));
            cudaStreamSynchronize(streams[idx]);
        }
        // cudaStreamSynchronize(streams[idx]);
        // // transfer_timer.start();
        // CUDA_CHECK(cudaMemcpyAsync(buffers[idx][0], input, PREDICT_BUFFER_SIZE * PLANES_SIZE * BOARD_SIZE * BOARD_SIZE * sizeof(float), cudaMemcpyHostToDevice, streams[idx]));
        // // transfer_timer.end();

        // // predict_timer.start();
        // context[idx]->enqueueV3(streams[idx]);
        // cudaStreamSynchronize(streams[idx]);
        // // predict_timer.end();
        
        // // transfer_timer.start();
        // CUDA_CHECK(cudaMemcpyAsync(pi, buffers[idx][1], PREDICT_BUFFER_SIZE * ACTION_SIZE * sizeof(float), cudaMemcpyDeviceToHost, streams[idx]));
        // CUDA_CHECK(cudaMemcpyAsync(val, buffers[idx][2], PREDICT_BUFFER_SIZE * sizeof(float), cudaMemcpyDeviceToHost, streams[idx]));

        // cudaStreamSynchronize(streams[idx]);
        // transfer_timer.end();


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

void test_predict(TensorRTComponent* tensorrt_component) {
    float data[PREDICT_BUFFER_SIZE * PLANES_SIZE * BOARD_SIZE * BOARD_SIZE];
    srand(time(NULL)); // Seed the random generator
    for (int i = 0; i < PREDICT_BUFFER_SIZE * PLANES_SIZE * BOARD_SIZE * BOARD_SIZE; i++) {
        data[i] = static_cast<float>(rand() % 2); // Generate either 0 or 1
    }
    float pi[PREDICT_BUFFER_SIZE * ACTION_SIZE];
    float val[PREDICT_BUFFER_SIZE * 1];
    tensorrt_component->predict(data, pi, val);
}

int main() {
    int step_num = 2000;

    Logger* logger = new Logger(5, "test.log");
    NetworkWrapper* network_wrapper = new NetworkWrapper(logger, PREDICT_BUFFER_SIZE);
    network_wrapper->load_weight("/home/work/file/mlzero/ckpt/weight_1.ckpt"); 

    // network_wrapper->save_weights();
    printf("load weight finished\n");
    fflush(stdout);


    printf("tensorrt start\n");
    fflush(stdout);
    TensorRTComponent tensorrt_component(network_wrapper);
    Timer tensorrt_timer = Timer();
    std::thread* threads[10000];
    tensorrt_timer.start();
    for(int i = 0; i < step_num; i++) {
        float data[PREDICT_BUFFER_SIZE * PLANES_SIZE * BOARD_SIZE * BOARD_SIZE];
        srand(time(NULL)); // Seed the random generator
        for (int i = 0; i < PREDICT_BUFFER_SIZE * PLANES_SIZE * BOARD_SIZE * BOARD_SIZE; i++) {
            data[i] = static_cast<float>(rand() % 2); // Generate either 0 or 1
        }
        float pi[PREDICT_BUFFER_SIZE * ACTION_SIZE];
        float val[PREDICT_BUFFER_SIZE * 1];
        // threads[i] = new std::thread(test_predict, &tensorrt_component);

        tensorrt_component.predict(data, pi, val);

        // for(int i = 0; i < PREDICT_BUFFER_SIZE; i++) {
        //     printf("tensorrt Predicted batch %d\n", i);
        //     printf("pi: \n"); 
        //     for(int j = 0; j < BOARD_SIZE; j++) {
        //         for(int k = 0; k < BOARD_SIZE; k++) {
        //             printf("%.3f ", pi[i * ACTION_SIZE + j * BOARD_SIZE + k]);
        //         }
        //         printf("\n");
        //     }
        //     printf("val: %.3f\n\n", val[i]);
        // }       
        // printf("Predicted time: %.3f ms\n", tensorrt_timer.get_duration_us() / 1000.0);

        // Generate random binary data (0 or 1) for validation
        // float data_ref[PREDICT_BUFFER_SIZE * PLANES_SIZE * BOARD_SIZE * BOARD_SIZE];
        // for (int i = 0; i < PREDICT_BUFFER_SIZE * PLANES_SIZE * BOARD_SIZE * BOARD_SIZE; i++) {
        //     data_ref[i] = data[i];
        // }
        // float pi_ref[PREDICT_BUFFER_SIZE * ACTION_SIZE];
        // float val_ref[PREDICT_BUFFER_SIZE * 1];
        // network_wrapper->predict_single(PREDICT_BUFFER_SIZE, data_ref, pi_ref, val_ref);

        // for(int i = 0; i < PREDICT_BUFFER_SIZE; i++) {
        //     printf("libtorch Predicted batch %d\n", i);
        //     printf("pi: \n"); 
        //     for(int j = 0; j < BOARD_SIZE; j++) {
        //         for(int k = 0; k < BOARD_SIZE; k++) {
        //             printf("%.3f ", pi_ref[i * ACTION_SIZE + j * BOARD_SIZE + k]);
        //         }
        //         printf("\n");
        //     }
        //     printf("val: %.3f\n\n", val_ref[i]);
        // }       


        // for(int i = 0; i < PREDICT_BUFFER_SIZE; i++) {
        //     for(int j = 0; j < ACTION_SIZE; j++) {
        //         if(fabs(pi[i * ACTION_SIZE + j] - pi_ref[i * ACTION_SIZE + j]) > 0.01) {
        //             printf("pi not equal i: %d j: %d, pi: %lf, pi_ref: %lf\n", i, j, pi[i * ACTION_SIZE + j], pi_ref[i * ACTION_SIZE + j]);
        //             exit(1);
        //         }
        //     }
        //     if(fabs(val[i] - val_ref[i]) > 0.1) {
        //         printf("val not equal i: %d\n", i);
        //         exit(1);
        //     }
        // }
    }

    // for(int i = 0; i < step_num; i++) {
    //     threads[i]->join();
    //     delete threads[i];
    // }
    tensorrt_timer.end();
    
    printf("tensorrt predict: %.3f ms\n", tensorrt_component.predict_timer.get_all_duration_us() / 1000.0);
    printf("tensorrt transfer: %.3f ms\n", tensorrt_component.transfer_timer.get_all_duration_us() / 1000.0);
    printf("tensorrt: %.3f ms\n", tensorrt_timer.get_all_duration_us() / 1000.0);
    fflush(stdout);

    if(USE_LIBTORCH_HALF) {
        network_wrapper->get_net()->to(torch::kHalf);
    }

    Timer libtorch_timer = Timer();
    libtorch_timer.start();
    for(int i = 0; i < step_num; i++) {
        
        float data[PREDICT_BUFFER_SIZE * PLANES_SIZE * BOARD_SIZE * BOARD_SIZE];
        for (int i = 0; i < PREDICT_BUFFER_SIZE * PLANES_SIZE * BOARD_SIZE * BOARD_SIZE; i++)
            data[i] = 1.0;
        float pi[PREDICT_BUFFER_SIZE * ACTION_SIZE];
        float val[PREDICT_BUFFER_SIZE * 1];
        network_wrapper->predict_single(PREDICT_BUFFER_SIZE, data, pi, val);
        // printf("Predicted time: %.3f ms\n", libtorch_timer.get_duration_us() / 1000.0);

    }
    libtorch_timer.end();
    printf("libtorch predict: %.3f ms\n", network_wrapper->predict_single_predict_timer.get_all_duration_us() / 1000.0);
    printf("libtorch transfer: %.3f ms\n", network_wrapper->predict_single_transfer_timer.get_all_duration_us() / 1000.0);
    printf("libtorch: %.3f ms\n", libtorch_timer.get_all_duration_us() / 1000.0);
    fflush(stdout);
    
    delete logger;
    delete network_wrapper;
}