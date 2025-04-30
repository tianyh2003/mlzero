#ifndef NETWORK_WRAPPER_H
#define NETWORK_WRAPPER_H

#include "common.h"
#include "logger.hpp"
#include "util.hpp"
#include <iostream>
#include <string>

#ifdef USE_RESNET
    #include "network_resnet.hpp"
    #include "tensorrt_resnet.hpp"
#else
    #include "network.hpp"
    #include "tensorrt.hpp"
#endif

namespace fs = std::filesystem;

class PredictBuffer {
private:
    Logger* logger;
    std::mutex buffer_mutex;
    std::mutex predict_libtorch_mutex;
    Timer timer_predict_buffer;
    TensorRTComponent* tensorrt_component;
    
    AlphaZeroResNetImpl* net;
    torch::Device device = torch::kCUDA;
    std::vector<float*> input_buffer;
    std::vector<float*> policy_buffer;
    std::vector<float*> value_buffer;
    std::chrono::_V2::system_clock::time_point start_time = std::chrono::high_resolution_clock::now();
    std::chrono::_V2::system_clock::time_point end_time = std::chrono::high_resolution_clock::now();
    
    int idx = 0; 
    int is_predicting = 0;
    int is_finished = 0; 
    int num_finished_back = 0;

    int use_tensorrt = USE_TENSORRT;

    float input_tensors[PREDICT_BUFFER_SIZE * PLANES_SIZE * BOARD_SIZE * BOARD_SIZE];
    float policy_tensors[PREDICT_BUFFER_SIZE * ACTION_SIZE];
    float value_tensors[PREDICT_BUFFER_SIZE];

public:
    PredictBuffer(
        AlphaZeroResNetImpl* _net, 
        torch::Device _device, 
        Logger* _logger, 
        bool _use_tensorrt = USE_TENSORRT
    ) {
        net = _net;
        device = _device;
        logger = _logger;
        timer_predict_buffer.start();
        use_tensorrt = _use_tensorrt;
    }

    ~PredictBuffer() {
        input_buffer.clear();
        policy_buffer.clear();
        value_buffer.clear();
        logger->log(0, "predict buffer time: " + std::to_string(timer_predict_buffer.get_all_duration_us() / 1000) + " ms");
    }

    void set_tensorrt_component(TensorRTComponent* _tensorrt_component) {
        tensorrt_component = _tensorrt_component;
    }

    void predict(
        float* input, 
        float* policy, 
        float* value
    ) {
        int this_idx; 
        while(true) {
            {
                std::lock_guard<std::mutex> lock(buffer_mutex);
                if(idx < PREDICT_BUFFER_SIZE && is_predicting == 0) {
                    this_idx = idx;
                    input_buffer.push_back(input);
                    policy_buffer.push_back(policy);
                    value_buffer.push_back(value);
                    idx++;
                    start_time = std::chrono::high_resolution_clock::now();
                    break;
                } 
            }
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
        while(true) {
            {
                std::lock_guard<std::mutex> lock(buffer_mutex);
                if(is_finished) break; 
                end_time = std::chrono::high_resolution_clock::now();
                long long duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
                if(idx >= PREDICT_BUFFER_SIZE || duration_us > 1000) {
                    is_predicting = 1;
                    _predict_buffer();
                    is_finished = 1;
                    break;
                }
            }
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
        {
            std::lock_guard<std::mutex> lock(buffer_mutex);
            num_finished_back++;
            if(num_finished_back == idx) {
                is_finished = 0;
                is_predicting = 0;
                num_finished_back = 0;
                idx = 0; 
                input_buffer.clear();
                policy_buffer.clear(); 
                value_buffer.clear(); 
            }
        }
        return;
    }

    void _predict_buffer() {
        for(int i = 0; i < idx; i++) {
            memcpy(input_tensors + i * PLANES_SIZE * BOARD_SIZE * BOARD_SIZE, input_buffer[i], PLANES_SIZE * BOARD_SIZE * BOARD_SIZE * sizeof(float));
        }
        if(use_tensorrt) {
            tensorrt_component->predict(PREDICT_BUFFER_SIZE, input_tensors, policy_tensors, value_tensors);
        }
        else {
            net->eval();
            std::lock_guard<std::mutex> lock(predict_libtorch_mutex);
            torch::NoGradGuard no_grad;
            if (idx <= 0) {
                logger->log(1, "idx < 0");
                exit(-1);
            }
            torch::Tensor input_tensor = torch::from_blob(input_tensors, {idx, PLANES_SIZE, BOARD_SIZE, BOARD_SIZE}).clone().to(device);
            input_tensor.set_requires_grad(false);
            // printf("predict buffer %d\n", idx);
            // printf("before\n");
            // check_cuda_memory();
            auto outputs = net->forward(input_tensor);
            // c10::cuda::CUDACachingAllocator::emptyCache();
            // c10::cuda::getCurrentCUDAStream().synchronize();
            // printf("after\n");
            // check_cuda_memory();
            torch::Tensor policy = std::get<0>(outputs).cpu();
            torch::Tensor value = std::get<1>(outputs).cpu();
            memcpy(policy_tensors, policy.data_ptr(), idx * ACTION_SIZE * sizeof(float));
            memcpy(value_tensors, value.data_ptr(), idx * sizeof(float));
        }
        
        for(int i = 0; i < idx; i++) {
            memcpy(policy_buffer[i], policy_tensors + i * ACTION_SIZE, ACTION_SIZE * sizeof(float));
            memcpy(value_buffer[i], value_tensors + i, sizeof(float));
        }
    }
};

class PredictTaskBlock {
public:
    float* input[PREDICT_BUFFER_SIZE];
    float* policy[PREDICT_BUFFER_SIZE];
    float* value[PREDICT_BUFFER_SIZE];
    int buffer_size = 0;
    int is_finished = 0;
    int is_ready = 0;
    int num_get_result = 0;
    std::mutex mutex;
};

class PredictManager {
private:
    Logger* logger;
    std::mutex manager_mutex;
    std::mutex manager_buffer_mutex;
    std::queue<PredictTaskBlock*> predict_task_blocks;
    AlphaZeroResNetImpl* net;
    TensorRTComponent* tensorrt_component;
    torch::Device device = torch::kCUDA;
    PredictTaskBlock* current_predict_task_block = nullptr;

    int manager_block_num = 0;
    int manager_buffer_in_use[MANAGER_BUFFER_SIZE] = {0};
    int each_input_buffer_size = MANAGER_CONBINE_MAX_BLOCK_NUM * PREDICT_BUFFER_SIZE * PLANES_SIZE * BOARD_SIZE * BOARD_SIZE;
    int each_policy_buffer_size = MANAGER_CONBINE_MAX_BLOCK_NUM * PREDICT_BUFFER_SIZE * ACTION_SIZE;
    int each_value_buffer_size = MANAGER_CONBINE_MAX_BLOCK_NUM * PREDICT_BUFFER_SIZE;
    float manager_input_tensors[MANAGER_BUFFER_SIZE * MANAGER_CONBINE_MAX_BLOCK_NUM * PREDICT_BUFFER_SIZE * PLANES_SIZE * BOARD_SIZE * BOARD_SIZE];
    float manager_policy_tensors[MANAGER_BUFFER_SIZE * MANAGER_CONBINE_MAX_BLOCK_NUM * PREDICT_BUFFER_SIZE * ACTION_SIZE];
    float manager_value_tensors[MANAGER_BUFFER_SIZE * MANAGER_CONBINE_MAX_BLOCK_NUM * PREDICT_BUFFER_SIZE];

    cudaStream_t transfer_stream[MANAGER_BUFFER_SIZE];
    void* cuda_buffers[MANAGER_BUFFER_SIZE][3];

    long long log_force_predict = 0;
    long long log_all_predict = 0;

    //protected by manager_mutex
    int block_is_predicting = 0;
    // long long gpu_task_num = 0;

public:
    PredictManager(
        AlphaZeroResNetImpl* _net, 
        torch::Device _device, 
        Logger* _logger
    ) {
        net = _net;
        device = _device;
        logger = _logger;
        for(int i = 0; i < MANAGER_BUFFER_SIZE; i++) {
            CUDA_CHECK(cudaStreamCreate(&transfer_stream[i]));
            CUDA_CHECK(cudaMalloc(&cuda_buffers[i][0], each_input_buffer_size * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&cuda_buffers[i][1], each_policy_buffer_size * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&cuda_buffers[i][2], each_value_buffer_size * sizeof(float)));
        }
    }

    ~PredictManager() {
        for(int i = 0; i < MANAGER_BUFFER_SIZE; i++) {
            CUDA_CHECK(cudaStreamDestroy(transfer_stream[i]));
            CUDA_CHECK(cudaFree(cuda_buffers[i][0]));
            CUDA_CHECK(cudaFree(cuda_buffers[i][1]));
            CUDA_CHECK(cudaFree(cuda_buffers[i][2]));
        }
    }

    void set_tensorrt_component(TensorRTComponent* _tensorrt_component) {
        tensorrt_component = _tensorrt_component;
    }

    void _block_predict_operation(
        PredictTaskBlock* task_block, 
        int manager_buffer_idx
    ) {
        {
            std::lock_guard<std::mutex> lock(task_block->mutex);
            float* _manager_input_tensors = manager_input_tensors + manager_buffer_idx * PREDICT_BUFFER_SIZE * PLANES_SIZE * BOARD_SIZE * BOARD_SIZE;
            float* _manager_policy_tensors = manager_policy_tensors + manager_buffer_idx * PREDICT_BUFFER_SIZE * ACTION_SIZE;
            float* _manager_value_tensors = manager_value_tensors + manager_buffer_idx * PREDICT_BUFFER_SIZE;

            for(int i = 0; i < task_block->buffer_size; i++) {
                memcpy(_manager_input_tensors + i * PLANES_SIZE * ACTION_SIZE, task_block->input[i], PLANES_SIZE * ACTION_SIZE * sizeof(float));
            }

            tensorrt_component->predict(PREDICT_BUFFER_SIZE, _manager_input_tensors, _manager_policy_tensors, _manager_value_tensors);
            
            for(int i = 0; i < task_block->buffer_size; i++) {
                memcpy(task_block->policy[i], _manager_policy_tensors + i * ACTION_SIZE, ACTION_SIZE * sizeof(float));
                memcpy(task_block->value[i], _manager_value_tensors + i, sizeof(float));
            }
            task_block->is_finished = 1;
        }
        {
            std::lock_guard<std::mutex> lock(manager_buffer_mutex);
            manager_buffer_in_use[manager_buffer_idx] = 0;
        }
    }

    void _block_predict_without_combine() {
        PredictTaskBlock* this_predict_task_block = nullptr;
        {
            std::lock_guard<std::mutex> lock(manager_mutex);
            int predict_task_blocks_size = predict_task_blocks.size();
            if(predict_task_blocks_size > 0 && block_is_predicting < MANAGER_BUFFER_SIZE) {
                this_predict_task_block = predict_task_blocks.front();
                predict_task_blocks.pop();
                block_is_predicting++;
            }
        }
        if(this_predict_task_block == nullptr) return;
        int manager_buffer_idx = -1; 
        while(manager_buffer_idx == -1) {
            {
                std::lock_guard<std::mutex> lock(manager_buffer_mutex);
                for(int i = 0; i < PREDICT_BUFFER_SIZE; i++) {
                    if(manager_buffer_in_use[i] == 0) {
                        manager_buffer_in_use[i] = 1;
                        manager_buffer_idx = i;
                        break; 
                    }
                }
            }
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
        _block_predict_operation(this_predict_task_block, manager_buffer_idx);
        {
            std::lock_guard<std::mutex> lock(manager_mutex);
            block_is_predicting--;
        }
        return; 
    }


    void _block_predict() {
        int combined_block_num = 0; 
        PredictTaskBlock* combined_task_block[MANAGER_CONBINE_MAX_BLOCK_NUM] = {nullptr};
        {
            std::lock_guard<std::mutex> lock(manager_mutex);
            int predict_task_blocks_size = predict_task_blocks.size(); //in queue
            // printf("predict_task_blocks_size: %d, block_is_predicting: %d\n", predict_task_blocks_size, block_is_predicting);
            if(predict_task_blocks_size > 0 && block_is_predicting < MANAGER_BUFFER_SIZE) {

                if(predict_task_blocks_size >= MANAGER_CONBINE_MIN_BLOCK) {
                    // combined_block_num = std::min(predict_task_blocks_size / 2, MANAGER_CONBINE_MAX_BLOCK_NUM); 
                    // combined_block_num = std::min(predict_task_blocks_size - 1, MANAGER_CONBINE_MAX_BLOCK_NUM); 
                    combined_block_num = std::min(predict_task_blocks_size, MANAGER_CONBINE_MAX_BLOCK_NUM); 
                    // printf("combined_block_num: %d\n", combined_block_num);
                    // combined_block_num = predict_task_blocks_size / 2; 
                    for(int i = 0; i < combined_block_num; i++) {
                        PredictTaskBlock* task_block = predict_task_blocks.front();
                        predict_task_blocks.pop();
                        combined_task_block[i] = task_block;
                    }
                }
                else {
                    combined_block_num = 1;
                    combined_task_block[0] = predict_task_blocks.front();
                    predict_task_blocks.pop();
                }
                block_is_predicting++;
            }
        }

        if(combined_block_num == 0) return;

        int manager_buffer_idx = -1; 
        while(manager_buffer_idx == -1) {
            {
                std::lock_guard<std::mutex> lock(manager_buffer_mutex);
                for(int i = 0; i < PREDICT_BUFFER_SIZE; i++) {
                    if(manager_buffer_in_use[i] == 0) {
                        manager_buffer_in_use[i] = 1;
                        manager_buffer_idx = i;
                        break; 
                    }
                }
            }
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }

        int batch_size = 0;
        float* _manager_input_tensors = manager_input_tensors + manager_buffer_idx * each_input_buffer_size;
        float* _manager_policy_tensors = manager_policy_tensors + manager_buffer_idx * each_policy_buffer_size;
        float* _manager_value_tensors = manager_value_tensors + manager_buffer_idx * each_value_buffer_size;

        for(int i = 0; i < combined_block_num; i++) {
            for(int j = 0; j < combined_task_block[i]->buffer_size; j++) {
                memcpy(_manager_input_tensors + batch_size * PLANES_SIZE * BOARD_SIZE * BOARD_SIZE, combined_task_block[i]->input[j], PLANES_SIZE * BOARD_SIZE * BOARD_SIZE * sizeof(float));
                batch_size++; 
            }
        }

        CUDA_CHECK(cudaMemcpyAsync(cuda_buffers[manager_buffer_idx][0], _manager_input_tensors, batch_size * PLANES_SIZE * BOARD_SIZE * BOARD_SIZE * sizeof(float), cudaMemcpyHostToDevice, transfer_stream[manager_buffer_idx]));

        tensorrt_component->predict_without_transfer(
            batch_size, 
            cuda_buffers[manager_buffer_idx][0], 
            cuda_buffers[manager_buffer_idx][1], 
            cuda_buffers[manager_buffer_idx][2], 
            transfer_stream[manager_buffer_idx]
        );
        CUDA_CHECK(cudaMemcpyAsync(_manager_policy_tensors, cuda_buffers[manager_buffer_idx][1], batch_size * ACTION_SIZE * sizeof(float), cudaMemcpyDeviceToHost, transfer_stream[manager_buffer_idx]));
        CUDA_CHECK(cudaMemcpyAsync(_manager_value_tensors, cuda_buffers[manager_buffer_idx][2], batch_size * sizeof(float), cudaMemcpyDeviceToHost, transfer_stream[manager_buffer_idx]));
        cudaStreamSynchronize(transfer_stream[manager_buffer_idx]);
        
        int _batch_size = 0;
        for(int i = 0; i < combined_block_num; i++) {
            std::lock_guard<std::mutex> lock(combined_task_block[i]->mutex);
            for(int j = 0; j < combined_task_block[i]->buffer_size; j++) {
                memcpy(combined_task_block[i]->policy[j], _manager_policy_tensors + _batch_size * ACTION_SIZE, ACTION_SIZE * sizeof(float));
                memcpy(combined_task_block[i]->value[j], _manager_value_tensors + _batch_size, sizeof(float));
                _batch_size++; 
            }
            combined_task_block[i]->is_finished = 1;
        }
        {
            std::lock_guard<std::mutex> lock(manager_buffer_mutex);
            manager_buffer_in_use[manager_buffer_idx] = 0;
        }
        {
            std::lock_guard<std::mutex> lock(manager_mutex);
            block_is_predicting--;
        }
        return; 
    }

    void predict(float* input, float* policy, float* value) {
        PredictTaskBlock* this_predict_task_block = nullptr;
        while(true) {
            {
                std::lock_guard<std::mutex> lock(manager_mutex);
                if(manager_block_num <=  MANAGER_BLOCK_NUM) {
                    if(current_predict_task_block == nullptr) {
                        current_predict_task_block = new PredictTaskBlock();
                        manager_block_num++;
                    }
                    this_predict_task_block = current_predict_task_block;
                    int idx = current_predict_task_block->buffer_size;
                    current_predict_task_block->input[idx] = input;
                    current_predict_task_block->policy[idx] = policy;
                    current_predict_task_block->value[idx] = value;
                    current_predict_task_block->buffer_size++;
                    if(current_predict_task_block->buffer_size >= PREDICT_BUFFER_SIZE) {
                        current_predict_task_block->is_ready = 1;
                        predict_task_blocks.push(current_predict_task_block);
                        log_all_predict++;
                        current_predict_task_block = nullptr;
                    }
                    break; 
                }
            }
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
        
        int wait_time = 0;
        while(true) {
            {
                std::lock_guard<std::mutex> lock(manager_mutex);
                if(this_predict_task_block->is_ready == 1) break;
                else {
                    wait_time++;
                    if(wait_time >= 3) {
                        current_predict_task_block->is_ready = 1;
                        predict_task_blocks.push(current_predict_task_block);
                        log_force_predict++;
                        log_all_predict++;
                        // printf("force predict %ld/%ld\n", log_force_predict, log_all_predict);
                        current_predict_task_block = nullptr;
                        break;
                    }
                    
                }
            }
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }

        int will_delete = 0;
        while(true) {
            _block_predict();
            {
                std::lock_guard<std::mutex> lock(this_predict_task_block->mutex);
                if(this_predict_task_block->is_finished == 1) {
                    this_predict_task_block->num_get_result++;
                    if(this_predict_task_block->num_get_result == this_predict_task_block->buffer_size) {
                        will_delete = 1;
                    }
                    break;
                }
            }
            std::this_thread::sleep_for(std::chrono::microseconds(5));
        }
        if(will_delete) {
            delete this_predict_task_block;
            {
                std::lock_guard<std::mutex> lock(manager_mutex);
                manager_block_num--;
            }
        }
    }

};
    
class TrainResult {
public:
    float loss; 
    float entropy; 
    float kl;

    std::string get_string() {
        std::stringstream ss;
        ss << "loss: " << loss << ", entropy: " << entropy;
        return ss.str();
    }
};

class NetworkWrapper {
private:
    std::shared_ptr<AlphaZeroResNetImpl> net;
    torch::Device device = torch::kCPU;
    std::unique_ptr<torch::optim::Adam> optimizer_;
    std::unique_ptr<torch::optim::StepLR> lr_scheduler_;
    std::vector<int> lr_milestones_;
    
    PredictBuffer* predict_buffer;
    PredictManager* predict_manager;

    std::mutex predict_single_mutex; 

    float init_lr_, sgd_momentum_, l2_reg_;
    float lr_decay_;

    int use_tensorrt = USE_TENSORRT;
    
public:
    std::mutex thread_lock;
    Logger logger = Logger("network.log");
    TensorRTComponent* tensorrt_component;

    NetworkWrapper(bool is_train = false) {
        net = std::make_shared<AlphaZeroResNetImpl>();
        if (USE_GPU && torch::cuda::is_available()) {
            logger.log(0, "learner use GPU");
            device = torch::kCUDA;
            net->to(device);
        }

        if (is_train) {
            net->train();
            use_tensorrt = false;
        } else {
            net->eval();
        }

        predict_buffer = new PredictBuffer(net.get(), device, &logger, use_tensorrt);
        predict_manager = new PredictManager(net.get(), device, &logger);

        if (use_tensorrt) {
            tensorrt_component = new TensorRTComponent(net.get());
            predict_buffer->set_tensorrt_component(tensorrt_component);
            predict_manager->set_tensorrt_component(tensorrt_component);
        }

        init_lr_ = 2e-3f;
        l2_reg_ = 1e-4f;
        optimizer_ = std::make_unique<torch::optim::Adam>(
            net->parameters(),
            torch::optim::AdamOptions(init_lr_).weight_decay(l2_reg_)
        );
        
        int64_t step_size = 1;
        double gamma = 0.5;  
        lr_scheduler_ = std::make_unique<torch::optim::StepLR>(*optimizer_, step_size, gamma);
    }

    ~NetworkWrapper() {
        if (use_tensorrt) {
            delete tensorrt_component;
        }
        delete predict_buffer;
        delete predict_manager;
    }
    
    AlphaZeroResNetImpl* get_net() {
        return net.get();
    }

    void eval() {
        net->eval();
    }

    void train() {
        net->train();
    }

    int load_weight(std::string weight_path, bool stop_update_tensorrt = false) {
        try {
            torch::load(net, weight_path, torch::Device(torch::kCUDA));
        } catch (const c10::Error& e) {
            logger.log(0, "Failed to load weights: " + std::string(e.what()));
            std::cerr << "Failed to load weights: " << e.what() << std::endl;
            return -1;
        }
        if (use_tensorrt && stop_update_tensorrt == false) {
            tensorrt_component->update_engine();
        }
        return 1; 
    }

    void update_tensorrt() {
        if (use_tensorrt) {
            tensorrt_component->update_engine();
        }
    }
    
    void save_weights(std::string weight_path) {
        net->to(torch::kCPU);
        torch::save(net, weight_path);
        net->to(torch::kCUDA);
        logger.log(0, "save weights to " + weight_path);
    }

    void predict_only_buffer(
        int batch_size, 
        float* input, 
        float* output_policy,
        float* output_value
    ) {
        if(USE_PREDICT_BUFFER && batch_size == 1) {
            predict_buffer->predict(input, output_policy, output_value);
        }
        else {
            std::lock_guard<std::mutex> lock(thread_lock);
            predict_single(batch_size, input, output_policy, output_value);
        }
        
    }

    void predict(
        int batch_size, 
        float* input, 
        float* output_policy,
        float* output_value
    ) {
        if(USE_PREDICT_BUFFER && batch_size == 1) {
            if(USE_PREDICT_MANAGER) {
                predict_manager->predict(input, output_policy, output_value);
            }
            else {
                predict_buffer->predict(input, output_policy, output_value);
            }
        }
        else {
            predict_single(batch_size, input, output_policy, output_value);
        }
        
    }

    void predict_single(
        int batch_size, 
        float* input, 
        float* output_policy,
        float* output_value
    ) {
        std::lock_guard<std::mutex> lock(predict_single_mutex);
        torch::NoGradGuard no_grad;

        torch::Tensor input_tensor = torch::from_blob(input, {batch_size, PLANES_SIZE, BOARD_SIZE, BOARD_SIZE}).to(device);
        
        auto outputs = net->forward(input_tensor);

        torch::Tensor policy = std::get<0>(outputs).cpu();
        torch::Tensor value = std::get<1>(outputs).cpu();

        memcpy(output_policy, policy.data_ptr(), batch_size * ACTION_SIZE * sizeof(float));
        memcpy(output_value, value.data_ptr(), batch_size * sizeof(float));
    }

    TrainResult train(
        int batch_size,
        float* input, 
        float* pi, 
        float* v
    ) {
        net->train();
        optimizer_->zero_grad();
        auto input_gpu = torch::from_blob(input, {batch_size, PLANES_SIZE, BOARD_SIZE, BOARD_SIZE}).clone().to(device);
        auto pi_gpu = torch::from_blob(pi, {batch_size, ACTION_SIZE}).clone().to(device);
        auto v_gpu = torch::from_blob(v, {batch_size}).clone().to(device);
        
        auto [pred_pi, pred_v] = net->forward(input_gpu);

        auto log_pred_pi = torch::log(pred_pi + 1e-10);
        auto pi_loss = -torch::mean(torch::sum(pi_gpu * log_pred_pi, 1));
        auto v_loss = torch::mse_loss(pred_v.view({-1}), v_gpu);
        auto total_loss = pi_loss + v_loss;
        total_loss.backward();
        optimizer_->step();
        
        auto entropy = -torch::mean(torch::sum(torch::exp(log_pred_pi) * log_pred_pi, 1));
        auto [new_pred_pi, new_pred_v] = net->forward(input_gpu);
        auto kl = torch::mean(torch::sum(pred_pi * (log_pred_pi - torch::log(new_pred_pi + 1e-10)), 1));

        return {
            .loss = total_loss.item<float>(),
            .entropy = entropy.item<float>(),
            .kl = kl.item<float>()
        };
    }

    void set_lr(float lr) {
        for (auto& param_group : optimizer_->param_groups()) {
            param_group.options().set_lr(lr);
        }
    }
};



#endif