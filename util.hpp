#ifndef UTIL_H
#define UTIL_H
#include "common.h"

void check_cuda_memory(const std::string& tag = "") {
    if (!USE_GPU || !torch::cuda::is_available()) return;
    
    size_t free_memory, total_memory;
    cudaMemGetInfo(&free_memory, &total_memory);
    
    std::string message = tag.empty() ? "CUDA内存状态" : tag;
    message += ": 空闲=" + std::to_string(free_memory/(1024*1024)) + 
               "MB, 总计=" + std::to_string(total_memory/(1024*1024)) + "MB";
    
    // logger.log(2, message);
    std::cout << message << std::endl;
}

int check_weight_id() {
    if (std::filesystem::exists(CKPT_DIR)) {
        int max_id = -1;
        for (const auto& entry : std::filesystem::directory_iterator(CKPT_DIR)) {
            if (entry.is_regular_file()) {
                std::string filename = entry.path().filename().string();
                std::regex pattern("weight_(\\d+)\\.ckpt");
                std::smatch matches;
                if (std::regex_search(filename, matches, pattern) && matches.size() > 1) {
                    int id = std::stoi(matches[1].str());
                    max_id = std::max(max_id, id);
                }
            }
        }
        return max_id;
    }
    return -1; 
}

std::string get_weight_path(int weight_id) {
    return CKPT_DIR + "/weight_" + std::to_string(weight_id) + ".ckpt";
}

void print_obs(float* _obs) {
    printf("obs: \n");
    for(int i = 0; i < PLANES_SIZE; i++) {
        for(int j = 0; j < BOARD_SIZE; j++){
            for(int k = 0; k < BOARD_SIZE; k++){
                printf("%.2f ", _obs[i * BOARD_SIZE * BOARD_SIZE + j * BOARD_SIZE + k]);
            }
            printf("\n");
        }
        printf("\n");
    }
    printf("\n");
}

void print_pi(float* _pi) {
    printf("pi: \n");
    for(int i = 0; i < BOARD_SIZE; i++) {
        for(int j = 0; j < BOARD_SIZE; j++) {
            printf("%.5f ", _pi[i * BOARD_SIZE + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void print_value(float* _value) {
    printf("value: %f\n", *_value);
    printf("\n");
}

void print_env_by_obs(float* _obs) {

}

void softmax(float* _pi, int size) {
    float sum = 0;
    for(int i = 0; i < size; i++) {
        sum += exp(_pi[i]);
    }
    for(int i = 0; i < size; i++) {
        _pi[i] = exp(_pi[i]) / sum;
    }
}

void print_obs_pi_val(float* _obs, float* _pi, float* _value) {
    printf("obs: \n");
    for(int i = 0; i < PLANES_SIZE; i++) {
        for(int j = 0; j < BOARD_SIZE; j++) {
            for(int k = 0; k < BOARD_SIZE; k++) {
                printf("%.2f ", _obs[i * BOARD_SIZE * BOARD_SIZE + j * BOARD_SIZE + k]);
            }
            printf("\n");
        }
        printf("\n");
    }
    printf("pi: \n");
    for(int i = 0; i < BOARD_SIZE; i++) {
        for(int j = 0; j < BOARD_SIZE; j++) {
            printf("%.2f ", _pi[i * BOARD_SIZE + j]);
        }
        printf("\n");
    }
    printf("\n");
    printf("value: %f\n", *_value);
    fflush(stdout);
}

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

#endif