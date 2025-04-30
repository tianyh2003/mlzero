#ifndef COMMON_H
#define COMMON_H

#include <cstdio>
#include <ctime>
#include <cerrno>
#include <array>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <memory>
#include <vector>
#include <cmath>
#include <random>
#include <cassert>
#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <unistd.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <queue>
#include <sstream>
#include <torch/torch.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <regex>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <atomic>
#include <c10/cuda/CUDACachingAllocator.h>

#define USE_RESNET
#define TRAIN_MODE

enum class ActorType {ApvMcts, PureMcts, Arena};
enum class PlayerType {NN, PureMcts, MinMax, People};

class TrainConfig {
public:
    int weight_save_interval = 5; // -1 not save, other save interval
    int init_weight_id = -1; // -1 means check weight id auto
};

#ifdef TRAIN_MODE
    const int TRAIN_ITERATIONS = 3000000;
    const int GAMES_PER_ITERATION = 5;
    const int ACT_SIMULATIONS = 800;
    const int NUM_THREADS_PER_MCTS = 8;
    const int TRAIN_STEPS = 8;
    const int TRAIN_BATCH_SIZE = 512;
    const bool USE_PREDICT_BUFFER = true;
    const bool USE_PREDICT_MANAGER = true; //if use, buffer must open
    const bool USE_TENSORRT = true;
#else
    const int TRAIN_ITERATIONS = 3;
    const int GAMES_PER_ITERATION = 5; 
    const int ACT_SIMULATIONS = 800; 
    const int NUM_THREADS_PER_MCTS = 1; 
    const int TRAIN_STEPS = 8;
    const int TRAIN_BATCH_SIZE = 512;
    const bool USE_PREDICT_BUFFER = false;
    const bool USE_PREDICT_MANAGER = false; 
    const bool USE_TENSORRT = false;
#endif

const nvinfer1::BuilderFlag TENSORRT_PREDICT_DATA_TYPE = nvinfer1::BuilderFlag::kFP16;
// const nvinfer1::BuilderFlag TENSORRT_PREDICT_DATA_TYPE = nvinfer1::BuilderFlag::kTF32;

const ActorType WARMBOOT_TYPE = ActorType::PureMcts;
const int WARMBOOT_NUM = 3;
const int PURE_MCTS_SIMS = 2000;
const int RESNET_SIZE = 3;
const bool ENABLE_TIMER = true; 
const bool USE_DATA_AUGMENTATION = true;
const int TENSORRT_MEMORY_POOL_LIMIT = 1 << 28;
const int BOARD_SIZE = 8;
const int ACTION_SIZE = 64;
const int HISTORY_SIZE = 12;
const int PLANES_SIZE = 4;
const int REPLAY_SIZE = 10000;
const bool USE_GPU = true;
const int LOG_RANK = 5;
const int NUM_CHANNELS = 256;
const float VIRTUAL_LOSS = 2.0f;
const int PREDICT_BUFFER_SIZE = 5;
const int EVALUATE_PERIOD = -1; // -1 means close periodically evaluate

const int MANAGER_BLOCK_NUM = 200;
const int MANAGER_BUFFER_SIZE = 2;
const int MANAGER_CONBINE_MIN_BLOCK = 5;
const int MANAGER_CONBINE_MAX_BLOCK_NUM = 20;
const int TENSORRT_MAX_BATCH_SIZE = MANAGER_CONBINE_MAX_BLOCK_NUM * PREDICT_BUFFER_SIZE;

const std::string CKPT_DIR = "../ckpt";
const std::string LOG_DIR = "../log";

#endif