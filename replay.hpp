#ifndef REPLAY_HPP
#define REPLAY_HPP

#include "common.h"
#include "util.hpp"
#include <memory>

class ReplayMemory {
private:
    int idx = 0;
    int count = 0;
    // float inputs[REPLAY_SIZE * PLANES_SIZE * BOARD_SIZE * BOARD_SIZE];
    // float pis[REPLAY_SIZE * ACTION_SIZE];  
    // float values[REPLAY_SIZE]; 
    float* inputs = new float[REPLAY_SIZE * PLANES_SIZE * BOARD_SIZE * BOARD_SIZE];
    float* pis = new float[REPLAY_SIZE * ACTION_SIZE];
    float* values = new float[REPLAY_SIZE];

public:
    ReplayMemory(){
    }

    ~ReplayMemory() {
        delete[] inputs;
        delete[] pis;
        delete[] values;
    }

    void add(float* obs, float* pi, float* value) {
        std::memcpy(
            inputs + idx * PLANES_SIZE * BOARD_SIZE * BOARD_SIZE, 
            obs, 
            PLANES_SIZE * BOARD_SIZE * BOARD_SIZE * sizeof(float)
        );
        std::memcpy(
            pis + idx * ACTION_SIZE, 
            pi, 
            ACTION_SIZE * sizeof(float)
        );
        std::memcpy(
            values + idx, 
            value, 
            sizeof(float)
        );

        idx = (idx + 1) % REPLAY_SIZE;
        count = std::min(count + 1, REPLAY_SIZE);
    }

    void sample(int batch_size, float* input_batch, float* pi_batch, float* value_batch) {

        
        const int valid_size = std::min(count, REPLAY_SIZE);
        if (valid_size == 0) {
            printf("replay memory is empty\n");
            exit(-1); 
        }

        std::random_device rd;
        std::mt19937 rng(rd());
        std::uniform_int_distribution<int> dist(0, valid_size-1);

        srand((unsigned int)time(NULL));

        for (int i = 0; i < batch_size; i++) {
            const int selected = dist(rng);
            std::memcpy(input_batch + i * PLANES_SIZE * BOARD_SIZE * BOARD_SIZE, 
                       inputs + selected * PLANES_SIZE * BOARD_SIZE * BOARD_SIZE,
                       PLANES_SIZE * BOARD_SIZE * BOARD_SIZE * sizeof(float));
            
            std::memcpy(pi_batch + i * ACTION_SIZE,
                       pis + selected * ACTION_SIZE,
                       ACTION_SIZE * sizeof(float));
            
            value_batch[i] = values[selected];
        }
    }

};

#endif