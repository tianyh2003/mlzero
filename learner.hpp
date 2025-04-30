#ifndef LEARNER_HPP
#define LEARNER_HPP

#include "common.h"
#include "replay.hpp"
#include "network_wrapper.hpp"
#include "logger.hpp"
#include "util.hpp"

class Learner {
public:
    Logger logger = Logger("learner.log");
    NetworkWrapper* network_wrapper;
    ReplayMemory* replay_memory;
    int train_iters = 0; 
    float lr_multipier = 1.0;
    float init_lr = 2e-3f; 

    Learner(
        ReplayMemory* _replay_memory,
        NetworkWrapper* _network_wrapper
    ) {
        replay_memory = _replay_memory;
        network_wrapper = _network_wrapper;
    }

    void run_learner_once(std::string weight_path) {
        float loss = 0; 
        float entropy = 0;
        float kl = 0; 
        int steps = 0;
        for(int i = 0; i < TRAIN_STEPS; i++) {
            float data_obs[TRAIN_BATCH_SIZE * PLANES_SIZE * BOARD_SIZE * BOARD_SIZE];
            float data_pi[TRAIN_BATCH_SIZE * ACTION_SIZE];
            float data_value[TRAIN_BATCH_SIZE];
            replay_memory->sample(
                TRAIN_BATCH_SIZE, 
                data_obs, 
                data_pi, 
                data_value
            );

            TrainResult train_result = network_wrapper->train(
                TRAIN_BATCH_SIZE, 
                data_obs, 
                data_pi, 
                data_value
            );
            loss = (loss * steps + train_result.loss) / (steps + 1);
            entropy = (entropy * steps + train_result.entropy) / (steps + 1);
            kl = (kl * steps + train_result.kl) / (steps + 1);
            steps++;
        }

        train_iters++; 
        if (train_iters == 100 || train_iters == 200 || train_iters == 300 || train_iters == 500) {
            lr_multipier = std::max(0.1f, lr_multipier * 0.5f);
            network_wrapper->set_lr(init_lr * lr_multipier);
        }

        std::stringstream ss;
        ss << "train_iters: " << train_iters << " loss: " << loss << " entropy: " << entropy << " kl: " << kl << " lr: " << init_lr * lr_multipier;
        logger.log(0, ss.str());
    }
};

#endif
