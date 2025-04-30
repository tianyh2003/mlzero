
#include "common.h"
#include "actor.hpp"
#include "replay.hpp"
#include "network_wrapper.hpp"
#include "logger.hpp"
#include "learner.hpp"
#include "util.hpp"
#include "evaluator.hpp"
#include <cstdio>

void train() {
    TrainConfig train_config;
    Logger logger = Logger("main.log");
    Timer timer_all, timer_learn, timer_act; 
    ReplayMemory replay_memory = ReplayMemory();
    std::mutex replay_mutex;
    NetworkWrapper network_wrapper = NetworkWrapper();
    int weight_id = (train_config.init_weight_id == -1) ? check_weight_id() : train_config.init_weight_id;
    if(weight_id == -1) {
        weight_id = 0;
    } else {
        network_wrapper.load_weight(get_weight_path(weight_id));
    }
    Learner learner = Learner(&replay_memory, &network_wrapper); 
    logger.log(0, "start train, weight id: " + std::to_string(weight_id));
    std::thread* threads[GAMES_PER_ITERATION];
    for(int iter = 0; iter < TRAIN_ITERATIONS; iter++) {
        timer_all.start();
        network_wrapper.eval();
        network_wrapper.update_tensorrt();

        if (EVALUATE_PERIOD != -1 && weight_id % EVALUATE_PERIOD == 0) {
            float score = Evaluator(&network_wrapper, weight_id).run();
            if (score > 0.6) {
                Logger("eval_main.log").log(0, "weight id: " + 
                    std::to_string(weight_id) + ", score: " + std::to_string(score));
                return;
            }
        }

        timer_act.start();
        for(int i = 0; i < GAMES_PER_ITERATION; i++) {
            threads[i] = new std::thread(
                run_actor_mp, 
                &network_wrapper,
                &replay_memory,
                &replay_mutex,
                iter
            );
        }
        for(int i = 0; i < GAMES_PER_ITERATION; i++) {
            threads[i]->join();
            delete threads[i];
        }

        timer_act.end();
        timer_learn.start();

        weight_id++; 
        network_wrapper.train();
        learner.run_learner_once(get_weight_path(weight_id));
        if (train_config.weight_save_interval != -1 && weight_id % train_config.weight_save_interval == 0) 
            network_wrapper.save_weights(get_weight_path(weight_id));
        
        timer_learn.end();
        timer_all.end();
        logger.log(0, "train iter: " + std::to_string(iter) + 
            ", all time: " + std::to_string(timer_all.get_all_duration_us() / 1000) + " ms"
            ", act time: " + std::to_string(timer_act.get_all_duration_us() / 1000) + " ms" + 
            ", learn time: " + std::to_string(timer_learn.get_all_duration_us() / 1000) + " ms");
    }
}

int main() {
    for (int i = 0; i < 100; i++) {
        train(); 
    }
    return 0;
}