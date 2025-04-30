#ifndef ACTOR_HPP
#define ACTOR_HPP

#include "common.h"
#include "logger.hpp"
#include "mcts_mp.hpp"
#include "replay.hpp"
#include "util.hpp"
#include "network_wrapper.hpp"
#include <cstdio>

void add_augmented_data(float* _obs, float* _pi, float* _value, ReplayMemory* replay_memory);

static void run_mcts_mp(
    MCTS* mcts, 
    NetworkWrapper* network_wrapper, 
    int* mcts_nums, 
    std::mutex* mcts_mutex, 
    int mcts_thread_idx
) {
    while(true) {
        {
            std::lock_guard<std::mutex> lock(*mcts_mutex);
            if((*mcts_nums) >= ACT_SIMULATIONS) break;
            (*mcts_nums)++;
        }
        mcts->run_mp(network_wrapper, mcts_thread_idx);
    }
}

class Actor {
public:
    Timer mcts_timer; 
    Timer actor_timer;

    Logger logger = Logger("actor.log");
    MCTS* mcts;
    NetworkWrapper* network_wrapper;
    ReplayMemory* replay_memory; //protected by replay_mutex
    std::mutex* replay_mutex;

    float obs_list[ACTION_SIZE * PLANES_SIZE * BOARD_SIZE * BOARD_SIZE]; 
    float pi_list[ACTION_SIZE * ACTION_SIZE];
    float value_list[ACTION_SIZE];
    int player_list[ACTION_SIZE];
    int result_list_idx = 0; 
    int done = 0;
    
    Actor(
        NetworkWrapper* _network_wrapper, 
        ReplayMemory* _replay_memory, 
        std::mutex* _replay_mutex
    ) {
        network_wrapper = _network_wrapper;
        replay_memory = _replay_memory;
        replay_mutex = _replay_mutex;
    }

    MCTSResult run_puremcts(Env& _env) {
        MCTSPure mcts = MCTSPure(&_env);
        mcts.run(PURE_MCTS_SIMS);
        return mcts.deal_result();
    }

    MCTSResult run_apvmcts(Env& _env, NetworkWrapper& network_wrapper) {
        MCTS mcts = MCTS(&_env);
        int mcts_nums = 0;
        std::mutex mcts_mutex;
        std::thread* threads[NUM_THREADS_PER_MCTS];
        for(int i = 0; i < NUM_THREADS_PER_MCTS; i++) {
            threads[i] = new std::thread(
                run_mcts_mp, 
                &mcts, 
                &network_wrapper, 
                &mcts_nums, 
                &mcts_mutex, 
                i
            );
        }
        for(int i = 0; i < NUM_THREADS_PER_MCTS; i++) {
            threads[i]->join();
            delete threads[i];
        }
        return mcts.deal_result();
    }

    void run_actor(int train_iter) {
        actor_timer.start();

        ActorType actor_type = ActorType::ApvMcts;
        if (train_iter < WARMBOOT_NUM) actor_type = WARMBOOT_TYPE;  
        
        result_list_idx = 0;
        done = 0;
        Env env;
        
        while(done == 0) {
            MCTSResult mcts_result;
            if (actor_type == ActorType::ApvMcts) 
                mcts_result = run_apvmcts(env, *network_wrapper);
            else if (actor_type == ActorType::PureMcts)
                mcts_result = run_puremcts(env);
            else if (actor_type == ActorType::Arena) {
                if (env.get_player() == 1) mcts_result = run_apvmcts(env, *network_wrapper);
                else mcts_result = run_puremcts(env);
            }
            
            memcpy(
                obs_list + result_list_idx * PLANES_SIZE * BOARD_SIZE * BOARD_SIZE, 
                mcts_result.obs, 
                sizeof(float) * PLANES_SIZE * BOARD_SIZE * BOARD_SIZE
            );
            memcpy(
                pi_list + result_list_idx * ACTION_SIZE, 
                mcts_result.pi, 
                sizeof(float) * ACTION_SIZE
            );
            player_list[result_list_idx] = env.get_player();
            result_list_idx++;

            done = env.move(mcts_result.action);
        }
        if(done == 2) {
            for(int i = 0; i < result_list_idx; i++) 
                value_list[i] = 0;
        }
        else {
            for(int i = 0; i < result_list_idx; i++) {
                value_list[i] = (player_list[i] == done ? 1 : -1);
            }
        }
        {
            std::lock_guard<std::mutex> lock(*replay_mutex);
            for(int i = 0; i < result_list_idx; i++) {
                float* _obs = obs_list + i * PLANES_SIZE * BOARD_SIZE * BOARD_SIZE; 
                float* _pi = pi_list + i * ACTION_SIZE;
                float* _value = value_list + i;
                replay_memory->add(_obs, _pi, _value);
                if(USE_DATA_AUGMENTATION) {
                    add_augmented_data(_obs, _pi, _value, replay_memory);
                }
            }
        }
        actor_timer.end();
        // logger.log(0, "actor time: " + std::to_string(actor_timer.get_all_duration_us()/1000) + " ms");
    }
};


void run_actor_mp(
    NetworkWrapper* _network_wrapper, 
    ReplayMemory* _replay_memory, 
    std::mutex* replay_mutex,
    int train_iter
) {
    Actor actor = Actor(
        _network_wrapper, 
        _replay_memory, 
        replay_mutex
    );
    actor.run_actor(train_iter);
}

void rotate90(float* src_obs, float* dst_obs, float* src_pi, float* dst_pi) {
    for (int p = 0; p < PLANES_SIZE; p++) {
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                dst_obs[p * BOARD_SIZE * BOARD_SIZE + j * BOARD_SIZE + (BOARD_SIZE - 1 - i)] = 
                    src_obs[p * BOARD_SIZE * BOARD_SIZE + i * BOARD_SIZE + j];
            }
        }
    }
    
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            dst_pi[j * BOARD_SIZE + (BOARD_SIZE - 1 - i)] = src_pi[i * BOARD_SIZE + j];
        }
    }
}

void rotate180(float* src_obs, float* dst_obs, float* src_pi, float* dst_pi) {
    for (int p = 0; p < PLANES_SIZE; p++) {
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                dst_obs[p * BOARD_SIZE * BOARD_SIZE + (BOARD_SIZE - 1 - i) * BOARD_SIZE + (BOARD_SIZE - 1 - j)] = 
                    src_obs[p * BOARD_SIZE * BOARD_SIZE + i * BOARD_SIZE + j];
            }
        }
    }
    
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            dst_pi[(BOARD_SIZE - 1 - i) * BOARD_SIZE + (BOARD_SIZE - 1 - j)] = src_pi[i * BOARD_SIZE + j];
        }
    }
}

void rotate270(float* src_obs, float* dst_obs, float* src_pi, float* dst_pi) {
    for (int p = 0; p < PLANES_SIZE; p++) {
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                // (i,j) -> (BOARD_SIZE-1-j, i) 旋转270度
                dst_obs[p * BOARD_SIZE * BOARD_SIZE + (BOARD_SIZE - 1 - j) * BOARD_SIZE + i] = 
                    src_obs[p * BOARD_SIZE * BOARD_SIZE + i * BOARD_SIZE + j];
            }
        }
    }
    
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            dst_pi[(BOARD_SIZE - 1 - j) * BOARD_SIZE + i] = src_pi[i * BOARD_SIZE + j];
        }
    }
}

void flip_horizontal(float* src_obs, float* dst_obs, float* src_pi, float* dst_pi) {
    for (int p = 0; p < PLANES_SIZE; p++) {
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                // (i,j) -> (i, BOARD_SIZE-1-j) 水平翻转
                dst_obs[p * BOARD_SIZE * BOARD_SIZE + i * BOARD_SIZE + (BOARD_SIZE - 1 - j)] = 
                    src_obs[p * BOARD_SIZE * BOARD_SIZE + i * BOARD_SIZE + j];
            }
        }
    }
    
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            dst_pi[i * BOARD_SIZE + (BOARD_SIZE - 1 - j)] = src_pi[i * BOARD_SIZE + j];
        }
    }
}

void flip_horizontal_rotate90(float* src_obs, float* dst_obs, float* src_pi, float* dst_pi) {
    float temp_obs[PLANES_SIZE * BOARD_SIZE * BOARD_SIZE];
    float temp_pi[ACTION_SIZE];
    
    flip_horizontal(src_obs, temp_obs, src_pi, temp_pi);
    
    rotate90(temp_obs, dst_obs, temp_pi, dst_pi);
}

void flip_horizontal_rotate180(float* src_obs, float* dst_obs, float* src_pi, float* dst_pi) {
    float temp_obs[PLANES_SIZE * BOARD_SIZE * BOARD_SIZE];
    float temp_pi[ACTION_SIZE];
    
    flip_horizontal(src_obs, temp_obs, src_pi, temp_pi);
    
    rotate180(temp_obs, dst_obs, temp_pi, dst_pi);
}

void flip_horizontal_rotate270(float* src_obs, float* dst_obs, float* src_pi, float* dst_pi) {
    float temp_obs[PLANES_SIZE * BOARD_SIZE * BOARD_SIZE];
    float temp_pi[ACTION_SIZE];
    
    flip_horizontal(src_obs, temp_obs, src_pi, temp_pi);
    
    rotate270(temp_obs, dst_obs, temp_pi, dst_pi);
}


void add_augmented_data(float* _obs, float* _pi, float* _value, ReplayMemory* replay_memory) {
    // by deepseek

    float transformed_obs[PLANES_SIZE * BOARD_SIZE * BOARD_SIZE];
    float transformed_pi[ACTION_SIZE];
    
    // 1. 旋转90度
    rotate90(_obs, transformed_obs, _pi, transformed_pi);
    replay_memory->add(transformed_obs, transformed_pi, _value);

    // 2. 旋转180度
    rotate180(_obs, transformed_obs, _pi, transformed_pi);
    replay_memory->add(transformed_obs, transformed_pi, _value);
    
    // 3. 旋转270度
    rotate270(_obs, transformed_obs, _pi, transformed_pi);
    replay_memory->add(transformed_obs, transformed_pi, _value);
    
    // 4. 水平翻转
    flip_horizontal(_obs, transformed_obs, _pi, transformed_pi);
    replay_memory->add(transformed_obs, transformed_pi, _value);
    
    // 5. 水平翻转后旋转90度
    flip_horizontal_rotate90(_obs, transformed_obs, _pi, transformed_pi);
    replay_memory->add(transformed_obs, transformed_pi, _value);
    
    // 6. 水平翻转后旋转180度
    flip_horizontal_rotate180(_obs, transformed_obs, _pi, transformed_pi);
    replay_memory->add(transformed_obs, transformed_pi, _value);
    
    // 7. 水平翻转后旋转270度
    flip_horizontal_rotate270(_obs, transformed_obs, _pi, transformed_pi);
    replay_memory->add(transformed_obs, transformed_pi, _value);

}

#endif