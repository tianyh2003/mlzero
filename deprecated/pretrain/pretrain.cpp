#include "../common.h"
#include "../replay.hpp"
#include "../network_wrapper.hpp"
#include "../util.hpp"
#include "../learner.hpp"

int main() {
    NetworkWrapper network = NetworkWrapper(true); 
    // ReplayMemory replay_memory;
    // Learner learner = Learner(&replay_memory, &network); 

    // for(int row = 0; row < BOARD_SIZE; row++) {
    //     for(int col = 0; col < BOARD_SIZE - 4; col++) {
    //         for(int i = 0; i < 5; i++) {
    //             //black
    //             float obs[PLANES_SIZE * BOARD_SIZE * BOARD_SIZE] = {0};
    //             float pi[ACTION_SIZE] = {0};
    //             float value[1] = {0};
    //             for(int j = 0; j < 5; j++) {
    //                 obs[row * BOARD_SIZE + col + j] = 1;
    //             }
    //             int idx = row * BOARD_SIZE + col + i;
    //             obs[idx] = 0; 
    //             for(int j = 0; j < BOARD_SIZE * BOARD_SIZE; j++) {
    //                 obs[2 * BOARD_SIZE * BOARD_SIZE + j] = 1;
    //             }
    //             pi[idx] = 1;
    //             value[0] = 1;
    //             replay_memory.add(obs, pi, value);
    //             print_obs(obs);
    //             print_pi(pi);
    //             print_value(value);
    //         }
    //     }
    // }


}
