// #include "replay.hpp"


// // 示例用法
// int main() {
//     // 创建经验池
//     ReplayMemory pool(REPLAY_SIZE);

//     // 创建模拟数据
//     float sample_state[PLANES_SIZE][BOARD_SIZE][BOARD_SIZE] = {0};
//     float sample_pi[ACTION_SIZE] = {0};
//     float sample_value = 0.5f;

//     sample_state[0][0][0] = 1;  
//     sample_state[1][0][1] = 1;
//     sample_state[2][0][2] = 1;
//     sample_state[0][1][0] = 1;  
//     sample_state[1][1][1] = 1;
//     sample_state[2][1][2] = 1;
//     sample_state[0][2][0] = 1;
//     sample_state[1][2][1] = 1;
//     sample_state[2][2][2] = 1;
//     sample_pi[0] = 0.1; 
//     sample_pi[1] = 0.2;
//     sample_pi[2] = 0.3;

//     // 添加经验
//     pool.add(sample_state,  // 三维数组转一维指针
//              sample_pi,
//              sample_value);

//     sample_pi[0] = 0.4; 
//     sample_pi[1] = 0.5;
//     sample_pi[2] = 0.6;
//     pool.add(sample_state,  // 三维数组转一维指针
//         sample_pi,
//         sample_value);

//     sample_pi[0] = 0.7; 
//     sample_pi[1] = 0.8;
//     sample_pi[2] = 0.9;
//     pool.add(sample_state,  // 三维数组转一维指针
//         sample_pi,
//         sample_value);
//     // 准备采样内存
//     float batch_input[BATCH_SIZE][PLANES_SIZE][BOARD_SIZE][BOARD_SIZE];
//     float batch_pi[BATCH_SIZE][ACTION_SIZE];
//     float batch_value[BATCH_SIZE];

//     // 执行采样
//     pool.sample(BATCH_SIZE, 
//                batch_input, 
//                batch_pi,
//                batch_value);

//     for(int i = 0; i < BATCH_SIZE; i++) {
//         for(int j = 0; j < PLANES_SIZE; j++) {
//             for(int k = 0; k < BOARD_SIZE; k++) {
//                 for(int l = 0; l < BOARD_SIZE; l++) {
//                     std::cout << batch_input[i][j][k][l] << " ";
//                 }
//                 std::cout << std::endl;
//             }
//             std::cout << std::endl;
//         }
//         for(int j = 0; j < ACTION_SIZE; j++) {
//             std::cout << batch_pi[i][j] << " ";
//         }
//         std::cout << std::endl;
//         std::cout << batch_value[i] << std::endl;
//     }

//     return 0;
// }