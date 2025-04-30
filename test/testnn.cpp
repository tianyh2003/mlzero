#include "../network_wrapper.hpp"
#include "../common.h"
#include "../logger.hpp"

// void testnn() {
//     try {
//         // 初始化网络
//         NetworkWrapper network(18, 19, 362); // 围棋参数示例
        
//         // 生成测试数据
//         auto input = torch::rand({1, 18, 19, 19}); // 批量大小1
//         auto pi = torch::randint(0, 362, {1});
//         auto v = torch::rand({1});
        
//         // 预测示例
//         auto [policy, value] = network.predict(input);
//         std::cout << "Predicted value: " << value << std::endl;
        
//         // 训练示例
//         for(int i = 0; i < 10; ++i) {
//             network.train(input, pi, v);
//             std::cout << "Training step " << i << " completed" << std::endl;
//         }
        
//         // 保存权重
//         network.save_weights("latest.pt");
        
//     } catch (const std::exception& e) {
//         std::cerr << "Error: " << e.what() << std::endl;
//         return 1;
//     }
// }

void testnnwrapper(NetworkWrapper* network) {
    try {
        
        float input[1][3][8][8] = {0};
        float pi[64] = {0};
        float v = 0.0f;
        pi[1] = 1.0f;
        v = 0.5f;
        float output_pi[1][64] = {0};
        float output_v[1] = {0};
        printf("into networkwrapper\n");
        network->predict(1, &input[0][0][0][0], &output_pi[0][0], output_v);
        std::cout << "Predicted value: " << output_v[0] << std::endl;
        std::cout << "Predicted pi" << std::endl;
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                printf("%.2f ", output_pi[0][i * 8 + j]);
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
        
        // for(int i = 0; i < 10; ++i) {
        //     network.train(input, pi, v);
        //     std::cout << "Training step " << i << " completed" << std::endl;
        // }
        
        // network.save_weights("latest.pt");
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return;
    }
}

int main() {
    Logger* logger = new Logger(5, "test.log");
    NetworkWrapper* network = new NetworkWrapper(logger);
    int thread_num = 15; 
    std::thread* threads[20];
    for(int i = 0; i < thread_num; i++) {
        threads[i] = new std::thread(testnnwrapper, network);
    }

    for(int i = 0; i < thread_num; i++) {
        threads[i]->join();
    }
    while(true) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}