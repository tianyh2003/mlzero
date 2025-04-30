#ifndef Network_H
#define Network_H

#include "common.h"
#include "util.hpp"

struct AlphaZeroResNetImpl: torch::nn::Module {
    torch::nn::Conv2d conv1{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr};
    std::vector<torch::nn::Sequential> residual_blocks;
    torch::nn::Sequential policy_head, value_head;

    AlphaZeroResNetImpl() {
        conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(4, NUM_CHANNELS, 3).padding(1)));
        bn1 = register_module("bn1", torch::nn::BatchNorm2d(NUM_CHANNELS));

        for (int i = 0; i < RESNET_SIZE; i++) {
            residual_blocks.push_back(
                register_module("res_block_" + std::to_string(i), torch::nn::Sequential(
                    torch::nn::Conv2d(torch::nn::Conv2dOptions(NUM_CHANNELS, NUM_CHANNELS, 3).padding(1)),
                    torch::nn::BatchNorm2d(NUM_CHANNELS),
                    torch::nn::ReLU(),
                    torch::nn::Conv2d(torch::nn::Conv2dOptions(NUM_CHANNELS, NUM_CHANNELS, 3).padding(1)),
                    torch::nn::BatchNorm2d(NUM_CHANNELS)
                )
            ));
        }

        policy_head = register_module("policy_head", 
            torch::nn::Sequential(
                torch::nn::Conv2d(torch::nn::Conv2dOptions(NUM_CHANNELS, 4, 1)),
                torch::nn::BatchNorm2d(4),
                torch::nn::ReLU(),
                torch::nn::Flatten(),
                torch::nn::Linear(4 * ACTION_SIZE, ACTION_SIZE),
                torch::nn::Softmax(1)
        ));

        value_head = register_module("value_head", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(NUM_CHANNELS, 2, 1)),
            torch::nn::BatchNorm2d(2),
            torch::nn::ReLU(),
            torch::nn::Flatten(),
            torch::nn::Linear(2 * ACTION_SIZE, ACTION_SIZE),
            torch::nn::ReLU(),
            torch::nn::Linear(ACTION_SIZE, 1),
            torch::nn::Tanh()
        ));
    }
    
    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
        x = torch::relu(bn1(conv1(x)));
        for (auto& block : residual_blocks) {
            auto residual = x;
            x = block->forward(x);
            x += residual;
            x = torch::relu(x);
        }

        auto pi = policy_head->forward(x);
        auto val = value_head->forward(x);

        return {pi, val};
    }
};

#endif