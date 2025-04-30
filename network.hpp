#ifndef Network_H
#define Network_H

#include "common.h"
#include "util.hpp"

struct AlphaZeroResNetImpl : torch::nn::Module {
    torch::nn::Conv2d conv1{nullptr};
    torch::nn::Conv2d conv2{nullptr};
    torch::nn::Conv2d conv3{nullptr};
    torch::nn::Conv2d conv4{nullptr};
    torch::nn::Sequential policy_head, value_head;

    AlphaZeroResNetImpl() { 
        conv1 = register_module("conv1", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(4, NUM_CHANNELS, 3).padding(1)));
        conv2 = register_module("conv2", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(NUM_CHANNELS, NUM_CHANNELS, 3).padding(1)));
        conv3 = register_module("conv3", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(NUM_CHANNELS, NUM_CHANNELS, 3).padding(1)));
        conv4 = register_module("conv4", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(NUM_CHANNELS, NUM_CHANNELS, 3).padding(1)));
        
        policy_head = register_module("policy_head", 
            torch::nn::Sequential(
                torch::nn::Conv2d(torch::nn::Conv2dOptions(NUM_CHANNELS, 4, 1)),
                torch::nn::ReLU(),
                torch::nn::Flatten(),
                torch::nn::Linear(4 * ACTION_SIZE, ACTION_SIZE), 
                torch::nn::Softmax(1)
        ));

        value_head = register_module("value_head", torch::nn::Sequential(
            torch::nn::Conv2d(
                torch::nn::Conv2dOptions(NUM_CHANNELS, 2, 1)),
                torch::nn::ReLU(),
                torch::nn::Flatten(),
                torch::nn::Linear(2 * ACTION_SIZE, ACTION_SIZE),
                torch::nn::ReLU(),
                torch::nn::Linear(ACTION_SIZE, 1),
                torch::nn::Tanh()
        ));
    }
    
    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
        x = torch::relu(conv1(x));
        x = torch::relu(conv2(x));
        x = torch::relu(conv3(x));
        x = torch::relu(conv4(x));
        auto pi = policy_head->forward(x);
        auto val = value_head->forward(x);

        return {pi, val};
    }
};
TORCH_MODULE(AlphaZeroResNet);

#endif