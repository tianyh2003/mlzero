#ifndef MCTS_H
#define MCTS_H

#include "common.h"
#include "network_wrapper.hpp"

class Env {
public:
    int position[ACTION_SIZE];
    int legal_move[ACTION_SIZE];
    int history[HISTORY_SIZE][ACTION_SIZE];
    int observation[PLANES_SIZE][BOARD_SIZE][BOARD_SIZE];
    int history_idx;
    int player;
    int current_idx;
    int done;

    Env(); 
    Env(const Env& src);
    int check_over();
    void update_legal_move();
    void set_position(int* _position, int player);
    void fast_move(int next_idx);
    int move(int next_idx);
    void copy_legal_move(int* dest);
    void copy_position(int* dest);
    int get_player();
    void copy_obs(float* obs);
    void print();
    void check_env(); 
};

class Node {
public:
    int player;
    int next_idx;
    bool is_expanded;
    Node* parent;
    Node* child[ACTION_SIZE];
    int valid[ACTION_SIZE];
    float child_pi[ACTION_SIZE];
    float child_v[ACTION_SIZE];
    int child_n[ACTION_SIZE];

    Node(int player, Node* parent = nullptr);
};

class MCTSResult {
public:
    int action;
    int done;
    float obs[PLANES_SIZE * BOARD_SIZE * BOARD_SIZE];
    float pi[ACTION_SIZE];
};

class MCTS {
private:
    Node* dummy_node;
    Node* root_node;
    bool use_gpu;
    bool warm_up;
    bool add_noise;
    Env env;
    std::vector<Node*> all_nodes; 
    float noise_eps = 0.25f;

    int __uct(Node* node);
    void __simulate(class NetworkWrapper* network_wrapper);
    void __update_root(int action);
    void __add_dirichlet_noise(Node* node);
    void __generate_optim_pi(float* output);

public:
    MCTS(bool use_gpu = false, bool warm_up = false, bool add_noise = true, Env* _env = nullptr);
    ~MCTS();
    MCTSResult run(class NetworkWrapper* network_wrapper, int sims = 10);
    Env get_env();
};

#endif