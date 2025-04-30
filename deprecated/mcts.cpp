#include "mcts.hpp"

Env::Env(){
    history_idx = 0;
    player = 1;
    current_idx = -1;
    done = 0;
    memset(position, 0, sizeof(position));
    memset(history, 0, sizeof(history));
    memset(observation, 0, sizeof(observation));
    update_legal_move();
}

Env::Env(const Env& src) {
    history_idx = src.history_idx; 
    player = src.player; 
    current_idx = src.current_idx; 
    done = src.done; 
    memcpy(position, src.position, sizeof(position)); 
    memcpy(history, src.history, sizeof(history)); 
    memcpy(observation, src.observation, sizeof(observation)); 
    update_legal_move();
}

int Env::check_over() {
    bool is_draw = true;
    for(int i = 0; i < ACTION_SIZE; ++i) {
        if(position[i] == 0) {
            is_draw = false;
            break;
        }
    }
    if(is_draw) return 2;

    int color = position[current_idx];
    int dx[8] = {0, -1, -1, 1, 0, 1, 1, -1}; 
    int dy[8] = {-1, 0, 1, 1, 1, 0, -1, -1}; 
    int counts[4] = {1, 1, 1, 1}; 
    
    int x = current_idx / BOARD_SIZE; 
    int y = current_idx % BOARD_SIZE;

    for(int dir = 0; dir < 8; ++dir) {
        int current_x = x + dx[dir];
        int current_y = y + dy[dir];
        int consecutive = 0;

        while(current_x >= 0 && current_x < BOARD_SIZE &&
                current_y >= 0 && current_y < BOARD_SIZE) {
            const int idx = current_x * BOARD_SIZE + current_y;
            if(position[idx] != color) break;
            consecutive++;
            current_x += dx[dir];
            current_y += dy[dir];
        }
        
        counts[dir % 4] += consecutive;
        
        if(counts[dir % 4] >= 5) {
            return color;
        }
    }
    return 0;
}
   
void Env::update_legal_move() {
    for(int i = 0; i < ACTION_SIZE; i++) 
        legal_move[i] = (position[i] == 0) ? 1 : 0; 
}

void Env::set_position(int* _position, int _player) {
    player = _player;
    history_idx = 0;
    current_idx = -1;
    done = 0;
    memset(history, 0, sizeof(history));
    memset(observation, 0, sizeof(observation));
    memcpy(position, _position, sizeof(int) * ACTION_SIZE); 
    update_legal_move();
}

void Env::fast_move(int next_idx) {
    position[next_idx] = player;
    player *= -1;
    current_idx = next_idx;
}

int Env::move(int next_idx) {
    if(position[next_idx] != 0) {
        exit(-1);
    }
    position[next_idx] = player;
    update_legal_move();
    player *= -1; 
    current_idx = next_idx; 
    done = check_over();
    memcpy(history[history_idx], position, sizeof(position));
    history_idx = (history_idx + 1) % HISTORY_SIZE;
    return done;
}

void Env::copy_legal_move(int* dest) {
    memcpy(dest, legal_move, sizeof(int) * ACTION_SIZE);
}

void Env::copy_position(int* dest) {
    memcpy(dest, position, sizeof(int) * ACTION_SIZE);
}

int Env::get_player(){
    return player;
}

void Env::copy_obs(float* obs) {
    float _obs[PLANES_SIZE][BOARD_SIZE][BOARD_SIZE];
    for(int i = 0; i < BOARD_SIZE; i++) {
        for(int j = 0; j < BOARD_SIZE; j++) {
            _obs[0][i][j] = (position[i * BOARD_SIZE + j] == 1) ? 1 : 0;
            _obs[1][i][j] = (position[i * BOARD_SIZE + j] == -1) ? 1 : 0;
            _obs[2][i][j] = 0;
            _obs[3][i][j] = (player == 1) ? 1 : 0; 
        }
    }
    _obs[2][current_idx / BOARD_SIZE][current_idx % BOARD_SIZE] = 1;
    memcpy(obs, &_obs[0][0][0], PLANES_SIZE * BOARD_SIZE * BOARD_SIZE * sizeof(float));
}

void Env::print() {
    int* pos = position;
    std::cout << "  ";
    for (int c = 0; c < BOARD_SIZE; ++c) {
        std::cout << c << ' ';
    }
    std::cout << '\n';
    for (int r = 0; r < BOARD_SIZE; ++r) {
        std::cout << r << ' ';  // 行号
        for (int c = 0; c < BOARD_SIZE; ++c) {
            const int idx = r * BOARD_SIZE + c;
            switch (pos[idx]) {
                case 1:  std::cout << "X "; break;
                case -1: std::cout << "O "; break;
                default: std::cout << "  "; 
            }
        }
        std::cout << '\n';
    }
    std::cout << std::endl; 
}

void Env::check_env() {
    std::cout << "postion:" << std::endl; 
    // for(int i = 0; i <  ACTION_SIZE; i++)   
    //     std::cout << position[i] << " ";
    this->print(); 
    std::cout << std::endl; 

    std::cout << "legal_move:" << std::endl; 
    for(int i = 0; i <  ACTION_SIZE; i++)   
        std::cout << legal_move[i] << " ";
    std::cout << std::endl; 

    std::cout << "history_idx:" << std::endl; 
    std::cout << history_idx << std::endl; 
    std::cout << "player:" << std::endl; 
    std::cout << player << std::endl; 
    std::cout << "current_idx:" << std::endl; 
    std::cout << current_idx << std::endl; 
    std::cout << "done:" << std::endl; 
    std::cout << done << std::endl;
}

Node::Node(int player, Node* parent): 
    player(player), is_expanded(false), parent(parent) {
    for (int i = 0; i < ACTION_SIZE; ++i) {
        child[i] = nullptr;
        valid[i] = false;
        child_pi[i] = 0.0f;
        child_v[i] = 0.0f;
        child_n[i] = 0;
    }
}

int MCTS::__uct(Node* node) {
    const int c_puct_base = 19652;
    const float c_puct_init = 1.25f;

    float child_q[ACTION_SIZE];
    for (int i = 0; i < ACTION_SIZE; ++i) {
        child_q[i] = node->child_v[i] / (node->child_n[i] > 0 ? node->child_n[i] : 1);
    }

    int node_n = node->parent->child_n[node->parent->next_idx];
    float pb_c = std::log((1.0f + node_n + c_puct_base) / c_puct_base) + c_puct_init;

    float child_u[ACTION_SIZE];
    for (int i = 0; i < ACTION_SIZE; ++i) {
        child_u[i] = pb_c * node->child_pi[i] * (std::sqrt(node_n) / (1 + node->child_n[i]));
    }

    float ucb_score[ACTION_SIZE];
    for (int i = 0; i < ACTION_SIZE; ++i) {
        ucb_score[i] = node->valid[i] ? (child_q[i] + child_u[i]) : -9999.0f;
    }

    if(node == root_node) {
        // std::cout << "va: " << std::endl;
        // for(int i = 0; i < BOARD_SIZE; i++) {
        //     for(int j = 0; j < BOARD_SIZE; j++) {
        //         printf("%d ", node->valid[i * BOARD_SIZE + j]);
        //     }
        //     std::cout << std::endl;   
        // }
        // std::cout << std::endl;
    
        // std::cout << "cq: " << std::endl;
        // for(int i = 0; i < BOARD_SIZE; i++) {
        //     for(int j = 0; j < BOARD_SIZE; j++) {
        //         printf("%.2f ", child_q[i * BOARD_SIZE + j]);
        //     }
        //     std::cout << std::endl;   
        // }
        // std::cout << std::endl;
    
        // std::cout << "cu: " << std::endl;
        // for(int i = 0; i < BOARD_SIZE; i++) {
        //     for(int j = 0; j < BOARD_SIZE; j++) {
        //         printf("%.2f ", child_u[i * BOARD_SIZE + j]);
        //     }
        //     std::cout << std::endl;   
        // }
        // std::cout << std::endl;
    
        // std::cout << "ucb_score: " << std::endl;
        // for(int i = 0; i < BOARD_SIZE; i++) {
        //     for(int j = 0; j < BOARD_SIZE; j++) {
        //         if(ucb_score[i * BOARD_SIZE + j] == -9999.0f) {
        //             printf("0.000 ");
        //         } else {
        //             printf("%.3f ", ucb_score[i * BOARD_SIZE + j]);
        //         }
                
        //     }
        //     std::cout << std::endl;   
        // }
        // std::cout << std::endl;
        // std::cout << "ucb_score end" << std::endl;
    
    }

    int max_idx = 0;
    for (int i = 1; i < ACTION_SIZE; ++i) {
        if (ucb_score[i] > ucb_score[max_idx]) max_idx = i;
    }

    // std::cout << "ucb_score: ";
    // for(int i = 0; i < ACTION_SIZE; i++) {
    //     std::cout << ucb_score[i] << " ";
    // }
    // std::cout << std::endl;
    // std::cout << "ucb_score end" << std::endl;

    if (!node->valid[max_idx]) throw std::runtime_error("Invalid UCT selection");

    return max_idx;
}

void MCTS::__simulate(NetworkWrapper* network_wrapper) {
    Node* current = root_node;
    Env current_env = env;
    float value = 0.0f;

    // Selection
    while (current->is_expanded) {
        int action = __uct(current);

        // std::cout << "action: " << action << std::endl;

        current->next_idx = action;
        current_env.fast_move(action);
        
        if (!current->child[action]) {
            current->child[action] = new Node(current_env.get_player(), current);
            all_nodes.push_back(current->child[action]);
        }
        current = current->child[action];
    }

    // Expansion
    // std::cout << "check expand: "<< std::endl;

    int done = current_env.check_over();

    // std::cout << "done: " << done << std::endl;

    if (done) {
        // std::cout << "game done" << std::endl;
        
        value = (done == 1 || done == -1) ? 1.0f : -1.0f;
    } else {
        current_env.update_legal_move();
        
        float obs[1][PLANES_SIZE][BOARD_SIZE][BOARD_SIZE];
        current_env.copy_obs(&obs[0][0][0][0]);

        // std::cout << "obs: " << std::endl;
        // for(int i = 0; i < PLANES_SIZE; i++) {
        //     for(int j = 0; j < BOARD_SIZE; j++) {
        //         for(int k = 0; k < BOARD_SIZE; k++) {
        //             std::cout << obs[0][i][j][k] << " ";
        //         }
        //         std::cout << std::endl;
        //     }
        //     std::cout << std::endl;
        // }

        network_wrapper->predict(1, &obs[0][0][0][0], current->child_pi, &value);

        // std::cout << "expand nn_data: " << std::endl;
        // std::cout << "nn_data: " << nn_data.val << std::endl;
        // for(int i = 0; i < ACTION_SIZE; i++) {
        //     std::cout << nn_data.pi[i] << " ";
        // }
        // std::cout << std::endl;
        
        current_env.copy_legal_move(current->valid);

        // std::cout << "child_legal_move: " << std::endl;
        // for(int i = 0; i < ACTION_SIZE; i++) {
        //     std::cout << current->valid[i] << " ";
        // }
        // std::cout << std::endl;

        current->is_expanded = true;
        value = -value;
    }

    // std::cout << "value: " << value << std::endl;
    // std::cout << "check expand end" << std::endl;

    // std::cout << "check backpropagation" << std::endl;

    // std::cout << "before" << std::endl;
    // Node* check_goal_node = current;
    // Node* check_current_node = this->root_node; 
    // while(check_current_node != check_goal_node) {
    //     std::cout << "check_trajectory_value: " << check_current_node->parent->child_v[check_current_node->parent->next_idx] << std::endl;
    //     check_current_node = check_current_node->child[check_current_node->next_idx];
    // }
    // std::cout << "check_goal_node end" << std::endl;


    // Backpropagation
    while (current->parent && current->parent != dummy_node) {
        current->parent->child_n[current->parent->next_idx]++;
        current->parent->child_v[current->parent->next_idx] += value;
        value = -value;
        current = current->parent;
    }

    // std::cout << "after" << std::endl;
    // check_current_node = this->root_node; 
    // while(check_current_node != check_goal_node) {
    //     std::cout << "check_trajectory_value: " << check_current_node->parent->child_v[check_current_node->parent->next_idx] << std::endl;
    //     check_current_node = check_current_node->child[check_current_node->next_idx];
    // }
    // std::cout << "check_goal_node end" << std::endl;

}

void MCTS::__update_root(int action) {
    if (root_node->child[action] == nullptr) {
        exit(-11);
    }
    root_node = root_node->child[action];
    root_node->parent = dummy_node;
    dummy_node->next_idx = 0; 
    dummy_node->child[0] = root_node;
    dummy_node->child_n[0] = 1;
    
}

void MCTS::__add_dirichlet_noise(Node* node) {
    float noise[ACTION_SIZE];
    float sum = 0.0f;

    std::gamma_distribution<float> gamma(0.3);
    std::random_device rd;

    for (int i=0; i < ACTION_SIZE; ++i) {
        noise[i] = gamma(rd);
        sum += noise[i];
    }

    for (int i = 0; i < ACTION_SIZE; ++i) {
        if (node->valid[i]) {
            noise[i] /= sum;
            node->child_pi[i] = node->child_pi[i] * (1 - noise_eps) + noise[i] * noise_eps;
        }
    }
}

void MCTS::__generate_optim_pi(float* output) {
    float total = 0.0f;
    float temp[ACTION_SIZE];
    float temperature = warm_up ? 1.0f : 0.1f;

    for (int i = 0; i < ACTION_SIZE; ++i) {
        temp[i] = pow(root_node->child_n[i] * root_node->valid[i], 1.0/temperature);
        total += temp[i];
    }

    for (int i = 0; i < ACTION_SIZE; ++i) {
        output[i] = temp[i] / total;
    }
}

MCTS::MCTS(bool use_gpu, bool warm_up, bool add_noise, Env* _env) :
    use_gpu(use_gpu), warm_up(warm_up), add_noise(add_noise) {
    dummy_node = new Node(0);
    root_node = new Node(env.get_player(), dummy_node);
    all_nodes.push_back(dummy_node);
    all_nodes.push_back(root_node);
    dummy_node->child[0] = root_node;
    dummy_node->valid[0] = true;
    dummy_node->child_n[0] = 1;
    dummy_node->is_expanded = true;
    dummy_node->next_idx = 0;
    if(_env != nullptr) {
        env.set_position(_env->position, _env->player);
        // printf("env is not null\n");
        // env.print();
    }
}

MCTS::~MCTS() {
    for (Node* node : all_nodes) {
        delete node;
    }
}

Env MCTS::get_env() {
    return env;
}

MCTSResult MCTS::run(NetworkWrapper* network_wrapper, int sims) {
    if (!root_node->is_expanded) {

        // std::cout << "# check root " << std::endl; 

        float val;
        float obs[1][PLANES_SIZE][BOARD_SIZE][BOARD_SIZE];

        env.copy_obs(&obs[0][0][0][0]);

        // std::cout << "obs " << std::endl; 
        // for(int i = 0; i < PLANES_SIZE; i++) {
        //     for(int j = 0; j < BOARD_SIZE; j++) {
        //         for(int k = 0; k < BOARD_SIZE; k++) {
        //             std::cout << obs[0][i][j][k] << " "; 
        //         }
        //         std::cout << std::endl; 
        //     }
        //     std::cout << std::endl; 
        // }
        
        network_wrapper->predict(1, &obs[0][0][0][0], root_node->child_pi, &val);
        // network_wrapper->predict_only_buffer(1, &obs[0][0][0][0], root_node->child_pi, &val);

        // for(int i = 0; i < BOARD_SIZE; i++) {
        //     for(int j = 0; j < BOARD_SIZE; j++) {
        //         printf("%.2f ", root_node->child_pi[i * BOARD_SIZE + j]);
        //     }
        //     printf("\n");
        // }
        // printf("\n");
        // printf("val: %.2f\n", val);
        
        // std::cout << "nn_data " << std::endl;
        // for(int i = 0; i < ACTION_SIZE; i++) {
        //     std::cout << nn_data.pi[i] << " "; 
        // }
        // std::cout << std::endl;
        // std::cout << nn_data.val << std::endl;

        env.copy_legal_move(root_node->valid);

        // std::cout << "valid " << std::endl;
        // for(int i = 0; i < ACTION_SIZE; i++) {
        //     std::cout << root_node->valid[i] << " ";
        // }
        // std::cout << std::endl;

        root_node->is_expanded = true;

        // std::cout << "child_pi " << std::endl;
        // for(int i = 0; i < ACTION_SIZE; i++) {
        //     std::cout << root_node->child_pi[i] << " ";
        // }
        // std::cout << std::endl;
        // std::cout << "# check root end" << std::endl; 
    }

    // std::cout << "# check dirichlet noise" << std::endl;
    // for(int i = 0; i < ACTION_SIZE; i++) {
    //     std::cout << root_node->child_pi[i] << " ";
    // }
    // std::cout << std::endl;
    
    if (add_noise) __add_dirichlet_noise(root_node);

    // std::cout << "after" << std::endl;
    // for(int i = 0; i < ACTION_SIZE; i++) {
    //     std::cout << root_node->child_pi[i] << " ";
    // }
    // std::cout << std::endl;
    // std::cout << "# check dirichlet noise end" << std::endl;

    // std::cout << "# check simulate" << std::endl;

    for (int i = 0; i < sims; ++i) {
        __simulate(network_wrapper);
    }

    // std::cout << "# check simulate end" << std::endl;

    MCTSResult result;
    result.action = 0;
    for (int i = 1; i < ACTION_SIZE; ++i) {
        if (root_node->child_n[i] > root_node->child_n[result.action]) {
            result.action = i;
        }
    }

    // printf("result child_n: \n");
    // for(int i = 0; i < BOARD_SIZE; i++) {
    //     for(int j = 0; j < BOARD_SIZE; j++) {
    //         printf("%d ", root_node->child_n[i * BOARD_SIZE + j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");

    env.copy_obs(result.obs);
    result.done = env.move(result.action);
    __generate_optim_pi(result.pi);

    // printf("result pi\n");
    // for(int i = 0; i < BOARD_SIZE; i++) {
    //     for(int j = 0; j < BOARD_SIZE; j++) {
    //         printf("%.2f ", result.pi[i * BOARD_SIZE + j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");

    __update_root(result.action);

    
    return result;
}
