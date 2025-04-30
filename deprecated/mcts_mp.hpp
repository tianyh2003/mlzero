#ifndef MCTS_MP_H
#define MCTS_MP_H

#include "common.h"
#include "network_wrapper.hpp"
#include "logger.hpp"
#include "util.hpp"

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

    Env(){
        history_idx = 0;
        player = 1;
        current_idx = -1;
        done = 0;
        memset(position, 0, sizeof(position));
        memset(history, 0, sizeof(history));
        memset(observation, 0, sizeof(observation));
        update_legal_move();
    }
    
    Env(const Env& src) {
        history_idx = src.history_idx; 
        player = src.player; 
        current_idx = src.current_idx; 
        done = src.done; 
        memcpy(position, src.position, sizeof(position)); 
        memcpy(history, src.history, sizeof(history)); 
        memcpy(observation, src.observation, sizeof(observation)); 
        update_legal_move();
    }

    int check_over(){
        bool is_draw = true;
        for(int i = 0; i < ACTION_SIZE; i++) {
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

    void update_legal_move() {
        for(int i = 0; i < ACTION_SIZE; i++) 
            legal_move[i] = (position[i] == 0) ? 1 : 0; 
    }


    void set_position(int* _position, int _player) {
        player = _player;
        history_idx = 0;
        current_idx = -1;
        done = 0;
        memset(history, 0, sizeof(history));
        memset(observation, 0, sizeof(observation));
        memcpy(position, _position, sizeof(int) * ACTION_SIZE); 
        update_legal_move();
    }

    void fast_move(int next_idx) {
        position[next_idx] = player;
        player *= -1;
        current_idx = next_idx;
    }

    int move(int next_idx) {
        if(position[next_idx] != 0) {
            std::cerr << ("no illegal move\n");
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

    void copy_legal_move(int* dest) {
        memcpy(dest, legal_move, sizeof(int) * ACTION_SIZE);
    }

    void copy_position(int* dest) {
        memcpy(dest, position, sizeof(int) * ACTION_SIZE);
    }

    int get_player() {
        return player;
    }

    void copy_obs(float* obs) {
        float _obs[PLANES_SIZE][BOARD_SIZE][BOARD_SIZE];
        for(int i = 0; i < BOARD_SIZE; i++) {
            for(int j = 0; j < BOARD_SIZE; j++) {
                _obs[0][i][j] = (position[i * BOARD_SIZE + j] == 1) ? 1 : 0;
                _obs[1][i][j] = (position[i * BOARD_SIZE + j] == -1) ? 1 : 0;
                _obs[2][i][j] = 0;
                _obs[3][i][j] = (player == 1) ? 1 : 0; 
            }
        }
        if(current_idx != -1)
            _obs[2][current_idx / BOARD_SIZE][current_idx % BOARD_SIZE] = 1;
        memcpy(obs, &_obs[0][0][0], PLANES_SIZE * BOARD_SIZE * BOARD_SIZE * sizeof(float));
    }

    void print() {
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

    void check_env() {
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
    
};

class Node {
public:
    int player;
    bool is_expanded;
    Node* parent;
    Node* child[ACTION_SIZE];
    int valid[ACTION_SIZE];
    float child_pi[ACTION_SIZE];
    float child_v[ACTION_SIZE];
    int child_n[ACTION_SIZE];
    std::mutex mutex;

    Node(int player, Node* parent = nullptr): player(player), is_expanded(false), parent(parent) {
        for (int i = 0; i < ACTION_SIZE; i++) {
            child[i] = nullptr;
            valid[i] = false;
            child_pi[i] = 0.0f;
            child_v[i] = 0.0f;
            child_n[i] = 0;
        }
    }
};

class AtomicNode {
public:
    int player;
    bool is_expanded;
    AtomicNode* parent;
    AtomicNode* child[ACTION_SIZE];
    int valid[ACTION_SIZE];
    float child_pi[ACTION_SIZE];
    std::mutex mutex;

    // float child_v[ACTION_SIZE];
    // int child_n[ACTION_SIZE];
    atomic<float> child_v[ACTION_SIZE];
    atomic<int> child_n[ACTION_SIZE];

    AtomicNode(int player, AtomicNode* parent = nullptr): player(player), is_expanded(false), parent(parent) {
        for (int i = 0; i < ACTION_SIZE; i++) {
            child[i] = nullptr;
            valid[i] = true;
            child_pi[i] = 0.0f;
            child_v[i] = 0.0f;
            child_n[i] = 0;
        }
    }
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
    std::mutex mutex;
    float noise_eps = 0.25f;
    

    int __uct(Node* node, int node_visit_time) {
        // const int c_puct_base = 19652;
        // const float c_puct_init = 1.25f;
        // float pb_c = std::log((1.0f + node_visit_time + c_puct_base) / c_puct_base) + c_puct_init;

        const float c_puct = 5.0f;
    
        float child_q[ACTION_SIZE];
        for (int i = 0; i < ACTION_SIZE; i++) {
            child_q[i] = node->child_v[i] / (float)(node->child_n[i] > 0 ? node->child_n[i] : 1);
        }
    
        float child_u[ACTION_SIZE];
        for (int i = 0; i < ACTION_SIZE; i++) {
            child_u[i] = c_puct * node->child_pi[i] * (std::sqrt(node_visit_time) / (1 + node->child_n[i]));
        }
    
        float ucb_score[ACTION_SIZE];
        for (int i = 0; i < ACTION_SIZE; i++) {
            ucb_score[i] = node->valid[i] ? (child_q[i] + child_u[i]) : -9999.0f;
        }
    
        int max_idx = 0;
        for (int i = 1; i < ACTION_SIZE; i++) {
            if (ucb_score[i] > ucb_score[max_idx]) max_idx = i;
        }
    
        if (!node->valid[max_idx]) throw std::runtime_error("Invalid UCT selection");
    
        return max_idx;
    }

    void __update_root(int action) {
        if (root_node->child[action] == nullptr) {
            exit(-11);
        }
        root_node = root_node->child[action];
        root_node->parent = dummy_node;
        dummy_node->child[0] = root_node;
        dummy_node->child_n[0] = 1;
    }

    void __add_dirichlet_noise(Node* node) {
        float noise[ACTION_SIZE];
        float sum = 0.0f;
    
        std::gamma_distribution<float> gamma(0.3);
        std::random_device rd;
    
        for (int i=0; i < ACTION_SIZE; i++) {
            noise[i] = gamma(rd);
            sum += noise[i];
        }
    
        for (int i = 0; i < ACTION_SIZE; i++) {
            if (node->valid[i]) {
                noise[i] /= sum;
                node->child_pi[i] = node->child_pi[i] * (1 - noise_eps) + noise[i] * noise_eps;
            }
        }
    }

    void __generate_optim_pi(float* output) {
        float total = 0.0f;
        float temp[ACTION_SIZE];
        float temperature = warm_up ? 1.0f : 0.1f;
    
        for (int i = 0; i < ACTION_SIZE; i++) {
            temp[i] = pow(root_node->child_n[i] * root_node->valid[i], 1.0/temperature);
            total += temp[i];
        }
    
        for (int i = 0; i < ACTION_SIZE; i++) {
            output[i] = temp[i] / total;
        }
    }

public:
    MCTS(bool use_gpu = false, bool warm_up = false, bool add_noise = true, Env* _env = nullptr):
        use_gpu(use_gpu), warm_up(warm_up), add_noise(add_noise) {
        if(_env != nullptr) {
            env = Env(*_env);
        }
        dummy_node = new Node(0);
        root_node = new Node(env.get_player(), dummy_node);
        all_nodes.push_back(dummy_node);
        all_nodes.push_back(root_node);
        dummy_node->child[0] = root_node;
        dummy_node->valid[0] = true;
        dummy_node->child_n[0] = 1;
        dummy_node->is_expanded = true;
    }
    
    ~MCTS() {
        for (Node* node : all_nodes) {
            if (node != nullptr) {
                delete node;
                node = nullptr;
            }
        }
    }

    Node* get_root_node() {
        return root_node;
    }
    
    Env get_env() {
        return env;
    }


    MCTSResult deal_result() {
        MCTSResult result;
        result.action = 0;
        for (int i = 0; i < ACTION_SIZE; i++) {
            if (root_node->child_n[i] > root_node->child_n[result.action]) {
                result.action = i;
            }
        }

        env.copy_obs(result.obs);

        result.done = env.move(result.action);
        __generate_optim_pi(result.pi);
        __update_root(result.action);
    
        return result;
    }

    void virtual_backup(std::vector<std::pair<Node*, int>>& _trajectory) {
        // std::string info = "virtual backup\n";
        for(std::vector<std::pair<Node*, int>>::reverse_iterator iter = _trajectory.rbegin(); iter != _trajectory.rend(); iter++) {

            Node* node = iter->first;
            int action = iter->second;
            Node* parent_node = node->parent;
            std::lock_guard<std::mutex> lock(parent_node->mutex);

            // info += "virtual backup parent_node child_n before: \n";
            // for(int i = 0; i < BOARD_SIZE; i++) {
            //     for(int j = 0; j < BOARD_SIZE; ++j) {
            //         info += std::to_string(parent_node->child_n[i * BOARD_SIZE + j]) + " ";
            //     }
            //     info += "\n";
            // }
            // info += "\n";
            // logger.log(1, info);
            // info.clear();

            parent_node->child_n[action]++;

            // info += "virtual backup parent_node child_n after: \n";
            // for(int i = 0; i < BOARD_SIZE; i++) {
            //     for(int j = 0; j < BOARD_SIZE; ++j) {
            //         info += std::to_string(parent_node->child_n[i * BOARD_SIZE + j]) + " ";
            //     }
            //     info += "\n";
            // }
            // info += "\n";
            // logger.log(1, info);
            // info.clear();
        }
    }

    void virtual_real_backup(std::vector<std::pair<Node*, int>>& _trajectory, float value) {
        // std::string info = "backup\n";
        for(std::vector<std::pair<Node*, int>>::reverse_iterator iter = _trajectory.rbegin(); iter != _trajectory.rend(); iter++) {
            Node* node = iter->first;
            int action = iter->second;
            Node* parent_node = node->parent;
            std::lock_guard<std::mutex> lock(parent_node->mutex);
            // node->parent->child_n[action]++;

            // info += " backup parent_node child_v before: \n";
            // for(int i = 0; i < BOARD_SIZE; i++) {
            //     for(int j = 0; j < BOARD_SIZE; ++j) {
            //         info += std::to_string(parent_node->child_v[i * BOARD_SIZE + j]) + " ";
            //     }
            //     info += "\n";
            // }
            // info += "\n";
            // logger.log(1, info);
            // info.clear();

            parent_node->child_v[action] += value;
            value = -value;

            // info += " backup parent_node child_v after: \n";
            // for(int i = 0; i < BOARD_SIZE; i++) {
            //     for(int j = 0; j < BOARD_SIZE; ++j) {
            //         info += std::to_string(parent_node->child_v[i * BOARD_SIZE + j]) + " ";
            //     }
            //     info += "\n";
            // }
            // info += "\n";
            // logger.log(1, info);
            // info.clear();
        }
    }

    void backup(std::vector<std::pair<Node*, int>>& _trajectory, float value) {
        for(std::vector<std::pair<Node*, int>>::reverse_iterator iter = _trajectory.rbegin(); iter != _trajectory.rend(); iter++) {
            Node* node = iter->first;
            int action = iter->second;
            Node* parent_node = node->parent;
            std::lock_guard<std::mutex> lock(parent_node->mutex);
            parent_node->child_n[action]++;
            parent_node->child_v[action] += value;
            value = -value;
        }
    }

    void run_mp(NetworkWrapper* network_wrapper, int thread_idx) {
        // Logger logger = Logger(LOG_RANK, "mcts" + std::to_string(thread_idx) + ".log");
        // std::string info = "thread " + std::to_string(thread_idx) + " start\n";

        {
            std::lock_guard<std::mutex> lock(root_node->mutex);
            if (!root_node->is_expanded) {
                float val;
                float obs[1][PLANES_SIZE][BOARD_SIZE][BOARD_SIZE];
                env.copy_obs(&obs[0][0][0][0]);
                
                // printf("obs: \n");
                // for(int i = 0; i < PLANES_SIZE; i++) {
                //     for(int j = 0; j < BOARD_SIZE; j++) {
                //         for(int k = 0; k < BOARD_SIZE; k++) {
                //             printf("%.2f ", obs[0][i][j][k]);
                //         }
                //         printf("\n");
                //     }
                //     printf("\n");
                // }

                network_wrapper->predict(1, &obs[0][0][0][0], root_node->child_pi, &val);
                env.copy_legal_move(root_node->valid);
                root_node->is_expanded = true;

                //print child pi
                // for(int i = 0; i < BOARD_SIZE; i++) {
                //     for(int j = 0; j < BOARD_SIZE; j++) {
                //         printf("%.2f ", root_node->child_pi[i * BOARD_SIZE + j]);
                //     }
                //     printf("\n");
                // }
                // printf("\n");
                // printf("val: %f\n", val);
                

                // info += " root node is expanded\n";

            } 
            if(add_noise) __add_dirichlet_noise(root_node);
        }

        Node* current = root_node;
        Env current_env = Env(env);
        float value = 0.0f;
        int done = 0;
        std::vector<std::pair<Node*, int>> node_trajectory;
        node_trajectory.push_back(std::pair<Node*, int>(current, 0));
        int node_visit_time = current->parent->child_n[0];

        while(true) {
            std::lock_guard<std::mutex> lock(current->mutex);
            
            if (!current->is_expanded) {

                // info += " arrived at leaf node\n";
                // logger.log(1, info);
                // info.clear(); 

                done = current_env.check_over();
                if (done) {
                    value = (done == 1 || done == -1) ? 1.0f : -1.0f;
                    backup(node_trajectory, value);
                    break;
                }
                current_env.update_legal_move();
                float obs[1][PLANES_SIZE][BOARD_SIZE][BOARD_SIZE];
                current_env.copy_obs(&obs[0][0][0][0]);
                virtual_backup(node_trajectory);
                network_wrapper->predict(1, &obs[0][0][0][0], current->child_pi, &value);

                // logger.log(1, "predict value: " + std::to_string(value) + "\n");

                current_env.copy_legal_move(current->valid);
                current->is_expanded = true;
                value = -value;
                virtual_real_backup(node_trajectory, value);
                break;
            }

            // current_env.print();

            int action = __uct(current, node_visit_time);
            
            // info += " node_visit_time: " + std::to_string(node_visit_time) + "\n";
            // info += " action: " + std::to_string(action) + "\n";

            current_env.fast_move(action);

            // for(int i = 0; i < BOARD_SIZE; i++) {
            //     for(int j = 0; j < BOARD_SIZE; j++) {
            //         info += std::to_string(current_env.position[i * BOARD_SIZE + j]) + " ";
            //     }
            //     info += "\n";
            // }

            // logger.log(1, info);
            // info = "";

            if (!current->child[action]) {
                current->child[action] = new Node(current_env.get_player(), current);
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    all_nodes.push_back(current->child[action]);
                }
            }
            node_visit_time = current->child_n[action];
            current = current->child[action];
            node_trajectory.push_back(std::pair<Node*, int>(current, action));
        }
        // logger.log(1, info);
        // logger.flush();
    }
};

#endif