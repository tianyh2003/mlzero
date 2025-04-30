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

class AtomicNode {
public:
    int player;
    bool is_expanded;
    AtomicNode* parent;
    AtomicNode* child[ACTION_SIZE];
    int valid[ACTION_SIZE];
    float child_pi[ACTION_SIZE];
    std::mutex mutex;

    std::atomic<float> child_v[ACTION_SIZE];
    std::atomic<int> child_n[ACTION_SIZE];

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
    AtomicNode* dummy_node;
    AtomicNode* root_node;
    bool use_gpu;
    bool warm_up;
    Env env;
    std::vector<AtomicNode*> all_nodes; 
    std::mutex mutex;
    float noise_eps = 0.25f;
    
    int __uct(AtomicNode* node, int node_visit_time) {

        const float c_puct = 1.0f;
    
        float child_q[ACTION_SIZE];
        float child_u[ACTION_SIZE];

        for (int i = 0; i < ACTION_SIZE; i++) {
            float child_v_value = node->child_v[i].load();
            int child_n_value = node->child_n[i].load();
            child_q[i] = child_v_value / (float)(child_n_value > 0 ? child_n_value : 1);
            child_u[i] = c_puct * node->child_pi[i] * (std::sqrt(node_visit_time) / (1 + child_n_value));
        }
    
        float ucb_score[ACTION_SIZE];
        for (int i = 0; i < ACTION_SIZE; i++) {
            ucb_score[i] = node->valid[i] ? (child_q[i] + child_u[i]) : -9999.0f;
        }
    
        int max_idx = 0;
        for (int i = 1; i < ACTION_SIZE; i++) {
            if (ucb_score[i] > ucb_score[max_idx]) max_idx = i;
        }
    
        if (!node->valid[max_idx]) {
            printf("\n");
            for (int i = 0; i < ACTION_SIZE; i++) {
                printf("%f ", node->child_pi[i]);
            }
            printf("\n");
            printf("node is expand %d\n", node->is_expanded);
            printf("node visit time %d\n", node_visit_time);
            std::cerr << ("no illegal move in uct\n");
            exit(-1);
        }
    
        return max_idx;
    }

    void __add_dirichlet_noise(AtomicNode* node) {
        float noise[ACTION_SIZE];
        float sum = 0.0f;
    
        std::gamma_distribution<float> gamma(0.3);
        std::random_device rd;
    
        for (int i = 0; i < ACTION_SIZE; i++) {
            noise[i] = gamma(rd);
            sum += noise[i];
        }
    
        for (int i = 0; i < ACTION_SIZE; i++) {
            if (node->valid[i]) {
                noise[i] /= sum;
                node->child_pi[i] = node->child_pi[i] * (1 - 0.1) + noise[i] * 0.1;
            }
        }
    }

    void __generate_optim_pi(float* output) {
        for (int i = 0; i < ACTION_SIZE; i++) {
            output[i] = log(root_node->child_n[i] * root_node->valid[i] + 1e-10);
        }
        softmax(output, ACTION_SIZE);
    }

public:
    MCTS(Env* _env = nullptr) {
        if(_env != nullptr) {
            env = Env(*_env);
        }
        dummy_node = new AtomicNode(0);
        root_node = new AtomicNode(env.get_player(), dummy_node);
        all_nodes.push_back(dummy_node);
        all_nodes.push_back(root_node);
        dummy_node->child[0] = root_node;
        dummy_node->valid[0] = true;
        dummy_node->child_n[0] = 1;
        dummy_node->is_expanded = true;
    }
    
    ~MCTS() {
        for (AtomicNode* node : all_nodes) {
            if (node != nullptr) {
                delete node;
                node = nullptr;
            }
        }
    }

    AtomicNode* get_root_node() {
        return root_node;
    }
    
    Env get_env() {
        return env;
    }

    MCTSResult deal_result(bool return_deterministic = false) {
        if(return_deterministic == true) {
            MCTSResult result;
            result.action = 0;
            for (int i = 0; i < ACTION_SIZE; i++) {
                if (root_node->child_n[i].load() > root_node->child_n[result.action].load()) {
                    result.action = i;
                }
            }
            env.copy_obs(result.obs);
            result.done = env.move(result.action);
            __generate_optim_pi(result.pi);
            return result;
        }

        MCTSResult result;
        env.copy_obs(result.obs);
        __generate_optim_pi(result.pi);

        std::vector<int> acts;
        std::vector<float> probs;
        for (int i = 0; i < ACTION_SIZE; i++) {
            if (root_node->valid[i]) {
                acts.push_back(i);
                probs.push_back(result.pi[i]);
            }
        }
        std::vector<float> noise(acts.size());
        float sum_noise = 0.0f;
        std::gamma_distribution<float> gamma(0.3);
        std::random_device rd;
        std::mt19937 gen(rd());
        for (size_t i = 0; i < acts.size(); ++i) {
            noise[i] = gamma(gen);
            sum_noise += noise[i];
        }
        for (size_t i = 0; i < acts.size(); ++i) {
            noise[i] /= sum_noise;
        }
        std::vector<float> combined_probs(acts.size());
        for (size_t i = 0; i < acts.size(); ++i) {
            combined_probs[i] = 0.75 * probs[i] + 0.25 * noise[i];
        }
        std::discrete_distribution<> dist(combined_probs.begin(), combined_probs.end());
        int selected_index = dist(gen);
        result.action = acts[selected_index];
        result.done = env.move(result.action);
        return result;
    }

    void virtual_backup(std::vector<std::pair<AtomicNode*, int>>& _trajectory, float value) {
        for(std::vector<std::pair<AtomicNode*, int>>::reverse_iterator iter = _trajectory.rbegin(); iter != _trajectory.rend(); iter++) {
            AtomicNode* node = iter->first;
            int action = iter->second;
            AtomicNode* parent_node = node->parent;
            parent_node->child_v[action].fetch_add(value);
            parent_node->child_v[action].fetch_add(VIRTUAL_LOSS);
            value = -value;
        }
    }

    void backup(std::vector<std::pair<AtomicNode*, int>>& _trajectory, float value) {
        for(std::vector<std::pair<AtomicNode*, int>>::reverse_iterator iter = _trajectory.rbegin(); iter != _trajectory.rend(); iter++) {
            AtomicNode* node = iter->first;
            int action = iter->second;
            AtomicNode* parent_node = node->parent;
            parent_node->child_v[action].fetch_add(value);
            value = -value;
        }
    }

    void run_mp(NetworkWrapper* network_wrapper, int thread_idx = 0) {
        AtomicNode* current = root_node;
        Env current_env = Env(env);
        float value = 0.0f;
        int done = 0;
        std::vector<std::pair<AtomicNode*, int>> node_trajectory;
        node_trajectory.push_back(std::pair<AtomicNode*, int>(current, 0));
        current->parent->child_n[0].fetch_add(1);
        int node_visit_time = current->parent->child_n[0].load();

        while(true) {
            {
                std::lock_guard<std::mutex> lock(current->mutex);
                if (current->is_expanded == false) {
                    done = current_env.check_over();
                    if (done) {
                        value = (done == 1 || done == -1) ? 1.0f : -1.0f;
                        virtual_backup(node_trajectory, value);
                        break;
                    }
                    current_env.update_legal_move();
                    float obs[1][PLANES_SIZE][BOARD_SIZE][BOARD_SIZE];
                    current_env.copy_obs(&obs[0][0][0][0]);
                    network_wrapper->predict(1, &obs[0][0][0][0], current->child_pi, &value);
                    current_env.copy_legal_move(current->valid);
                    current->is_expanded = true;
                    value = -value;
                    virtual_backup(node_trajectory, value);
                    break;
                }
            }

            int action = __uct(current, node_visit_time);
            current->child_n[action].fetch_add(1);
            current->child_v[action].fetch_sub(VIRTUAL_LOSS);
            node_visit_time = current->child_n[action].load();
            current_env.fast_move(action);

            {
                std::lock_guard<std::mutex> lock(current->mutex);
                if (!current->child[action]) {
                    current->child[action] = new AtomicNode(current_env.get_player(), current);
                    {
                        std::lock_guard<std::mutex> lock(mutex);
                        all_nodes.push_back(current->child[action]);
                    }
                }
            }

            current = current->child[action];
            node_trajectory.push_back(std::pair<AtomicNode*, int>(current, action));
        }
    }
};

class MCTSNode {
public:
    int player;
    bool is_expanded;
    MCTSNode* parent = nullptr;
    MCTSNode* child[ACTION_SIZE];
    int valid[ACTION_SIZE];
    float child_pi[ACTION_SIZE];
    float child_v[ACTION_SIZE];
    int child_n[ACTION_SIZE];
    int next_idx = -1;

    MCTSNode(int player, MCTSNode* parent = nullptr): player(player), is_expanded(false), parent(parent) {
        for (int i = 0; i < ACTION_SIZE; i++) {
            child[i] = nullptr;
            valid[i] = true;
            child_pi[i] = 0.0f;
            child_v[i] = 0.0f;
            child_n[i] = 0;
        }
    }
};

class MCTSPure {
public:
    Env env;
    MCTSNode* dummy_node;
    MCTSNode* root_node;
    std::vector<MCTSNode*> all_nodes;
    int result_action = -1; 

    MCTSPure(Env* _env = nullptr) {
        if(_env != nullptr) {
            env = Env(*_env);
        }
        dummy_node = new MCTSNode(1);
        root_node = new MCTSNode(env.get_player(), dummy_node);
        all_nodes.push_back(dummy_node);
        all_nodes.push_back(root_node);
        dummy_node->child[0] = root_node;
        dummy_node->valid[0] = true;
        dummy_node->child_n[0] = 1;
        dummy_node->is_expanded = true;
        srand((int)time(0));
    }

    ~MCTSPure() {
        for (MCTSNode* node : all_nodes) {
            if (node != nullptr) {
                delete node;
                node = nullptr;
            }
        }
    }

    int __uct(MCTSNode* node) {
        const float c_puct = 3.0f;
        float child_q[ACTION_SIZE];
        float child_u[ACTION_SIZE];
        int node_visit_time = node->parent->child_n[node->parent->next_idx];
        for (int i = 0; i < ACTION_SIZE; i++) {
            float child_v_value = node->child_v[i];
            int child_n_value = node->child_n[i];
            child_q[i] = child_v_value / (float)(child_n_value > 0 ? child_n_value : 1);
            child_u[i] = c_puct * node->child_pi[i] * (std::sqrt(node_visit_time) / (1 + child_n_value));
        }
        float ucb_score[ACTION_SIZE];
        for (int i = 0; i < ACTION_SIZE; i++) {
            ucb_score[i] = node->valid[i] ? (child_q[i] + child_u[i]) : -9999.0f;
        }
        int max_idx = 0;
        for (int i = 0; i < ACTION_SIZE; i++) {
            if (ucb_score[i] > ucb_score[max_idx]) max_idx = i;
        }
        if(!node->valid[max_idx]) {
            std::cerr << ("no illegal move in uct pure\n");
            exit(-1);
        }
        return max_idx;
    }

    void __generate_optim_pi(float* output) {
        // for (int i = 0; i < ACTION_SIZE; i++) {
        //     output[i] = log(root_node->child_n[i] * root_node->valid[i] + 1e-10);
        // }
        // softmax(output, ACTION_SIZE);
        float sum = 0.0f;
        for (int i = 0; i < ACTION_SIZE; i++) {
            output[i] = root_node->child_n[i] * root_node->valid[i];
            sum += output[i];
        }
        for (int i = 0; i < ACTION_SIZE; i++) {
            output[i] /= sum;
        }
    }

    MCTSResult deal_result() {
        MCTSResult result;
        if(result_action == -1) {
            std::cerr << ("no action in deal result\n");
            exit(-1);
        }
        result.action = result_action;
        env.copy_obs(result.obs);
        result.done = env.move(result.action);
        __generate_optim_pi(result.pi);
        return result;
    }

    void run_once() {
        MCTSNode* node = root_node;
        Env current_env = Env(env);
        int done = 0;

        while(node->is_expanded) {
            int action = __uct(node);
            node->next_idx = action;
            done = current_env.move(action);
            if(node->child[action] == nullptr) {
                node->child[action] = new MCTSNode(current_env.get_player(), node);
                all_nodes.push_back(node->child[action]);
            }
            node = node->child[action];
        }

        current_env.copy_legal_move(node->valid);
        if(done == 0) {
            float all_legal_move = 0; 
            for(int i = 0; i < ACTION_SIZE; i++) 
                if(node->valid[i]) all_legal_move += 1;
            float _each_pi = 1.0 / all_legal_move;
            for(int i = 0; i < ACTION_SIZE; i++)
                if(node->valid[i]) node->child_pi[i] = _each_pi;
                else node->child_pi[i] = 0.0f;
            node->is_expanded = true;
        }

        while(done == 0) {
            float action_random_pi[ACTION_SIZE] = {0};
            int valid[ACTION_SIZE] = {0};
            current_env.copy_legal_move(valid);
            for (int i = 0; i < ACTION_SIZE; i++) {
                if(valid[i]) action_random_pi[i] = rand() % 1000 / 1000.0;   
                else action_random_pi[i] = -9999.0f;   
            }
            int max_idx = 0;
            for (int i = 0; i < ACTION_SIZE; i++) {
                if (action_random_pi[i] > action_random_pi[max_idx]) max_idx = i;
            }
            if (!valid[max_idx]) {
                std::cerr << ("no illegal move in random predict\n");
                exit(-1);
            }
            current_env.move(max_idx);
            done = current_env.check_over();
        }

        while(node != dummy_node) {
            node = node->parent;
            node->child_n[node->next_idx] += 1;
            if(done == (node->player)) node->child_v[node->next_idx] += 1.0f;
            else if(done == -(node->player)) node->child_v[node->next_idx] -= 1.0f;
        }
    }

    int run(int n_sims = PURE_MCTS_SIMS) {
        while(n_sims--) run_once();
        int action_idx = 0;
        for(int i = 0; i < ACTION_SIZE; i++) 
            if (root_node->child_n[i] > root_node->child_n[action_idx]) action_idx = i;
        result_action = action_idx;
        return action_idx;
    }
};

#endif