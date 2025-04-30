#ifndef EVALUATOR_HPP
#define _EVALUATOR_HPP

#include "common.h"
#include "network_wrapper.hpp"
#include "mcts_mp.hpp"
#include "logger.hpp"

const float MINMAX_SCORE_RATIO = 1;

class MinMax {
public:
    int max_depth = 1; 
    int next_x = -1; 
    int next_y = -1;
    
    MinMax(int _max_depth = 1) {
        max_depth = _max_depth;
    }

    int eval_shape[15][8] = {
        {5, 5, 0, 1, 1, 0, 0, -1}, 
        {5, 5, 0, 0, 1, 1, 0, -1},
        {5, 20, 1, 1, 0, 1, 0, -1},
        {5, 50, 0, 0, 1, 1, 1, -1},
        {5, 50, 1, 1, 1, 0, 0, -1},
        {5, 500, 0, 1, 1, 1, 0, -1},
        {6, 500, 0, 1, 0, 1, 1, 0},
        {6, 500, 0, 1, 1, 0, 1, 0},
        {5, 500, 1, 1, 1, 0, 1, -1},
        {5, 500, 1, 1, 0, 1, 1, -1},
        {5, 500, 1, 0, 1, 1, 1, -1},
        {5, 500, 1, 1, 1, 1, 0, -1},
        {5, 500, 0, 1, 1, 1, 1, -1},
        {6, 5000, 0, 1, 1, 1, 1, 0},
        {5, 99999999, 1, 1, 1, 1, 1, -1}
    }; 

    int direction[8][2] = {
        {1, 0}, {0, 1}, {-1, 0}, {0, -1},
        {1, 1}, {1, -1}, {-1, 1}, {-1, -1}
    };

    inline int _get_x(int idx) {
        return idx / BOARD_SIZE;
    }

    inline int _get_y(int idx) {
        return idx % BOARD_SIZE;
    }

    inline int _get_idx(int x, int y) {
        return x * BOARD_SIZE + y;
    }

    int _cal_subop(int* pos, int player, int x, int y, int dx, int dy, int& done) {
        if (x + 4 * dx < 0 || x + 4 * dx >= BOARD_SIZE || y + 4 * dy < 0 || y + 4 * dy >= BOARD_SIZE) {
            return 0; 
        }
        int score = 0;
        for (int i = 0; i < 15; i++) {
            int len = eval_shape[i][0];
            int shape_score = eval_shape[i][1];
            if (len > 5) {
                if (x + (len - 1) * dx < 0 || x + (len - 1) * dx >= BOARD_SIZE || y + (len - 1) * dy < 0 || y + (len - 1) * dy >= BOARD_SIZE) {
                    continue;
                }
            }
            for (int j = 0; j < len; j++) {
                int nx = x + j * dx;
                int ny = y + j * dy;
                int state = pos[_get_idx(nx, ny)];
                if (state == player) state = 1; 
                else if (state == -player) state = -1;
                else state = 0; 
                if (state != eval_shape[i][j + 2]) break; 
                if (j == len - 1) score += shape_score;
                if (j == len - 1 && i == 15 - 1) done = player;
            }
        }
        return score;
    }

    double eval(int* pos, int player, int& done) {
        int black_score = 0;
        int white_score = 0; 
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                int _score = 0; 
                for (int k = 0; k < 8; k++) {
                    int dx = direction[k][0];
                    int dy = direction[k][1];
                    black_score += _cal_subop(pos, 1, i, j, dx, dy, done);
                    white_score += _cal_subop(pos, -1, i, j, dx, dy, done);
                }
            }
        }
        if (player == 1) return black_score - white_score * MINMAX_SCORE_RATIO * 0.1;
        else return white_score - black_score * MINMAX_SCORE_RATIO * 0.1;
    }

    void copy_pos(int* dst, int* src) {
        for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; i++) {
            dst[i] = src[i];
        }
    }

    double minmax(int* pos, int player, int depth, double alpha, double beta) {
        int done = 0; 
        double state_val = eval(pos, player, done);
        if (depth == 0 || done != 0) {
            return state_val;
        }
        int cur_pos[BOARD_SIZE * BOARD_SIZE];
        copy_pos(cur_pos, pos);
        for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; i++) {
            if (cur_pos[i] != 0) continue; 
            if (depth == max_depth && next_x == -1 && next_y == -1) {
                next_x = _get_x(i);
                next_y = _get_y(i);
            }
            cur_pos[i] = player;
            double val = -minmax(cur_pos, -player, depth - 1, -beta, -alpha);
            cur_pos[i] = 0;
            if (val > alpha) {
                alpha = val;
                if (depth == max_depth) {
                    next_x = _get_x(i);
                    next_y = _get_y(i);
                }
                if (alpha >= beta) return beta;
            }
        }
        return alpha;
    }

    int check_over(int* pos, int x, int y){
        bool is_draw = true;
        for(int i = 0; i < BOARD_SIZE * BOARD_SIZE; i++) {
            if(pos[i] == 0) {
                is_draw = false;
                break;
            }
        }
        if(is_draw) return 2;
    
        int color = pos[_get_idx(x, y)];
        int dx[8] = {0, -1, -1, 1, 0, 1, 1, -1}; 
        int dy[8] = {-1, 0, 1, 1, 1, 0, -1, -1}; 
        int counts[4] = {1, 1, 1, 1}; 
    
        for(int dir = 0; dir < 8; ++dir) {
            int current_x = x + dx[dir];
            int current_y = y + dy[dir];
            int consecutive = 0;
    
            while(current_x >= 0 && current_x < BOARD_SIZE &&
                    current_y >= 0 && current_y < BOARD_SIZE) {
                int idx = current_x * BOARD_SIZE + current_y;
                if(pos[idx] != color) break;
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

    void print_pos(int* pos) {
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                int player = pos[_get_idx(i, j)];
                if (player == 0) printf("  ");
                else if (player == 1) printf("X ");
                else if (player == -1) printf("O ");
            }
            printf("\n");
        }
    }

    void test() {
        int pos[BOARD_SIZE * BOARD_SIZE] = {
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0
        };
        int done = 0; 
        int player = 1; 
        while(done == 0) {
            minmax(pos, player, max_depth, -99999999, 99999999);
            if (next_x == -1 || next_y == -1 || pos[_get_idx(next_x, next_y)] != 0) {
                printf("illegal move, x: %d, y: %d\n", next_x, next_y);
                exit(0);
            }
            pos[_get_idx(next_x, next_y)] = player;
            done = check_over(pos, next_x, next_y);
            player = -player;
            std::cout << "next_x: " << next_x << ", next_y: " << next_y << std::endl;
            print_pos(pos);
            next_x = -1, next_y = -1; 
        }
    }

    void run(int* pos, int player, int& done, int& _next_x, int& _next_y) {
        int cur_pos[BOARD_SIZE * BOARD_SIZE];
        copy_pos(cur_pos, pos);
        minmax(cur_pos, player, max_depth, -99999999, 99999999);
        if (next_x == -1 || next_y == -1 || cur_pos[_get_idx(next_x, next_y)] != 0) {
            print_pos(cur_pos);
            printf("illegal move, x: %d, y: %d\n", next_x, next_y);
            exit(0);
        }
        cur_pos[_get_idx(next_x, next_y)] = player;
        done = check_over(cur_pos, next_x, next_y);
        _next_x = next_x, _next_y = next_y;
        next_x = -1, next_y = -1; 
    }
}; 

class EvaluatorConfig {
public:
    int MinMaxMaxDepth = 1;
};

class Evaluator {
public:
    NetworkWrapper* nn = nullptr; 
    int weight_id = -1;

    Evaluator(NetworkWrapper* _nn, int _weight_id = -1) {
        nn = _nn;
        weight_id = _weight_id;
    }

    int run_once(Env& env, PlayerType type, EvaluatorConfig config = EvaluatorConfig()) {
        if (type == PlayerType::NN) {
            MCTS mcts(&env);
            for(int i = 0; i < ACT_SIMULATIONS; i++) mcts.run_mp(nn);
            MCTSResult mcts_result = mcts.deal_result(true);
            return mcts_result.action;
        } else if (type == PlayerType::PureMcts) {
            MCTSPure mcts(&env);
            return mcts.run(PURE_MCTS_SIMS);
        } else if (type == PlayerType::MinMax) {
            MinMax min_max = MinMax(config.MinMaxMaxDepth);
            int done = 0, next_x = -1, next_y = -1; 
            int pos[ACTION_SIZE]; 
            env.copy_position(pos); 
            min_max.run(pos, env.get_player(), done, next_x, next_y); 
            return next_x * BOARD_SIZE + next_y;
        } else if (type == PlayerType::People) {
            env.print();
            std::cout << "move: " << std::endl;
            int x, y; 
            std::cin >> x >> y; 
            return x * BOARD_SIZE + y;
        }  else {
            std::cerr << "invalid player type" << std::endl;
            exit(-1);
        }
    }

    float run() {
        Logger logger = Logger("evaluator.log");
        int compare_times = 1; 
        int nn_win = 0; 
        int adversary_win = 0; 
        for(int times = 0; times < compare_times; times++) {
            int done = 0; 
            Env env;
            while(done == 0) {
                int action_idx = 0;
                if(env.get_player() == 1) action_idx = run_once(env, PlayerType::NN);
                else action_idx = run_once(env, PlayerType::MinMax);
                done = env.move(action_idx);
            }
            if(done == 1) nn_win++;
            else if(done == -1) adversary_win++; 
        }
        int draw = compare_times - nn_win - adversary_win;
        float score = (nn_win + 0.5 * draw) / (float)compare_times;
        std::string info = "weight id: " + std::to_string(weight_id) + 
            ", nn / ad: " + std::to_string(nn_win) + " / " + std::to_string(adversary_win) + 
            ", score: " + std::to_string(score);
        logger.log(0, info);
        return score;
    }
};
    

#endif