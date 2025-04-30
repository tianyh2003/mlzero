#include "../common.h"
#include "../network_wrapper.hpp"
#include "../logger.hpp"    
#include "../mcts_mp.hpp"
#include "../actor.hpp"
#include "../evaluator.hpp"
#include <cstddef>
#include <new>

const PlayerType PLAYER_1_TYPE = PlayerType::NN;
const PlayerType PLAYER_2_TYPE = PlayerType::MinMax;

const int EVAL_INTERVAL = 10; 
const int EVAL_SIMS = 800;
const int EVAL_FORWARD_TIME = 5; 
const int EVAL_BACKWARD_TIME = 5;
const int EVAL_MIN_WEIGHT_ID = 10;
const int EVAL_MAX_WEIGHT_ID = 200;
const int EVAL_PURE_SIMS = 2000;
const int EVAL_MINMAX_MAX_DEPTH = 1;
const int EVAL_GAME_TIMES = EVAL_FORWARD_TIME + EVAL_BACKWARD_TIME;
const std::string EVAL_CKPT_DIR = "/home/work/file/mlzeros/mlzero_res/ckpt";

inline std::string eval_weight_path(int weight_id) {
    return EVAL_CKPT_DIR + "/weight_" + std::to_string(weight_id) + ".ckpt";
}

bool eval_load_weight(int weight_id, NetworkWrapper& nn) {
    std::string weight_path = eval_weight_path(weight_id);
    if(!std::filesystem::exists(weight_path)) {
        std::cout << "weight not exist: " << weight_path << std::endl;
        return false;
    }
    if(nn.load_weight(weight_path, true) == -1) {
        std::cout << "load weight failed: " << weight_path << std::endl;
        return false; 
    }
    return true; 
}

int init_move[5] = {18, 21, 27, 42, 45};

int main() {
    Logger logger = Logger("evaluate.log");
    std::string weight_path = eval_weight_path(EVAL_MIN_WEIGHT_ID);
    int forward_time = EVAL_FORWARD_TIME; 
    int backward_time = EVAL_BACKWARD_TIME;;

    for(int weight_id = EVAL_MIN_WEIGHT_ID; weight_id <= EVAL_MAX_WEIGHT_ID; weight_id += EVAL_INTERVAL) {
        NetworkWrapper nn = NetworkWrapper();
        if(!eval_load_weight(weight_id, nn)) continue;
        Evaluator player1 = Evaluator(&nn, weight_id);
        Evaluator player2 = Evaluator(&nn, weight_id);
        int player1_win = 0;
        int player2_win = 0;
        EvaluatorConfig eval_config = EvaluatorConfig();
        eval_config.MinMaxMaxDepth = EVAL_MINMAX_MAX_DEPTH;

        for(int times = 0; times < forward_time; times++) {
            int done = 0; 
            Env env;
            env.move(init_move[times % 5]);
            while(done == 0) {
                int action_idx = 0;
                if(env.get_player() == 1) action_idx = player1.run_once(env, PLAYER_1_TYPE, eval_config);
                else action_idx = player2.run_once(env, PLAYER_2_TYPE, eval_config);
                done = env.move(action_idx);
                env.print();
            }
            if(done == 1) player1_win++; 
            else if(done == -1) player2_win++; 
        }
        for(int times = 0; times < backward_time; times++) {
            int done = 0; 
            Env env;
            env.move(init_move[times % 5]);
            while(done == 0) {
                int action_idx = 0;
                if(env.get_player() == 1) action_idx = player2.run_once(env, PLAYER_2_TYPE, eval_config);
                else action_idx = player1.run_once(env, PLAYER_1_TYPE, eval_config);
                done = env.move(action_idx);
                env.print();
            }
            if(done == 1) player2_win++;
            else if(done == -1) player1_win++;
        }

        int draw = EVAL_GAME_TIMES - player1_win - player2_win;
        float player1_win_ratio = (player1_win + 0.5 * draw) / EVAL_GAME_TIMES;

        std::string log_info =  "[Weight " + std::to_string(weight_id) + "] " 
            + std::to_string(player1_win) + " : " + std::to_string(player2_win) 
            + " ratio: " + std::to_string(player1_win_ratio);
        logger.log(0, log_info);
    }
}