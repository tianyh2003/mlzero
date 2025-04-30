#include "../common.h"
#include "../network_wrapper.hpp"
#include "../logger.hpp"    
#include "../mcts_mp.hpp"
#include "../rating.hpp"
#include "../actor.hpp"

int main() {
    printf("start evaluate\n");
    int sims = 800; 
    std::string ckpt_dir = "/hy-tmp/mlzero/ckpt";
    Logger* logger = new Logger("evaluate.log");

    EloRating black_rating;
    EloRating white_rating;

    int min_weight_id = 668;
    int max_weight_id = 669; 

    printf("max idx is %d\n", max_weight_id);

    std::string weight_path = ckpt_dir + "/weight_" + std::to_string(min_weight_id) + ".ckpt";
    std::string last_weight_path = ckpt_dir + "/weight_" + std::to_string(min_weight_id) + ".ckpt";

    for(int weight_id = min_weight_id + 1; weight_id <= max_weight_id; weight_id += 1) {
        NetworkWrapper black_nn = NetworkWrapper();
        NetworkWrapper white_nn = NetworkWrapper();
        std::string weight_path = ckpt_dir + "/weight_" + std::to_string(weight_id) + ".ckpt";

        if(!std::filesystem::exists(weight_path)) {
            std::cout << "weight not exist: " << weight_path << std::endl;
            fflush(stdout);
            continue;
        }

        if(black_nn.load_weight(weight_path) == -1) {
            std::cout << "load weight failed: " << weight_path << std::endl;
            fflush(stdout);
            continue;
        }
        if(white_nn.load_weight(last_weight_path) == -1) {
            std::cout << "load last weight failed: " << last_weight_path << std::endl;
            fflush(stdout);
            continue;
        }
        
        std::cout << "black_nn: " << weight_path << std::endl;
        std::cout << "white_nn: " << last_weight_path << std::endl;
        fflush(stdout);

        int done = 0; 
        Env env;
        std::mutex mcts_mutex;
        while(done == 0) {
            
            env.print();
            MCTS mcts(true, false, true, &env);
            MCTSResult result;
            int mcts_times = 0;

            if(env.get_player() == 1) {
                run_mcts_mp(&mcts, &black_nn, &mcts_times, &mcts_mutex, 0);
                result = mcts.deal_result(true);
            } else {
                run_mcts_mp(&mcts, &white_nn, &mcts_times, &mcts_mutex, 0);
                result = mcts.deal_result(true);
            }
            done = result.done;
            env.move(result.action);
            printf("move (%d, %d)\n", result.action / BOARD_SIZE, result.action % BOARD_SIZE);
        }
        if(done == 1) {
            black_rating.update_rating(white_rating.get_rating(), 1.0f);
            white_rating.update_rating(black_rating.get_rating(), 0.0f);
        } else if(done == -1) {
            white_rating.update_rating(black_rating.get_rating(), 1.0f);
            black_rating.update_rating(white_rating.get_rating(), 0.0f);
        }
        last_weight_path = weight_path;
        // std::string log_info = "done: " + std::to_string(done) +" Weight " + std::to_string(weight_id) + ": Black " + std::to_string(black_rating.get_rating()) + ", White " + std::to_string(white_rating.get_rating());
        std::string log_info = "done: " + std::to_string(done) +" Weight " + std::to_string(weight_id) + ": Black " + std::to_string(black_rating.get_rating());
        std::cout << log_info << std::endl;
        logger->log(1, log_info);
        white_rating.set_rating(black_rating.get_rating());
        
    }
    delete logger;



}