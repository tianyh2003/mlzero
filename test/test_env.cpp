
#include "../network_wrapper.hpp"
#include "../common.h"
#include "../logger.hpp"
#include "../mcts_mp.hpp"
int main() {
    int pos[ACTION_SIZE] = {0};
    pos[0] = -1; 
    pos[1] = -1; 
    pos[2] = 0; 
    pos[3] = 0; 
    pos[4] = 0; 
    pos[5] = 0; 
    pos[6] = -1; 
    pos[7] = -1; 

    pos[8] = -1;
    pos[9] = 0;
    pos[10] = 1;
    pos[11] = 0;
    pos[12] = -1;
    pos[13] = 1;
    pos[14] = 1;
    pos[15] = 1;
    
    pos[16] = 1;
    pos[17] = 1;
    pos[18] = -1;
    pos[19] = 1;
    pos[20] = 1;
    pos[21] = 1;
    pos[22] = 0;
    pos[23] = 0;

    pos[24] = 0;
    pos[25] = 0;
    pos[26] = -1;
    pos[27] = 1;
    pos[28] = 1;
    pos[29] = -1;
    pos[30] = 0;
    pos[31] = 0;

    pos[32] = -1;
    pos[33] = 1;
    pos[34] = 1;
    pos[35] = 1;
    pos[36] = 1;
    pos[37] = -1;
    pos[38] = -1;
    pos[39] = -1;

    pos[40] = -1;
    pos[41] = 0;
    pos[42] = -1;
    pos[43] = -1;
    pos[44] = 0;
    pos[45] = 0;
    pos[46] = 0;
    pos[47] = 1;

    pos[48] = -1;
    pos[49] = 0;
    pos[50] = 0;
    pos[51] = 0;
    pos[52] = 0;
    pos[53] = 0;
    pos[54] = -1;
    pos[55] = 0;

    pos[56] = 1;
    pos[57] = 1;
    pos[58] = 0;
    pos[59] = -1;
    pos[60] = 1;
    pos[61] = -1;
    pos[62] = 1;
    pos[63] = 0;

    
    Env env = Env();
    env.set_position(pos, -1);
    env.print(); 
    int done = env.move(41); 

    MCTS mcts = MCTS(false, false, false, &env); 
    int sims = 1000; 
    for(int i = 0; i < sims; i++) {
        // mcts.run_pure(); 
    }
    


    
    printf("done: %d\n", done);
    env.print(); 
}