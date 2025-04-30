import os
import signal
import multiprocessing as mp
import queue
import numpy as np
import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()},
                  reload_support=True)
from mcts import MCTS
from logger import Logger, Timer
from network_wrapper import NetworkWrapper
from actor import Actor
from replay import ReplayMemory
from evaluator import Evaluator

def play_with_human():
    eval_sims = 160
    log_rank = 5

    batch_size = 512
    ckpt_dir = "./ckpt"
    ckpt_file = "/weight_178.ckpt"
    use_gpu = True

    plane_size = 3
    board_size = 8
    action_size = 64

    logger = Logger(rank=log_rank, log_file="./log/play_with_human_log.log")

    def term(sig_num, addtion):
        logger.flush()
        os.killpg(os.getpgid(os.getpid()), signal.SIGKILL)
    signal.signal(signal.SIGTERM, term)

    eva = Evaluator(
        plane_size=plane_size, 
        board_size=board_size, 
        action_size=action_size,
        ckpt_dir=ckpt_dir, 
        log_rank=log_rank, 
        use_gpu=use_gpu, 
        sims=eval_sims, 
        default_elo=0, 
        timeout=0
    )
    network_wrapper = NetworkWrapper(plane_size, board_size, action_size, logger, ckpt_dir, use_gpu)
    id = network_wrapper.update_weight(ckpt_file)
    print(f"file id: {id}")
    eva.play_with_human(network_wrapper)
    logger.flush()

if __name__ == '__main__':
    play_with_human()
