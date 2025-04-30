import numpy as np
import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()},
                  reload_support=True)
import multiprocessing as mp
import config
from logger import Logger, Timer
from mcts import MCTS, Env
from network_wrapper import NetworkWrapper
import util

class Actor:
    def __init__(
        self, 
        data_queue: mp.Queue(), 
        weight_id: int
    ):
        self.data_queue = data_queue
        self.nn = NetworkWrapper()
        self.nn.load_weight(weight_id)
        self.logger = Logger(log_file="./log/actor.log")
        self.timer = Timer("mcts")

    def __del__(self):
        self.logger.log(0, f"mcts time: {self.timer.get_sum()}")
        self.logger.flush()

    def run(self):
        player = 1
        done = 0
        obs_list = []
        pi_list = []
        result_list = []
        step = 0
        env = Env()
        while done == 0:
            self.timer.start()
            next_idx, done, obs, opti_pi = MCTS(env).run(self.nn, config.ACT_SIMS)
            self.timer.end()
            step += 1
            obs_list.append(obs)
            pi_list.append(opti_pi)
            done = env.move(next_idx)
        if done == 2:
            for _ in range(step):
                result_list.append(0)
        else: 
            for _ in range(step):
                result_list.append(done)
                done *= -1

        extended_obs_list = []
        extended_pi_list = []
        extended_result_list = []
        for obs, pi, result in zip(obs_list, pi_list, result_list):
            for i in [1, 2, 3, 4]:
                # 逆时针旋转
                equi_obs = np.array([np.rot90(s, i) for s in obs])
                equi_pi = np.rot90(np.flipud(pi.reshape(8, 8)), i) 
                extended_obs_list.append(equi_obs)
                extended_pi_list.append(np.flipud(equi_pi).flatten())
                extended_result_list.append(result)
                
                # 水平翻转
                equi_obs = np.array([np.fliplr(s) for s in equi_obs])
                equi_pi = np.fliplr(equi_pi)
                extended_obs_list.append(equi_obs)
                extended_pi_list.append(np.flipud(equi_pi).flatten())
                extended_result_list.append(result)
        
        for _data in zip(extended_obs_list, extended_pi_list, extended_result_list):
            self.data_queue.put(_data)