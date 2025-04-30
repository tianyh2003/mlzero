import os
import signal
import multiprocessing as mp
import queue
import math
import numpy as np
import config
import util
from logger import Logger, Timer
from network_wrapper import NetworkWrapper
from actor import Actor
from replay import ReplayMemory
from datetime import datetime

def run_actor(data_queue, weight_id):
    actor = Actor(data_queue, weight_id)
    actor.run()

def run_learner( 
    weight_id: int,
    data_queue: mp.Queue(),
    replay: ReplayMemory
):
    logger = Logger(log_file="./log/learner.log")
    network_wrapper = NetworkWrapper()
    network_wrapper.load_weight(weight_id)
    lr_multi = 1
    if weight_id > 100 and weight_id <= 200:
        lr_multi = 0.5
    elif weight_id > 200 and weight_id <= 300:
        lr_multi = 0.25
    elif weight_id > 300 and weight_id <= 500:
        lr_multi = 0.125
    elif weight_id > 500:
        lr_multi = 0.0625
    
    lr = config.INIT_LR * lr_multi
    network_wrapper.set_learning_rate(lr)
    loss = 0
    entropy = 0
    lr = 0
    for _ in range(config.TRAIN_STEPS):
        inputs, pi, v = replay.sample(config.BATCH_SIZE)
        if inputs is not None:
            _loss, _entropy, _cur_lr = network_wrapper.train(inputs, pi, v)
        loss += _loss
        entropy += _entropy
    loss /= config.TRAIN_STEPS
    entropy /= config.TRAIN_STEPS
    logger.log(0, f"train {weight_id} loss {loss} entropy {entropy} lr {_cur_lr}")
    network_wrapper.save_weight(weight_id + 1)

def main():
    data_queue = mp.Queue()
    replay = ReplayMemory()
    logger = Logger(log_file="./log/log.log")
    weight_id = 0
    if os.path.exists(util.get_weight_path(weight_id)) == False:
        NetworkWrapper().save_weight(weight_id)
    actor_timer = Timer("actor")
    learner_timer = Timer("learner")
    for i in range(config.NUM_TRAIN):
        actor_timer.start()
        played_games = 0; 
        while played_games < config.GAME_PER_ITER:
            actors = []
            for _ in range(config.NUM_ACTOR):
                actor = mp.Process(
                    target=run_actor,
                    args=(data_queue, weight_id)
                )
                actor.daemon = True
                actor.start()
                actors.append(actor)
                played_games += 1
                if played_games >= config.GAME_PER_ITER:
                    break
            while True:
                try:
                    _data = data_queue.get(timeout=1)
                    replay.add(_data)
                except queue.Empty:
                    num_alive = 0
                    for actor in actors:
                        if actor.is_alive():
                            num_alive += 1
                    if num_alive == 0:
                        break
            for actor in actors:
                actor.join()

        actor_timer.end()
        logger.log(0, f"[actor] time {actor_timer.get_sum()}")
        
        learner_timer.start()
        run_learner(weight_id, data_queue, replay)
        learner_timer.end()
        logger.log(0, f"[{learner_timer.get_name()}] time {learner_timer.get_sum()}")
        weight_id += 1

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
