import numpy as np
import random
import torch
import math
import copy
import time
import util
import config

class Env: 
    def __init__(self):
        self.pos = np.zeros(config.ACTION_SIZE, dtype=np.int32)
        self.legal = np.ones(config.ACTION_SIZE, dtype=np.int32)
        self.player = 1
        self.current_idx = 0
        self.update_legal_move()

    def check_over(self):
        draw = True
        idx = self.current_idx
        for i in range(config.ACTION_SIZE):
            if self.pos[i] == 0:
                draw = False
                break
        if draw:
            return 2

        color = self.pos[idx]
        direction = [[0, -1], [-1, 0], [-1, 1], [1, 1], [0, 1], [1, 0], [1, -1], [-1, -1]]
        num = [1, 1, 1, 1]
        n = config.BOARD_SIZE
        x = idx // n
        y = idx % n
        
        for i in range(8):
            count = 0
            dx = direction[i][0]
            dy = direction[i][1]
            current_x, current_y = x + dx, y + dy
            while 0 <= current_x < n and 0 <= current_y < n:
                if self.pos[current_x * n + current_y] == color:
                    count += 1
                    current_x += dx
                    current_y += dy
                else:
                    break
            num[i % 4] += count
            if num[i % 4] >= 5:
                return color
        return 0
    
    def update_legal_move(self):
        for i in self.pos:
            if i == 0:
                self.legal[i] = 1
            else:
                self.legal[i] = 0
        return self.legal
    
    def move(self, idx):
        assert self.legal[idx] == 1, "illegal move"
        self.pos[idx] = self.player
        self.player *= -1
        self.current_idx = idx
        self.update_legal_move()
        return self.check_over()
    
    def get_obs(self):
        origin_data = np.array(self.pos).reshape(config.BOARD_SIZE, config.BOARD_SIZE)
        plane1 = np.where(origin_data == 1, 1.0, 0.0)
        plane2 = np.where(origin_data == -1, 1.0, 0.0)
        plane3 = np.zeros_like(origin_data)
        if self.current_idx != -1:
            plane3[self.current_idx // config.BOARD_SIZE][self.current_idx % config.BOARD_SIZE] = 1.0
        plane4 = np.full_like(origin_data, 1 if self.player == 1 else -1)
        return np.array([plane1, plane2, plane3, plane4], dtype=np.float32)

class Node:
    def __init__(self, player, parent):
        self.parent = parent
        self.is_expanded = False
        self.player = player
        self.child = [None] * config.ACTION_SIZE
        self.valid = np.zeros(config.ACTION_SIZE, dtype=np.int32)
        self.child_v = np.zeros(config.ACTION_SIZE, dtype=np.float32)
        self.child_n = np.zeros(config.ACTION_SIZE, dtype=np.int32)
        self.child_pi = None
        self.next_idx = -1
    
class MCTS:
    def __init__(self, env: Env = None):
        self.env = copy.deepcopy(env) if env != None else Env()
        self.dummy_node = Node(0, None)
        self.root_node = Node(1, self.dummy_node)
        self.dummy_node.child[0] = self.root_node
        self.dummy_node.valid[0] = 1
        self.dummy_node.child_n[0] = 1
        self.dummy_node.is_expanded = True
        self.dummy_node.next_idx = 0

    def uct(self, node) :
        q = node.child_v / np.where(node.child_n == 0, 1, node.child_n)
        u = 5 * node.child_pi * np.sqrt(node.parent.child_n[node.parent.next_idx]) / (1 + node.child_n)
        score = q + u
        score = np.where(node.valid == 1, score, -9999)
        next_idx = np.argmax(score)
        assert node.valid[next_idx] == 1, "invalid move"
        return next_idx
    
    def simulate(self, network_wrapper):
        cur_node = self.root_node
        env = copy.deepcopy(self.env)
        next_idx = -1
        val = 0
        while cur_node.is_expanded:
            next_idx = self.uct(cur_node)
            cur_node.next_idx = next_idx
            done = env.move(next_idx)
            if cur_node.child[next_idx] is None:
                cur_node.child[next_idx] = Node(env.player, cur_node)
            cur_node = cur_node.child[next_idx]
        done = env.check_over()
        if done != 0:
            val = 1 if done == 1 or done == -1 else -1
        else:
            obs = np.expand_dims(env.get_obs(), axis=0)
            pi, val = network_wrapper.predict(obs)
            val = -val
            cur_node.child_pi = pi
            cur_node.valid = np.array(env.legal, dtype=np.int32)
            cur_node.is_expanded = True
        while cur_node.parent != self.dummy_node:
            cur_node.parent.child_n[cur_node.parent.next_idx] += 1
            cur_node.parent.child_v[cur_node.parent.next_idx] += val
            cur_node = cur_node.parent
            val *= -1

    def get_noise_action(self, origin_pi):
        noise = np.random.dirichlet(0.3 * np.ones(len(origin_pi)))
        noise_pi = 0.75 * origin_pi + 0.25 * noise
        noise_pi = np.where(self.root_node.valid == 1, noise_pi, 0)
        valid_actions = np.where(self.root_node.valid)[0]
        valid_probs = noise_pi[valid_actions]
        valid_probs = valid_probs / np.sum(valid_probs)
        next_idx = np.random.choice(valid_actions, p=valid_probs)
        return next_idx

    def softmax(self, x):
        pi = np.exp(x - np.max(x))
        pi /= np.sum(pi)
        return pi

    def generate_optim_pi(self):
        child_n = (self.root_node.child_n * self.root_node.valid).astype(np.float32)
        pi = np.log(child_n + 1e-10)
        pi = self.softmax(pi)
        return pi

    def run(self, network_wrapper, sims = config.ACT_SIMS):
        if self.root_node.is_expanded == False:
            pi, _ = network_wrapper.predict(np.expand_dims(self.env.get_obs(), axis=0))
            self.root_node.valid = np.array(self.env.legal, dtype=np.int32)
            self.root_node.child_pi = np.array(pi, dtype=np.float32)
            self.root_node.is_expanded = True
        for _ in range(sims):
            self.simulate(network_wrapper)
        
        optim_pi = self.generate_optim_pi()
        last_obs = self.env.get_obs()
        next_idx = self.get_noise_action(optim_pi)
        assert self.root_node.valid[next_idx] == 1, "invalid move"
        self.env.move(next_idx)
        return next_idx, self.env.check_over(), last_obs, optim_pi


