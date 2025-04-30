# cython: language_level=3
cdef extern from *:
    "#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION"

import numpy as np
cimport numpy as np
import random
import torch
import math
import copy
import time
import util
import config

DEF BOARD_SIZE = 8
DEF ACTION_SIZE = 64
DEF HISTORY_SIZE = 6
DEF PLANES_SIZE = 3

from libc.string cimport memset, memcpy

cdef class Env:
    cdef int position[ACTION_SIZE]
    cdef int legal_move[ACTION_SIZE]
    cdef int player
    cdef int current_idx
    cdef int done

    def __init__(self):
        memset(self.position, 0, sizeof(self.position))
        self.player = 1
        self.current_idx = -1
        self.done = 0
        self.update_legal_move()

    cpdef int check_over(self):
        #需要修改 万一最后一步使得游戏结束，此时场上没有空位
        cdef bint draw = True
        cdef int i 
        cdef idx = self.current_idx
        for i in range(ACTION_SIZE):
            if self.position[i] == 0:
                draw = False
                break
        if draw == True:
            return 2
        cdef int n, x, y, current_x, current_y
        cdef int color = self.position[idx]
        cdef int[8] directions_x
        cdef int[8] directions_y
        cdef int[4] num 
        directions_x = [0, -1, -1, 1, 0, 1, 1, -1]
        directions_y = [-1, 0, 1, 1, 1, 0, -1, -1]
        num = [1] * 4
        n = BOARD_SIZE
        x = idx // n
        y = idx % n
        for i in range(8):
            count = 0
            dx = directions_x[i]
            dy = directions_y[i]
            current_x, current_y = x + dx, y + dy
            while 0 <= current_x < n and 0 <= current_y < n:
                if self.position[current_x * n + current_y] == color:
                    count += 1
                    current_x += dx
                    current_y += dy
                else:
                    break
            num[i % 4] += count
            if num[i % 4] >= 5:
                return color
        return 0

    cpdef int is_over(self):
        return self.done

    cpdef void update_legal_move(self):
        for i in range(ACTION_SIZE):
            self.legal_move[i] = 1 if self.position[i] == 0 else 0 

    cpdef void set_position(self, int[:] _position, int player):
        cdef int i = 0
        for i in range(ACTION_SIZE):
            self.position[i] = _position[i]
        self.update_legal_move()
        self.player = player
        self.current_idx = -1
        self.done = 0 

    cpdef void fast_move(self, int next_idx):
        self.position[next_idx] = self.player
        self.player *= -1
        self.current_idx = next_idx

    cpdef int move(self, int next_idx):
        if self.position[next_idx] != 0:
            assert False, "illegal move"
        else:
            self.position[next_idx] = self.player
            self.update_legal_move()
            self.player *= -1
            self.current_idx = next_idx
            self.done = self.check_over()
            return self.done

    cpdef np.ndarray get_legal_move(self):
        return np.array(self.legal_move)

    cpdef np.ndarray get_position(self):
        return np.array(self.position)

    cpdef int get_player(self):
        return self.player

    cpdef np.ndarray get_obs(self):
        plane1 = np.array(self.position).reshape((BOARD_SIZE, BOARD_SIZE))
        plane2 = np.where(plane1 == 1, 1., 0)
        plane3 = np.where(plane1 == -1, 1., 0)
        plane4 = np.zeros_like(plane1)
        if self.current_idx != -1:
            plane4[self.current_idx // BOARD_SIZE][self.current_idx % BOARD_SIZE] = 1
        plane5 = np.full_like(plane1, self.player)
        return np.array([plane2, plane3, plane4, plane5], dtype=np.float32)

    cpdef printf(self):
        pos = self.get_position()
        print(" ", end=' ')
        for i in range(BOARD_SIZE):
            print(i, end=' ')
        print(" ")
        for i in range(BOARD_SIZE):
            print(i, end=' ')
            for j in range(BOARD_SIZE):
                if pos[i * BOARD_SIZE + j] == 0:
                    print(" ", end=' ')
                elif pos[i * BOARD_SIZE + j] == 1:
                    print("X", end=' ')
                elif pos[i * BOARD_SIZE + j] == -1:
                    print("O", end=' ')
            print("")
        print("")
        
cdef class Node:
    cdef public int player
    cdef public int next_idx
    cdef public bint is_expanded
    cdef public Node parent
    cdef public list child
    cdef public np.ndarray valid
    cdef public np.ndarray child_pi
    cdef public np.ndarray child_v
    cdef public np.ndarray child_n

    def __init__(self, int player, Node parent = None):
        self.player = player
        self.is_expanded = False
        self.parent = parent
        self.child = [None] * ACTION_SIZE
        self.valid = np.zeros(ACTION_SIZE, dtype=bool)
        self.child_pi
        self.child_v = np.zeros(ACTION_SIZE, dtype=np.float32)
        self.child_n = np.zeros(ACTION_SIZE, dtype=np.intc)

cdef class MCTS:
    cdef public Node dummy_node
    cdef public Node root_node
    cdef public Env env

    def __init__(self, Env env):
        if config.USE_GPU == True:
            assert torch.cuda.is_available(), "cannot use cuda"
        self.env = copy.deepcopy(env)
        self.dummy_node = Node(0, None)
        self.root_node = Node(self.env.player, self.dummy_node)
        self.dummy_node.child[0] = self.root_node
        self.dummy_node.valid[0] = True
        self.dummy_node.child_n[0] = 1
        self.dummy_node.is_expanded = True
        self.dummy_node.next_idx = 0

    cdef int __uct(self, Node node):
        cdef int next_idx
        cdef float c_puct = 5
        child_q = node.child_v / np.where(node.child_n > 0, node.child_n, 1)
        node_n = node.parent.child_n[node.parent.next_idx]
        child_u = c_puct * node.child_pi * (math.sqrt(node_n) / (1 + node.child_n))
        ucb_score = child_q + child_u
        ucb_score = np.where(node.valid == 1, ucb_score, -9999)
        next_idx = np.argmax(ucb_score)
        if node.valid[next_idx] == 0:
            assert False, "error in uct"
        return next_idx

    cdef void __simulate(self, object network_wrapper):
        cdef Node _current_node = self.root_node
        cdef Env env = copy.deepcopy(self.env)
        cdef int next_idx = -1
        cdef float val = 0
        while _current_node.is_expanded:
            next_idx = self.__uct(_current_node)
            _current_node.next_idx = next_idx
            env.fast_move(next_idx)
            if _current_node.child[next_idx] is None:
                _current_node.child[next_idx] = Node(env.get_player(), _current_node)
            _current_node = _current_node.child[next_idx]
        cdef int done = env.check_over()
        if done != 0:
            val = 1 if done == 1 or done == -1 else -1
        else:
            env.update_legal_move()  
            with torch.no_grad():
                obs = np.expand_dims(env.get_obs(), 0)
                pi, val = network_wrapper.predict(obs)
                val = -val
                _current_node.child_pi = np.asarray(pi)
            _current_node.valid = env.get_legal_move()
            _current_node.is_expanded = True
        while _current_node != self.dummy_node:
            _current_node.parent.child_n[_current_node.parent.next_idx] += 1
            _current_node.parent.child_v[_current_node.parent.next_idx] += val
            _current_node = _current_node.parent
            val *= -1


    cdef int get_noise_action(self, np.ndarray origin_pi):
        cdef np.ndarray noise = np.random.dirichlet(0.3 * np.ones(len(origin_pi)))
        noise_pi = 0.75 * origin_pi + 0.25 * noise
        for i in range(ACTION_SIZE):
            noise_pi[i] = noise_pi[i] if self.root_node.valid[i] == 1 else 0
        valid_actions = np.where(self.root_node.valid)[0]
        valid_probs = [noise_pi[i] for i in valid_actions if self.root_node.valid[i] == 1]
        valid_probs = valid_probs / np.sum(valid_probs)
        next_idx = np.random.choice(valid_actions, p=valid_probs)
        return next_idx

    cdef __softmax(self, np.ndarray[dtype=np.float32_t, ndim=1] x):
        probs = np.exp(x - np.max(x))
        probs /= np.sum(probs)
        return probs

    cdef __generate_optim_pi(self):
        cdef np.ndarray[dtype=np.float32_t, ndim=1] _child_n, pi_probs
        _child_n = (self.root_node.child_n * self.root_node.valid).astype(np.float32)
        pi_probs = np.log(_child_n + 1e-10)
        pi_probs = self.__softmax(pi_probs)
        return pi_probs

    cpdef tuple[int, int, np.ndarray, np.ndarray] run(self, object network_wrapper, int sims):
        if self.root_node.is_expanded == False:
            pi, _ = network_wrapper.predict(np.expand_dims(self.env.get_obs(), 0))
            self.root_node.valid = self.env.get_legal_move()
            self.root_node.child_pi = np.asarray(pi)
            self.root_node.is_expanded = True
        for _ in range(sims):
            self.__simulate(network_wrapper)
        
        optim_pi = self.__generate_optim_pi()
        last_obs = self.env.get_obs()
        next_idx = self.get_noise_action(optim_pi)

        if self.root_node.valid[next_idx] == False:
            assert False, f"error in noise pi, next idx{next_idx}"
        self.env.move(next_idx)
        return next_idx, self.env.is_over(), last_obs, optim_pi