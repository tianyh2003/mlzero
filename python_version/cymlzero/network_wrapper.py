import os
import re
import copy
import torch
import numpy as np
import config
import util
from network import MlzeroResNet as Network
import torch.nn.functional as F
from logger import Logger, Timer

class NetworkWrapper:
    def __init__(self):
        self.nn = Network()
        self.device = "cuda" if config.USE_GPU == True else "cpu"
        self.nn.to(self.device)
        self.l2_const = 1e-4 
        self.optimizer = torch.optim.Adam(self.nn.parameters(), weight_decay=self.l2_const)
        self.timer = Timer("predict")
        self.logger = Logger(log_file="./log/nw.log")
    
    def __del__(self):
        self.logger.log(0, f"predict time: {self.timer.get_sum()}")
        self.logger.flush()

    def eval_mode(self):
        self.nn.eval()
    
    def train_mode(self):
        self.nn.train()

    def load_weight(self, weight_id):
        weight_path = util.get_weight_path(weight_id)
        if os.path.exists(weight_path):
            self.nn.load_state_dict(torch.load(weight_path))
        else:
            assert False, f"weight {weight_id} not exists"

    def save_weight(self, weight_id):
        weight_path = util.get_weight_path(weight_id)
        if os.path.exists(weight_path):
            assert False, f"weight {weight_id} already exists"
        else:
            torch.save(self.nn.state_dict(), weight_path)

    def predict(self, in_data: np.ndarray) -> tuple[np.ndarray, float]:
        self.timer.start()
        self.nn.eval()
        with torch.no_grad():
            in_data = torch.FloatTensor(in_data.astype(np.float32)).cuda()
            out_data, val = self.nn(in_data)
            out_data = out_data.detach().cpu().numpy()
            val = val.detach().cpu().item()
        self.timer.end()
        return out_data, val

    def set_learning_rate(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def train(self, inputs: np.ndarray, pi: np.ndarray, v: np.ndarray):
        self.nn.train()
        self.optimizer.zero_grad()
        inputs = torch.FloatTensor(inputs.astype(np.float32)).cuda()
        pi = torch.FloatTensor(pi.astype(np.float32)).cuda()
        v = torch.FloatTensor(v.astype(np.float32)).cuda()
        pred_pi, pred_v = self.nn(inputs)
        log_pred_pi = torch.log(pred_pi + 1e-10)
        pi_loss = -torch.mean(torch.sum(pi*log_pred_pi, 1))
        v_loss = F.mse_loss(pred_v.squeeze(), v.squeeze(), reduction='mean')
        loss = pi_loss + v_loss
        loss.backward()
        self.optimizer.step()
        entropy = -torch.mean(torch.sum(torch.exp(log_pred_pi) * log_pred_pi, 1))
        current_lr = self.optimizer.param_groups[0]['lr']
        return loss.item(), entropy.item(), current_lr