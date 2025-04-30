# https://github.com/junxiaosong/AlphaZero_Gomoku

import math
import torch
from torch import nn
import config

class MlzeroNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.policy_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        self.policy_fc1 = nn.Linear(4 * 8 * 8, 8 * 8)

        self.value_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.value_fc1 = nn.Linear(2 * 8 * 8, 64)
        self.value_fc2 = nn.Linear(64, 1)


    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        policy_x = torch.relu(self.policy_conv1(x))
        policy_x = policy_x.view(-1, 4 * 8 * 8)
        policy_x = torch.softmax(self.policy_fc1(policy_x), dim=1)
        value_x = torch.relu(self.value_conv1(x))
        value_x = value_x.view(-1, 2 * 8 * 8)
        value_x = torch.relu(self.value_fc1(value_x))
        value_x = torch.tanh(self.value_fc2(value_x))
        return policy_x, value_x


class MlzeroResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, config.NUM_CHANNELS, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(config.NUM_CHANNELS)

        self.residual_blocks = nn.ModuleList()
        for i in range(config.RESBLOCK_NUM):
            self.residual_blocks.append(nn.Sequential(
                nn.Conv2d(config.NUM_CHANNELS, config.NUM_CHANNELS, kernel_size=3, padding=1),
                nn.BatchNorm2d(config.NUM_CHANNELS),
                nn.ReLU(),
                nn.Conv2d(config.NUM_CHANNELS, config.NUM_CHANNELS, kernel_size=3, padding=1),
                nn.BatchNorm2d(config.NUM_CHANNELS)
            ))
        
        self.policy_head = nn.Sequential(
            nn.Conv2d(config.NUM_CHANNELS, 4, kernel_size=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4 * config.ACTION_SIZE, config.ACTION_SIZE),
            nn.Softmax(dim=1)
        )
        
        self.value_head = nn.Sequential(
            nn.Conv2d(config.NUM_CHANNELS, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * config.ACTION_SIZE, config.ACTION_SIZE),
            nn.ReLU(),
            nn.Linear(config.ACTION_SIZE, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        
        for block in self.residual_blocks:
            residual = x
            x = block(x)
            x += residual
            x = torch.relu(x)
            
        pi = self.policy_head(x)
        val = self.value_head(x)
        
        return pi, val


