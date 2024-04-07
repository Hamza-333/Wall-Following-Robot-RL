import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import relu
from utils import SEED

torch.manual_seed(SEED)
np.random.seed(SEED)

class Q_function(nn.Module):
    def __init__(self, state_dim, action_dim, l1_dim=100, l2_dim=100):
        super(Q_function, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, l1_dim)
        self.l2 = nn.Linear(l1_dim, l2_dim)
        self.l3 = nn.Linear(l2_dim, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)

        x = relu(self.l1(x))
        x = relu(self.l2(x))
        x = self.l3(x)

        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, q1_l1_dim=100, q1_l2_dim=100,
                 q2_l1_dim=100, q2_l2_dim=100):
        super(Critic, self).__init__()

        # Q1
        self.q1 = Q_function(state_dim, action_dim, q1_l1_dim, q1_l2_dim)
        # Q2
        self.q2 = Q_function(state_dim, action_dim, q2_l1_dim, q2_l2_dim)
    
    def forward(self, x, a):
        x1 = self.q1(x, a)
        x2 = self.q2(x, a)
        
        return x1, x2