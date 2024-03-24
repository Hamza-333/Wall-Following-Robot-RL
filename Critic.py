import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import relu

class Critic(nn.module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self)
        
        Q1_l1_dim=200, 
        Q1_l2_dim=100,
        Q2_l1_dim= 200, 
        Q2_l2_dim= 100
        
        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, Q1_l1_dim)
        self.l2 = nn.Linear(Q1_l1_dim, Q1_l2_dim)
        self.l3 = nn.Linear(Q1_l2_dim, 1)
        
        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, Q2_l1_dim)
        self.l5 = nn.Linear(Q2_l1_dim, Q2_l2_dim)
        self.l6 = nn.Linear(Q2_l2_dim, 1)
    
    def forward(self, x, a):
        xa = torch.cat([x, a], 1)
        x1 = relu(self.l1(xa))
        x1 = relu(self.l2(x1))
        x1 = self.l3(x1)
        
        x2 = relu(self.l4(xa))
        x2 = relu(self.l5(x2))
        x2 = self.l6(x2)
        return x1, x2


    