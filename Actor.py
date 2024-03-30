import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import relu


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, l1_dim=400, l2_dim=300):
        super(Actor, self).__init__()
 
        self.l1 = nn.Linear(state_dim, l1_dim)
        self.l2 = nn.Linear(l1_dim, l2_dim)
        self.l3 = nn.Linear(l2_dim, action_dim)

        self.max_action = max_action
        self.tanh = nn.Tanh()

    def forward(self, x):

        x = relu(self.l1(x))
        x = relu(self.l2(x))
        x = self.max_action * self.tanh(self.l3(x))

        return x
    

