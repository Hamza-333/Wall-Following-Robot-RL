import numpy as np
import sys

import logging

SEED = 0

#Files and directories for different settings
file_name = "TD3_"
file_name_var = "TD3_VAR_"
file_name_accl = "TD3_ACCL_"

policies_dir = "./policies"
policies_dir_var_speed = "./policies/varPolicies"
policies_dir_accl = "./policies/acclPolicies"

logs_filepath = './benchmarks/logs/TD3_log.csv'
logs_filepath_var_speed = './benchmarks/logs/TD3_log_VAR.csv'
logs_filepath_accl = './benchmarks/logs/TD3_log_ACCL.csv'

model_dir = "./pytorch_models"



np.random.seed(SEED)

class ReplayBuffer(object):
    def __init__(self, max_size=1e4):
        self.storage = []
        self.max_size = max_size
        self.ind = 0

    def add(self, state, next_state, action, reward, done):
        data = [state, next_state, action, reward, done]

        # if there is still space in storage, add data
        if len(self.storage) < self.max_size:
            self.storage.append(data)
            # space met, reset index back to 0
        else:
            self.storage[self.ind] = data
            self.ind += 1
            if self.ind == self.max_size:
                self.ind = 0

    def sample(self, batch_size):
        # randomly sample batch size number of past events
        indices = np.random.randint(0, len(self.storage), size=batch_size)
        states, next_states, actions, rewards, done = [], [], [], [], []
        
        for i in indices:

            s, ns, ac, r, d = self.storage[i]
            states.append(np.array(s, copy=False))
            next_states.append(np.array(ns, copy=False))
            actions.append(np.array(ac, copy=False))
            rewards.append(np.array(r, copy=False))
            done.append(np.array(d, copy=False))
        
        return np.array(states), np.array(next_states), np.array(actions), \
            np.array(rewards).reshape(-1, 1), np.array(done).reshape(-1, 1)

