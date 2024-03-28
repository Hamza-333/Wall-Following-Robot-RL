import numpy as np

class ReplayBuffer(object):
    def __init__(self, max_size):
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
        indices = np.random.randint(0, self.max_size , size=batch_size)
        states, next_states, actions, rewards, dones = [], [], [], [], []
        for i in indices:
            self.storage[i, :]
            s, ns, a, r, d = self.storage[i, :]
            states.append(np.array(s, copy=False))
            next_states.append(np.array(ns, copy=False))
            actions.append(np.array(a, copy=False))
            rewards.append(r)
            dones.append(d)
        
        return np.array(states), np.array(next_states), np.array(actions), np.array(rewards).reshape(-1,1), np.array(dones).reshape(-1,1)

