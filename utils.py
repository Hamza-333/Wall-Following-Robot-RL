import numpy as np

class ReplayBuffer(object):
    def __init__(self, max_size):
        self.storage = []
        self.max_size = max_size
        self.ind = 0

    def add(self, state, terminated, truncated, reward, done):
        data = [state, reward, terminated, truncated]

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
        indices = np.random.randint(0, self.max_size, size=batch_size)
        states, rewards, terminateds, truncateds = [], [], [], [], []
        for i in indices:
            self.storage[i, :]
            s, r, trunc, term = self.storage[i, :]
            states.append(np.array(s, copy=False))
            rewards.append(r)
            terminateds.append(np.array(term, copy=False))
            truncateds.append(np.array(trunc, copy=False))

        
        return np.array(states), np.array(rewards), np.array(terminateds), np.array(truncateds)

