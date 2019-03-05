from collections import deque
import numpy as np
import random

class ReplayBuffer():

    def __init__(self, maxlen=None):
        self.buffer = deque(maxlen=maxlen)

    def append(self, item):
        self.buffer.append(item)

    def __len__(self):
        return len(self.buffer)

    def sample(self, batch_size):
        return np.random.choice(self.buffer, batch_size, replace=False)
