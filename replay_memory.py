import random
import numpy as np


class ReplayMemory:
    def __init__(self, capacity, seed):
        """
        Init ReplayMemory
        :param capacity: max size of replay buffer
        :param seed:     seed of random, make sure results of each experiment are the same
        """
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def remember(self, state, action, reward, next_state, mask):
        """
        Store a new experience in the replay memory.
        Overwrite the oldest experience if memory is at capacity.
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, mask)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Sample a batch of experiences from memory.
        """
        batch = random.sample(self.buffer, batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = map(np.stack, zip(*batch))
        return state_batch, action_batch, reward_batch, next_state_batch, mask_batch
