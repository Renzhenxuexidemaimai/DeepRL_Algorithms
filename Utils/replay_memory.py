import random
from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'mask'))


class Memory(object):
    def __init__(self):
        self.memory = []

    # save item
    def push(self, *args):
        self.memory.append(Transition(*args))

    def clear(self):
        self.memory.clear()

    # sample a mini_batch
    def sample(self, batch_size=None):
        if batch_size is None:
            return Transition(*zip(*self.memory))
        else:
            random_batch = random.sample(self.memory, batch_size)
            return Transition(*zip(*random_batch))

    def __len__(self):
        return len(self.memory)
