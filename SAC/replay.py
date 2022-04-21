import numpy as np
import random
from configs import *

class memeory:

    def __init__(self):
        self.buffer = list()
        self.length = Replay_size
        self.pointer = 0 

    def push(self,state,next_state,action,reward,done):
        if len(self.buffer) < self.length: self.buffer.append(None)
        self.buffer[self.pointer] = (state,next_state,action,reward,done)
        self.pointer = (self.pointer + 1) % self.length

    def peek(self):
        batch = random.sample(self.buffer,Batch_Size)
        return [np.stack(i) for i in zip(*batch)]


