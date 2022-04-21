import torch.nn as nn
import torch
import numpy as np
from configs import max_size

class Actor(nn.Module):
    
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.seq = nn.Sequential(
                nn.Linear(state_dim, 400),
                nn.ReLU(),
                nn.Linear(400, 300),
                nn.ReLU(),
                nn.Linear(300, action_dim),
                nn.Tanh()
                )
        self.max_action = max_action

    def forward(self, x):
        return self.seq(x) * self.max_action


class Critic(nn.Module):
 
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.seq = nn.Sequential(
                nn.Linear(state_dim + action_dim, 400),
                nn.ReLU(),
                nn.Linear(400, 300),
                nn.ReLU(),
                nn.Linear(300, 1),
                )

    def forward(self, x, u):
        xu = torch.cat([x, u], 1)
        return self.seq(xu)

class ReplayBuffer:
    def __init__(self):
        self.storage = []

    def add(self, data):
        if len(self.storage) == max_size:
            self.storage.pop(0)
        self.storage.append(data)
    
    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        s, s1, a, r, d = [], [], [], [], []
        for i in ind: 
            for x,y in zip([s,s1,a,r,d],self.storage[i]):
                x.append(np.array(y,copy=False))
        return np.array(s), np.array(s1), np.array(a), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)