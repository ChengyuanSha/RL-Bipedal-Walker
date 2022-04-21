import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from configs import *
from utils import *

class Q(nn.Module):

    def __init__(self,inputs,actions):
        super(Q,self).__init__()

        self.linear1 = nn.Linear(inputs+actions,Hidden_Size)
        self.linear1_1 = nn.Linear(Hidden_Size,Hidden_Size)
        self.linear1_2 = nn.Linear(Hidden_Size,1)

        self.linear2 = nn.Linear(inputs+actions,Hidden_Size)
        self.linear2_1 = nn.Linear(Hidden_Size,Hidden_Size)
        self.linear2_2 = nn.Linear(Hidden_Size,1)

        self.apply(init_weight)

    def forward(self,state,action):
        x = torch.cat([state,action],1)

        output1 = F.relu(self.linear1(x))
        output1 = F.relu(self.linear1_1(output1))
        output1 = self.linear1_2(output1)

        output2 = F.relu(self.linear2(x))
        output2 = F.relu(self.linear2_1(output2))
        output2 = self.linear2_2(output2)

        return output1,output2

class Policy(nn.Module):

    def __init__(self,inputs,actions,action_space):
        super(Policy,self).__init__()

        self.linear1 = nn.Linear(inputs,Hidden_Size)
        self.linear2 = nn.Linear(Hidden_Size,Hidden_Size)
        self.linear3 = nn.Linear(Hidden_Size,actions)
        self.linear4 = nn.Linear(Hidden_Size,actions)

        self.apply(init_weight)

        self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2.).to(Device)
        self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2.).to(Device)

    def forward(self,state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        mean = self.linear3(output)
        output = self.linear4(mean)
        output = torch.clamp(output,min=Log_sig_max,max=Log_sig_max)
        return mean,output

    def peek(self,state):
        mean,output = self.forward(state)
        value = torch.tanh(Normal(mean,output.exp()).rsample())
        action = value * self.action_scale + self.action_bias
        log_prob = Normal(mean,output.exp()).log_prob(Normal(mean,output.exp()))
        log_prob = log_prob - torch.log(self.action_scale * (1 - value.pow(2))+Epsilon)
        log_prob = log_prob.sum(1,keepdim=True)
        return action,log_prob


class DPolicy(nn.Module):
    def __init__(self,inputs,actions,action_space):
        super(DPolicy,self).__init__()
        self.linear1 = nn.Linear(inputs,Hidden_Size)
        self.linear2 = nn.Linear(Hidden_Size,Hidden_Size)

        self.linear3 = nn.Linear(Hidden_Size,actions)
        self.noise = torch.Tensor(actions)

        self.apply(init_weight)

        self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2.).to(Device)
        self.action_bias = torch.FloatTensor((action_space.high - action_space.low) / 2.).to(Device)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        output = torch.tanh(self.linear3(output)) * self.action_scale + self.action_bias
        return output

    def peek(self,state):
        output = self.forward(state)
        noise = self.noise.normal_(0.,std=0.1)
        noise = noise.clamp(-0.25,0.25)
        action = output + noise
        return action
