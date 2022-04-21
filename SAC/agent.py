import torch
import torch.nn.functional as F
from torch.optim import Adam
from model import *
from configs import *

class SAC:

    def __init__(self,state,action):
        
        self.gamma = Gamma
        self.tau = Tau
        self.alpha = Alpha

        self.critic = Q(state,action.shape[0]).to(Device)
        self.critic_optim,self.critic_target = Adam(self.critic.parameters(),lr=Learning_Rate),Q(state,action.shape[0]).to(Device)
        hard_update(self.critic_target,self.critic)

        self.log_alpha = torch.zeros(1,requires_grad=True,device=Device)
        self.log_alpha_optim = Adam([self.log_alpha],lr=Learning_Rate)

        self.policy =  Policy(state,action.shape[0],action)
        self.policy_optim = Adam(self.policy.parameters(),lr=Learning_Rate)

    def next_action(self,state):
        return self.policy.peek(torch.FloatTensor(state).to(Device).unsqueeze(0)).detach().cpu().numpy()[0]

    def update(self, memory):
        state,next_state,action,reward,done = memory.peek()

        state,next_state,action = map(lambda x: torch.FloatTensor(x).to(Device),[state,next_state,action])
        reward,done = map(lambda x:torch.FloatTensor(x).to(Device).unsqueeze(1),[reward,done])

        with torch.no_grad():
            next_state_action, next_state_log_pi = self.policy.peek(next_state)
            qf1_next_target, qf2_next_target = self.critic_target(next_state, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward + done * self.gamma * (min_qf_next_target)

        qf1, qf2 = self.critic(state, action) 
        qf1_loss,qf2_loss = F.mse_loss(qf1, next_q_value),F.mse_loss(qf2,next_q_value)
  
        pi, log_pi  = self.policy.peek(state)

        qf1_pi, qf2_pi = self.critic(state, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() 

        map(lambda x: process(x[0],x[1]),[(self.critic_optim,qf1_loss),(self.critic_optim,qf2_loss),(self.policy_optim,policy_loss)]) 

        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

        map(lambda x:process(x[0],x[1]),[(self.log_alpha_optim,alpha_loss)])

        self.alpha = self.log_alpha.exp()
         
        soft_update(self.critic_target, self.critic, self.tau)    

