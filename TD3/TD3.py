from models import Actor,Critic
from configs import device,torch
import torch.nn.functional as F

class TD3: 

    def __init__(self, state_dim, action_dim, max_action):
        self.ac = Actor(state_dim, action_dim, max_action).to(device)
        self.ac_target = Actor(state_dim, action_dim, max_action).to(device)
        self.ac_target.load_state_dict(self.ac.state_dict())
        self.ac_optimizer = torch.optim.Adam(self.ac.parameters())

        self.cr1 = Critic(state_dim, action_dim).to(device)
        self.cr1_target = Critic(state_dim, action_dim).to(device)
        self.cr1_target.load_state_dict(self.cr1.state_dict())
        self.cr1_optimizer = torch.optim.Adam(self.cr1.parameters())
        
        self.cr2 = Critic(state_dim, action_dim).to(device)
        self.cr2_target = Critic(state_dim, action_dim).to(device)
        self.cr2_target.load_state_dict(self.cr2.state_dict())
        self.cr2_optimizer = torch.optim.Adam(self.cr2.parameters())

        self.max_action = max_action
        self.action_dim = action_dim

    def select_action(self, state):
        return self.ac(self.tensor(state.reshape(1, -1))).cpu().data.numpy().flatten()

    def tensor(self, val):
        return torch.FloatTensor(val).to(device)
    
    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, \
              tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        
        for it in range(iterations):
            s, s1, a, r, d = replay_buffer.sample(batch_size)
            #state, next_state, action, reward, done = list(map(lambda x:torch.FloatStorage(x).to(device),[s,s1,a,r,1-d]))
            state, next_state, action, reward, done = self.tensor(s), self.tensor(s1), self.tensor(a), self.tensor(r), self.tensor(1 - d)

            # Select action according to policy and add clipped noise 
            noise = (
                    torch.FloatTensor(a).data.normal_(0, policy_noise).to(device)
            ).clamp(-noise_clip, noise_clip)

            next_action = (self.ac_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.cr1_target(next_state, next_action), self.cr2_target(next_state, next_action)
            target_Q = reward + (done * discount * torch.min(target_Q1, target_Q2)).detach()

            # Get current Q estimates
            current_Q1, current_Q2 = self.cr1(state, action), self.cr2(state, action)

            # Compute the loss of two critics
            cr1_loss, cr2_loss = map(lambda x: F.mse_loss(x, target_Q), [current_Q1,current_Q2])

            # Optimize the critic
            def optim(cr,loss):
                cr.zero_grad()
                loss.backward()
                cr.step()

            map(lambda x,y:optim(x,y),zip([self.cr1_optimizer,self.cr2_optimizer],[cr1_loss,cr2_loss]))

            # Update the actor every two steps
            if it % policy_freq == 0:

                # Compute actor loss
                ac_loss = -self.cr1.forward(state, self.ac(state)).mean()

                # Optimize the actor 
                optim(self.ac_optimizer,ac_loss)

                self.update_param(tau)

    # Update the networks
    def update_param(self, tau):
        if tau == None:
            return
        
        cr1_param, cr1_param_target = self.cr1.parameters(), self.cr1_target.parameters()
        while True:
            try:
                elem1 = next(cr1_param)
                elem2 = next(cr1_param_target)
            except StopIteration:
                break
            elem2.data.copy_(tau * elem1.data + (1 - tau) * elem2.data)

        cr2_param, cr2_param_target = self.cr2.parameters(), self.cr2_target.parameters()
        while True:
            try:
                elem3 = next(cr2_param)
                elem4 = next(cr2_param_target)
            except StopIteration:
                break
            elem4.data.copy_(tau * elem3.data + (1 - tau) * elem4.data)

        ac_param, ac_param_target = self.ac.parameters(), self.ac_target.parameters()
        while True:
            try:
                elem5 = next(ac_param)
                elem6 = next(ac_param_target)
            except StopIteration:
                break
            elem6.data.copy_(tau * elem5.data + (1 - tau) * elem6.data) 
