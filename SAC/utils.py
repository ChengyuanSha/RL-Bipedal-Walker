from configs import *
import torch.nn as nn
import torch

def train(m_steps,env,agent,memory):

    rewards,steps = [],[]

    for _ in range(Episodes):

        total_reward, total_steps, done = 0,0,False

        state = env.reset()
        for _ in range(m_steps):

            action = agent.next_action(state) if Steps < sum(steps) else env.action_space.sample()

            if len(memory.buffer) > Batch_Size: agent.update(memory)

            next_state,reward,done,_ = env.step(action)

            total_reward += reward
            total_steps += 1

            memory.push(state,next_state,action,reward,done)

            state = next_state

            if done:
                break
            
        rewards.append(total_reward)
        steps.append(total_steps)
    
    return rewards,steps

def save(agent, directory, filename, suffix):
    torch.save(agent.policy.state_dict(), '%s/%s_actor_%s.pth' % (directory, filename, suffix))
    torch.save(agent.critic.state_dict(), '%s/%s_critic_%s.pth' % (directory, filename, suffix))

def init_weight(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight,gain = 1)
        torch.nn.init.constant_(m.bias,0)

def process(optim,loss):
    optim.zero_grad()
    loss.backward()
    optim.step()

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

