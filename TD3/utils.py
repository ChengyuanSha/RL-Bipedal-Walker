from collections import deque
from models import ReplayBuffer
from configs import n_episodes,max_len,start_timestep,std_noise
import numpy as np

def train(env,agent):
    file_log = open('TD3_record.csv', "w+")
    file_log.write('episode,step,reward\n')
    scores_deque = deque(maxlen = max_len)
    scores_array = []

    replay_buf = ReplayBuffer()                 # Init ReplayBuffer

    total_timesteps = 0

    for i_episode in range(1, n_episodes+1):

        timestep = 0
        total_reward = 0

        # Reset environment and the done flag
        state, done = env.reset(), False

        while True:

            # Select action randomly while below the target time step(for exploration)
            if total_timesteps < start_timestep:
                action = env.action_space.sample()
            else:
                # Select a certain action
                action = agent.select_action(np.array(state))
                if std_noise != 0: 
                    shift_action = np.random.normal(0, std_noise, size=agent.action_dim)
                    action = (action + shift_action).clip(env.action_space.low, env.action_space.high)

            # Perform action
            new_state, reward, done, _ = env.step(action) 
            done_bool = 0 if timestep + 1 == env._max_episode_steps else float(done)
            total_reward += reward                          # full episode reward

            # Store every timestep in replay buffer
            replay_buf.add((state, new_state, action, reward, done_bool))
            state = new_state

            timestep += 1     
            total_timesteps += 1

            if done:
                break

        scores_deque.append(total_reward)
        scores_array.append(total_reward)

        avg_score = np.mean(scores_deque)

        to_print = '{},{},{}\n'.format(i_episode, total_timesteps, avg_score)
        print('Ep. {}, Timestep {},  Ep.Timesteps {}, Avg.Score: {:.2f} '\
                .format(i_episode, total_timesteps, timestep, avg_score))
        file_log.write(to_print)
        file_log.flush()
        agent.train(replay_buf, timestep)

    return scores_array

