import torch

max_size = 1e6
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_episodes=2000
save_every=10

max_len = 100
start_timestep = 1e4
std_noise = 0.1 