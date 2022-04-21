from torch import torch

# Train
Episodes = 10000
Batch = 256
Seed = 0

Learning_Rate = 0.0005
Steps = 10000
Replay_size = 1000000

# Agent
Device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Hidden_Size = 256
Gamma = 0.99
Tau = 0.005
Alpha = 0.2

# Memory
Batch_Size = 256
Replay_size=1000000

# Model
Log_sig_max = 2
Log_sig_min = -20
Epsilon = 1e-6