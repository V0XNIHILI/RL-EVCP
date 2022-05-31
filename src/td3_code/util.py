import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_sample_dict(sample_dict):
    
    state = sample_dict['state']
    state_next = sample_dict['state_next']
    observations = sample_dict['observations']
    observations_next = sample_dict['observations_next']
    actions = sample_dict['actions']
    dones = sample_dict['done']
    reset_mask = sample_dict['reset_mask']
    rewards = sample_dict['reward']

    observations_extended = np.concatenate([observations, observations_next[:, -1:]], axis=1)
    reset_mask_extended = np.concatenate([reset_mask, dones[:, -1:]], axis=1)
    state_extended = np.concatenate([state, state_next[:, -1:]], axis=1)

    state_extended = torch.tensor(state_extended, dtype=torch.float32).to(DEVICE)
    observations_extended = torch.tensor(observations_extended, dtype=torch.float32).to(DEVICE)
    actions = torch.tensor(actions, dtype=torch.float32).to(DEVICE)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)
    dones = torch.tensor(dones, dtype=torch.float32).to(DEVICE)
    reset_mask_extended = torch.tensor(reset_mask_extended, dtype=torch.float32).to(DEVICE)

    return state_extended, observations_extended, actions, rewards, dones, reset_mask_extended
