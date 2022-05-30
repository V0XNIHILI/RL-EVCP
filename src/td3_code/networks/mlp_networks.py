import torch.nn.functional as F
import torch.nn as nn
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = "cpu"

class Actor(nn.Module):

    def __init__(self, observation_dim, action_dim, max_action_val, hidden_dims_list,):
        super(Actor, self).__init__()
        self.recurrent = False
        self.max_action_val = max_action_val
        self.l1 = nn.Linear(observation_dim, hidden_dims_list[0],)
        self.l2 = nn.Linear(hidden_dims_list[0], hidden_dims_list[1])
        self.l3 = nn.Linear(hidden_dims_list[1], action_dim)
        self.parameters_list = list(self.parameters())

    def get_initial_state(self, batch_size):
        return torch.zeros(1, batch_size, 0).to(DEVICE)

    def forward(self, state, h):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = torch.tanh(self.l3(a)) * self.max_action_val
        return a, h


class Critic(nn.Module):
    def __init__(self, observation_dim, action_dim, hidden_dims_list):
        super(Critic, self).__init__()
        self.recurrent = False
        self.l1 = nn.Linear(observation_dim + action_dim, hidden_dims_list[0])
        self.l2 = nn.Linear(hidden_dims_list[0], hidden_dims_list[1])
        self.l3 = nn.Linear(hidden_dims_list[1], 1)
        self.parameters_list = list(self.parameters())

    def get_initial_state(self, batch_size):
        return torch.zeros(1, batch_size, 0).to(DEVICE)

    def forward(self, observation, action, h):
        sa = torch.cat([observation, action], -1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1, h

