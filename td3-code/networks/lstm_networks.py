import torch.nn.functional as F
import torch.nn as nn
import torch


class LSTMActor(nn.Module):

    def __init__(self, state_dim, action_dim, n_lstm, hidden_dim):
        super(LSTMActor, self).__init__()
        self.recurrent = False
        self.l1 = nn.LSTM(state_dim, n_lstm, batch_first=True)
        self.l2 = nn.Linear(n_lstm, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state, h):
        self.l1.flatten_parameters()
        a, h = self.l1(state, h)
        a = F.relu(self.l2(a))
        a = torch.tanh(self.l3(a))
        return a, h


class LSTMCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_lstm, hidden_dim):
        super(LSTMCritic, self).__init__()
        self.recurrent = False
        # Q1 architecture
        self.l1 = nn.LSTM(state_dim + action_dim, n_lstm, batch_first=True)
        self.l2 = nn.Linear(n_lstm, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)
        # Q2 architecture
        self.l4 = nn.LSTM(state_dim + action_dim, n_lstm, batch_first=True)
        self.l5 = nn.Linear(n_lstm, hidden_dim)
        self.l6 = nn.Linear(hidden_dim, 1)

    def Q1(self, state, action, h):
        sa = torch.cat([state, action], -1)
        self.l1.flatten_parameters()
        q1, h = self.l1(sa, h)
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1, h

    def Q2(self, state, action, h):
        sa = torch.cat([state, action], -1)
        self.l4.flatten_parameters()
        q2, h = self.l4(sa, h)
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q2, h
