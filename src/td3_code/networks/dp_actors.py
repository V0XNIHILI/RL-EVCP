import torch.nn.functional as F
import torch.nn as nn
import torch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = "cpu"

class MLPActor(nn.Module):

    def __init__(self, observation_dim, action_dim, max_action_val, hidden_dims_list, ):
        super(MLPActor, self).__init__()
        self.recurrent = False
        self.max_action_val = max_action_val
        self.linear_layers = nn.ModuleList()
        self.linear_activations = []
        self.parameters_list = []
        input_shape = observation_dim
        for hidden_dim in hidden_dims_list:
            self.linear_layers.append(nn.Linear(input_shape, hidden_dim).to(DEVICE))
            self.linear_activations.append(torch.relu)
            input_shape = hidden_dim
        self.linear_layers.append(nn.Linear(input_shape, action_dim).to(DEVICE))
        self.parameters_list = list(self.parameters())
        self.linear_activations.append(torch.tanh)

    def get_initial_state(self, batch_size):
        return torch.zeros(1, batch_size, 0).to(DEVICE)

    def forward(self, observation, hidden_state=None):
        """ Input must be tensors """
        x = observation
        hidden_state_new = hidden_state
        for layer, activation in zip(self.linear_layers, self.linear_activations):
            x = activation(layer(x))
        return x * self.max_action_val, hidden_state_new


class LSTMActor(nn.Module):

    def __init__(self, observation_dim, action_dim, max_action_val, lstm_dims_list, hidden_dims_list):
        super(LSTMActor, self).__init__()
        self.recurrent = True
        self.max_action_val = max_action_val
        self.lstm_layers = nn.ModuleList()
        self.lstm_activations = []
        self.linear_layers = nn.ModuleList()
        self.linear_activations = []

        input_shape = observation_dim
        for n_lstm in lstm_dims_list:
            self.lstm_layers.append(nn.LSTM(input_shape, n_lstm, batch_first=True).to(DEVICE))
            self.lstm_activations.append(lambda x: x)
            input_shape = n_lstm
        for hidden_dim in hidden_dims_list:
            self.linear_layers.append(nn.Linear(input_shape, hidden_dim).to(DEVICE))
            self.linear_activations.append(F.relu)
            input_shape = hidden_dim
        self.linear_layers.append(nn.Linear(input_shape, action_dim).to(DEVICE))
        self.linear_activations.append(torch.tanh)
        self.parameters_list = list(self.parameters())

    def get_initial_state(self, batch_size):
        hidden_state = []
        for layer in self.lstm_layers:
            h = torch.zeros((1, batch_size, layer.hidden_size), dtype=torch.float).to(DEVICE)
            c = torch.zeros((1, batch_size, layer.hidden_size), dtype=torch.float).to(DEVICE)
            hidden_state.append((h, c))
        return hidden_state

    def forward(self, observation, hidden_state):
        x = observation
        hidden_state_new = []
        for layer, activation, h in zip(self.lstm_layers, self.lstm_activations, hidden_state):
            x, h_new = layer(x, h)
            x = activation(x)
            hidden_state_new.append(h_new)

        for layer, activation in zip(self.linear_layers, self.linear_activations):
            x = activation(layer(x))
        #print('h new', hidden_state_new)
        return x * self.max_action_val, hidden_state_new
