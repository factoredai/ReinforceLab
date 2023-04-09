import torch
import torch.nn as nn
from itertools import zip_longest
from .brain import Brain

from reinforcelab.utils import soft_update


class DuelingQNetwork(nn.Module, Brain):
    def __init__(self, state_size, action_size, hidden_layers=[], learning_rate=0.001, alpha=0.001):
        super(DuelingQNetwork, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.model, self.value_head, self.advantage_head = self.__build_model()

    def __build_model(self):
        layers_sizes = [self.state_size] + \
            self.hidden_layers + [self.action_size]
        activations = [nn.ReLU()]*(len(layers_sizes) - 2)
        layers_tuples = list(zip(layers_sizes, layers_sizes[1:]))
        linear_layers = [nn.Linear(in_size, out_size)
                         for in_size, out_size in layers_tuples[:-1]]
        value_head = nn.Linear(layers_tuples[-1][0], 1)
        advantage_head = nn.Linear(*layers_tuples[-1])
        layers = [x for layer in zip_longest(
            linear_layers, activations) for x in layer if x is not None]
        return nn.Sequential(*layers), value_head, advantage_head

    def forward(self, state):
        hidden_activations = self.model(state)
        value = self.value_head(hidden_activations)
        advantage = self.advantage_head(hidden_activations)
        return value + advantage

    def update(self, _, pred, target):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss = nn.MSELoss()

        loss_val = loss(pred, target)
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

    def update_from(self, brain: "DuelingQNetwork"):
        soft_update(self.model, brain.model, self.alpha)
