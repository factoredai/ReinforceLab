import torch
import torch.nn as nn
from itertools import zip_longest

from .brain import Brain
from reinforcelab.utils import soft_update


class QNetwork(nn.Module, Brain):
    def __init__(self, state_size, action_size, hidden_layers=[], learning_rate=0.01, alpha=0.001):
        super(QNetwork, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.model = self.__build_model()

    def __build_model(self):
        layers_sizes = [self.state_size] + \
            self.hidden_layers + [self.action_size]
        activations = [nn.ReLU()]*(len(layers_sizes) - 2)
        layers_tuples = list(zip(layers_sizes, layers_sizes[1:]))
        linear_layers = [nn.Linear(in_size, out_size)
                         for in_size, out_size in layers_tuples]
        layers = [x for layer in zip_longest(
            linear_layers, activations) for x in layer if x is not None]
        return nn.Sequential(*layers)

    def forward(self, state):
        return self.model(state)

    def update(self, _, pred, target):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss = nn.MSELoss()

        loss_val = loss(pred, target)
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

    def update_from(self, brain: "QNetwork"):
        soft_update(brain.model, self.model, self.alpha)
