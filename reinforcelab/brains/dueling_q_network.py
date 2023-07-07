import torch
import torch.nn as nn
from .brain import Brain
import gymnasium as gym
from typing import Tuple

from reinforcelab.utils import soft_update, build_fcnn, space_is_type, get_state_action_sizes


class DuelingQNetwork(nn.Module, Brain):
    def __init__(self, env: gym.Env, hidden_layers=[], learning_rate=0.001, alpha=0.001):
        super(DuelingQNetwork, self).__init__()
        self.state_size, self.action_size = self.__get_state_action_sizes(env)
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.model, self.value_head, self.advantage_head = self.__build_model()

    def __get_state_action_sizes(self, env: gym.Env) -> Tuple[int, int]:
        """Returns the state and action sizes established by the environment.
        If the ennvironment isn't state continuous raises an error

        Args:
            env (gym.Env): Gymnasium environment

        Returns:
            Tuple[int, int]: State and action sizes
        """
        obs_cont = space_is_type(env.observation_space, gym.spaces.Box)
        if not obs_cont:
            raise RuntimeError("Incompatible observation space")

        return get_state_action_sizes(env)

    def __build_model(self):
        layers_sizes = [self.state_size] + \
            self.hidden_layers + [self.action_size]
        linear_layers = build_fcnn(layers_sizes[:-1])
        value_head = build_fcnn([layers_sizes[-2], 1])
        advantage_head = build_fcnn(layers_sizes[-2:])
        return linear_layers, value_head, advantage_head

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
        soft_update(brain.model, self.model, self.alpha)
