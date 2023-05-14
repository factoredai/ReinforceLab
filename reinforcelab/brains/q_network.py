import torch
import torch.nn as nn
import gymnasium as gym
from typing import Tuple

from .brain import Brain
from reinforcelab.utils import soft_update, build_fcnn, space_is_type, get_state_action_sizes


class QNetwork(nn.Module, Brain):
    def __init__(self, env: gym.Env, hidden_layers=[], learning_rate=0.01, alpha=0.001):
        super(QNetwork, self).__init__()
        self.state_size, self.action_size = self.__get_state_action_sizes(env)
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.model = self.__build_model()

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
        return build_fcnn(layers_sizes)

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
