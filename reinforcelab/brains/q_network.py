import torch
import torch.nn as nn
from torch import Tensor
import gymnasium as gym
from typing import Tuple
from copy import deepcopy

from .brain import Brain
from reinforcelab.estimators import Estimator
from reinforcelab.experience import Experience
from reinforcelab.utils import soft_update, build_fcnn, space_is_type, get_state_action_sizes


class QNetwork(Brain):
    def __init__(self, model: nn.Module, estimator: Estimator, learning_rate=0.01, alpha=0.001):
        super(QNetwork, self).__init__()
        self.local_model = model
        self.target_model = deepcopy(model)
        self.estimator = estimator
        self.learning_rate = learning_rate
        self.alpha = alpha

    def __call__(self, state):
        return self.local_model(state)

    def target(self, state):
        return self.target_model(state)

    def action_value(self, state: Tensor, action: Tensor, target: bool = False) -> Tensor:
        if target:
            q = self.target(state)
        else:
            q =self.local(state)
        return q.gather(1, action).squeeze()

    def max_action(self, state: Tensor, target: bool = False) -> Tensor:
        if target:
            return self.target(state).max(dim=1)
        return self(state).max(dim=1)

    def update(self, experience: Experience):
        optimizer = torch.optim.Adam(self.local_model.parameters(), lr=self.learning_rate)
        loss = nn.MSELoss()

        target = self.estimator(experience, self)

        states, actions, *_ = experience
        pred = self.local(states).gather(1, actions).squeeze()
        loss_val = loss(pred, target)
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        soft_update(self.local_model, self.target_model, self.alpha)