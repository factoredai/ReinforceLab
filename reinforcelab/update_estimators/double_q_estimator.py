from typing import Tuple
import torch
from torch import Tensor
from torch.nn import Module
import numpy as np

from reinforcelab.experience import Experience
from .update_estimator import UpdateEstimator


class DoubleQEstimator(UpdateEstimator):
    def __init__(self, local_nn: Module, target_nn: Module, gamma: float):
        self.local_nn = local_nn
        self.target_nn = target_nn
        self.gamma = gamma

    def __call__(self, experience: Experience) -> Tuple[Tensor, Tensor]:
        """Computes the action-value estimation for an experience tuple with the given local and
        target networks. It computes the value estimation directly from the local target, as well
        as the bellman equation value estimation with the target network.

        Args:
            experience (Experienc): An experience instance, containing tensors for states, actions, 
            rewards, next_states and dones

        Returns:
            Tuple[Tensor, Tensor]: a list containing value estimation from the local network and the bellman update.
        """

        states, actions, rewards, next_states, dones, *_ = experience

        with torch.no_grad():
            # Implement DQN
            max_actions = np.argmax(self.local_nn(
                next_states), axis=1).unsqueeze(1)
            max_vals = self.target_nn(next_states)
            max_vals = max_vals.gather(1, max_actions).squeeze()
            target = rewards + self.gamma * max_vals * (1-dones)
        pred_values = self.local_nn(states)
        pred = pred_values.gather(1, actions).squeeze()

        return pred, target