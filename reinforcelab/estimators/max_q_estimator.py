from typing import Tuple
import torch
from torch import Tensor
import gymnasium as gym

from .estimator import Estimator
from reinforcelab.experience import Experience
from reinforcelab.brains import Brain
from reinforcelab.utils import space_is_type


class MaxQEstimator(Estimator):
    def __init__(self, env: gym.Env, gamma: float, transforms=None):
        """Creates a Q estimator

        Args:
            gamma (float): Gamma parameter or discount factor
        """
        self.gamma = gamma
        self.transforms = transforms

    def __call__(self, experience: Experience, brain: Brain) -> Tensor:
        """Computes the action-value estimation for an experience tuple with the given local and
        target networks. It computes the value estimation directly from the local target, as well
        as the bellman equation value estimation with the target network.

        Args:
            experience (Experience): An experience instance or batch

        Returns:
            Tensor: Max Q Value estimation for the given experience
        """

        if self.transforms:
            experience = self.transforms(experience)

        states, actions, rewards, next_states, dones, *_ = experience

        with torch.no_grad():
            # Implement DQN
            max_vals = brain.max_action(next_states, target=True).values
            target = rewards.squeeze() + self.gamma * max_vals * (1-dones.squeeze())

        return target
