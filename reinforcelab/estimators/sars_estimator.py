from typing import Tuple
import torch
from torch import Tensor
import gymnasium as gym

from .estimator import Estimator
from reinforcelab.experience import Experience
from reinforcelab.brains import Brain
from reinforcelab.utils import space_is_type


class SARSEstimator(Estimator):
    def __init__(self, env: gym.Env, gamma: float):
        """Creates a SARS estimator

        Args:
            gamma (float): Gamma parameter or discount factor
        """
        self.gamma = gamma

    def __call__(self, experience: Experience, brain: Brain) -> Tensor:
        """Computes the action-value estimation using the SARS algorith. It
        approximates the SARSA algorithm by generating the next actions with
        the current policy

        Args:
            experience (Experience): An ordered experience batch

        Returns:
            Tensor: SARSA Value estimation for the given experience
        """

        states, actions, rewards, next_states, dones, *_ = experience
        # Generate the next actions according to the current policy

        with torch.no_grad():
            # Implement SARSA
            next_actions = brain.target(next_states)
            next_vals = brain.action_value(next_states, next_actions, target=True)
        target = rewards + self.gamma * next_vals * (1-dones)

        return target
