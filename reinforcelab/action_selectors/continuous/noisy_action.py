
import torch
from .continuous_action_selector import ContinuousActionSelector

import gymnasium as gym

from reinforcelab.utils import tie_breaker, get_space_size, space_is_type


class NoisyAction(ContinuousActionSelector):
    def __init__(self, env: gym.Env, epsilon=0.0):
        self.action_space = env.action_space
        self.epsilon = epsilon

    def __call__(self, action: torch.Tensor, epsilon=None) -> torch.Tensor:
        """Selects an action according to the epsilon-greedy algorithm

        Args:
            action (Tensor): action to condition epsilon-greecy from

        Returns:
            Tensor: The selected action
        """
        min_vals = torch.tensor(self.action_space.low)
        max_vals = torch.tensor(self.action_space.high)
        if epsilon is not None:
            self.epsilon = epsilon

        # Vectorized implementation of epsilon-greedy
        action = torch.normal(action, epsilon)
        action = torch.clamp(action, min_vals, max_vals)
        return action

    def distribution(self, action: torch.Tensor) -> torch.Tensor:
        """Returns 

        Args:
            action_values (Tensor): Action values obtained for a given state

        Returns:
            Tensor: Probabilities for taking each action on the given state. Must sum to 1
        """

        return torch.distributions.Normal(action, self.epsilon)
