from typing import Tuple
import torch
from torch import Tensor
import gymnasium as gym

from .estimator import Estimator
from reinforcelab.experience import Experience
from reinforcelab.brains import Brain
from reinforcelab.action_selectors import DiscreteActionSelector
from reinforcelab.utils import space_is_type


class ExpectedSARSAEstimator(Estimator):
    def __init__(self, env: gym.Env, action_selector: DiscreteActionSelector, gamma: float):
        """Creates an estimator instance

        Args:
            action_selector (ActionSelector): the target, more stable brain
            gamma (float): Gamma parameter or discount factor
        """
        self.__validate_env(env)
        self.action_selector = action_selector
        self.gamma = gamma

    def __validate_env(self, env: gym.Env):
        """Determines if the environment is compatible with the estimator.
        If the action space is not Dsicrete, raises an error

        Args:
            env (gym.Env): Gym environment
        """
        act_disc = space_is_type(env.action_space, gym.spaces.Discrete)
        if not act_disc:
            raise RuntimeError("Incompatible action space")

    def __call__(self, experience: Experience, brain: Brain) -> Tensor:
        """Computes the action-value estimation using the Expected SARSA algorith.
        It uses the action distribution (provided by the action_selector) to compute the
        expected value for the next state

        Args:
            experience (Experience): An ordered experience batch

        Returns:
            Tensor: Expected SARSA estimation for the given experience and policy
        """

        states, actions, rewards, next_states, dones, *_ = experience

        with torch.no_grad():
            # Implement SARSA
            # TODO: Can we generalize this?
            next_qs = brain.target(next_states)
            distributions = self.action_selector.distribution(next_qs)
            next_vals = torch.mul(next_qs, distributions).sum(dim=-1)
            target = rewards + self.gamma * next_vals * (1-dones)
        return target
