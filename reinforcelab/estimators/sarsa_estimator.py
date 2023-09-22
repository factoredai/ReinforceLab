from typing import Tuple
import torch
from torch import Tensor
import gymnasium as gym

from .estimator import Estimator
from reinforcelab.experience import Experience
from reinforcelab.brains import Brain
from reinforcelab.utils import space_is_type


class SARSAEstimator(Estimator):
    def __init__(self, env: gym.Env, gamma: float):
        """Creates a vanilla estimator

        Args:
            gamma (float): Gamma parameter or discount factor
        """
        self.__validate_env(env)
        self.gamma = gamma

    def __validate_env(self, env: gym.Env):
        """Determines if the environment is compatible with the estimator.
        If the action space is not Dsicrete, raises an error

        Args:
            env (gym.Env): Gym environment
        """
        act_discrete = space_is_type(env.action_space, gym.spaces.Discrete)
        if not act_discrete:
            raise RuntimeError("Incompatible action space")

    def __call__(self, experience: Experience, brain: Brain) -> Tensor:
        """Computes the action-value estimation using the SARSA algorith. It expects
        the experience to come in order so that it can extract the next_action.

        Args:
            experience (Experience): An ordered experience batch

        Returns:
            Tensor: SARSA Value estimation for the given experience
        """

        # TODO: Use n-step instead of ordered experience
        # Use the last step for retrieving next actions
        # This in turn transforms the n-step into (n-1)-step
        # So, any SARSA implementation would require at least
        # n=2 n-step
        states, actions, rewards, next_states, dones, *_ = experience
        next_actions = actions[1:]
        states = states[:-1]
        actions = actions[:-1]
        rewards = rewards[:-1]
        next_states = next_states[:-1]
        dones = dones[:-1]

        with torch.no_grad():
            # Implement SARSA
            next_vals = brain.action_value(next_states, next_actions, target=True)
            target = rewards + self.gamma * next_vals * (1-dones)

        return target
