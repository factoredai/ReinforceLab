from typing import Tuple
import torch
from torch import Tensor
import gymnasium as gym

from .update_estimator import UpdateEstimator
from reinforcelab.experience import Experience
from reinforcelab.brains import Brain
from reinforcelab.modules.utils import space_is_type


class SARSAEstimator(UpdateEstimator):
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

    def __call__(self, experience: Experience, brain: Brain) -> Tuple[Tensor, Tensor]:
        """Computes the action-value estimation using the SARSA algorith. It expects
        the experience to come in order so that it can extract the next_action.

        Args:
            experience (Experience): An ordered experience batch

        Returns:
            List[Tensor]: a list containing value estimation from the local network and the bellman update.
        """

        states, actions, rewards, next_states, dones, *_ = experience
        next_actions = actions[1:]
        states = states[:-1]
        actions = actions[:-1]
        rewards = rewards[:-1]
        next_states = next_states[:-1]
        dones = dones[:-1]

        with torch.no_grad():
            # Implement SARSA
            next_qs = brain.target(next_states)
            next_vals = next_qs.gather(1, next_actions).squeeze()
            target = rewards + self.gamma * next_vals * (1-dones)
        pred_values = brain.local(states)
        pred = pred_values.gather(1, actions).squeeze()

        return pred, target
