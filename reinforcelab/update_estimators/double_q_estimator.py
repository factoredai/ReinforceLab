from typing import Tuple
import torch
from torch import Tensor
from torch.nn import Module
import numpy as np
import gymnasium as gym

from reinforcelab.experience import Experience
from reinforcelab.utils import space_is_type
from reinforcelab.brains import Brain
from .update_estimator import UpdateEstimator


class DoubleQEstimator(UpdateEstimator):
    def __init__(self, env: gym.Env, brain: Brain, gamma: float):
        self.__validate_env(env)
        self.brain = brain
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

    def __call__(self, experience: Experience) -> Tuple[Tensor, Tensor]:
        """Computes the action-value estimation for an experience tuple with the given local and
        target networks. It computes the value estimation directly from the local target, as well
        as the bellman equation value estimation with the target network.

        Args:
            experience (Experience): An experience instance, containing tensors for states, actions, 
            rewards, next_states and dones

        Returns:
            Tuple[Tensor, Tensor]: a list containing value estimation from the local network and the bellman update.
        """

        states, actions, rewards, next_states, dones, *_ = experience

        with torch.no_grad():
            # Implement DQN
            max_actions = np.argmax(self.brain.local(
                next_states), axis=1).unsqueeze(1)
            max_vals = self.brain.target(next_states)
            max_vals = max_vals.gather(1, max_actions).squeeze()
            target = rewards + self.gamma * max_vals * (1-dones)
        pred_values = self.brain.local(states)
        pred = pred_values.gather(1, actions).squeeze()

        return pred, target
