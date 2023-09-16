from typing import Tuple
import torch
from torch import Tensor
import gymnasium as gym

from .update_estimator import UpdateEstimator
from reinforcelab.experience import Experience
from reinforcelab.brains import Brain
from reinforcelab.utils import space_is_type


class MaxQEstimator(UpdateEstimator):
    def __init__(self, env: gym.Env, brain: Brain, gamma: float, transforms=None):
        """Creates a Q estimator

        Args:
            local_brain (Module): the local, more frequently updated brain
            target_brain (Module): the target, more stable brain
            gamma (float): Gamma parameter or discount factor
        """
        self.__validate_env(env)
        self.brain = brain
        self.gamma = gamma
        self.transforms = transforms

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
            experience (Experience): An experience instance or batch

        Returns:
            List[Tensor]: a list containing value estimation from the local network and the bellman update.
        """

        if self.transforms:
            experience = self.transforms(experience)

        states, actions, rewards, next_states, dones, *_ = experience

        with torch.no_grad():
            # Implement DQN
            max_vals = self.brain.target(next_states).max(dim=1).values
            target = rewards.squeeze() + self.gamma * max_vals * (1-dones.squeeze())
        pred_values = self.brain.local(states)
        pred = pred_values.gather(1, actions).squeeze()

        return pred, target
