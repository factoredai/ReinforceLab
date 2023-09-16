from typing import Tuple
import torch
from torch import Tensor
import gymnasium as gym

from .update_estimator import UpdateEstimator
from reinforcelab.experience import Experience
from reinforcelab.brains import Brain
from reinforcelab.action_selectors import DiscreteActionSelector
from reinforcelab.utils import space_is_type


class ExpectedSARSAEstimator(UpdateEstimator):
    def __init__(self, env: gym.Env, brain: Brain, action_selector: DiscreteActionSelector, gamma: float):
        """Creates an estimator instance

        Args:
            local_brain (Brain): the local, more frequently updated brain
            target_brain (Brain): the target, more stable brain
            action_selector (ActionSelector): the target, more stable brain
            gamma (float): Gamma parameter or discount factor
        """
        self.__validate_env(env)
        self.brain = brain
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

    def __call__(self, experience: Experience) -> Tuple[Tensor, Tensor]:
        """Computes the action-value estimation using the Expected SARSA algorith.
        It uses the action distribution (provided by the action_selector) to compute the
        expected value for the next state

        Args:
            experience (Experience): An ordered experience batch

        Returns:
            List[Tensor]: a list containing value estimation from the local network and the bellman update.
        """

        states, actions, rewards, next_states, dones, *_ = experience

        with torch.no_grad():
            # Implement SARSA
            next_qs = self.brain.target(next_states)
            distributions = self.action_selector.distribution(next_qs)
            next_vals = torch.mul(next_qs, distributions).sum(dim=-1)
            target = rewards + self.gamma * next_vals * (1-dones)
        pred_values = self.brain.local(states)
        pred = pred_values.gather(1, actions).squeeze()

        return pred, target
