from typing import Tuple
import torch
from torch import Tensor

from .update_estimator import UpdateEstimator
from reinforcelab.experience import Experience
from reinforcelab.brains import Brain


class SARSAEstimator(UpdateEstimator):
    def __init__(self, local_brain: Brain, target_brain: Brain, gamma: float):
        """Creates a vanilla estimator

        Args:
            local_brain (Module): the local, more frequently updated brain
            target_brain (Module): the target, more stable brain
            gamma (float): Gamma parameter or discount factor
        """
        self.local_brain = local_brain
        self.target_brain = target_brain
        self.gamma = gamma

    def __call__(self, experience: Experience) -> Tuple[Tensor, Tensor]:
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
            next_qs = self.target_brain(next_states)
            next_vals = next_qs.gather(1, next_actions).squeeze()
            target = rewards + self.gamma * next_vals * (1-dones)
        pred_values = self.local_brain(states)
        pred = pred_values.gather(1, actions).squeeze()

        return pred, target