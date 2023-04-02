from typing import Tuple
import torch
from torch import Tensor

from .update_estimator import UpdateEstimator
from reinforcelab.experience import Experience
from reinforcelab.brains import Brain

class VanillaEstimator(UpdateEstimator):
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
        """Computes the action-value estimation for an experience tuple with the given local and
        target networks. It computes the value estimation directly from the local target, as well
        as the bellman equation value estimation with the target network.

        Args:
            experience (Experience): An experience instance
            rewards, next_states and dones
            gamma (float): discount factor

        Returns:
            List[Tensor]: a list containing value estimation from the local network and the bellman update.
        """

        states, actions, rewards, next_states, dones, *_ = experience

        with torch.no_grad():
            # Implement DQN
            max_vals = self.target_brain(next_states).max(dim=1).values
            target = rewards + self.gamma * max_vals * (1-dones)
        pred_values = self.local_brain(states)
        pred = pred_values.gather(1, actions).squeeze()

        return pred, target