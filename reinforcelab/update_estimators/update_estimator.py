from typing import Tuple
from torch import Tensor
from abc import ABC, abstractmethod

from reinforcelab.experience import Experience

class UpdateEstimator(ABC):
    @abstractmethod
    def __call__(self, experience: Experience, gamma: float) -> Tuple[Tensor, Tensor]:
        """Computes the the bellman update for an experience tuple with the given local and
        target networks. It computes the update estimation directly from the local target, as well
        as the bellman equation value estimation with the target network.

        Args:
            experience (Experience): An instance of experience
            rewards, next_states and dones
            gamma (float): discount factor

        Returns:
            Tuple[Tensor, Tensor]: a list containing value estimation from the local network and the bellman update.
        """