from typing import Tuple
from torch import Tensor
from abc import ABC, abstractmethod

from reinforcelab.experience import Experience


class UpdateEstimator(ABC):
    @abstractmethod
    def __call__(self, experience: Experience) -> Tuple[Tensor, Tensor]:
        """Computes the the bellman update for an experience tuple with the given local and
        target brain. It computes the update estimation directly from the local brain, as well
        as the bellman equation value estimation with the target brain.

        Args:
            experience (Experience): An instance of experience
            rewards, next_states and dones

        Returns:
            Tuple[Tensor, Tensor]: a list containing value estimation from the local network and the bellman update.
        """
