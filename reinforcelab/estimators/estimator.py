from typing import Tuple
from torch import Tensor
from abc import ABC, abstractmethod

from reinforcelab.experience import Experience
from reinforcelab.brains import Brain


class Estimator(ABC):
    @abstractmethod
    def __call__(self, experience: Experience, brain: Brain) -> Tensor:
        """Computes a value estimation for the current experience

        Args:
            experience (Experience): An instance of experience
            rewards, next_states and dones
            brain (Brain): A brain instance for computing estimations from it

        Returns:
            Tensor: Estimation of the experience value according to the current policy
        """
