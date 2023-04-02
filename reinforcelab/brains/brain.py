from torch import Tensor
from abc import ABC, abstractmethod

from reinforcelab.experience import Experience

class Brain(ABC):
    @abstractmethod
    def __call__(self, state: Tensor) -> Tensor:
        """Performs a computation over a state to determine the next actions.
        This could be a state value (V), state action value (Q), or a distribution
        over actions (pi)

        Args:
            state (Tensor): a tensor description of the state

        Returns:
            Tensor: Result of the computation over the state
        """

    @abstractmethod
    def update(self, experience: Experience, pred: Tensor, target: Tensor):
        """Updates the brain estimation given the current state computation and
        the expected state computation.

        Args:
            experience (Experience): An experience instance
            pred (Tensor): Current estimation for a given state
            target (Tensor): Expected estimation for the same given state
        """