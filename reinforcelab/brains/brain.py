from torch import Tensor
from abc import ABCMeta, abstractmethod

from reinforcelab.experience import Experience


class Brain(metaclass=ABCMeta):
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
    def target(self, state: Tensor) -> Tensor:
        """Performs a computation over a state using the target model. Used for
        computing error estimation and obtaining training gradients

        Args:
            state (Tensor): a tensor description of the state

        Returns:
            Tensor: Result of the computation over the state
        """

    def local(self, state: Tensor) -> Tensor:
        """Alias for __call__

        Args:
            state (Tensor): a tensor description of the state

        Returns:
            Tensor: Result of the computation over the state
        """
        return self.__call__(state)

    @abstractmethod
    def update(self, experience: Experience):
        """Updates the brain estimation given the passed experience

        Args:
            experience (Experience): An experience instance
        """