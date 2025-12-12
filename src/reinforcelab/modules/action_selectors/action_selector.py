from torch import Tensor
from abc import ABC, abstractmethod


class ActionSelector(ABC):
    @abstractmethod
    def __call__(self, action_values: Tensor, **kwargs) -> Tensor:
        """Selects an action according to the action values or scores.

        Args:
            action_values (Tensor): Representation of the action values. This could be action-state values,
                action policy distribution, or any representation of the action preference for the current state.

        Returns:
            Tensor: The selected action
        """
