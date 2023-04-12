from torch import Tensor
from abc import ABC, abstractmethod
from ..action_selector import ActionSelector


class TabularActionSelector(ActionSelector):
    @abstractmethod
    def distribution(self, action_values: Tensor) -> Tensor:
        """The distribution of action probabilities to be taken
        given the passed action_values

        Args:
            action_values (Tensor): Action values obtained for a given state

        Returns:
            Tensor: Probabilities for taking each action on the given state. Must sum to 1
        """
