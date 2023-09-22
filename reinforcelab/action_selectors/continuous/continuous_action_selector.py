from torch import Tensor
from abc import ABC, abstractmethod
from ..action_selector import ActionSelector


class ContinuousActionSelector(ActionSelector):
    @abstractmethod
    def distribution(self, action: Tensor) -> Tensor:
        """The distribution of action probabilities to be taken
        given the passed action_values

        Args:
            action (Tensor): Action to base the distribution from

        Returns:
            Tensor: Probability distribution over the action space, conditioned on the current action
        """
