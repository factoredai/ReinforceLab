import torch
from .action_selector import ActionSelector

from reinforcelab.utils import tie_breaker


class EpsilonGreedy(ActionSelector):
    def __init__(self, action_size: int, epsilon=0.0):
        self.epsilon = epsilon
        self.action_size = action_size

    def __call__(self, action_values: torch.Tensor, epsilon=None) -> torch.Tensor:
        """Selects an action according to the epsilon-greedy algorithm

        Args:
            action_values (Tensor): Representation of the action values. This could be action-state values,
                action policy distribution, or any representation of the action preference for the current state.

        Returns:
            Tensor: The selected action
        """
        if epsilon is None:
            epsilon = self.epsilon

        qvalues = tie_breaker(action_values)
        greedy_action = torch.argmax(qvalues, dim=-1)

        # Vectorized implementation of epsilon-greedy
        random_action = torch.randint_like(greedy_action, self.action_size)
        mask = (torch.rand_like(greedy_action, dtype=torch.float32)
                > epsilon).type(torch.int)
        action = greedy_action * mask + random_action * (1 - mask)
        return action
