import torch
from .tabular_action_selector import TabularActionSelector

from reinforcelab.utils import tie_breaker


class EpsilonGreedy(TabularActionSelector):
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
        if epsilon is not None:
            self.epsilon = epsilon

        qvalues = tie_breaker(action_values)
        greedy_action = torch.argmax(qvalues, dim=-1)

        # Vectorized implementation of epsilon-greedy
        random_action = torch.randint_like(greedy_action, self.action_size)
        mask = (torch.rand_like(greedy_action, dtype=torch.float32)
                > self.epsilon).type(torch.int)
        action = greedy_action * mask + random_action * (1 - mask)
        return action

    def distribution(self, action_values: torch.Tensor) -> torch.Tensor:
        """Computes the probability of taking each action following the
        epsilon-greedy policy. All actions have a probability of epsilon/action_size,
        but the remaining probability goes to the greedy action

        Args:
            action_values (Tensor): Action values obtained for a given state

        Returns:
            Tensor: Probabilities for taking each action on the given state. Must sum to 1
        """
        qvalues = tie_breaker(action_values)
        greedy_action = torch.argmax(qvalues, dim=-1)

        random_prob = self.epsilon / self.action_size
        greedy_prob = 1 - self.epsilon + random_prob

        dist = torch.zeros_like(qvalues) + random_prob
        dist[torch.arange(dist.size(0)), greedy_action] = greedy_prob

        return dist
