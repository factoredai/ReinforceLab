import torch
from .discrete_action_selector import DiscreteActionSelector

import gymnasium as gym

from reinforcelab.utils import tie_breaker, get_space_size, space_is_type


class EpsilonGreedy(DiscreteActionSelector):
    def __init__(self, env: gym.Env, epsilon=0.0):
        self.action_size = self.__get_action_size(env)
        self.epsilon = epsilon

    def __get_action_size(self, env: gym.Env) -> int:
        """Returns the environment's action size. If the action space is
        not discrete, raises an error.

        Args:
            env (gym.Env): Gymansium environment

        Returns:
            int: Action size
        """
        act_disc = space_is_type(env.action_space, gym.spaces.Discrete)
        if not act_disc:
            raise RuntimeError("Incompatible action space type")

        return get_space_size(env.action_space)

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
