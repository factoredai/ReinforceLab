import torch
from typing import Tuple
from gymnasium import Env


def tie_breaker(values: torch.Tensor) -> torch.Tensor:
    """Provides a new version of the values tensor, where ties between maximal
    values are broken by adding random noise to those maximal values.

    Args:
        values (Tensor): Value tensor, where the first dimension is the batch dimension

    Returns:
        Tensor: Noisy types value tensor, which only differs to the input values by random noise on maximal values
    """
    # Get the maximal value for each row
    maximal_values = values.max(dim=-1).values
    # Get the cells that are tied for each row
    max_mask = (values.T == maximal_values).T.type(torch.int)
    # Generate some noise to apply to the values
    noise = torch.rand_like(values, dtype=torch.float32)
    # Add noise only to the maximal values
    noisy_maximal_values = values + noise * max_mask
    return noisy_maximal_values


def epsilon_greedy(qvalues: torch.Tensor, epsilon=0.0) -> torch.Tensor:
    """Selects an optilmal action 1-epsilon times, and a random one otherwise

    Args:
        qvalues (torch.Tensor): A batched tensor of qvalues

    Returns:
        torch.Tensor: A batched tensor of actions to take
    """
    qvalues = tie_breaker(qvalues)
    greedy_action = torch.argmax(qvalues, dim=-1)

    # Vectorized implementation of epsilon-greedy
    random_action = torch.randint_like(greedy_action, self.action_size)
    mask = (torch.rand_like(greedy_action, dtype=torch.float32)
            > epsilon).type(torch.int)
    action = greedy_action * mask + random_action * (1 - mask)
    return action


def get_state_action_sizes(env: Env) -> Tuple[int, int]:
    """Returns the state and action sizes from an environment.
    This method standardizes how to get those values from different environments,
    each of which could have different Space definition.

    Args:
        env (Env): Gymnasium environment from which to get state/action sizes

    Returns:
        Tuple[int, int]: A tuple with state_size, action_size respectively
    """
    if isinstance(env.observation_space, Tuple):
        state_size = len(env.observation_space)
    else:
        state_size = env.observation_space.n

    action_size = env.action_space.n
    return state_size, action_size


def soft_update(local_model: torch.nn.Module, target_model: torch.nn.Module, alpha: float):
    """Updates a target model parameters towards the local model parameters by alpha

    Args:
        local_model (torch.nn.Module): More frequently updated.
        target_model (torch.nn.Module): Less frequently updated model.
        alpha (float): amount by which to update the target model
    """
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(
            alpha * local_param.data + (1.0-alpha) * target_param.data)
