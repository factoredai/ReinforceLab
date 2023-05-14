import torch
from typing import Tuple
import gymnasium as gym
from typing import List, Type
import torch.nn as nn
from itertools import zip_longest


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


def get_space_size(space: gym.spaces.Space) -> int:
    """This method standardizes how to get those values from different environments,
    each of which could have different Space definition.

    Args:
        space (gym.spaces.Space): Gymnasium space

    Returns:
        (int): the space size
    """
    if isinstance(space, gym.spaces.Tuple):
        size = len(space)
    elif isinstance(space, gym.spaces.Box):
        size = space.shape[0]
    else:
        size = space.n

    return size


def get_state_action_sizes(env: gym.Env) -> Tuple[int, int]:
    """Returns the state and action sizes for a given environment

    Args:
        env (gym.Env): Gymnasium environment

    Returns:
        Tuple[int, int]: State and action sizes
    """
    state_size = get_space_size(env.observation_space)
    action_size = get_space_size(env.action_space)

    return state_size, action_size


def space_is_type(space: gym.spaces.Space, space_type: Type[gym.spaces.Space]) -> bool:
    """Determines if a given space is constructed by a given space type. It unrolls tuple
    spaces and only returns True if all contained spaces are of the specified type.

    Args:
        space (gym.spaces.Space): Space to analyze
        space_type (Type[gym.spaces.Space]): Space type to identify

    Returns:
        bool: Wether the space is composed by the given type or not
    """
    spaces = [space]
    while len(spaces) > 0:
        space = spaces.pop()
        if isinstance(space, gym.spaces.Tuple):
            spaces += list(space)
            continue

        if not isinstance(space, space_type):
            return False

    return True


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
        continue


def build_fcnn(layers_sizes: List[int], activation_fn: nn.Module = nn.ReLU()) -> nn.Sequential:
    """Builds a fully connected neural network based on the layer sizes passed.
    The last layer are linear logits.

    Args:
        layers_sizes (List[int]): A list that specifies the size of each layer in the network
        activation_fn (nn.Module): Activation function to use in between layers

    Returns:
        nn.Module: a sequential network with the specified layers.
    """
    activations = [activation_fn]*(len(layers_sizes) - 2)
    layers_tuples = list(zip(layers_sizes, layers_sizes[1:]))
    linear_layers = [nn.Linear(in_size, out_size)
                     for in_size, out_size in layers_tuples]
    layers = [x for layer in zip_longest(
        linear_layers, activations) for x in layer if x is not None]
    return nn.Sequential(*layers)
