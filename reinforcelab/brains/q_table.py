import torch
from torch import Tensor
from typing import Tuple
from collections import defaultdict
import gymnasium as gym

from .brain import Brain
from reinforcelab.utils import space_is_type, get_state_action_sizes


class QTable(Brain):
    def __init__(self, env: gym.Env, alpha=0.01):
        self.state_size, self.action_size = self.__get_state_action_sizes(env)
        self.alpha = alpha
        self.table = defaultdict(lambda: torch.zeros(self.action_size))

    def __get_state_action_sizes(self, env: gym.Env) -> Tuple[int, int]:
        """Returns the state and action sizes established by the environment.
        If the ennvironment isn't discrete in both state and action, raises an error

        Args:
            env (gym.Env): Gymnasium environment

        Returns:
            Tuple[int, int]: State and action sizes
        """
        obs_disc = space_is_type(env.observation_space, gym.spaces.Discrete)
        action_disc = space_is_type(env.action_space, gym.spaces.Discrete)
        if not obs_disc or not action_disc:
            raise RuntimeError("Incompatible action/observation space")

        return get_state_action_sizes(env)

    def __call__(self, state: Tensor) -> Tensor:
        # Assume that a batch of states were passed
        results = []
        for single_state in state:
            idx = self.__state2idx(single_state)
            result = self.table[idx]
            results.append(result)
        return torch.vstack(results)

    def __state2idx(self, state: Tensor) -> Tuple:
        state_list = state.tolist()
        if isinstance(state_list, list):
            return tuple(state_list)
        else:
            return tuple([state_list])

    def update(self, experience, pred, target):
        state, action, *_ = experience
        td_error = target - pred
        new_val = pred + self.alpha * td_error
        # Assume that a batch was passed
        for single_state, single_action, new_single_val in zip(state, action, new_val):
            idx = self.__state2idx(single_state)
            self.table[idx][single_action] = new_single_val

    def update_from(self, brain: "QTable"):
        # QTables don't need an inter-brain update procedure
        pass
