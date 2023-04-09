import torch
from torch import Tensor
from typing import Tuple
from collections import defaultdict

from .brain import Brain


class QTable(Brain):
    def __init__(self, state_size, action_size, alpha=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha
        self.table = defaultdict(lambda: torch.zeros(action_size))

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
