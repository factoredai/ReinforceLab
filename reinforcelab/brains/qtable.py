import torch
from torch import Tensor
from collections import defaultdict

from .brain import Brain

class QTable(Brain):
    def __init__(self, state_size, action_size, alpha = 0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha
        self.table = defaultdict(lambda: torch.zeros(action_size))

    def __call__(self, state: Tensor) -> Tensor:
        # Assume that a batch of states were passetate
        results = [self.table[single_state] for single_state in state]
        return torch.vstack(results)

        return self.table[state]

    def update(self, experience, pred, target):
        state, action, *_ = experience
        td_error = target - pred
        new_val = pred + self.alpha * td_error
        # Assume that a batch was passed
        for single_state, single_action, new_single_val in zip(state, action, new_val):
            self.table[single_state][single_action] = new_single_val