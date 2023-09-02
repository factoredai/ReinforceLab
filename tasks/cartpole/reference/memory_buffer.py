import numpy as np
import torch
from collections import namedtuple, deque


class MemoryBuffer:
    def __init__(self, max_size, n_steps):
        self.experience = namedtuple(
            'Experience', ['state', 'action', 'reward', 'next_state', 'done'])
        self.experience_buffer = deque(maxlen=max_size)
        self.n_steps = n_steps

    def add(self, state, action, reward, next_state, done):
        exp = self.experience(state, action, reward, next_state, done)
        self.experience_buffer.append(exp)

    def sample(self, batch_size):
        batch_idxs = np.random.choice(range(len(self.experience_buffer) - self.n_steps), size=batch_size, replace=False)
        experiences = [self.experience_buffer[idx] for idx in batch_idxs]
        states = torch.vstack([torch.tensor(exp.state)
                              for exp in experiences]).float()
        actions = torch.tensor(
            [exp.action for exp in experiences]).long().unsqueeze(1)
        rewards = torch.tensor([exp.reward for exp in experiences]).float()
        next_states = torch.vstack(
            [torch.tensor(exp.next_state) for exp in experiences]).float()
        dones = torch.tensor([float(exp.done) for exp in experiences]).float()

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.experience_buffer)
