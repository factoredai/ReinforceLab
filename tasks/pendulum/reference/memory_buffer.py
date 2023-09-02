import torch
import random
from collections import namedtuple, deque


class MemoryBuffer:
    def __init__(self, max_size):
        self.experience = namedtuple(
            'Experience', ['state', 'action', 'reward', 'next_state', 'done'])
        self.experience_buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        exp = self.experience(state, action, reward, next_state, done)
        self.experience_buffer.append(exp)

    def sample(self, batch_size):
        experiences = random.sample(list(self.experience_buffer), batch_size)
        states = torch.vstack([torch.tensor(exp.state)
                              for exp in experiences]).float()
        actions = torch.tensor(
            [exp.action for exp in experiences]).float().reshape((-1, 1))
        rewards = torch.tensor([exp.reward for exp in experiences]).float()
        next_states = torch.vstack(
            [torch.tensor(exp.next_state) for exp in experiences]).float()
        dones = torch.tensor([float(exp.done) for exp in experiences]).float().unsqueeze(-1)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.experience_buffer)
