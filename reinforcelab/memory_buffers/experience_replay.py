import torch
import random
from collections import namedtuple, deque

from memory_buffer import MemoryBuffer

class ExperienceReplay(MemoryBuffer):
    def __init__(self, config):
        self.experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])
        self.experience_buffer = deque(maxlen=config['max_size'])

    def add(self, state, action, reward, next_state, done):
        """Adds an experience tuple to the memory buffer

        Args:
            state (tensor): A tensor describing the state of the environment
            action (int): The action taken by the agent
            reward (float): The reward obtained after interaction
            next_state (tensor): A tensor describing the obtained state after interaction
            done (bool): A bool indicating wether the end of the episode was reached
        """
        exp = self.experience(state, action, reward, next_state, done)
        self.experience_buffer.append(exp)

    def sample(self, batch_size):
        """Retrieves a random set of experience tuples from the memory buffer

        Args:
            batch_size (int): The number of samples to retrieve in a single batch
        """
        experiences = random.sample(list(self.experience_buffer), batch_size)
        states = torch.vstack([torch.tensor(exp.state) for exp in experiences]).float()
        actions = torch.tensor([exp.action for exp in experiences]).long().unsqueeze(1)
        rewards = torch.tensor([exp.reward for exp in experiences]).float()
        next_states = torch.vstack([torch.tensor(exp.next_state) for exp in experiences]).float()
        dones = torch.tensor([float(exp.done) for exp in experiences]).float()

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.experience_buffer)