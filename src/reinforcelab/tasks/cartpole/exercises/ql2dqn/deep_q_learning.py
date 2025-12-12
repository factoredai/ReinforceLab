import os
from copy import deepcopy
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from reinforcelab.modules.agents.agent import Agent
from reinforcelab.modules.memory_buffers.memory_buffer import MemoryBuffer


class DeepQLearningAgent(Agent):
    def __init__(self, env, gamma, alpha, learning_rate=0.0003, batch_size=128, update_every=10):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.update_every = update_every
        self.step = 0

        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        ########## START CODING HERE ############
        # For Q-Learning, we would instantiate our Q-table
        # What do we need to instantiate instead for DQN?

        ########## END CODING HERE ##############

    def act(self, state, epsilon=0.0):
        state = torch.tensor(state).float().unsqueeze(0)
        ########## START CODING HERE ############
        # For Q-Learning we would retrieve the Q-Values
        # from the table, indexing by the state
        # How should we retrieve the Q-Values here?

        ########## END CODING HERE ##############

        action = np.argmax(qvalues)

        # Randomly choose an action with p=epsilon
        if np.random.rand() < epsilon:
            action = np.random.choice(self.action_size)
        return action

    def update(self, state, action, reward, next_state, done):
        ########## START CODING HERE ############
        # Updating was pretty straightforward for Q-Learning
        # For Deep Q Learning, many things have to happen
        
        # We need to
        # 1. Keep track of previously seen experience
        # 2. Retrieve a batch of experiences if possible
        # 3. Update our QNetwork with gradient ascent
        # 4. Update our target QNetwork weights (hard or soft update)

        # Since this is too big, we recommend splitting this into two functions

        ########## END CODING HERE ##############

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        local_path = os.path.join(path, "local.pth")
        target_path = os.path.join(path, "target.pth")

        torch.save(self.local_nn.state_dict(), local_path)
        torch.save(self.target_nn.state_dict(), target_path)

    def load(self, path):
        local_path = os.path.join(path, "local.pth")
        target_path = os.path.join(path, "target.pth")

        self.local_nn.load_state_dict(torch.load(local_path))
        self.target_nn.load_state_dict(torch.load(target_path))

    def display_policy(self):
        pass


class Network(nn.Module):
    def __init__(self, input_size, output_size):
        super(Network, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, output_size)
        )

    def forward(self, x):
        return self.network(x)
