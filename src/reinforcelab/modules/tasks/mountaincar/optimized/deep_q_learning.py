import os
from copy import deepcopy
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from reinforcelab.agents.agent import Agent
from memory_buffer import MemoryBuffer


class DeepQLearningAgent(Agent):
    def __init__(self, env, gamma, alpha, learning_rate=0.0003, batch_size=128, update_every=10):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.update_every = update_every
        self.step = 0
        self.num_epochs = 0

        self.rng = np.random.RandomState()
        self.rng.seed(0) # Set random seed
        torch.manual_seed(0)

        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.memory = MemoryBuffer(20000)
        self.local_nn = Network(self.state_size, self.action_size)
        self.target_nn = deepcopy(self.local_nn)

    def act(self, state, epsilon=0.0):
        state = torch.tensor(state).float().unsqueeze(0)
        qvalues = self.local_nn(state).detach().numpy()
        action = np.argmax(qvalues)

        # Randomly choose an action with p=epsilon
        if self.rng.rand() < epsilon:
            action = self.rng.choice(self.action_size)
        return action

    def update(self, state, action, reward, next_state, done):
        # Add the latest experience to the experience replay
        self.memory.add(state, action, reward, next_state, done)

        # Determine if we should update the value estimate
        exp_len = len(self.memory)
        enough_exp = exp_len >= self.batch_size
        should_update = self.step % self.update_every == 0

        if enough_exp and should_update:
            self.update_step()

        self.step += 1

    def update_step(self):
        optimizer = torch.optim.Adam(
            self.local_nn.parameters(), lr=self.learning_rate)
        loss = nn.MSELoss()

        # Get a batch of experience
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = batch

        # Compute loss with the td error
        with torch.no_grad():
            # QNetwork update. Use the max value
            max_vals = self.target_nn(next_states).max(dim=-1).values
            target = rewards + self.gamma * max_vals * (1 - dones)
        # Get the predicted value of the actions taken
        predicted_values = self.local_nn(states).gather(1, actions).squeeze()

        loss_val = loss(predicted_values, target)
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        self.__soft_update()

    def __soft_update(self):
        alpha = self.alpha
        for target_param, local_param in zip(self.target_nn.parameters(), self.local_nn.parameters()):
            target_param.data.copy_(
                alpha * local_param.data + (1.0-alpha) * target_param.data)

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
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )

    def forward(self, x):
        return self.network(x)
