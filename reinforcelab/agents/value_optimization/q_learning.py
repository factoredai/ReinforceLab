import os
import pickle
import numpy as np
import torch
from torch import Tensor
from gymnasium.spaces.tuple import Tuple

from reinforcelab.agents.agent import Agent
from reinforcelab.brains import QTable
from reinforcelab.update_estimators import VanillaEstimator
from reinforcelab.experience import Experience, BatchExperience


class QLearningAgent(Agent):
    def __init__(self, env, gamma, alpha):
        self.env = env

        if isinstance(env.observation_space, Tuple):
            self.state_size = len(env.observation_space)
        else:
            self.state_size = env.observation_space.n

        self.action_size = env.action_space.n

        self.brain = QTable(self.state_size, self.action_size, alpha)
        # Since QTables don't use function approximation, there's no need for two brains.
        # We can reuse the vanilla estimator by passing the same brain twice
        self.estimator = VanillaEstimator(self.brain, self.brain, gamma)

    def act(self, state: Tensor, epsilon=0.0):
        if not isinstance(state, Tensor):
            state = torch.tensor(state).unsqueeze(0)
        qvalues = self.brain(state)
        action = torch.argmax(qvalues, dim=0)

        # Randomly choose an action with p=epsilon
        if np.random.random() < epsilon:
            action = torch.tensor(np.random.choice(self.action_size))
        return action.item()

    def update(self, state, action, reward, next_state, done):
        experience = Experience(state, action, reward, next_state, done)
        # leverage on BatchExperience to reshape the experience in the expected dimensions
        batch = BatchExperience([experience])
        pred, target = self.estimator(batch)
        self.brain.update(batch, pred, target)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        filepath = os.path.join(path, "checkpoint.pkl")
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    def load(self, path):
        filepath = os.path.join(path, "checkpoint.npy")
        with open(filepath, "rb") as f:
            loaded_agent = pickle.load(f)

        self.__dict__.clear()
        self.__dict__.update(loaded_agent.__dict__)
