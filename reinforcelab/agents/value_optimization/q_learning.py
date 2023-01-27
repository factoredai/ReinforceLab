import os
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from gymnasium.spaces import Space
from gymnasium.spaces.tuple import Tuple

from reinforcelab.agents.agent import Agent


class QLearningAgent(Agent):
    def __init__(self, env, gamma, alpha):
        self.env = env

        if isinstance(env.observation_space, Tuple):
            self.state_size = [space.n for space in env.observation_space]
        else:
            self.state_size = [env.observation_space.n]

        self.action_size = env.action_space.n

        dims = self.state_size + [self.action_size]
        self.qtable = np.zeros(dims)
        self.gamma = gamma
        self.alpha = alpha

    def __state2idx(self, state):
        if len(self.state_size) > 1:
            idx = tuple([int(val) for val in state])
        else:
            idx = tuple([state])
        return idx

    def act(self, state, epsilon=0.0):
        idx = self.__state2idx(state)
        qvalues = self.qtable[idx]
        action = np.argmax(qvalues)

        # Randomly choose an action with p=epsilon
        if np.random.random() < epsilon:
            action = np.random.choice(self.action_size)
        return action

    def update(self, state, action, reward, next_state, done):
        idx = self.__state2idx(state)
        next_idx = self.__state2idx(next_state)
        qvalue = self.qtable[idx][action]
        next_qvalue = 0
        if not done:
            next_qvalue = np.max(self.qtable[next_idx])

        # Compute td error
        td_error = reward + self.gamma * next_qvalue - qvalue

        # Update Q table
        new_val = qvalue + self.alpha * td_error
        self.qtable[idx][action] = new_val

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        filepath = os.path.join(path, "checkpoint.npy")
        np.save(filepath, self.qtable)

    def load(self, path):
        filepath = os.path.join(path, "checkpoint.npy")
        self.qtable = np.load(filepath)
