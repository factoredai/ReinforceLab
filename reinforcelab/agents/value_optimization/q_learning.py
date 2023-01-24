import os
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt


class QLearningAgent:
    def __init__(self, env, gamma, alpha):
        self.env = env

        self.state_size = [space.n for space in env.observation_space]
        self.action_size = env.action_space.n
        dims = self.state_size + [self.action_size]
        self.qtable = np.zeros(dims)
        self.gamma = gamma
        self.alpha = alpha

    def __state2idx(self, state):
        idx = tuple([int(val) for val in state])
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
        filepath = os.path.join(path, "checkpoint.npy")
        np.save(filepath, self.qtable)

    def load(self, path):
        filepath = os.path.join(path, "checkpoint.npy")
        self.qtable = np.load(filepath)

    def display_policy(self):
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        X = np.arange(1, 11)
        Y = np.arange(12, 23)
        X, Y = np.meshgrid(X, Y)
        Z = self.qtable[11:22, 1:, 0].max(axis=-1)

        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
        plt.show()
