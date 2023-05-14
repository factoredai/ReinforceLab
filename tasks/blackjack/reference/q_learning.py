import os
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import seaborn as sns

from reinforcelab.agents.agent import Agent


class QLearningAgent(Agent):
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
        os.makedirs(path, exist_ok=True)
        filepath = os.path.join(path, "checkpoint.npy")
        np.save(filepath, self.qtable)

    def load(self, path):
        filepath = os.path.join(path, "checkpoint.npy")
        self.qtable = np.load(filepath)

    def display_policy(self):
        fig = plt.figure(figsize=(9, 9))
        fig.suptitle("Agent Policy")
        ax1 = fig.add_subplot(2, 2, 1, projection="3d")
        self.__display_value(ax1, usable_ace=False)
        fig.add_subplot(2, 2, 2)
        self.__display_policy(usable_ace=False)
        ax3 = fig.add_subplot(2, 2, 3, projection="3d")
        self.__display_value(ax3, usable_ace=True)
        fig.add_subplot(2, 2, 4)
        self.__display_policy(usable_ace=True)
        plt.show()

    def __display_value(self, ax, usable_ace):
        X = np.arange(2, 12)
        Y = np.arange(13, 23)
        X, Y = np.meshgrid(X, Y)
        Z = self.qtable[11:21, 1:, int(usable_ace)].max(axis=-1).T

        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                               cmap="viridis", edgecolor="none")
        plt.xticks(range(2, 12), range(12, 22))
        plt.yticks(range(13, 23), ["A"] + list(range(2, 11)))
        ax.set_title(
            f"State values: {'No' if not usable_ace else ''} Usable Ace")
        ax.set_xlabel("Player sum")
        ax.set_ylabel("Dealer showing")
        ax.zaxis.set_rotate_label(False)
        ax.set_zlabel("Value", fontsize=14, rotation=90)

    def __display_policy(self, usable_ace):
        policy_grid = self.qtable[11:21, 1:, int(usable_ace)].argmax(axis=-1).T
        ax = sns.heatmap(policy_grid, linewidth=0,
                         annot=True, cmap="Accent_r", cbar=False)
        ax.set_title(f"Policy: {'No' if not usable_ace else ''} Usable Ace")
        ax.set_xlabel("Player sum")
        ax.set_ylabel("Dealer showing")
        ax.set_xticklabels(range(12, 22))
        ax.set_yticklabels(["A"] + list(range(2, 11)), fontsize=12)
