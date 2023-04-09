import os
import dill
import torch
from torch import Tensor
from gymnasium.spaces.tuple import Tuple

from reinforcelab.agents.agent import Agent
from reinforcelab.brains import QTable
from reinforcelab.update_estimators import QEstimator
from reinforcelab.experience import Experience, BatchExperience
from reinforcelab.utils import get_state_action_sizes, epsilon_greedy


class QAgent(Agent):
    def __init__(self, env, gamma, alpha):
        self.env = env
        self.state_size, self.action_size = get_state_action_sizes(env)

        self.brain = QTable(self.state_size, self.action_size, alpha)
        # Since QTables don't use function approximation, there's no need for two brains.
        # We can reuse the q estimator by passing the same brain twice
        self.estimator = QEstimator(self.brain, self.brain, gamma)

    def act(self, state: Tensor, epsilon=0.0) -> Tensor:
        qvalues = self.brain(state)
        return epsilon_greedy(qvalues, epsilon=epsilon)

    def update(self, state, action, reward, next_state, done):
        experience = Experience(state, action, reward, next_state, done)
        # leverage on BatchExperience to reshape the experience in the expected dimensions
        batch = BatchExperience([experience])
        pred, target = self.estimator(batch)
        self.brain.update(batch, pred, target)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        filepath = os.path.join(path, "checkpoint.dill")
        with open(filepath, "wb") as f:
            dill.dump(self, f)

    def load(self, path):
        filepath = os.path.join(path, "checkpoint.dill")
        with open(filepath, "rb") as f:
            loaded_agent = dill.load(f)

        self.__dict__.clear()
        self.__dict__.update(loaded_agent.__dict__)
