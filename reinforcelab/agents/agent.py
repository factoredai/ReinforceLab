import os
import dill
from torch import Tensor
from typing import Union
from abc import ABC, abstractmethod

from reinforcelab.brains import Brain
from reinforcelab.update_estimators import UpdateEstimator
from reinforcelab.memory_buffers import MemoryBuffer
from reinforcelab.action_selectors import ActionSelector
from reinforcelab.experience import Experience


class BaseAgent(ABC):
    @abstractmethod
    def act(self, state: Tensor, epsilon: float = 0.0):
        """Choose an action given a state

        Args:
            state (Any): A representation of the state
            epsilon (float, optional): Probability of taking an exploratory action. Defaults to 0.0.
        """

    @abstractmethod
    def update(self, experience: Experience):
        """Update the policy given the current experience

        Args:
            experience (Experience): An experience instance
        """

    @abstractmethod
    def save(self, path: str):
        """Store the current agent in memory

        Args:
            path (str): Where to store the agent
        """

    @abstractmethod
    def load(self, path: str):
        """Load an agent from memory

        Args:
            path (str): Where the agent is stored
        """


class Agent(BaseAgent):
    def __init__(self, local_brain: Brain, target_brain: Brain, update_estimator: UpdateEstimator, action_selector: ActionSelector, memory_buffer: MemoryBuffer, update_every=1):
        self.local_brain = local_brain
        self.target_brain = target_brain
        self.estimator = update_estimator
        self.action_selector = action_selector
        self.memory_buffer = memory_buffer
        self.update_every = update_every
        self.update_step = 0

    def act(self, state: Tensor, **kwargs) -> Tensor:
        qvalues = self.local_brain(state)
        return self.action_selector(qvalues, **kwargs)

    def update(self, experience):
        self.memory_buffer.add(experience)
        if self.update_step % self.update_every == 0:
            try:
                batch = self.memory_buffer.sample()
            except RuntimeError:
                # If the batch can't be obtained, skip the update proc
                return
            pred, target = self.estimator(batch)
            self.local_brain.update(batch, pred, target)
            self.target_brain.update_from(self.local_brain)
        self.update_step += 1

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        filepath = os.path.join(path, "checkpoint.dill")
        brains = [self.local_brain, self.target_brain]
        with open(filepath, "wb") as f:
            dill.dump(brains, f)

    def load(self, path):
        filepath = os.path.join(path, "checkpoint.dill")
        with open(filepath, "rb") as f:
            loaded_brains = dill.load(f)

        local_brain, target_brain = loaded_brains

        self.local_brain = local_brain
        self.target_brain = target_brain
