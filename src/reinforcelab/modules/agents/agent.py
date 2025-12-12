import os
import dill
from torch import Tensor
from typing import Union
from abc import ABC, abstractmethod

from reinforcelab.brains import Brain
from reinforcelab.estimators import Estimator
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
    def __init__(self, brain: Brain, action_selector: ActionSelector, memory_buffer: MemoryBuffer, update_every=1):
        self.brain = brain
        self.action_selector = action_selector
        self.memory_buffer = memory_buffer
        self.update_every = update_every
        self.update_step = 0

    def act(self, state: Tensor, **kwargs) -> Tensor:
        qvalues = self.brain(state).detach()
        return self.action_selector(qvalues, **kwargs)

    def update(self, experience):
        self.memory_buffer.add(experience)
        if self.update_step % self.update_every == 0:
            try:
                batch = self.memory_buffer.sample()
            except RuntimeError:
                # If the batch can't be obtained, skip the update proc
                return
            self.brain.update(batch)
        self.update_step += 1

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        filepath = os.path.join(path, "checkpoint.dill")
        with open(filepath, "wb") as f:
            dill.dump(self.brain, f)

    def load(self, path):
        filepath = os.path.join(path, "checkpoint.dill")
        with open(filepath, "rb") as f:
            brain = dill.load(f)

        self.brain = brain
