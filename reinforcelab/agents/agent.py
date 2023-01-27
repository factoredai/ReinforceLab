from typing import Any
from abc import ABC, abstractmethod


class Agent(ABC):
    @abstractmethod
    def act(self, state: Any, epsilon: float = 0.0):
        """Choose an action given a state

        Args:
            state (Any): A representation of the state
            epsilon (float, optional): Probability of taking an exploratory action. Defaults to 0.0.
        """

    @abstractmethod
    def update(self, state: Any, action: Any, reward: float, next_state: Any, done: bool):
        """Update the policy given the current experience

        Args:
            state (Any): A representation of the current state
            action (Any): Action taken at the current state
            reward (float): Reward obtained from taking action at the current state
            next_state (Any): State reached after acting on the current state
            done (bool): Wether the episode is over at the next state
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
