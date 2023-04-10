from gymnasium import Env

from reinforcelab.agents.agent import Agent
from reinforcelab.brains import QTable
from reinforcelab.update_estimators import QEstimator
from reinforcelab.action_selectors import EpsilonGreedy
from reinforcelab.memory_buffers import ExperienceReplay
from reinforcelab.utils import get_state_action_sizes


class QLearningAgent(Agent):
    """A Q Learning Agent uses a Q Table to store the values
    of each action for each state. It updates the value estimates
    by assuming that the agents will act optimistically after the first
    action (This is implemented by the QEstimator). To enable exploration,
    the agent uses an epsilon-greedy policy. Q Learning agents don't need
    experience replay, so they learn from the immediate experience.
    """

    def __init__(self, env: Env, discount_factor: float = 0.999):
        state_size, action_size = get_state_action_sizes(env)
        brain = QTable(state_size, action_size)
        estimator = QEstimator(brain, brain, discount_factor)
        action_selector = EpsilonGreedy(action_size)
        buffer = ExperienceReplay({"batch_size": 1, "max_size": 1})
        super().__init__(brain, brain, estimator, action_selector, buffer)
