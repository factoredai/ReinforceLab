from gymnasium import Env

from reinforcelab.agents.agent import Agent
from reinforcelab.brains import QTable
from reinforcelab.update_estimators import QEstimator
from reinforcelab.action_selectors import EpsilonGreedy
from reinforcelab.memory_buffers import OnlineBuffer
from reinforcelab.utils import get_state_action_sizes


class QLearningAgent(Agent):
    def __init__(self, env: Env, discount_factor: float = 0.999):
        state_size, action_size = get_state_action_sizes(env)
        brain = QTable(state_size, action_size)
        estimator = QEstimator(brain, brain, discount_factor)
        action_selector = EpsilonGreedy(action_size)
        buffer = OnlineBuffer()
        super().__init__(brain, brain, estimator, action_selector, buffer)
