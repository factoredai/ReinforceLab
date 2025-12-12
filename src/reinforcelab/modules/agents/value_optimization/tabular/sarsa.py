from gymnasium import Env

from reinforcelab.agents.agent import Agent
from reinforcelab.brains import QTable
from reinforcelab.estimators import SARSAEstimator
from reinforcelab.action_selectors import EpsilonGreedy
from reinforcelab.memory_buffers import OrderedBuffer
from reinforcelab.utils import get_state_action_sizes


class SARSA(Agent):
    """A SARSA Learning Agent uses a Q Table to store the values
    of each action for each state. It updates the value estimates
    by gather the (S)tate, (A)ction, (R)eward, Next (S)tate and Next (A)ction. 
    To enable exploration, the agent uses an epsilon-greedy policy. 
    Q Learning agents don't need experience replay, so they learn from
    the immediate experience.
    """

    def __init__(self, env: Env, discount_factor: float = 0.999, alpha=0.01):
        action_selector = EpsilonGreedy(env)
        estimator = SARSAEstimator(env, discount_factor)
        brain = QTable(env, estimator, alpha=alpha)
        buffer = OrderedBuffer({"batch_size": 2, "max_size": 2})

        super().__init__(brain, action_selector, buffer)
