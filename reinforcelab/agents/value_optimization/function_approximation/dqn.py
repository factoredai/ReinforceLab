from gymnasium import Env
from copy import deepcopy
from torch import nn

from reinforcelab.agents.agent import Agent
from reinforcelab.brains import QNetwork
from reinforcelab.estimators import MaxQEstimator
from reinforcelab.action_selectors import EpsilonGreedy
from reinforcelab.memory_buffers import ExperienceReplay


class DQN(Agent):
    """ Deep Q Network implementation. A DQN uses the same Max Q estimation procedure, but using a Deep Neural Network
    as a brain instead of a QTable. Because of this, two brains are required for learning stability. Additionally, due
    to the sequential nature of RL, experience must be stored and retrieved in random order, to unbias the learning
    procedure.
    """

    def __init__(self, env: Env, model: nn.Module, learning_rate=0.01, discount_factor: float = 0.999, alpha=0.03, batch_size=128, update_every=4, max_buffer_size=2**12):
        action_selector = EpsilonGreedy(env)
        estimator = MaxQEstimator(env, discount_factor)
        brain = QNetwork(model, estimator, learning_rate=learning_rate, alpha=alpha)
        buffer = ExperienceReplay(
            {"batch_size": batch_size, "max_size": max_buffer_size})

        super().__init__(brain,
                         action_selector, buffer, update_every=update_every)
