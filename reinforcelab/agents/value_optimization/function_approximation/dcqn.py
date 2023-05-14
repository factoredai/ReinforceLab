from gymnasium import Env
from copy import deepcopy

from reinforcelab.agents.agent import Agent
from reinforcelab.brains import QNetwork
from reinforcelab.update_estimators import MaxQEstimator
from reinforcelab.transforms.experience import IntrinsicCuriosityModule
from reinforcelab.action_selectors import EpsilonGreedy
from reinforcelab.memory_buffers import ExperienceReplay
from reinforcelab.utils import get_state_action_sizes


class DCQN(Agent):
    """ Deep Q Network implementation. A DQN uses the same Max Q estimation procedure, but using a Deep Neural Network
    as a brain instead of a QTable. Because of this, two brains are required for learning stability. Additionally, due
    to the sequential nature of RL, experience must be stored and retrieved in random order, to unbias the learning
    procedure.
    """

    def __init__(self, env: Env, hidden_layers=[], learning_rate=0.01, discount_factor: float = 0.999, alpha=0.03, batch_size=128, update_every=4):
        state_size, action_size = get_state_action_sizes(env)

        local_brain = QNetwork(state_size, action_size, hidden_layers=hidden_layers,
                               learning_rate=learning_rate, alpha=alpha)
        target_brain = QNetwork(state_size, action_size, hidden_layers=hidden_layers,
                                learning_rate=learning_rate, alpha=alpha)
        action_selector = EpsilonGreedy(action_size)
        icm = IntrinsicCuriosityModule(
            state_size, action_size, 4, learning_rate=0.0001, state_transform_hidden_layers=[4, 4])
        buffer = ExperienceReplay(
            {"batch_size": batch_size, "max_size": 2**12, "transform": icm})
        estimator = MaxQEstimator(
            local_brain, target_brain, discount_factor)

        super().__init__(local_brain, target_brain, estimator,
                         action_selector, buffer, update_every=update_every)
