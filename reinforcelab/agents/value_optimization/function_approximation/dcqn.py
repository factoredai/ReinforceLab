from gymnasium import Env
from torch import nn

from reinforcelab.agents.agent import Agent
from reinforcelab.brains import QNetwork
from reinforcelab.estimators import MaxQEstimator
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

    def __init__(self, env: Env, model: nn.Module, learning_rate=0.01, discount_factor: float = 0.999, alpha=0.03, batch_size=128, update_every=4, max_buffer_size=2**12):
        action_selector = EpsilonGreedy(env)
        icm = IntrinsicCuriosityModule(
            env, 4, learning_rate=0.0001, state_transform_hidden_layers=[4, 4])
        buffer = ExperienceReplay(
            {"batch_size": batch_size, "max_size": max_buffer_size, "transform": icm})
        estimator = MaxQEstimator(env, discount_factor)
        brain = QNetwork(model, estimator, learning_rate, alpha)

        super().__init__(brain,
                         action_selector, buffer, update_every=update_every)
