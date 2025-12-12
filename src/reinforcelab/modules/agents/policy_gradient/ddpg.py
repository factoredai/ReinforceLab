from gymnasium import Env
from torch import nn


from reinforcelab.modules.agents import Agent
from reinforcelab.modules.brains import ActorCritic
from reinforcelab.modules.action_selectors.continuous import ContinuousEpsilonGreedy
from reinforcelab.modules.estimators import SARSEstimator
from reinforcelab.modules.memory_buffers import ExperienceReplay

class DDPG(Agent):
    def __init__(self, env: Env, actor_model: nn.Module, critic_model: nn.Module, learning_rate=0.01, discount_factor=0.999, alpha=0.03, batch_size=128, update_every=4, max_buffer_size=2**12):
        action_selector = ContinuousEpsilonGreedy(env)
        estimator = SARSEstimator(env, discount_factor)
        brain = ActorCritic(actor_model, critic_model, estimator, learning_rate, alpha)
        buffer = ExperienceReplay({"batch_size": batch_size, "max_size": max_buffer_size})

        super().__init__(brain, action_selector, buffer, update_every)