import torch
from torch import Tensor
import gymnasium as gym

from reinforcelab.modules.experience import Experience
from reinforcelab.modules.brains import Brain
from reinforcelab.modules.estimators import Estimator


class DoubleQEstimator(Estimator):
    def __init__(self, env: gym.Env, gamma: float):
        self.gamma = gamma

    def __call__(self, experience: Experience, brain: Brain) -> Tensor:
        """Computes the action-value estimation for an experience tuple with the given local and
        target networks. It computes the value estimation directly from the local network, as well
        as the bellman equation value estimation with the target network.

        Args:
            experience (Experience): An experience instance, containing tensors for states, actions, 
            rewards, next_states and dones

        Returns:
            Tensor: Double Q Value estimation for the given experience and policy
        """

        _, _, rewards, next_states, dones, *_ = experience

        with torch.no_grad():
            # Implement DQN
            max_actions = brain.max_action(next_states).unsqueeze(1)
            max_vals = brain.action_value(next_states, max_actions, target=True).squeeze()
            target = rewards + self.gamma * max_vals * (1-dones)

        return target
