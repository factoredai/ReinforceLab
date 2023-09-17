import torch
from torch import nn
from torch import Tensor
from copy import deepcopy

from .brain import Brain
from reinforcelab.experience import Experience
from reinforcelab.update_estimators import UpdateEstimator
from reinforcelab.utils import soft_update

class ActorCritic(Brain):
    def __init__(self, actor_model: nn.Module, critic_model: nn.Module, estimator: UpdateEstimator, learning_rate: 0.01, alpha: 0.001):
        super(ActorCritic, self).__init__()
        self.local_actor_model = actor_model
        self.target_actor_model = deepcopy(actor_model)
        self.local_critic_model = critic_model
        self.target_critic_model = deepcopy(critic_model)
        self.estimator = estimator
        self.learning_rate = learning_rate
        self.alpha = alpha

    def __call__(self, state):
        return self.local_actor_model(state)
    
    def target(self, state):
        return self.target_actor_model(state)

    def action_value(self, state: Tensor, action: Tensor, target: bool = False) -> Tensor:
        if target:
            return self.target_critic_model(state, action)
        return self.local_critic_model(state, action)
    
    def update(self, experience: Experience):
        loss_fn = nn.MSELoss()
        actor_optimizer = torch.optim.Adam(self.local_actor_model.parameters(), lr=self.learning_rate)
        critic_optimizer = torch.optim.Adam(self.local_critic_model.parameters(), lr=self.learning_rate)

        critic_target = self.estimator(experience, self)

        states, actions, *_ = experience
        critic_pred = self.local_critic_model(states, actions)
        critic_loss = loss_fn(critic_pred, critic_target)

        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        # Update the actor critic by maximizing the value estimate from the critic
        pred_actions = self.local(states)
        actor_loss = -self.target_critic_model(states, pred_actions).mean()

        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        soft_update(self.local_actor_model, self.target_actor_model, self.alpha)
        soft_update(self.local_critic_model, self.target_critic_model, self.alpha)