import torch
from torch import nn
from torch import Tensor
from copy import deepcopy

from .brain import Brain
from reinforcelab.experience import Experience
from reinforcelab.estimators import Estimator
from reinforcelab.utils import soft_update

class ActorCritic(Brain):
    def __init__(self, actor_model: nn.Module, critic_model: nn.Module, estimator: Estimator, learning_rate: 0.01, alpha: 0.001):
        super(ActorCritic, self).__init__()
        self.local_actor_model = actor_model
        self.target_actor_model = deepcopy(actor_model)
        self.local_critic_model = critic_model
        self.target_critic_model = deepcopy(critic_model)
        self.estimator = estimator
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.actor_optimizer = torch.optim.Adam(self.local_actor_model.parameters(), lr=self.learning_rate, weight_decay=1.e-5)
        self.critic_optimizer = torch.optim.Adam(self.local_critic_model.parameters(), lr=self.learning_rate)

    def __call__(self, state):
        return self.local_actor_model(state)
    
    def target(self, state):
        return self.target_actor_model(state)

    def action_value(self, state: Tensor, action: Tensor, target: bool = False) -> Tensor:
        if target:
            return self.target_critic_model(state, action)
        return self.local_critic_model(state, action)
    
    def max_action(self, state: Tensor, target: bool = False, iters: int = 200, learning_rate: float = 0.1) -> Tensor:
        """Compute the action that maximizes the expected return by means
        of backpropagation to the action input of the critic

        Args:
            state (Tensor): state(s) to compute action from
            target (bool, optional): Wether to use the target networks. Defaults to False.
            iters (int, optional): How many backpropagation iterations to do. Defaults to 200
            learning (float, optional): Learning rate for backpropagation. Defaults to 0.1

        Returns:
            Tensor: maximal action
        """
        # Initialize the maximal action through the actor
        actor = self.local_actor_model
        critic = self.local_critic_model
        if target:
            actor = self.target_actor_model
            critic = self.target_critic_model

        with torch.no_grad():
            action = actor(state)

        max_action = torch.nn.Parameter(action, requires_grad=True)
        optim = torch.optim.SGD([max_action], lr=learning_rate)

        for _ in range(iters):
            loss = -critic(state, max_action)
            optim.zero_grad()
            loss.backward()
            optim.step()

        return max_action.data.detach()

    def update(self, experience: Experience):
        loss_fn = nn.MSELoss()

        critic_target = self.estimator(experience, self)

        states, actions, *_ = experience
        critic_pred = self.local_critic_model(states, actions)
        critic_loss = loss_fn(critic_pred, critic_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update the actor critic by maximizing the value estimate from the critic
        pred_actions = self.local(states)
        actor_loss = -self.target_critic_model(states, pred_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        soft_update(self.local_actor_model, self.target_actor_model, self.alpha)
        soft_update(self.local_critic_model, self.target_critic_model, self.alpha)