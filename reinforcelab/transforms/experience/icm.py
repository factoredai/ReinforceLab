import torch
import torch.nn as nn
from typing import List

from .experience_transform import ExperienceTransform
from reinforcelab.utils import build_fcnn
from reinforcelab.experience import Experience


class IntrinsicCuriosityModule(ExperienceTransform):
    """Intrinsic Curiosity Module (ICM)

    Enhances exploration by providing an intrinsic reward to
    unfamiliar states. It does so by predicting the next state
    features given the current state and action. The error of this
    prediction is fed as additional reward to the agent, effectively
    motivating the agent to explore states that have not been trained on
    or observed enough. By predicting state features instead of the raw state,
    this module is robust to noisy states by only learning the features of the
    state that are relevant to the agents actions.

    https://arxiv.org/pdf/1705.05363.pdf
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        state_embedding_size: int,
        state_transform_hidden_layers: List[int] = [],
        inverse_dynamics_hidden_layers: List[int] = [],
        forward_dynamics_hidden_layers: List[int] = [],
        learning_rate: float = 0.001,
        update_every: int = 1,
        curiosity_amount: float = 0.5,
        beta: float = 0.5
    ):
        """Creates an ICM transform, which adds an intrinsic curiosity reward to
        a given set of experiences

        Args:
            state_size (int): The size of the environment state
            action_size (int): The number of actions an agent can take
            state_embedding_size (int): The size of the state embedding
            state_transform_hidden_layers (List[int], optional): List of layers to use for computing the state features. Defaults to [].
            inverse_dynamics_hidden_layers (List[int], optional): List of layers to use to compute the inverse dynamics. Defaults to [].
            forward_dynamics_hidden_layers (List[int], optional): List of layers to use to compute the forward dynamics. Defaults to [].
            learning_rate (float, optional): learning rate for updating all models. Defaults to 0.001.
            update_every (int, optional): Determines the training rate, or at how many calls it should train itself. Defaults to 1.
            curiosity_amount (float, optional): Scaling factor for how much curiosity reward to give . Defaults to 0.01.
            beta (float, optional): How much relevance to give to inverse vs forward error during training. Defaults to 0.5, or equal weight.
        """
        super(IntrinsicCuriosityModule, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.state_embedding_size = state_embedding_size
        self.state_transform_hidden_layers = state_transform_hidden_layers
        self.inverse_dynamics_hidden_layers = inverse_dynamics_hidden_layers
        self.forward_dynamics_hidden_layers = forward_dynamics_hidden_layers
        self.learning_rate = learning_rate
        self.update_every = update_every
        self.curiosity_amount = curiosity_amount
        self.beta = beta
        self.current_step = 0

        self.state_nn, self.inv_dyn_nn, self.fwd_dyn_nn = self.__build_models()

    def __build_models(self):
        state_layers_sizes = [self.state_size] + \
            self.state_transform_hidden_layers + [self.state_embedding_size]
        inv_dyn_layers_sizes = [self.state_embedding_size * 2] + \
            self.inverse_dynamics_hidden_layers + [self.action_size]
        fwd_dyn_layers_sizes = [self.state_embedding_size + self.action_size] + \
            self.forward_dynamics_hidden_layers + [self.state_embedding_size]

        state_nn = build_fcnn(state_layers_sizes)
        inv_dyn_nn = build_fcnn(inv_dyn_layers_sizes)
        fwd_dyn_nn = build_fcnn(fwd_dyn_layers_sizes)

        return state_nn, inv_dyn_nn, fwd_dyn_nn

    def __call__(self, experience: Experience) -> Experience:
        """Applies an intrinsic curiosity reward to the given experiences.
        Additionally, updates the internal networks every `update_every` steps.

        Args:
            experience (Experience): Experience to transform

        Returns:
            Experience: Transformed experience with intrinsic curiosity reward
        """
        states, actions, rewards, next_states, *extra = experience

        if self.current_step % self.update_every == 0:
            self.__train(experience)

        self.current_step += 1

        with torch.no_grad():
            states_emb = self.state_nn(states)
            next_states_emb = self.state_nn(next_states)
            oh_actions = nn.functional.one_hot(
                actions, num_classes=self.action_size)

            fwd_input = torch.cat([states_emb, oh_actions.squeeze()], dim=-1)
            pred_next_states_emb = self.fwd_dyn_nn(fwd_input)

            loss_fn = nn.MSELoss(reduce=False)
            loss_val = loss_fn(pred_next_states_emb,
                               next_states_emb).sum(dim=-1)

        rewards += self.curiosity_amount * loss_val.reshape(rewards.shape)

        experience.reward = rewards

        return experience

    def __train(self, experience: Experience):
        """Trains the internal models with the given experience tuple

        Args:
            experience (Experience): Experience tuple to train on
        """
        states, actions, _, next_states, *_ = experience
        inv_loss_fn = nn.CrossEntropyLoss()
        fwd_loss_fn = nn.MSELoss()
        params = list(self.state_nn.parameters()) + \
            list(self.inv_dyn_nn.parameters()) + \
            list(self.fwd_dyn_nn.parameters())
        optimizer = torch.optim.Adam(params, lr=self.learning_rate)

        states_emb = self.state_nn(states)
        next_states_emb = self.state_nn(next_states)

        inv_input = torch.cat([states_emb, next_states_emb], dim=-1)
        inv_logits = self.inv_dyn_nn(inv_input)
        pred_actions = torch.softmax(inv_logits, dim=-1)

        inv_loss = inv_loss_fn(pred_actions, actions.squeeze())

        oh_actions = nn.functional.one_hot(
            actions, num_classes=self.action_size)
        fwd_input = torch.cat([states_emb, oh_actions.squeeze()], dim=-1)
        pred_next_states_emb = self.fwd_dyn_nn(fwd_input)

        fwd_loss = fwd_loss_fn(pred_next_states_emb, next_states_emb)

        loss_val = (1 - self.beta) * inv_loss + self.beta * fwd_loss

        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
