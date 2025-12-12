import os
from copy import deepcopy
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F

from reinforcelab.agents.agent import Agent
from memory_buffer import MemoryBuffer


class DDPGAgent(Agent):
    def __init__(self, env, gamma, alpha, critic_lr=0.0003, actor_lr=0.0003, batch_size=128, update_every=10):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.critic_lr = critic_lr
        self.actor_lr = actor_lr
        self.batch_size = batch_size
        self.update_every = update_every
        self.step = 0

        self.rng = np.random.RandomState()
        self.rng.seed(0) # Set random seed
        torch.manual_seed(0)

        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.shape[0]
        self.memory = MemoryBuffer(2**14)
        self.local_actor = ActorNetwork(self.state_size, self.action_size)
        self.target_actor = deepcopy(self.local_actor)
        self.local_critic = CriticNetwork(self.state_size + self.action_size)
        self.target_critic = deepcopy(self.local_critic)

    def act(self, state, epsilon=0.0):
        state = torch.tensor(state).float().unsqueeze(0)
        action = self.local_actor(state)

        noise = 2 * torch.randn_like(action)
        if np.random.random() < epsilon:
            return noise.detach().numpy()
        return action.detach().numpy()

    def update(self, state, action, reward, next_state, done):
        # Add the latest experience to the experience replay
        self.memory.add(state, action, reward, next_state, done)

        # Determine if we should update the value estimate
        exp_len = len(self.memory)
        enough_exp = exp_len >= self.batch_size
        should_update = self.step % self.update_every == 0

        if enough_exp and should_update:
            self.update_step()

        self.step += 1

    def update_step(self):
        loss_fn = nn.MSELoss()
        critic_optimizer = optim.Adam(self.local_critic.parameters(), lr=self.critic_lr, weight_decay=1.e-5)
        actor_optimizer = optim.Adam(self.local_actor.parameters(), lr=self.actor_lr)

        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = batch

        # Update the critic with the TD estimate
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            q_next = self.target_critic(next_states, next_actions)
        value_estimate = rewards.unsqueeze(-1) + (1 - dones) * self.gamma * q_next
        value_pred = self.local_critic(states, actions)
        critic_loss = loss_fn(value_pred, value_estimate)

        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        # Update the actor critic by maximizing the value estimate from the critic
        pred_actions = self.local_actor(states)
        actor_loss = -self.target_critic(states, pred_actions).mean()

        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        # Soft-update target models
        self.__soft_update()

    def __soft_update(self):
        alpha = self.alpha

        for target_param, local_param in zip(self.target_actor.parameters(), self.local_actor.parameters()):
            target_param.data.copy_(
                alpha * local_param.data + (1.0-alpha) * target_param.data)
            
        for target_param, local_param in zip(self.target_critic.parameters(), self.local_critic.parameters()):
            target_param.data.copy_(
                alpha * local_param.data + (1.0-alpha) * target_param.data)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        local_actor_path = os.path.join(path, "local_actor.pth")
        target_actor_path = os.path.join(path, "target_actor.pth")
        local_critic_path = os.path.join(path, "local_critic.pth")
        target_critic_path = os.path.join(path, "target_critic.pth")

        torch.save(self.local_actor.state_dict(), local_actor_path)
        torch.save(self.target_actor.state_dict(), target_actor_path)
        torch.save(self.local_critic.state_dict(), local_critic_path)
        torch.save(self.target_critic.state_dict(), target_critic_path)

    def load(self, path):
        local_actor_path = os.path.join(path, "local_actor.pth")
        target_actor_path = os.path.join(path, "target_actor.pth")
        local_critic_path = os.path.join(path, "local_critic.pth")
        target_critic_path = os.path.join(path, "target_critic.pth")

        self.local_actor.load_state_dict(torch.load(local_actor_path))
        self.target_actor.load_state_dict(torch.load(target_actor_path))
        self.local_critic.load_state_dict(torch.load(local_critic_path))
        self.target_critic.load_state_dict(torch.load(target_critic_path))

    def display_policy(self):
        pass



class ActorNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(ActorNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )

    def forward(self, x):
        # Pendulum restricts action to a range of [-2, 2]
        return 2 * torch.sigmoid(self.network(x))

class CriticNetwork(nn.Module):
    def __init__(self, input_size):
        super(CriticNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, state, actions):
        x = torch.concat([state, actions], dim=-1)
        return self.network(x)