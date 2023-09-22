import gymnasium as gym
import torch
from torch import nn
from copy import deepcopy

from reinforcelab.procedures.train import Train
from reinforcelab.agents import DDPG

env = gym.make('Pusher-v4', render_mode="rgb_array")

class Actor(nn.Module):
    def __init__(self, input_size, output_size):
        super(Actor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, state):
        x = self.model(state.float())
        return 2 * torch.tanh(x)

class Critic(nn.Module):
    def __init__(self, input_size):
        super(Critic, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, state, action):
        x = torch.concat([state, action], dim=-1)
        return self.model(x.float())

state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]
actor_model = Actor(state_size, action_size)
critic_model = Critic(state_size + action_size)
agent = DDPG(env, actor_model, critic_model, 0.003, alpha=0.01, update_every=16, batch_size=128, max_buffer_size=2**14)

train = Train(env, agent, "Pusher-DDPG")
train.run(epsilon=1.0, min_epsilon=0.1, epsilon_decay=1e-5, num_epochs=20000)