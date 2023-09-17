import gymnasium as gym
import torch
from torch import nn

from reinforcelab.procedures.train import Train
from reinforcelab.agents import DQN

env = gym.make('LunarLander-v2')
model = nn.Sequential(
    nn.Linear(8, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 4)
)
agent = DQN(env, model, 0.0001, alpha=0.001, update_every=8)

train = Train(env, agent, "LunarLander-DQN")
train.run(epsilon=1.0, min_epsilon=0.1, epsilon_decay=1e-5)