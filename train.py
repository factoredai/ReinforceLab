import gymnasium as gym

from reinforcelab.procedures.train import Train
from reinforcelab.agents.value_optimization.tabular import QLearning

env = gym.make('Blackjack-v1', sab=True)
agent = QLearning(env, 0.99, 0.001)

train = Train(env, agent, "blackjack_qlearning")
train.run()