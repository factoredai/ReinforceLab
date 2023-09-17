from datetime import datetime
from tqdm import tqdm
from reinforcelab.utils.leaderboard_utils import load_leaderboard, add_score, save_leaderboard, update_readme
import dill
import gymnasium as gym
import numpy as np
import os
import pandas as pd
import sys
import torch

def test(env, agent, num_episodes=100):
    """
    This test function should be specific for the environment.
    This test will be done using GitHub Actions and the score
    will be stored in a leaderboard.
    """
    
    rng = np.random.RandomState()
    rng.seed(0) # Set random seed
    cum_reward = 0
    for _ in tqdm(range(num_episodes)):
        state, info = env.reset(seed=rng.randint(10**6))
        ep_cum_reward = 0
        while True:
            action = agent.act(state)
            next_state, reward, done, *_ = env.step(action)
            ep_cum_reward += reward
            state = next_state

            if done:
                break

        cum_reward += ep_cum_reward
    avg_reward = cum_reward / num_episodes
    print(
        f"The agent obtained an average reward of {avg_reward} over {num_episodes}")
    return avg_reward

if __name__ == "__main__":

    # Load env
    env = gym.make('MountainCar-v0', render_mode="rgb_array")

    # Load agent
    agent_path = "/".join((sys.argv[0]).split('/')[:-1])
    with open(f'{agent_path}/optimized_agent.dill', 'rb') as file:
        agent = dill.load(file) 

    # Test agent
    avg_reward = test(env, agent)

    # Update Leaderboard
    leaderboard = load_leaderboard()
    leaderboard = add_score(leaderboard, user=sys.argv[1], score=avg_reward, num_epochs=agent.num_epochs)
    save_leaderboard(leaderboard)
    update_readme(leaderboard.head(10))
