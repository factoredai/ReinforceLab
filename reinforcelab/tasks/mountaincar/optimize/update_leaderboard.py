from datetime import datetime
from tqdm import tqdm
import dill
import gymnasium as gym
import numpy as np
import os
import pandas as pd
import sys
import torch

def load_leaderboard(path='/leaderboard/leaderboard.csv'):

    path = "/".join((sys.argv[0]).split('/')[:-2]) + path

    if os.path.exists(path):
        print(pd.read_csv(path))
        print(path)
        return pd.read_csv(path)
    else:
        return pd.DataFrame(columns=['User', 'Score', 'Date'])

def add_score(leaderboard, user="test_user", score=0):
    
    new_score = pd.DataFrame.from_dict({'User': [user], 'Score': [score], 'Date': [datetime.utcnow()]})
    leaderboard = pd.concat([leaderboard, new_score], axis=0)

    # Sort leaderboard
    leaderboard = leaderboard.sort_values(by=['Score'], ascending=[0]).reset_index(drop=True)

    return leaderboard

def save_leaderboard(leaderboard, path='/leaderboard/leaderboard.csv'):
    path = "/".join((sys.argv[0]).split('/')[:-2]) + path
    directory = "/".join(path.split('/')[:-1])
    if not os.path.isdir(directory):
        os.mkdir(directory)

    leaderboard.to_csv(path, index=False)

def update_readme(leaderboard):
    path = "/".join((sys.argv[0]).split('/')[:-2])
    f = open(f"{path}/README.md", "w")
    f.write("# Leaderboard \n\n")
    f.write(f"Last Update (UTC): {datetime.utcnow()}\n\n")
    f.write(leaderboard.to_markdown())
    f.write("\n")
    f.close()

def test(env, agent, num_episodes=100):
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
    leaderboard = add_score(leaderboard, user=sys.argv[1], score=avg_reward)
    save_leaderboard(leaderboard)
    update_readme(leaderboard.head(10))
