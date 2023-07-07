import cv2
from tqdm import tqdm
import gymnasium as gym
from deep_q_learning import DeepQLearningAgent
import numpy as np
import torch
import dill

def train(env, agent, path, num_epochs=5000, epsilon=0.1, epsilon_decay=1e-5, min_epsilon=0):
    loop = tqdm(range(num_epochs))
    best_avg_reward = float("-inf")
    rewards_history = []
    render_every = 1000

    rng = np.random.RandomState()
    rng.seed(42) # Set random seed
    torch.manual_seed(42) # Set torch seed

    for epoch in loop:
        state, info = env.reset(seed=rng.randint(10**6))
        epoch_cum_reward = 0
        while True:
            # Generate a RL interaction
            if epoch % render_every == 0:
                img = env.render()
                cv2.imshow(f'MountainCar | Epoch {epoch}', img)
                cv2.waitKey(40)
            else:
                cv2.destroyAllWindows()
            action = agent.act(state, epsilon=epsilon)
            next_state, reward, done, truncated, info = env.step(action)

            # Modify Reward
            reward += abs(next_state[1])*10

            agent.update(state, action, reward, next_state, done)

            # Update epsilon
            epsilon = max(min_epsilon, epsilon * (1 - epsilon_decay))

            epoch_cum_reward += reward
            state = next_state

            if done or truncated:
                break

        # Show performance
        rewards_history.append(epoch_cum_reward)
        rewards_window = rewards_history[-100:]
        avg_reward = sum(rewards_window)/len(rewards_window)

        # Save best model
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            agent.save(path)
            agent.num_epochs = epoch

        loop.set_description(
            f"Avg 100eps Reward: {round(avg_reward, 4)} | Epsilon: {round(epsilon, 3)}")
    return rewards_history


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


if __name__ == "__main__":
    env = gym.make('MountainCar-v0', render_mode="rgb_array")
    agent = DeepQLearningAgent(env, gamma=0.999, alpha=0.0001)
    path = f"{env.spec.id}-{agent.__class__.__name__}"

    # Train agent
    train(env, agent, path, epsilon=0.5, num_epochs=6000, epsilon_decay=8e-6)
    
    # Test and save agent
    agent.load(path)
    agent.env = None
    
    with open(f'./optimized_agent.dill', 'wb') as file:
        dill.dump(agent, file)

    test_env = gym.make('MountainCar-v0', render_mode="rgb_array")
    test(test_env, agent)
    agent.display_policy()
