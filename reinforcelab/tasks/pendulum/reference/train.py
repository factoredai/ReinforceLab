import cv2
from tqdm import tqdm
import gymnasium as gym
from ddpg import DDPGAgent
import numpy as np
import torch


def train(env, agent, path, num_epochs=5000, epsilon=0.1, epsilon_decay=1e-5, min_epsilon=.01):
    loop = tqdm(range(num_epochs))
    best_avg_reward = float("-inf")
    rewards_history = []
    render_every = 500

    rng = np.random.RandomState()
    rng.seed(42) # Set random seed
    torch.manual_seed(42) # Set torch seed

    for epoch in loop:
        state, info = env.reset(seed=rng.randint(10**6))
        epoch_cum_reward = 0
        while True:
            if epoch % render_every == 0:
                img = env.render()
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imshow(f'MountainCar | Epoch {epoch}', img)
                cv2.waitKey(40)
            else:
                cv2.destroyAllWindows()
            # Generate a RL interaction
            action = agent.act(state, epsilon=epsilon)
            next_state, reward, done, truncated, info = env.step(action[0])
            next_state = next_state.squeeze()

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

        loop.set_description(
            f"Avg 100eps Reward: {round(avg_reward, 4)} | Epsilon: {round(epsilon, 3)}")
    return rewards_history


def test(env, agent, num_episodes=100):
    rng = np.random.RandomState()
    rng.seed(0) # Set random seed
    cum_reward = 0
    max_timesteps=200
    for _ in tqdm(range(num_episodes)):
        state, info = env.reset(seed=rng.randint(10**6))
        ep_cum_reward = 0
        for _ in range(max_timesteps):
            action = agent.act(state)
            next_state, reward, done, *_ = env.step(action[0])
            ep_cum_reward += reward
            state = next_state

            if done:
                break

        cum_reward += ep_cum_reward
    avg_reward = cum_reward / num_episodes
    print(
        f"The agent obtained an average reward of {avg_reward} over {num_episodes}")


if __name__ == "__main__":
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    agent = DDPGAgent(env, gamma=0.999, alpha=0.0003, actor_lr=0.001, critic_lr=0.001)
    path = f"{env.spec.id}-{agent.__class__.__name__}"

    train(env, agent, path, epsilon=1., num_epochs=10000, epsilon_decay=2e-6, min_epsilon=0.1)
    agent.load(path)
    env = gym.make("Pendulum-v1", render_mode="human")
    test(env, agent)
    agent.display_policy()
