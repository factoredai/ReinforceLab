from tqdm import tqdm
import gymnasium as gym
from reinforcelab.agents.value_optimization import QLearningAgent


def train(env, agent, num_epochs=5000, epsilon=0.1, epsilon_decay=1e-6, min_epsilon=.01):
    loop = tqdm(range(num_epochs))
    best_cum_reward = float("-inf")
    rewards_history = []

    for _ in loop:
        state, info = env.reset()
        epoch_cum_reward = 0
        while True:
            # Generate a RL interaction
            action = agent.act(state, epsilon=epsilon)
            next_state, reward, done, truncated, info = env.step(action)
            agent.learn(state, action, reward, next_state, done)

            # Update epsilon
            epsilon = max(min_epsilon, epsilon * (1 - epsilon_decay))

            epoch_cum_reward += reward
            state = next_state

            if done or truncated:
                break

        # Save best model
        if epoch_cum_reward > best_cum_reward:
            best_cum_reward = epoch_cum_reward
            agent.save(f"{env.spec.id}-{agent.__class__.__name__}")

        # Show performance
        rewards_history.append(epoch_cum_reward)
        rewards_window = rewards_history[-100:]
        avg_reward = sum(rewards_window)/len(rewards_window)
        loop.set_description(
            f"Avg 100eps Reward: {round(avg_reward, 4)} | Epsilon: {round(epsilon, 3)}")
    return rewards_history


if __name__ == '__main__':
    env = gym.make('Blackjack-v1', sab=True)
    agent = QLearningAgent(env, gamma=0.99, alpha=0.01)

    train(env, agent, epsilon=1., epsilon_decay=1e-5,
          num_epochs=100000, min_epsilon=0.1)
    agent.display_policy()
