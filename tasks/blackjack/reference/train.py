from tqdm import tqdm
import gymnasium as gym
from q_learning import QLearningAgent


def train(env, agent, path, num_epochs=5000, epsilon=0.1, epsilon_decay=1e-6, min_epsilon=.01, ):
    loop = tqdm(range(num_epochs))
    best_avg_reward = float("-inf")
    rewards_history = []

    for _ in loop:
        state, info = env.reset()
        epoch_cum_reward = 0
        while True:
            # Generate a RL interaction
            action = agent.act(state, epsilon=epsilon)
            next_state, reward, done, truncated, info = env.step(action)
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
    cum_reward = 0
    for _ in tqdm(range(num_episodes)):
        state, info = env.reset()
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
        f"The agent obtained an average reward of {avg_reward} over {num_episodes} episode(s)")


if __name__ == '__main__':
    env = gym.make('Blackjack-v1', sab=True)
    agent = QLearningAgent(env, gamma=0.999, alpha=0.001)
    path = f"{env.spec.id}-{agent.__class__.__name__}"

    train(env, agent, path, epsilon=1., epsilon_decay=1e-6,
          num_epochs=100000, min_epsilon=0.1)
    agent.load(path)
    test(env, agent)
    agent.display_policy()
