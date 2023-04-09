import torch
from tqdm import tqdm
from gymnasium import Env
from reinforcelab.agents.agent import Agent


class Train:
    def __init__(self, env: Env, agent: Agent, checkpoint_path: str):
        self.env = env
        self.agent = agent
        self.path = checkpoint_path

    def run(self, num_epochs: int = 5000, epsilon: float = 0.1, epsilon_decay: float = 1e-6, min_epsilon: float = .01):
        env = self.env
        agent = self.agent
        loop = tqdm(range(num_epochs))
        best_avg_reward = float("-inf")
        rewards_history = []
        avg_reward = 0

        for _ in loop:
            state, info = env.reset()
            epoch_cum_reward = 0
            while True:
                # Generate a RL interaction
                actionable_state = torch.tensor(state).unsqueeze(0)
                action = agent.act(actionable_state, epsilon=epsilon)
                next_state, reward, done, truncated, info = env.step(
                    action.item())
                agent.update(state, action, reward, next_state, done)

                epoch_cum_reward += reward
                state = next_state

                # Update epsilon
                epsilon = max(min_epsilon, epsilon * (1 - epsilon_decay))

                if done or truncated:
                    break

                loop.set_description(
                    f"Avg 100eps Reward: {round(avg_reward, 4)} | Epsilon: {round(epsilon, 3)}")

            # Show performance
            rewards_history.append(epoch_cum_reward)
            rewards_window = rewards_history[-100:]
            avg_reward = sum(rewards_window)/len(rewards_window)

            # Save best model
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                agent.save(self.path)

        return rewards_history
