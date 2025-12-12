import cv2
import torch
from tqdm import tqdm
from gymnasium import Env
from reinforcelab.agents.agent import Agent
from reinforcelab.experience import Experience


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
        len_history = []
        avg_reward = 0
        avg_len = 0

        for epoch in loop:
            state, info = env.reset()
            epoch_cum_reward = 0
            epoch_len = 0

            while True:
                if epoch % 200 == 0:
                    img = env.render()
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    cv2.imshow(f'Epoch {epoch}', img)
                    cv2.waitKey(20)
                else:
                    cv2.destroyAllWindows()
                # Generate a RL interaction
                actionable_state = torch.tensor(state).unsqueeze(0)
                action = agent.act(actionable_state, epsilon=epsilon)
                action = action.reshape((-1)).numpy()
                next_state, reward, done, truncated, info = env.step(action)
                experience = Experience(
                    state, action, reward, next_state, done)
                agent.update(experience)

                epoch_cum_reward += reward
                epoch_len += 1
                state = next_state

                # Update epsilon
                epsilon = max(min_epsilon, epsilon * (1 - epsilon_decay))

                if done or truncated:
                    break

                loop.set_description(
                    "Len: {:.2f} | Avg: {:.3f} | Best: {:.3f} | Eps: {:.3f} | R: {:.3f}".format(avg_len, avg_reward, best_avg_reward, epsilon, reward))

            # Show performance
            rewards_history.append(epoch_cum_reward)
            rewards_window = rewards_history[-100:]
            avg_reward = sum(rewards_window)/len(rewards_window)

            len_history.append(epoch_len)
            len_window = len_history[-100:]
            avg_len = sum(len_window)/len(len_window)

            # Save best model
            if avg_reward >= best_avg_reward:
                best_avg_reward = avg_reward
                agent.save(self.path)

        return rewards_history
