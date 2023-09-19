import cv2
import torch
from tqdm import tqdm
from gymnasium import Env
from reinforcelab.agents.agent import Agent


class Test:
    def __init__(self, env: Env, agent: Agent, render_every=100):
        self.env = env
        self.agent = agent
        self.render_every = render_every

    def run(self, num_epochs: int = 100):
        env = self.env
        agent = self.agent
        loop = tqdm(range(num_epochs))
        rewards_history = []

        for epoch in loop:
            state, info = env.reset()
            epoch_cum_reward = 0
            while True:
                if epoch % self.render_every == 0:
                    img = env.render()
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    cv2.imshow(f'Epoch {epoch}', img)
                    cv2.waitKey(20)
                else:
                    cv2.destroyAllWindows()
                actionable_state = torch.tensor(state).unsqueeze(0)
                action = agent.act(actionable_state, epsilon=0.0)
                action = action.reshape((-1)).numpy()
                next_state, reward, done, truncated, info = env.step(
                    action)

                epoch_cum_reward += reward
                state = next_state

                if done or truncated:
                    break

            rewards_history.append(epoch_cum_reward)

        return rewards_history
