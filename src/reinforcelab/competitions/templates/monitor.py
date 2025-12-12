import gymnasium as gym
import numpy as np


class ConvergenceMonitor(gym.Wrapper):
    """
    Wrapper that monitors training progress and detects convergence.
    
    Convergence is detected when the average reward over the last 
    `stability_window` episodes meets or exceeds `goal_reward`.
    
    When converged or max_steps reached, raises StopIteration to signal
    the agent to stop training.
    """
    
    def __init__(self, env, goal_reward, stability_window, max_steps):
        super().__init__(env)
        self.goal = goal_reward
        self.window = stability_window
        self.max_steps = max_steps
        
        self.episode_rewards = []
        self.current_ep_reward = 0.0
        self.total_steps = 0
        self.converged = False
        self.convergence_step = None  # Step at which convergence was detected

    def step(self, action):
        if self.total_steps >= self.max_steps:
            raise StopIteration("Max steps reached")
        
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.current_ep_reward += reward
        self.total_steps += 1
        
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        # Check for convergence when episode ends (before reset)
        if self.current_ep_reward != 0 or len(self.episode_rewards) > 0:
            self.episode_rewards.append(self.current_ep_reward)
            
            if len(self.episode_rewards) >= self.window:
                recent = self.episode_rewards[-self.window:]
                avg = np.mean(recent)
                if avg >= self.goal and not self.converged:
                    self.converged = True
                    self.convergence_step = self.total_steps
                    print(f"Converged at step {self.total_steps}! "
                          f"Avg reward: {avg:.2f} >= {self.goal}")
                    raise StopIteration("Convergence achieved")
        
        self.current_ep_reward = 0.0
        return self.env.reset(**kwargs)
    
    def get_stats(self):
        """Return training statistics."""
        return {
            "total_steps": self.total_steps,
            "episodes": len(self.episode_rewards),
            "converged": self.converged,
            "convergence_step": self.convergence_step,
            "recent_avg": np.mean(self.episode_rewards[-self.window:]) if self.episode_rewards else 0.0
        }
