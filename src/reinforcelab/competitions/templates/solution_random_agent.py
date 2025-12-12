"""
Random Agent - Baseline Submission for ReinforceLab Competitions

This agent takes random actions from the environment's action space.
It serves as a baseline and demonstrates the expected interface.

Your submission should have an agent.py file with an Agent class that implements:
- __init__(env): Initialize with the environment
- act(observation): Return an action given an observation
- train(): Train the agent using self.env
- load(): Load your model (optional - you decide how)
- save(): Save your model (optional - you decide how)
"""
import gymnasium as gym


class Agent:
    """
    Random Agent - A baseline agent that takes random actions.
    
    This agent samples random actions from the environment's action space.
    It does not learn anything, making it a useful baseline for comparison.
    """
    
    def __init__(self, env: gym.Env):
        """Initialize the agent with the environment."""
        self.env = env

    def act(self, observation):
        """
        Return a random action.
        
        Args:
            observation: The current observation from the environment.
            
        Returns:
            A random action sampled from the action space.
        """
        return self.env.action_space.sample()

    def train(self):
        """
        Train the agent using self.env.
        
        For a random agent, we don't actually learn anything.
        We just run episodes until the monitor signals to stop.
        """
        try:
            while True:
                obs, _ = self.env.reset()
                done = False
                while not done:
                    action = self.act(obs)
                    obs, reward, terminated, truncated, info = self.env.step(action)
                    done = terminated or truncated
        except StopIteration:
            # Training complete - convergence reached or max steps hit
            pass
    
    def load(self):
        """
        Load model weights.
        
        Random agent has no weights to load.
        In your implementation, load your model here, e.g.:
            self.model = torch.load('model.pt')
        """
        pass
    
    def save(self):
        """
        Save model weights.
        
        Random agent has no weights to save.
        In your implementation, save your model here, e.g.:
            torch.save(self.model.state_dict(), 'model.pt')
        """
        pass
