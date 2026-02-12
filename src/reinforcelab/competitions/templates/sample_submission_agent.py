"""
Sample Submission Agent - Demonstrates loading from checkpoint files

This agent shows how to load additional files from your submission directory.
In a real submission, you would load your trained model weights.

Key point: The load() method has no parameters - you decide where to load from.
Files in your submission zip will be extracted to the same directory as agent.py.
"""
import os
import gymnasium as gym


# Get the directory where this agent.py file is located
# This is where your submission files will be
SUBMISSION_DIR = os.path.dirname(os.path.abspath(__file__))


class Agent:
    """
    Sample Agent that demonstrates loading from a checkpoint file.
    
    This agent takes random actions but shows the pattern for loading
    saved model weights or other data from your submission.
    """
    
    def __init__(self, env: gym.Env):
        """Initialize the agent with the environment."""
        self.env = env
        self.checkpoint_data = None
    
    def act(self, observation):
        """
        Return a random action.
        
        In your implementation, this would use your trained model
        to select actions based on the observation.
        """
        return self.env.action_space.sample()
    
    def load(self):
        """
        Load model weights or data from files in the submission directory.
        
        This demo loads a text file to show the pattern.
        In your implementation, you would load your trained model:
        
        Example for PyTorch:
            model_path = os.path.join(SUBMISSION_DIR, 'model.pt')
            self.model = torch.load(model_path)
            
        Example for multiple files:
            config_path = os.path.join(SUBMISSION_DIR, 'config.json')
            weights_path = os.path.join(SUBMISSION_DIR, 'weights.pkl')
        """
        checkpoint_file = os.path.join(SUBMISSION_DIR, "checkpoint.txt")
        
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                self.checkpoint_data = f.read()
            print(f"Loaded checkpoint from: {checkpoint_file}")
            print(f"Checkpoint contents:\n{self.checkpoint_data}")
        else:
            print(f"No checkpoint.txt found in {SUBMISSION_DIR}")
            print(f"Available files: {os.listdir(SUBMISSION_DIR)}")
    
    def save(self):
        """
        Save model weights to the submission directory.
        
        Example for PyTorch:
            model_path = os.path.join(SUBMISSION_DIR, 'model.pt')
            torch.save(self.model.state_dict(), model_path)
        """
        print(f"Save called - would save to {SUBMISSION_DIR}")
    
    def train(self):
        """
        Train the agent using self.env.
        
        In your implementation, this would contain your training loop.
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
            pass
