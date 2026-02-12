"""
Base Agent Template for ReinforceLab Competitions

Implement your agent by creating an Agent class with the following interface.
The agent receives the environment in __init__ and is responsible for its own
loading/saving logic.
"""
import gymnasium as gym


class Agent:
    """
    Base Agent Template - Implement your RL agent here.
    
    Required methods:
    - __init__(env): Initialize with the environment
    - act(observation): Return an action given an observation
    - train(): Train the agent using self.env
    - load(): Load your model (you decide the format/location)
    - save(): Save your model (you decide the format/location)
    """
    
    def __init__(self, env: gym.Env):
        """
        Initialize your agent with the environment.
        
        Args:
            env: The gymnasium environment to interact with.
        """
        self.env = env

    def act(self, observation):
        """
        Return an action given an observation.
        
        This is used in both Phase 1 (evaluation) and Phase 2 (training).
        
        Args:
            observation: The current observation from the environment.
            
        Returns:
            An action to take in the environment.
        """
        raise NotImplementedError("You must implement the act() method")

    def train(self):
        """
        Train the agent using self.env.
        
        Used in Phase 2 (Convergence). The environment is wrapped with a
        ConvergenceMonitor that will raise StopIteration when:
        - The goal reward is achieved for the stability window, OR
        - Max steps is reached
        
        Your training loop should catch StopIteration to exit gracefully.
        """
        raise NotImplementedError("You must implement the train() method")

    def load(self):
        """
        Load your trained model.
        
        Used in Phase 1 (Evaluation). You decide where and how to load:
        - Load from a file in the same directory (e.g., 'model.pt', 'weights.pkl')
        - Load from multiple files
        - Use any format you prefer
        
        This method is called before evaluation begins.
        """
        pass  # Override to load your model

    def save(self):
        """
        Save your trained model.
        
        Optional - useful for checkpointing during training.
        You decide where and how to save.
        """
        pass  # Override to save your model
