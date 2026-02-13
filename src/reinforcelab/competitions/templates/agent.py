"""
Submission Agent - Default implementation with checkpoint save/load demonstration.

Your submission must have an agent.py with an Agent class implementing:
- __init__(env): Initialize with the environment
- act(observation): Return an action given an observation
- train(): Train the agent using self.env (until StopIteration from monitor)
- load(): Load your model (artifacts MUST be in the submission directory)
- save(): Save your model (artifacts MUST be in the submission directory)

This default agent takes random actions and demonstrates save/load with a pickle checkpoint.
"""
import os
import pickle
import gymnasium as gym


# Directory where this agent.py lives - your submission files are extracted here
SUBMISSION_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CHECKPOINT = os.path.join(SUBMISSION_DIR, "checkpoint.pkl")


class Agent:
    """
    Default agent: random actions with toy save/load for demonstration.

    Replace this with your trained model. The interface is:
    - act(obs): Return action for evaluation and during training
    - train(): Run training loop; catch StopIteration when monitor signals done
    - save(): Persist model/weights into the submission directory
    - load(): Load from files in the submission directory
    """

    def __init__(self, env: gym.Env):
        """
        Initialize the agent with the environment.

        Args:
            env: The gymnasium environment. In Phase 2 (convergence), this is
                 wrapped with ConvergenceMonitor which raises StopIteration
                 when the goal is reached or max steps hit.
        """
        self.env = env
        self.checkpoint_data = None

    def act(self, observation):
        """
        Return an action given the current observation.

        Used in both Phase 1 (evaluation) and Phase 2 (training).
        In your implementation, use your trained policy/model here.

        Args:
            observation: Current observation from the environment.

        Returns:
            An action valid for env.action_space.
        """
        return self.env.action_space.sample()

    def train(self):
        """
        Train the agent using self.env.

        The environment may be wrapped with ConvergenceMonitor, which raises
        StopIteration when the goal reward is achieved for the stability window
        or when max steps is reached. Catch StopIteration to exit gracefully.

        In your implementation, run your training loop here.
        """
        try:
            for _ in range(10):
                obs, _ = self.env.reset()
                done = False
                while not done:
                    action = self.act(obs)
                    obs, reward, terminated, truncated, info = self.env.step(action)
                    done = terminated or truncated
        except StopIteration:
            pass

    def load(self):
        """
        Load model weights or checkpoint.

        You decide how to load (pickle, torch, etc.), but the artifact MUST be
        in the submission directory (same directory as this agent.py).
        Example: path = os.path.join(SUBMISSION_DIR, 'model.pt')
        """
        if os.path.exists(DEFAULT_CHECKPOINT):
            with open(DEFAULT_CHECKPOINT, "rb") as f:
                self.checkpoint_data = pickle.load(f)
            print(f"Loaded checkpoint from {DEFAULT_CHECKPOINT}: {self.checkpoint_data}")
        else:
            print(f"No checkpoint found at {DEFAULT_CHECKPOINT}")

    def save(self):
        """
        Save model weights or checkpoint.

        You decide how to save (pickle, torch, etc.), but the artifact MUST be
        in the submission directory (same directory as this agent.py).
        Example: path = os.path.join(SUBMISSION_DIR, 'model.pt')
        """
        toy_data = {"episode": 0, "toy": True, "message": "Demo checkpoint"}
        with open(DEFAULT_CHECKPOINT, "wb") as f:
            pickle.dump(toy_data, f)
        print(f"Saved checkpoint to {DEFAULT_CHECKPOINT}")
