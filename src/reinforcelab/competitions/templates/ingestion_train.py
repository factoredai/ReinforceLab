import os
import sys
import json
import random
import subprocess
import numpy as np
import gymnasium as gym
from monitor import ConvergenceMonitor

# --- CONFIGURATION (Injected by Framework) ---
ENV_ID = "___ENV_ID___"
GOAL = "___GOAL___"
WINDOW = "___WINDOW___"
MAX_STEPS = "___MAX_STEPS___"
NUM_RUNS = "___NUM_RUNS___"
NUM_EPISODES = "___NUM_EPISODES___"
SEED = ___SEED___
# ---------------------------------------------


class ReseedWrapper(gym.Wrapper):
    """Injects deterministic seeds on env.reset() for reproducibility."""

    def __init__(self, env, base_seed):
        super().__init__(env)
        self.base_seed = base_seed
        self.reset_count = 0

    def reset(self, **kwargs):
        if "seed" not in kwargs:
            kwargs["seed"] = self.base_seed + self.reset_count
            self.reset_count += 1
        return self.env.reset(**kwargs)


def install_requirements(submission_dir):
    """Install requirements from submission's requirements.txt if it exists."""
    requirements_file = os.path.join(submission_dir, "requirements.txt")
    if os.path.exists(requirements_file):
        print(f"Installing requirements from {requirements_file}...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "-q", "--no-cache-dir", "-r", requirements_file
            ])
            print("Requirements installed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Warning: Failed to install some requirements: {e}")
    else:
        print("No requirements.txt found in submission.")


def run():
    print("--- Starting Phase 2: Convergence ---")
    
    # Codabench uses fixed paths under /app (no argv)
    input_dir = "/app/input_data"
    output_dir = "/app/output"
    program_dir = "/app/program"
    submission_dir = "/app/ingested_program"

    print(f"Submission dir: {submission_dir}")
    print(f"Files in submission: {os.listdir(submission_dir)}")
    
    # Install submission requirements BEFORE importing agent
    install_requirements(submission_dir)
    
    # Change to submission directory so agent can find its files
    original_dir = os.getcwd()
    os.chdir(submission_dir)
    
    # Add submission to path
    sys.path.insert(0, submission_dir)
    
    try:
        from agent import Agent
    except ImportError as e:
        print(f"CRITICAL: Could not import Agent. {e}")
        sys.exit(1)

    step_counts = []
    eval_returns = []
    runs = int(NUM_RUNS)
    num_episodes = int(NUM_EPISODES)
    goal = float(GOAL)
    window = int(WINDOW)
    max_s = int(MAX_STEPS)

    seed = int(SEED)
    for i in range(runs):
        print(f"\nRun {i + 1}/{runs}")
        
        # Seed for reproducibility (per run)
        run_seed = seed + i
        random.seed(run_seed)
        np.random.seed(run_seed)

        # Create environment with convergence monitor
        try:
            raw_env = gym.make(ENV_ID)
        except Exception:
            sys.path.append(program_dir)
            import importlib
            module = importlib.import_module("custom_env")
            raw_env = module.make_env()

        raw_env.action_space.seed(run_seed)
        raw_env = ReseedWrapper(raw_env, base_seed=run_seed)
        env = ConvergenceMonitor(raw_env, goal, window, max_s)
        
        # Create fresh agent for each run
        agent = Agent(env)
        
        # Train the agent (no parameters - agent uses self.env)
        try:
            agent.train()
        except StopIteration:
            pass  # Normal termination from monitor
        except Exception as e:
            print(f"Training Error: {e}")
            
        if env.converged:
            conv_score = env.convergence_step if env.convergence_step else env.total_steps
            print(f"Converged at step {conv_score}")
        else:
            conv_score = max_s * 1.5  # Penalty for not converging
            print(f"Did not converge (penalty: {conv_score:.0f})")
        step_counts.append(conv_score)

        # Evaluate trained agent's performance
        try:
            eval_env = gym.make(ENV_ID)
        except Exception:
            sys.path.append(program_dir)
            import importlib
            module = importlib.import_module("custom_env")
            eval_env = module.make_env()
        eval_env.action_space.seed(run_seed)
        ep_scores = []
        for ep in range(num_episodes):
            obs, _ = eval_env.reset(seed=run_seed + 1000 + ep)
            done = False
            ep_score = 0
            while not done:
                action = agent.act(obs)
                obs, reward, terminated, truncated, _ = eval_env.step(action)
                done = terminated or truncated
                ep_score += reward
            ep_scores.append(ep_score)
        eval_return = np.mean(ep_scores)
        eval_env.close()
        eval_returns.append(eval_return)
        print(f"Eval return: {eval_return:.2f}")

    convergence_score = np.mean(step_counts)
    eval_score = np.mean(eval_returns)
    print(f"\nConvergence Score: {convergence_score:.4f} (lower is better)")
    print(f"Eval Score: {eval_score:.4f} (higher is better)")
    
    # Restore directory and write output
    os.chdir(original_dir)
    with open(os.path.join(output_dir, "scores.json"), "w") as f:
        json.dump({
            "convergence_score": float(convergence_score),
            "eval_score": float(eval_score),
        }, f)


if __name__ == "__main__":
    run()
