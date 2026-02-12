import os
import sys
import json
import random
import subprocess
import numpy as np
import gymnasium as gym

# --- CONFIGURATION (Injected by Framework) ---
ENV_ID = "___ENV_ID___"
NUM_EPISODES = "___NUM_EPISODES___"
SEED = ___SEED___
# ---------------------------------------------


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
    print("--- Starting Phase 1: Evaluation ---")
    
    # Codabench directory structure parsing
    # Default args: program.py input output program submission
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    program_dir = sys.argv[3]
    submission_dir = sys.argv[4]

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
        print(f"CRITICAL: Could not import Agent from submission. {e}")
        sys.exit(1)

    # Setup Environment
    try:
        env = gym.make(ENV_ID)
    except Exception:
        sys.path.append(program_dir)
        import importlib
        module = importlib.import_module("custom_env")
        env = module.make_env()

    # Seed for reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    env.action_space.seed(SEED)

    # Create and load agent
    agent = Agent(env)
    
    # Let the agent load its own model (no path parameter)
    print("Calling agent.load()...")
    agent.load()

    # Run evaluation episodes
    scores = []
    for ep in range(int(NUM_EPISODES)):
        obs, _ = env.reset(seed=SEED + ep)
        done = False
        ep_score = 0
        while not done:
            action = agent.act(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_score += reward
        scores.append(ep_score)
        if (ep + 1) % 10 == 0:
            print(f"Episode {ep + 1}/{NUM_EPISODES}: Score = {ep_score:.2f}")

    final_score = np.mean(scores)
    print(f"Final Score: {final_score:.4f}")

    # Restore directory and write output
    os.chdir(original_dir)
    with open(os.path.join(output_dir, "scores.json"), "w") as f:
        json.dump({"score": final_score}, f)


if __name__ == "__main__":
    run()
