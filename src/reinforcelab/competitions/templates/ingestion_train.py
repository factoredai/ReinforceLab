import os
import sys
import json
import numpy as np
import gymnasium as gym
from monitor import ConvergenceMonitor

# --- CONFIGURATION (Injected by Framework) ---
ENV_ID = "___ENV_ID___"
GOAL = "___GOAL___"
WINDOW = "___WINDOW___"
MAX_STEPS = "___MAX_STEPS___"
NUM_RUNS = "___NUM_RUNS___"
# ---------------------------------------------


def run():
    print("--- Starting Phase 2: Convergence ---")
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    program_dir = sys.argv[3]
    submission_dir = sys.argv[4]

    print(f"Submission dir: {submission_dir}")
    print(f"Files in submission: {os.listdir(submission_dir)}")
    
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
    runs = int(NUM_RUNS)
    goal = float(GOAL)
    window = int(WINDOW)
    max_s = int(MAX_STEPS)

    for i in range(runs):
        print(f"\nRun {i + 1}/{runs}")
        
        # Create environment with convergence monitor
        try:
            raw_env = gym.make(ENV_ID)
        except Exception:
            sys.path.append(program_dir)
            import importlib
            module = importlib.import_module("custom_env")
            raw_env = module.make_env()
            
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
            score = env.convergence_step if env.convergence_step else env.total_steps
            print(f"Converged at step {score}")
        else:
            score = max_s * 1.5  # Penalty for not converging
            print(f"Did not converge (penalty: {score:.0f})")
            
        step_counts.append(score)

    final_score = np.mean(step_counts)
    print(f"\nFinal Score: {final_score:.4f}")
    
    # Restore directory and write output
    os.chdir(original_dir)
    with open(os.path.join(output_dir, "scores.json"), "w") as f:
        json.dump({"score": final_score}, f)


if __name__ == "__main__":
    run()
