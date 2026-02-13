"""
Local Runner - Simulates competition evaluation phases locally.

This script runs the same evaluation that will happen when you submit:
- Phase 1 (Evaluation): Tests your pre-trained agent over multiple episodes
- Phase 2 (Convergence): Trains your agent from scratch and measures convergence time

Usage:
    python run_local.py                    # Run all phases
    python run_local.py --phase eval       # Run only evaluation phase
    python run_local.py --phase train     # Run only training/convergence phase
"""
import argparse
import sys
from pathlib import Path

# Ensure starting_kit root is on path so "from utils.monitor" works
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import gymnasium as gym
from utils.monitor import ConvergenceMonitor

# --- CONFIGURATION (Injected by Builder) ---
ENV_ID = "___ENV_ID___"
NUM_EPISODES = "___NUM_EPISODES___"
GOAL_REWARD = "___GOAL_REWARD___"
STABILITY_WINDOW = "___STABILITY_WINDOW___"
MAX_STEPS = "___MAX_STEPS___"
NUM_RUNS = "___NUM_RUNS___"
# --------------------------------------------


def run_evaluation_phase():
    """
    Phase 1: Evaluation
    
    Loads your pre-trained agent and evaluates it over multiple episodes.
    Score = Average return over all episodes (higher is better)
    """
    print("=" * 60)
    print("PHASE 1: EVALUATION")
    print("=" * 60)
    print(f"Environment: {ENV_ID}")
    print(f"Episodes: {NUM_EPISODES}")
    print("-" * 60)
    
    from submission_contents.agent import Agent
    
    env = gym.make(ENV_ID)
    agent = Agent(env)
    
    # Let the agent load its model (agent decides where/how to load)
    print("Calling agent.load()...")
    agent.load()
    
    scores = []
    for ep in range(int(NUM_EPISODES)):
        obs, _ = env.reset()
        done = False
        ep_score = 0
        steps = 0
        
        while not done:
            action = agent.act(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_score += reward
            steps += 1
        
        scores.append(ep_score)
        print(f"  Episode {ep + 1}/{NUM_EPISODES}: Return = {ep_score:.2f} ({steps} steps)")
    
    env.close()
    
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    
    print("-" * 60)
    print(f"EVALUATION RESULTS:")
    print(f"  Average Return: {mean_score:.2f} (+/- {std_score:.2f})")
    print(f"  Min/Max: {np.min(scores):.2f} / {np.max(scores):.2f}")
    print(f"  Phase 1 Score: {mean_score:.4f}")
    print("=" * 60)
    
    return mean_score


def run_convergence_phase():
    """
    Phase 2: Training Convergence
    
    Trains your agent from scratch and measures how quickly it converges.
    Score = Average steps to reach goal reward (lower is better)
    """
    print("=" * 60)
    print("PHASE 2: TRAINING CONVERGENCE")
    print("=" * 60)
    print(f"Environment: {ENV_ID}")
    print(f"Goal Reward: {GOAL_REWARD}")
    print(f"Stability Window: {STABILITY_WINDOW} episodes")
    print(f"Max Steps: {MAX_STEPS}")
    print(f"Number of Runs: {NUM_RUNS}")
    print("-" * 60)
    
    from submission_contents.agent import Agent
    
    step_counts = []
    num_runs = int(NUM_RUNS)
    goal = float(GOAL_REWARD)
    window = int(STABILITY_WINDOW)
    max_steps = int(MAX_STEPS)
    
    for run in range(num_runs):
        print(f"\nRun {run + 1}/{num_runs}:")
        
        raw_env = gym.make(ENV_ID)
        env = ConvergenceMonitor(raw_env, goal, window, max_steps)
        agent = Agent(env)  # Fresh agent for each run
        
        try:
            agent.train()
        except StopIteration:
            pass  # Normal termination from monitor
        except Exception as e:
            print(f"  Training error: {e}")
        
        if env.converged:
            steps = env.convergence_step
            print(f"  Converged at step {steps}")
        else:
            steps = max_steps * 1.5  # Penalty for not converging
            print(f"  Did not converge (penalty score: {steps:.0f})")
        
        step_counts.append(steps)
        env.close()
    
    mean_steps = np.mean(step_counts)
    std_steps = np.std(step_counts)
    
    print("-" * 60)
    print(f"CONVERGENCE RESULTS:")
    print(f"  Average Steps: {mean_steps:.0f} (+/- {std_steps:.0f})")
    print(f"  Min/Max: {np.min(step_counts):.0f} / {np.max(step_counts):.0f}")
    print(f"  Converged: {sum(1 for s in step_counts if s < max_steps * 1.5)}/{num_runs} runs")
    print(f"  Phase 2 Score: {mean_steps:.4f} (lower is better)")
    print("=" * 60)
    
    return mean_steps


def main():
    parser = argparse.ArgumentParser(description="Run local competition evaluation")
    parser.add_argument(
        "--phase", 
        choices=["eval", "train", "all"], 
        default="all",
        help="Which phase to run (default: all)"
    )
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("LOCAL COMPETITION RUNNER")
    print(f"Environment: {ENV_ID}")
    print("=" * 60 + "\n")
    
    results = {}
    
    if args.phase in ["eval", "all"]:
        results["evaluation"] = run_evaluation_phase()
        print()
    
    if args.phase in ["train", "all"]:
        results["convergence"] = run_convergence_phase()
        print()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if "evaluation" in results:
        print(f"Phase 1 (Evaluation):  {results['evaluation']:.4f} (higher is better)")
    if "convergence" in results:
        print(f"Phase 2 (Convergence): {results['convergence']:.4f} (lower is better)")
    print("=" * 60)


if __name__ == "__main__":
    main()
