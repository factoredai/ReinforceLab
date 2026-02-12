"""
Training Script - Train your agent locally before submission.

This script trains your agent and saves the model for evaluation.
Customize the training loop to match your agent's implementation.

Usage:
    python train.py                     # Train with defaults
    python train.py --episodes 500      # Train for specific episodes

After training, run evaluation with:
    python run_local.py --phase eval
"""
import argparse
import gymnasium as gym
import numpy as np
from agent import Agent

# --- CONFIGURATION (Injected by Builder) ---
ENV_ID = "___ENV_ID___"
GOAL_REWARD = "___GOAL_REWARD___"
# --------------------------------------------


def train(num_episodes=500, print_every=10):
    """
    Train the agent.
    
    Args:
        num_episodes: Maximum episodes to train
        print_every: Print stats every N episodes
    """
    env = gym.make(ENV_ID)
    agent = Agent(env)
    
    goal = float(GOAL_REWARD)
    
    print("=" * 60)
    print(f"TRAINING AGENT ON {ENV_ID}")
    print("=" * 60)
    print(f"Episodes: {num_episodes}")
    print(f"Goal: {goal}")
    print("-" * 60)
    
    episode_rewards = []
    best_avg = float('-inf')
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            episode_reward += reward
        
        episode_rewards.append(episode_reward)
        
        # Calculate running average
        avg_reward = np.mean(episode_rewards[-100:]) if episode_rewards else 0
        
        # Print progress
        if (episode + 1) % print_every == 0:
            print(f"Episode {episode + 1:4d} | "
                  f"Reward: {episode_reward:6.1f} | "
                  f"Avg(100): {avg_reward:6.1f}")
        
        # Save best model
        if avg_reward > best_avg and len(episode_rewards) >= 100:
            best_avg = avg_reward
            agent.save()
            print(f"  -> New best: {best_avg:.1f} - Saved!")
        
        # Early stopping if solved
        if avg_reward >= goal and len(episode_rewards) >= 100:
            print("-" * 60)
            print(f"SOLVED! Avg {avg_reward:.1f} >= {goal}")
            break
    
    env.close()
    agent.save()
    
    print("=" * 60)
    print("TRAINING COMPLETE")
    print(f"Best average: {best_avg:.1f}")
    print("Model saved. Run: python run_local.py --phase eval")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Train agent")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--print-every", type=int, default=10)
    args = parser.parse_args()
    
    train(num_episodes=args.episodes, print_every=args.print_every)


if __name__ == "__main__":
    main()

