# ___TITLE___

___DESCRIPTION___

## Competition Overview

This reinforcement learning competition evaluates your agent's performance across two distinct phases:

### Phase 1: Evaluation
In the first phase, your trained agent will be evaluated on its **average return** over multiple episodes. The agent will run for **___NUM_EPISODES___** episodes in the `___ENV_ID___` environment, and the average return across all episodes will be used as the score. Higher average returns indicate better performance.

**Key Details:**
- Your agent should be trained and saved as `model.pt` in your submission
- The agent will be loaded and evaluated without further training
- Performance is measured by the mean return across all evaluation episodes
- Higher scores are better (descending order on leaderboard)

### Phase 2: Training Convergence
In the second phase, your agent will be trained from scratch. We measure both how quickly it converges and the optimized performance it achieves after training.

**Key Details:**
- Your agent will be trained from scratch (no pre-trained model)
- Training will run for up to **___MAX_STEPS___** steps
- The goal is to reach a reward of **___GOAL_REWARD___** and maintain it for **___STABILITY_WINDOW___** episodes
- After each training run, the trained agent is evaluated for **___NUM_EPISODES___** episodes
- **Convergence** (lower is better): average steps to reach convergence across **___NUM_RUNS___** runs
- **Eval** (higher is better): average return of the trained agent during evaluation

## Submission Format

Your submission should include:
- `agent.py`: A file containing an `Agent` class with:
  - `act(observation)`: Method that returns an action given an observation
  - `train()` (optional): Method for training the agent
  - `load(model_path)`: Method to load a saved model
  - `save(model_path)`: Method to save the current model
- `model.pt` (for Phase 1): Pre-trained model file that will be loaded during evaluation

## Getting Started

1. Download the starting kit from the competition page
2. Implement your `Agent` class in `agent.py`
3. Train your agent and save it as `model.pt` for Phase 1 evaluation
4. Submit your code and model to Phase 1
5. For Phase 2, ensure your agent can train effectively from scratch

Good luck!

