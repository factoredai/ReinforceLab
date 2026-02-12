# Evaluation

This competition uses a two-phase evaluation system to comprehensively assess your reinforcement learning agent.

## Phase 1: Average Return Evaluation

### Objective
Measure the **average return** of your pre-trained agent over multiple evaluation episodes.

### Process
1. Your agent is loaded from the `model.pt` file in your submission
2. The agent runs for **___NUM_EPISODES___** episodes in the `___ENV_ID___` environment
3. For each episode:
   - The environment is reset
   - The agent interacts with the environment until the episode terminates
   - The cumulative reward for the episode is recorded
4. The final score is the **mean return** across all **___NUM_EPISODES___** episodes

### Scoring
- **Score**: Mean return over all evaluation episodes
- **Higher is better**: Agents with higher average returns rank higher
- **Leaderboard**: Sorted in descending order

### Example
If your agent achieves returns of [450, 480, 470, 460, 490] over 5 episodes, your score would be:
```
Score = (450 + 480 + 470 + 460 + 490) / 5 = 470.0
```

## Phase 2: Training Convergence Evaluation

### Objective
Measure both how efficiently your agent learns and the optimized performance it achieves after training.

### Process
1. Your agent is initialized from scratch (no pre-trained model)
2. Training begins in the `___ENV_ID___` environment
3. The agent trains for up to **___MAX_STEPS___** steps
4. Convergence is detected when:
   - The agent achieves a reward of at least **___GOAL_REWARD___**
   - This performance is maintained for **___STABILITY_WINDOW___** consecutive episodes
5. After each training run, the trained agent is evaluated for **___NUM_EPISODES___** episodes to measure its optimized performance
6. The process is repeated **___NUM_RUNS___** times
7. Two scores are reported: **Convergence** (average steps to converge) and **Eval** (average return of trained agent)

### Scoring
- **Convergence**: Average number of training steps to reach convergence across runs
  - **Lower is better**: Agents that converge faster rank higher
  - **Timeout**: If convergence is not reached within **___MAX_STEPS___** steps, a penalty is applied
- **Eval**: Average return of the trained agent over **___NUM_EPISODES___** evaluation episodes
  - **Higher is better**: Agents that achieve higher returns after training rank higher

### Convergence Criteria
Convergence is achieved when:
1. An episode achieves a reward ≥ **___GOAL_REWARD___**
2. The following **___STABILITY_WINDOW___** consecutive episodes also achieve reward ≥ **___GOAL_REWARD___**
3. The step count at which this stability is first achieved is recorded

### Example
If your agent converges in runs requiring [5000, 5200, 4800, 5100, 4900] steps with eval returns [470, 485, 460, 475, 480]:
```
Convergence = (5000 + 5200 + 4800 + 5100 + 4900) / 5 = 5000.0 steps
Eval = (470 + 485 + 460 + 475 + 480) / 5 = 474.0
```

## Final Ranking

Your final ranking is determined by your performance in both phases. The leaderboard displays:
- **Phase 1 Score**: Average return (higher is better)
- **Phase 2 Convergence**: Steps to converge (lower is better)
- **Phase 2 Eval**: Trained agent's return (higher is better)

All metrics are important for a complete assessment of your agent's capabilities.

