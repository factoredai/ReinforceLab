# ReinforceLab Competitions

This module provides a **builder** and **templates** system for creating RL competitions that run on [CodaBench](https://www.codabench.org/). You define your competition in a short Python config, run the builder, and get a ready-to-upload competition bundle.

---

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [How the Builder Works](#how-the-builder-works)
- [How Templates Work](#how-templates-work)
- [Customizing Your Competition](#customizing-your-competition)
- [Template Reference](#template-reference)
- [Build Output Structure](#build-output-structure)

---

## Overview

The competitions system has three main pieces:

1. **Configuration API** (`RLCompetition`, `PhaseConfig`) — Declarative description of your competition (title, phases, environment, scoring).
2. **Builder** (`CompetitionBuilder`) — Takes your config and template files, renders them, and produces a CodaBench-compatible bundle.
3. **Templates** — Files in `templates/` that are copied or rendered into the bundle. They support placeholders (`___KEY___`) that get replaced with phase-specific values.

When you call `comp.build()`, the builder:

1. Cleans and creates the bundle directory structure
2. For each phase, generates an ingestion program (evaluation or convergence logic)
3. Renders markdown pages and scripts with competition-specific values
4. Builds the starting kit (agent template, run_local, train, sample submission)
5. Zips the bundle for upload to CodaBench

---

## Quick Start

Create a competition definition file (e.g. `my_competition.py`):

```python
from reinforcelab.competitions.builder import RLCompetition, PhaseConfig

comp = RLCompetition(
    title="My RL Competition",
    description="Train and evaluate agents on MyEnv.",
)

comp.add_phase(PhaseConfig(
    name="Eval",
    env_id="CartPole-v1",
    phase_type="evaluation",
    num_episodes=100,
    start="1-1-2025",
    end="1-31-2025",
))

comp.add_phase(PhaseConfig(
    name="Train",
    env_id="CartPole-v1",
    phase_type="convergence",
    goal_reward=475.0,
    max_steps=10000,
    start="2-1-2025",
    end="2-28-2025",
))

comp.build()
```

Run it:

```bash
uv run python my_competition.py
```

Upload the generated `build/bundle.zip` to CodaBench.

---

## How the Builder Works

### Configuration Classes

#### `RLCompetition`

Top-level competition configuration.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `title` | str | — | Competition name |
| `description` | str | — | Short description |
| `docker_image` | str | `aristizabal95/codalab-reinforcelab:gpu310` | Docker image for runs |
| `phases` | List[PhaseConfig] | [] | Competition phases |

Methods:

- `add_phase(phase: PhaseConfig)` — Append a phase.
- `build(build_dir="./build", override_dir=None)` — Run the builder. If `override_dir` is omitted, the directory of the calling script is used for template overrides.

#### `PhaseConfig`

Configuration for a single phase (task).

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | str | — | Phase name (e.g. "Eval", "Train") |
| `env_id` | str | — | Gymnasium environment ID (e.g. `CartPole-v1`) |
| `phase_type` | str | — | `"evaluation"` or `"convergence"` |
| `num_episodes` | int | 100 | Episodes for evaluation phase |
| `goal_reward` | float | 0.0 | Target reward for convergence phase |
| `stability_window` | int | 100 | Episodes to maintain goal for convergence |
| `max_steps` | int | 100000 | Max training steps for convergence phase |
| `num_runs` | int | 5 | Runs for convergence phase (score = mean) |
| `start` | str \| None | None | Phase start date (M-D-YYYY); None = today |
| `end` | str \| None | None | Phase end date (M-D-YYYY); None = start + 2 weeks |

### Phase Types

- **`evaluation`** — Load a pre-trained agent, run `act()` for `num_episodes` episodes, score = mean return (higher is better).
- **`convergence`** — Train agent from scratch; score = mean steps to reach `goal_reward` for `stability_window` episodes (lower is better).

### Build Flow

1. **Clean** — Remove previous `bundle/`.
2. **Data dirs** — Create `input_data/` and `reference_data/` (CodaBench requirements).
3. **Scoring program** — Copy `scoring.py` into `scoring_program/`.
4. **Tasks** — For each phase:
   - Create `ingestion_program_N/` with `program.py` (rendered from `ingestion_eval.py` or `ingestion_train.py`), `monitor.py`, `metadata.yaml`.
   - Register task and phase in the bundle metadata.
5. **competition.yaml** — Generate CodaBench config (tasks, phases, pages, leaderboards).
6. **Logo** — Copy `logo.png` (override or default).
7. **Solution** — Copy `solution_random_agent.py` as `solution/agent.py`.
8. **Pages** — Render `overview.md`, `evaluation.md`, copy `terms.md`.
9. **Starting kit** — Build agent template, run_local, train, requirements, sample submission, notebook.
10. **Zip** — Create `bundle.zip`.

---

## How Templates Work

### Template Resolution (Override System)

For each template file (e.g. `agent.py`, `requirements.txt`), the builder:

1. Looks for `override_dir / filename` (e.g. your competition folder).
2. If found, uses that file.
3. Otherwise, uses `templates / filename`.

So you can customize any template by placing a file with the same name in your competition directory.

### Placeholder Rendering

Templates can contain placeholders: `___KEY___` or `"___KEY___"` (for quoted strings in code).

The builder replaces them with values from a `replacements` dict. Example:

```python
self._render("ingestion_eval.py", {
    "ENV_ID": phase.env_id,
    "NUM_EPISODES": phase.num_episodes
})
```

In the template:

```python
ENV_ID = "___ENV_ID___"
NUM_EPISODES = "___NUM_EPISODES___"
```

After render:

```python
ENV_ID = "CartPole-v1"
NUM_EPISODES = "100"
```

### Which Templates Are Rendered vs Copied

| Template | Treatment | Placeholders |
|----------|-----------|--------------|
| `ingestion_eval.py` | Rendered | ENV_ID, NUM_EPISODES |
| `ingestion_train.py` | Rendered | ENV_ID, GOAL, WINDOW, MAX_STEPS, NUM_RUNS |
| `overview.md` | Rendered | TITLE, DESCRIPTION, NUM_EPISODES, ENV_ID, GOAL_REWARD, STABILITY_WINDOW, MAX_STEPS, NUM_RUNS |
| `evaluation.md` | Rendered | NUM_EPISODES, ENV_ID, GOAL_REWARD, STABILITY_WINDOW, MAX_STEPS, NUM_RUNS |
| `run_local.py` | Rendered | ENV_ID, NUM_EPISODES, GOAL_REWARD, STABILITY_WINDOW, MAX_STEPS, NUM_RUNS |
| `train.py` | Rendered | ENV_ID, GOAL_REWARD |
| `getting_started.ipynb` | Rendered | ENV_ID, TITLE |
| `agent.py`, `monitor.py`, `scoring.py`, etc. | Copied as-is | — |

For page templates, values come from the first evaluation and first convergence phase found (or defaults if a phase type is missing).

---

## Customizing Your Competition

### 1. Create a New Competition Directory

```
competitions/
├── my_env/
│   ├── my_env_competition.py   # Config + comp.build()
│   ├── requirements.txt        # Optional override
│   ├── logo.png                # Optional override
│   └── agent.py                # Optional override (custom base agent)
└── templates/
    └── ...
```

### 2. Override Templates

Put a file with the **same name** as a template in your competition directory. It will be used instead of the default.

**Example: Custom `requirements.txt`**

If your env needs extra packages (e.g. `gymnasium[classic-control]` for CartPole):

```
# my_env/requirements.txt
gymnasium
gymnasium[classic-control]
numpy
torch
```

**Example: Custom logo**

Place `logo.png` in your competition directory. It will be used in the bundle and starting kit.

### 3. Add or Reorder Phases

```python
# Evaluation-only competition
comp.add_phase(PhaseConfig(
    name="Eval",
    env_id="LunarLander-v2",
    phase_type="evaluation",
    num_episodes=50,
))

# Convergence-only competition
comp.add_phase(PhaseConfig(
    name="Train",
    env_id="LunarLander-v2",
    phase_type="convergence",
    goal_reward=200.0,
    stability_window=20,
    max_steps=50000,
))

# Multiple environments
comp.add_phase(PhaseConfig(name="CartPole", env_id="CartPole-v1", phase_type="evaluation", ...))
comp.add_phase(PhaseConfig(name="LunarLander", env_id="LunarLander-v2", phase_type="evaluation", ...))
```

### 4. Custom Agent Interface

Override `agent.py` in your competition folder to change the base agent interface (e.g. different method signatures). Participants will receive this as the starting template.

### 5. Custom Ingestion or Scoring

For non-standard behavior:

1. Copy `ingestion_eval.py` or `ingestion_train.py` (or both) into your competition directory.
2. Edit the logic and keep the `___KEY___` placeholders for values the builder injects.
3. Optionally override `scoring.py` if you need different score aggregation.

### 6. Custom Build Directory or Override Directory

```python
comp.build(build_dir="./output")
comp.build(override_dir="/path/to/my/competition/files")
```

### 7. Custom Environment (Non-Gymnasium)

If your env is not in Gymnasium, provide a `custom_env.py` in your competition directory that defines `make_env()` returning a Gymnasium-like env. The ingestion programs will import it when `gym.make(ENV_ID)` fails.

---

## Template Reference

### Templates Directory Structure

```
templates/
├── agent.py                 # Base Agent class for participants
├── evaluation.md            # Evaluation page (rendered)
├── getting_started.ipynb    # Getting started notebook (rendered)
├── ingestion_eval.py        # Evaluation phase ingestion (rendered)
├── ingestion_train.py       # Convergence phase ingestion (rendered)
├── logo.png                 # Default logo
├── monitor.py               # ConvergenceMonitor wrapper
├── overview.md              # Overview page (rendered)
├── requirements.txt         # Default Python dependencies
├── run_local.py             # Local runner (rendered)
├── sample_checkpoint.txt    # Example checkpoint for sample submission
├── sample_submission_agent.py # Sample submission agent
├── scoring.py               # CodaBench scoring program
├── solution_random_agent.py # Baseline/random agent
├── terms.md                 # Competition terms
└── train.py                 # Training script (rendered)
```

### Agent Interface (Expected by Ingestion)

Participants implement an `Agent` class with:

| Method | Required | Usage |
|--------|----------|-------|
| `__init__(env)` | Yes | Initialize with the environment |
| `act(observation)` | Yes | Return action given observation |
| `train()` | Yes (convergence) | Train using `self.env`; stop on `StopIteration` |
| `load()` | Yes (evaluation) | Load model from submission directory |
| `save()` | Optional | Save model (for training/checkpointing) |

`load()` and `save()` take no path arguments; the agent is responsible for resolving paths (e.g. `os.path.dirname(__file__)` for the submission dir).

---

## Build Output Structure

After `comp.build()`, you get:

```
build/
├── bundle.zip          # Upload this to CodaBench
└── bundle/
    ├── competition.yaml
    ├── logo.png
    ├── input_data/
    ├── reference_data/
    ├── scoring_program/
    │   ├── program.py
    │   └── metadata.yaml
    ├── ingestion_program_1/   # One per phase
    │   ├── program.py
    │   ├── monitor.py
    │   └── metadata.yaml
    ├── ingestion_program_2/
    │   └── ...
    ├── solution/
    │   └── agent.py
    ├── pages/
    │   ├── overview.md
    │   ├── evaluation.md
    │   └── terms.md
    └── starting_kit/
        ├── agent.py
        ├── random_agent.py
        ├── monitor.py
        ├── run_local.py
        ├── train.py
        ├── requirements.txt
        ├── getting_started.ipynb
        ├── logo.png
        ├── sample_submission/
        │   ├── agent.py
        │   ├── checkpoint.txt
        │   └── requirements.txt
        └── sample_submission.zip
```

---

## Example: CartPole Competition

The `cartpole/` directory is a full example:

```
cartpole/
├── cartpole_competition.py   # Competition definition
├── requirements.txt          # Override: adds gymnasium[classic-control]
├── logo.png                  # Override: custom logo
└── ...
```

To build:

```bash
uv run python -c "
from reinforcelab.competitions.cartpole.cartpole_competition import comp
comp.build()
"
```

Or run the competition script directly if it is set up to be executable.
