from reinforcelab.competitions.builder import RLCompetition, PhaseConfig

comp = RLCompetition(
    title="My Structured RL Comp",
    description="A cleaner implementation.",
)

comp.add_phase(PhaseConfig(
    name="Eval",
    env_id="CartPole-v1",
    phase_type="evaluation",
    start="12-12-2025",
    end="1-12-2026"
))

comp.add_phase(PhaseConfig(
    name="Train",
    env_id="CartPole-v1",
    phase_type="convergence",
    goal_reward=475.0,
    max_steps=10000,
    start="1-13-2026",
    end="1-31-2026",
))

comp.build()