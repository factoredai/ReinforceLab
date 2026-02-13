from reinforcelab.competitions.builder import RLCompetition, PhaseConfig

comp = RLCompetition(
    title="ReinforceLab: Tornado Cliff Competition",
    description="Train and evaluate reinforcement learning agents on the Tornado Cliff environment.",
)

comp.add_phase(PhaseConfig(
    name="Performance & Convergence Evaluation",
    env_id="tornadocliff_env:factoredai/TornadoCliff-v0",
    phase_type="convergence",
    goal_reward=-50.0,
    max_steps=1_000_000,
))

comp.build()