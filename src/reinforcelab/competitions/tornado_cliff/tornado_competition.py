from reinforcelab.competitions.builder import RLCompetition, PhaseConfig

comp = RLCompetition(
    title="ReinforceLab: Tornado Cliff Competition",
    description="Train and evaluate reinforcement learning agents on the Tornado Cliff environment.",
)

comp.add_phase(PhaseConfig(
    name="Performance Evaluation",
    env_id="tornadocliff_env:factoredai/TornadoCliff-v0",
    phase_type="evaluation",
    start="1-1-2026",
    end="1-31-2026",
))

comp.build()