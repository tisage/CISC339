"""Experiment prompt configurations for the Multi-Agent LLM Mafia demo.

Each sibling module (baseline.py, persona.py, strategic_mafia.py) defines
one complete, self-contained ExperimentConfig - the full set of prompt text
and model-assignment rules for one batch of generated games. Keeping these
as separate files (rather than one shared set of constants with scattered
if/else branches in mafia_engine.py) makes each experiment's exact prompt
wording diffable and independently reproducible: to see exactly what
changed between the "baseline" and "strategic_mafia" batches, diff the two
files directly.

This also sets up the natural lead-in to the later LLM lab section: the
game engine and rules are held completely fixed across all three
experiments, and only the prompt text changes - yet agent behavior changes
systematically. That's the empirical case for why prompt engineering
matters, made concrete instead of asserted.

Usage:
    from prompts import load_experiment
    config = load_experiment("strategic_mafia")
    game = MafiaGame(caller, GameConfig(experiment=config))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    description: str
    commitment_instruction: str
    role_goals: dict[str, str]
    use_personas: bool = False
    # If set, the Mafia agent's model slot is fixed to this model instead of
    # being randomly assigned from the roster along with everyone else.
    mafia_model_override: Optional[str] = None


def load_experiment(name: str) -> ExperimentConfig:
    if name == "baseline":
        from prompts.baseline import CONFIG
        return CONFIG
    if name == "persona":
        from prompts.persona import CONFIG
        return CONFIG
    if name == "strategic_mafia":
        from prompts.strategic_mafia import CONFIG
        return CONFIG
    raise ValueError(
        f"Unknown experiment {name!r}. Expected one of: "
        f"baseline, persona, strategic_mafia."
    )
