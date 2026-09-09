"""Persona experiment: same role_goals as baseline, but each agent also gets
a randomized flavor persona (voice/temperament, not correlated with role -
see PERSONA_POOL) for a more entertaining watch. Used for the 5-game
"flavor" batch.

Deliberately reuses baseline's COMMITMENT_INSTRUCTION and ROLE_GOALS
unchanged - the only variable this experiment isolates is "personas on vs
off," not any change in role strategy advice.
"""

from __future__ import annotations

from prompts import ExperimentConfig
from prompts.baseline import COMMITMENT_INSTRUCTION, ROLE_GOALS

# Each persona is a voice/temperament only - deliberately NOT correlated
# with any role, and deliberately free of tells like "loves accusing people"
# that would function as a hidden role hint. Persona is assigned per agent
# per game, independent of role and independent of model.
PERSONA_POOL: list[str] = [
    "blunt and impatient - you get to the point and don't sugarcoat your reads",
    "warm and diplomatic - you soften disagreements and look for common ground",
    "dryly sarcastic - you needle people with dry humor, but you're still playing to win",
    "anxious and over-explaining - you second-guess yourself out loud before landing on a position",
    "theatrical and confident - you narrate your own reasoning like it's obviously correct",
    "quiet and analytical - you speak rarely, but when you do it's dense with specifics",
    "folksy and roundabout - you tell little stories or analogies before making your point",
    "competitive and scorekeeping - you track who's been right or wrong so far and say so",
]

CONFIG = ExperimentConfig(
    name="persona",
    description=(
        "Same role_goals as baseline, plus a randomized flavor persona per "
        "agent per game (voice/temperament only, independent of role and "
        "model) for a more entertaining, varied watch."
    ),
    commitment_instruction=COMMITMENT_INSTRUCTION,
    role_goals=ROLE_GOALS,
    use_personas=True,
    mafia_model_override=None,
)
