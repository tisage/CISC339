"""Strategic-Mafia experiment: the Mafia seat is pinned to the strongest
available model, and the Mafia's own role prompt is rewritten with more
concrete, more deceptive tradecraft than baseline's - to see whether a
stronger model given more explicit strategic tools can meaningfully close
the Village-side win-rate gap observed across the baseline and persona
batches (11/14 games so far won by Village; see DESIGN.md / session
history for the underlying analysis).

Detective, Doctor, and Villager keep baseline's role_goals unchanged and
personas stay off - this experiment isolates exactly one pair of variables
(Mafia's model, Mafia's prompt) rather than bundling in unrelated changes,
so any shift in outcome is attributable to those two things specifically.
"""

from __future__ import annotations

from prompts import ExperimentConfig
from prompts.baseline import COMMITMENT_INSTRUCTION, ROLE_GOALS

# Pinned to the strongest model in the roster at time of writing (see
# mafia_engine.ROSTER / DESIGN.md Section 5). If the roster is ever
# re-verified and a stronger model replaces this one, update here - this
# override is deliberately kept in one place rather than duplicated.
MAFIA_MODEL = "anthropic/claude-opus-5"

STRATEGIC_MAFIA_GOAL = """\
You are the Mafia. Only you (and no one else) know this. Your goal is to \
avoid detection and eliminate Villagers each night until Mafia are no longer \
outnumbered.

You have consistently lost more often than you've won in games like this \
one, and the reason is almost always the same: you get caught reacting \
instead of directing. Village players who cross-check timelines and press \
on inconsistencies win most games. Beat that specifically:

1. **Plant your cover story before you need it.** On Day 1 - before anyone \
has a reason to suspect you - say something mildly analytical that sounds \
like a townsperson thinking out loud (a theory about the kill pattern, a \
soft question to someone else). This gives you a paper trail of "reasonable \
early behavior" to point back to later, instead of only reacting once \
you're accused.

2. **Never flat-out deny with nothing behind it.** A bare "I'm not Mafia" is \
exactly what a guilty player says, and good Village players know that. \
Instead, counter with something structural: point out a timeline problem in \
your accuser's story, offer your own "read" on a different player (even if \
fabricated), or - if someone claims Detective and names you - counter-claim \
a role yourself rather than only denying theirs. A counter-claim forces the \
Village to adjudicate between two stories instead of just believing the one \
accusation.

3. **Track who is being unusually careful, not just who is loud.** A player \
who goes quiet right after a no-kill night, or who waits an extra round \
before claiming a role, is very often the real Detective or Doctor managing \
their exposure - that patience pattern is itself a tell. Prioritize \
eliminating that player over an easier, louder target, even if it feels \
riskier in the moment.

4. **Match your target's own standard of evidence back at them.** If a \
Village player says "I'd need two consistent results before I'd trust a \
Detective claim," and someone later claims Detective with only one result, \
you can legitimately hold them to the standard *they* set - this reads as \
sharp, skeptical town play, not as Mafia deflection.

5. **Don't overcommit to a lie you can't sustain.** If you fabricate a claim \
(e.g. claiming Doctor), be ready to answer follow-up questions about it \
consistently in later rounds - a fabricated claim that falls apart under \
a second round of questioning is worse than not claiming at all. If you're \
not confident you can hold a specific lie together for multiple rounds, \
raise doubt about someone else instead of inventing a role you'll have to \
defend later.

You are still bound by the general rules every player follows: stay \
internally consistent with your own past statements (see below), and use \
your genuine private night-action information privately - never let your \
private reasoning leak into what you say publicly."""

CONFIG = ExperimentConfig(
    name="strategic_mafia",
    description=(
        "Mafia model pinned to the strongest available model in the roster "
        "(currently anthropic/claude-opus-5), and the Mafia's role prompt "
        "replaced with more concrete deceptive tradecraft (planting cover "
        "early, counter-claiming instead of flat denial, targeting players "
        "who show 'patient reveal' behavior, matching an accuser's own "
        "evidence standard back at them, and not overcommitting to an "
        "unsustainable lie). Detective/Doctor/Villager prompts and personas "
        "are unchanged from baseline, isolating the Mafia model+prompt as "
        "the only variables."
    ),
    commitment_instruction=COMMITMENT_INSTRUCTION,
    role_goals={**ROLE_GOALS, "Mafia": STRATEGIC_MAFIA_GOAL},
    use_personas=False,
    mafia_model_override=MAFIA_MODEL,
)
