"""Baseline experiment: no flavor personas, no special Mafia model pinning.

Every agent's prompt is identical except for role - this is the "clean
model-vs-model comparison" configuration used for the first 9 games in the
library (games generated before the prompts/ split existed used this exact
wording; this file is the reproducible record of it going forward).
"""

from __future__ import annotations

from prompts import ExperimentConfig

COMMITMENT_INSTRUCTION = """\
Commit fully to your assigned role. Reason as cleverly as you can about what \
other players might know or be hiding. When your role or situation calls for \
it, actively try to deceive other players - including bluffing confidence \
you don't actually have, not just outright lying. Keep your voice and your \
public positions internally consistent from round to round: do not silently \
contradict something you said earlier unless you have an in-game reason to \
change your story, and if you do change it, address the change rather than \
pretending it didn't happen."""

ROLE_GOALS = {
    "Mafia": """\
You are the Mafia. Only you (and no one else) know this. Your goal is to \
avoid detection and eliminate Villagers each night until Mafia are no longer \
outnumbered. You are explicitly permitted and encouraged to lie: deny being \
Mafia, accuse innocent players, fabricate a claim (e.g. falsely claim to be \
the Detective or Doctor) if it helps you survive the vote. Don't just play \
defense - the Village's biggest advantage is the Detective's information, so \
actively work to blunt it: seed suspicion onto an innocent player early so \
you have a ready scapegoat before anyone claims anything; if someone claims \
Detective and names you, consider a counter-claim (claiming Detective \
yourself, or claiming Doctor and "confirming" the real Detective's target as \
guilty when they're actually innocent) rather than a flat denial, since flat \
denials are exactly what a guilty player would say. Watch for a player who \
goes quiet right after a no-kill night or who claims a role late - that's \
often the real Detective or Doctor timing a reveal, and getting them voted \
out (or killed) before they can act again is usually worth more to you than \
another kill on a random Villager.""",
    "Detective": """\
You are the Detective, on the Village team. Each night you investigate one \
player and privately learn whether they are Mafia. Your goal is to use that \
private knowledge to guide the Village to vote out the Mafia - but revealing \
your role also makes you the Mafia's next kill target, so weigh when (or \
whether) to claim it publicly versus acting on your knowledge more subtly. \
Claiming immediately after your very first result is rarely your strongest \
play: a single result is easy for the Mafia to muddy ("they're lying," "I'm \
actually the Detective and my result disagrees"), and once you claim, you \
won't survive the next night to investigate anyone else. A more patient \
Detective often waits to accumulate two or more results (or waits for \
strong circumstantial support from the discussion) before revealing - or \
reveals selectively (e.g. only naming who's innocent, not confirming who's \
guilty) to stay useful for longer. There is no fixed rule here: judge the \
actual game state each round, including how many rounds are likely left \
and how convinced the Village already seems.""",
    "Doctor": """\
You are the Doctor, on the Village team. Each night you may protect one \
player (including yourself) from elimination. Your goal is to help the \
Village survive and identify the Mafia through the day discussion and vote, \
using your protection choices strategically.""",
    "Villager": """\
You are a Villager, on the Village team, with no special night action. Your \
goal is to help identify and vote out the Mafia through discussion and \
reasoning alone. You may bluff confidence or float theories you're not fully \
sure of if it helps draw out the truth - just stay internally consistent.""",
}

CONFIG = ExperimentConfig(
    name="baseline",
    description=(
        "Clean model-vs-model comparison: identical prompt wording for "
        "every agent regardless of model, no flavor personas, no model "
        "pinned to any role. Any behavioral difference between agents is "
        "attributable to the model, not the prompt."
    ),
    commitment_instruction=COMMITMENT_INSTRUCTION,
    role_goals=ROLE_GOALS,
    use_personas=False,
    mafia_model_override=None,
)
