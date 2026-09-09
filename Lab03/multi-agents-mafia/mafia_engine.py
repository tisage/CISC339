"""Game engine for the Multi-Agent LLM Mafia demo (Lab 03).

Implements the fixed 6-player ruleset from DESIGN.md Section 2:
  1 Mafia, 1 Detective, 1 Doctor, 3 Villagers.

This module is pure game logic + prompt construction + OpenRouter calls.
It has no notion of "replay" or HTML rendering (see replay_utils.py) and no
CLI/budget-loop concerns (see generate_games.py) - it just knows how to run
one game to completion given an OpenAI-compatible client.
"""

from __future__ import annotations

import json
import random
import re
import string
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from openai import OpenAI

# ---------------------------------------------------------------------------
# Fixed roster (DESIGN.md Section 5) - 3 flagship + 3 budget, 4 vendors.
# ---------------------------------------------------------------------------

ROSTER: list[str] = [
    "openai/gpt-6-astra",
    "anthropic/claude-sonnet-5",
    "anthropic/claude-opus-5",
    "google/gemini-3.8-flash",
    "deepseek/deepseek-v4-flash-0731",
    "openai/gpt-5.4-mini",
]

ROLES: list[str] = ["Mafia", "Detective", "Doctor", "Villager", "Villager", "Villager"]

VILLAGE_ROLES = {"Detective", "Doctor", "Villager"}

ROUND_CAP = 8

DISPLAY_NAMES = [
    "Alice", "Bruno", "Carmen", "Devon", "Elena", "Farid",
    "Grace", "Hiro", "Ines", "Jonah",
]

# Optional flavor personas (DESIGN.md Section 5 update: kept as an opt-in,
# not the default). The default (GameConfig.use_personas=False) roster is
# still fully persona-free, for the strict cross-model comparison batch -
# every agent gets identical wording regardless of model, so any difference
# in play is attributable to the model, not the prompt. This pool exists for
# a separate, explicitly-requested "more entertaining to watch" batch, where
# variety in voice matters more than a clean model-vs-model comparison.
#
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

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


# ---------------------------------------------------------------------------
# Shared prompt fragments (DESIGN.md Section 5)
# ---------------------------------------------------------------------------

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

OUTPUT_CONTRACT = """\
Always respond with a single JSON object and nothing else (no markdown \
fences, no commentary outside the JSON). Every response must have exactly \
these two fields:
  "reasoning": your private reasoning, never shown to other players.
  "output": your public statement, vote, or action target as instructed \
below - this WILL be shown to other living players (except night actions, \
which stay private to you)."""


def build_system_prompt(
    agent_name: str,
    role: str,
    living_players: list[str],
    game_log: list[str],
    persona: Optional[str] = None,
) -> str:
    log_block = "\n".join(game_log) if game_log else "(Nothing has happened yet - this is the start of the game.)"
    persona_block = (
        f"Your personal voice/temperament (this is flavor, not a role hint - "
        f"it has nothing to do with whether you're Mafia or Village): "
        f"you are {persona}. Let this come through in how you talk, not just "
        f"what you say.\n\n"
        if persona
        else ""
    )
    return (
        f"You are playing a game of Mafia. You are {agent_name}.\n\n"
        f"Game rules: 6 players - 1 Mafia, 1 Detective, 1 Doctor, 3 Villagers. "
        f"Mafia wins if Mafia count >= remaining Villagers. Village wins if all "
        f"Mafia are eliminated. The game proceeds in rounds of: night actions, "
        f"a morning death announcement, a day discussion (each living player "
        f"speaks once), then a vote to eliminate one player.\n\n"
        f"Currently living players: {', '.join(living_players)}.\n\n"
        f"{ROLE_GOALS[role]}\n\n"
        f"{persona_block}"
        f"{COMMITMENT_INSTRUCTION}\n\n"
        f"Everything that has happened in the game so far, in order (this is "
        f"your memory - use it; do not contradict your own past statements or "
        f"votes without acknowledging the change):\n{log_block}\n\n"
        f"{OUTPUT_CONTRACT}"
    )


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Agent:
    agent_id: str
    display_name: str
    model: str
    role: str
    alive: bool = True
    # Detective-only: {target_agent_id: bool_is_mafia} accumulated across nights.
    private_knowledge: dict = field(default_factory=dict)
    # Persistent memory of everything this agent has witnessed or done across
    # the whole game so far (public events + its own private night actions/
    # reasoning), in chronological order. Re-sent on every call so the agent
    # has real cross-round memory instead of only seeing the current round's
    # partial transcript. Never includes other agents' private reasoning.
    game_log: list[str] = field(default_factory=list)
    # None in the default persona-free mode (GameConfig.use_personas=False).
    persona: Optional[str] = None


@dataclass
class GameConfig:
    round_cap: int = ROUND_CAP
    seed: Optional[int] = None
    # Opt-in flavor personas (see PERSONA_POOL) - off by default so the
    # baseline game library stays a clean model-vs-model comparison.
    use_personas: bool = False


@dataclass
class UsageTotals:
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0

    def add(self, usage_obj: Any) -> None:
        if usage_obj is None:
            return
        raw = usage_obj.model_dump() if hasattr(usage_obj, "model_dump") else {}
        self.total_tokens += raw.get("total_tokens") or 0
        self.estimated_cost_usd += raw.get("cost") or 0.0


class LLMCaller:
    """Thin wrapper so mafia_engine can be dry-run with a mock in tests."""

    def __init__(self, client: OpenAI):
        self.client = client

    def call(self, model: str, system_prompt: str, user_prompt: str) -> dict:
        # The client has a hard per-request timeout (see generate_games.py /
        # test_models.py). A timeout or transient network error here is
        # retried once - most such failures are one-off upstream hiccups,
        # not a reason to burn an entire in-progress game.
        last_exc: Optional[Exception] = None
        for attempt in range(2):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                )
                break
            except Exception as exc:  # noqa: BLE001 - network/timeout errors from the SDK
                last_exc = exc
                if attempt == 0:
                    continue
                raise
        else:
            raise last_exc  # pragma: no cover - unreachable, loop always breaks or raises
        raw_text = response.choices[0].message.content
        parsed = _safe_parse_json(raw_text)
        return {"parsed": parsed, "raw_text": raw_text, "usage": response.usage}


_INVALID_JSON_ESCAPE_RE = re.compile(r'\\(.)')


def _fix_invalid_json_escapes(text: str) -> str:
    """Repair backslash-escapes that are illegal in JSON but that models
    routinely emit anyway - most commonly \\' (escaping an apostrophe the
    way you would in Python/JS source, which JSON does not allow and does
    not need). JSON only permits \\" \\\\ \\/ \\b \\f \\n \\r \\t \\uXXXX;
    anything else after a backslash gets the backslash dropped, since that's
    almost always what the model meant (an unescaped literal character).
    This was added after a real Claude Sonnet 5 response failed to parse -
    and after all three retries also failed the same way - purely because
    of a stray \\' around a possessive ("Alice\\'s")."""

    def repl(match: "re.Match[str]") -> str:
        ch = match.group(1)
        return match.group(0) if ch in '"\\/bfnrtu' else ch

    return _INVALID_JSON_ESCAPE_RE.sub(repl, text)


def _try_json_loads(text: str) -> Optional[dict]:
    """json.loads that also retries once with invalid-escape repair, since
    that repair is occasionally too aggressive to try as the *first* attempt
    (it would mangle a genuinely-valid \\uXXXX or similar), so it's a
    fallback within each parse attempt rather than applied unconditionally
    up front."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    try:
        return json.loads(_fix_invalid_json_escapes(text))
    except (json.JSONDecodeError, TypeError):
        return None


def _safe_parse_json(raw_text: str) -> dict:
    """Parse the model's structured-output JSON, tolerating common deviations
    (markdown code fences, leading/trailing prose, illegal escape sequences
    like \\') some OpenRouter upstreams add even when
    response_format=json_object is requested.

    Falls back to treating the raw text as the public output only as a last
    resort - and even then, if the raw text still looks like an embedded
    {"reasoning": ..., "output": ...} object (parse just failed to extract
    it), we must not let it leak whole into the public-facing "output" field,
    since that would leak an agent's private reasoning as if it were their
    public statement.
    """
    text = (raw_text or "").strip()

    result = _try_json_loads(text)
    if result is not None:
        return result

    # Strip ```json ... ``` or ``` ... ``` fences and retry.
    if text.startswith("```"):
        stripped = text.strip("`")
        stripped = stripped[4:] if stripped.startswith("json") else stripped
        result = _try_json_loads(stripped.strip())
        if result is not None:
            return result

    # Try to extract the first {...} block from surrounding prose.
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        result = _try_json_loads(text[start : end + 1])
        if result is not None:
            return result

    # Genuine fallback: no JSON object could be recovered at all. Do not
    # surface this raw text as public "output" - it may contain private
    # reasoning verbatim (as happened when this safeguard was added). Mark it
    # as a parse failure instead so callers can decide how to handle it
    # (e.g. skip the turn) rather than silently leaking content.
    return {
        "reasoning": "",
        "output": "",
        "_parse_failed": True,
        "_raw_text": text,
    }


# ---------------------------------------------------------------------------
# Game class
# ---------------------------------------------------------------------------


class MafiaGame:
    def __init__(self, caller: LLMCaller, config: Optional[GameConfig] = None):
        self.caller = caller
        self.config = config or GameConfig()
        self.rng = random.Random(self.config.seed)
        self.usage = UsageTotals()
        self.agents: dict[str, Agent] = self._assign_agents()
        self.rounds: list[dict] = []
        self.winner: Optional[str] = None
        self.end_reason: Optional[str] = None
        self.ended_round: Optional[int] = None

    # -- setup ---------------------------------------------------------

    def _assign_agents(self) -> dict[str, Agent]:
        models = ROSTER.copy()
        roles = ROLES.copy()
        names = DISPLAY_NAMES[: len(models)]
        self.rng.shuffle(models)
        self.rng.shuffle(roles)

        personas: list[Optional[str]] = [None] * len(models)
        if self.config.use_personas:
            personas = self.rng.sample(PERSONA_POOL, k=len(models))

        agents = {}
        for i, (name, model, role, persona) in enumerate(zip(names, models, roles, personas)):
            agent_id = f"agent_{i + 1}"
            agents[agent_id] = Agent(
                agent_id=agent_id,
                display_name=name,
                model=model,
                role=role,
                persona=persona,
            )
        return agents

    def living_ids(self) -> list[str]:
        return [aid for aid, a in self.agents.items() if a.alive]

    def living_names(self) -> list[str]:
        return [self.agents[aid].display_name for aid in self.living_ids()]

    def name_to_id(self, name: str) -> Optional[str]:
        for aid, a in self.agents.items():
            if a.display_name.lower() == str(name).lower():
                return aid
        return None

    # -- LLM call helper -------------------------------------------------

    def _ask(self, agent: Agent, user_prompt: str, retries: int = 2) -> dict:
        system_prompt = build_system_prompt(
            agent.display_name,
            agent.role,
            self.living_names(),
            agent.game_log,
            persona=agent.persona,
        )

        last_parsed: dict = {}
        for attempt in range(retries + 1):
            result = self.caller.call(agent.model, system_prompt, user_prompt)
            self.usage.add(result["usage"])
            parsed = result["parsed"]
            last_parsed = parsed
            if not parsed.get("_parse_failed"):
                return {
                    "reasoning": str(parsed.get("reasoning", "")),
                    "output": parsed.get("output", ""),
                }
            # Structured output could not be recovered - retry rather than
            # ever surfacing raw (possibly private-reasoning-containing) text
            # as this agent's public output.
            if attempt < retries:
                continue

        # All retries exhausted: fail loudly rather than silently leaking
        # unparseable raw text into a "public" field.
        raise ValueError(
            f"Could not parse structured JSON output from {agent.model} "
            f"for agent {agent.agent_id} after {retries + 1} attempt(s). "
            f"Last raw text: {last_parsed.get('_raw_text', '')[:500]!r}"
        )

    # -- shared memory -------------------------------------------------

    def _broadcast_public(self, text: str) -> None:
        """Append a public event to every living agent's persistent memory."""
        for aid in self.living_ids():
            self.agents[aid].game_log.append(text)

    def _remember_private(self, agent: Agent, text: str) -> None:
        """Append a private event to only one agent's persistent memory."""
        agent.game_log.append(text)

    # -- game loop ---------------------------------------------------------

    def run(self) -> dict:
        for round_num in range(1, self.config.round_cap + 1):
            round_log = {"round": round_num}
            self._broadcast_public(f"--- Round {round_num}: night falls. ---")

            night_log = self._run_night()
            round_log["night"] = night_log

            dead_this_round = night_log.get("_eliminated_by_night")
            if dead_this_round:
                self.agents[dead_this_round].alive = False
                round_log["morning_announcement"] = (
                    f"{self.agents[dead_this_round].display_name} was found dead."
                )
            else:
                round_log["morning_announcement"] = "No one died last night."
            self._broadcast_public(
                f"--- Round {round_num} morning: {round_log['morning_announcement']} ---"
            )

            if self._check_winner(round_log, round_num):
                self.rounds.append(round_log)
                break

            # Round 1 is a discussion-only round: no vote. With zero
            # information on Day 1, a vote is a pure guess and routinely
            # lynches the Detective or Doctor before they get a second
            # night action - see DESIGN.md for the rationale (this mirrors
            # a common real-world "no Day 1 lynch" house rule).
            skip_vote = round_num == 1
            round_log["day_discussion"] = self._run_day_discussion(skip_vote=skip_vote)

            if self._check_winner(round_log, round_num):
                self.rounds.append(round_log)
                break

            if skip_vote:
                round_log["vote_skipped"] = True
                self._broadcast_public(
                    "No vote today - the village agreed to gather information "
                    "on Day 1 before eliminating anyone."
                )
                self.rounds.append(round_log)
                continue

            vote_log = self._run_vote()
            round_log["vote"] = vote_log
            if vote_log.get("eliminated"):
                self.agents[vote_log["eliminated"]].alive = False

            self.rounds.append(round_log)

            if self._check_winner(round_log, round_num):
                break
        else:
            self.winner = "Draw"
            self.end_reason = f"Round cap ({self.config.round_cap}) reached."
            self.ended_round = self.config.round_cap

        return self._serialize()

    # -- phases ---------------------------------------------------------

    def _run_night(self) -> dict:
        night: dict = {}

        doctor = self._find_living_by_role("Doctor")
        protected_id = None
        if doctor:
            result = self._ask(
                doctor,
                "It is night. Choose one living player to protect from "
                "elimination tonight (you may choose yourself). Living "
                f"players: {', '.join(self.living_names())}. Set \"output\" "
                "to the exact name of the player you protect.",
            )
            protected_id = self.name_to_id(result["output"]) or doctor.agent_id
            night["doctor_protect"] = {
                "agent": doctor.agent_id,
                "target": protected_id,
                "reasoning": result["reasoning"],
            }
            self._remember_private(
                doctor,
                f"[Your private night action] You chose to protect "
                f"{self.agents[protected_id].display_name}. Your reasoning: "
                f"{result['reasoning']}",
            )

        mafia = self._find_living_by_role("Mafia")
        killed_id = None
        if mafia:
            candidates = [aid for aid in self.living_ids() if aid != mafia.agent_id]
            result = self._ask(
                mafia,
                "It is night. Choose one living player (not yourself) to "
                f"eliminate. Living players: {', '.join(self.living_names())}. "
                "Set \"output\" to the exact name of your target.",
            )
            target_id = self.name_to_id(result["output"])
            if target_id not in candidates:
                target_id = self.rng.choice(candidates)
            killed_id = target_id
            night["mafia_kill"] = {
                "agent": mafia.agent_id,
                "target": killed_id,
                "reasoning": result["reasoning"],
            }
            self._remember_private(
                mafia,
                f"[Your private night action] You chose to attack "
                f"{self.agents[killed_id].display_name}. Your reasoning: "
                f"{result['reasoning']}",
            )

        detective = self._find_living_by_role("Detective")
        if detective:
            candidates = [aid for aid in self.living_ids() if aid != detective.agent_id]
            if candidates:
                result = self._ask(
                    detective,
                    "It is night. Choose one living player to investigate. "
                    "You will privately learn whether they are Mafia. Living "
                    f"players: {', '.join(self.living_names())}. Set "
                    "\"output\" to the exact name of the player you investigate.",
                )
                target_id = self.name_to_id(result["output"]) or self.rng.choice(candidates)
                is_mafia = self.agents[target_id].role == "Mafia"
                detective.private_knowledge[target_id] = is_mafia
                night["detective_check"] = {
                    "agent": detective.agent_id,
                    "target": target_id,
                    "result": is_mafia,
                    "reasoning": result["reasoning"],
                }
                verdict = "IS Mafia" if is_mafia else "is NOT Mafia"
                self._remember_private(
                    detective,
                    f"[Your private night action] You investigated "
                    f"{self.agents[target_id].display_name} and learned they "
                    f"{verdict}. Your reasoning: {result['reasoning']}",
                )

        eliminated_by_night = None
        if killed_id and killed_id != protected_id:
            eliminated_by_night = killed_id
        night["_eliminated_by_night"] = eliminated_by_night
        return night

    def _run_day_discussion(self, skip_vote: bool = False) -> list[dict]:
        speaking_order = self.living_ids()
        self.rng.shuffle(speaking_order)
        discussion = []

        self._broadcast_public("Day discussion begins.")
        vote_notice = (
            " There will be NO vote today - the village has agreed to spend "
            "Day 1 gathering information only, since a vote with zero "
            "information tends to guess wrong. Speak accordingly (you can "
            "still share claims, suspicions, or ask questions)."
            if skip_vote
            else ""
        )
        for aid in speaking_order:
            agent = self.agents[aid]
            result = self._ask(
                agent,
                "It is the day discussion phase. Make your public statement "
                "now (accuse, defend, share a claim, or ask a question)."
                f"{vote_notice} Set \"output\" to that statement.",
            )
            statement = str(result["output"])
            discussion.append(
                {
                    "agent": aid,
                    "public_statement": statement,
                    "private_reasoning": result["reasoning"],
                }
            )
            self._broadcast_public(f"{agent.display_name} said: {statement}")

        return discussion

    def _run_vote(self) -> dict:
        living = self.living_ids()
        ballots: dict[str, str] = {}
        justifications: dict[str, str] = {}

        self._broadcast_public("Voting begins.")
        for aid in living:
            agent = self.agents[aid]
            others = ", ".join(self.agents[o].display_name for o in living if o != aid)
            result = self._ask(
                agent,
                "It is the vote phase. Cast your vote to eliminate one "
                f"living player (not yourself): {others}. Set \"output\" to "
                "the exact name of who you vote to eliminate, and put your "
                "public justification for that vote in \"reasoning\" - note "
                "that in this phase only, \"reasoning\" WILL be announced to "
                "everyone along with your vote (real votes are cast out loud "
                "with a reason, not secretly), so do not put anything in it "
                "you don't want revealed.",
            )
            target_id = self.name_to_id(result["output"])
            if target_id not in living or target_id == aid:
                target_id = self.rng.choice([o for o in living if o != aid])
            ballots[aid] = target_id
            justifications[aid] = result["reasoning"]
            self._broadcast_public(
                f"{agent.display_name} voted to eliminate "
                f"{self.agents[target_id].display_name} "
                f"(reason given: {result['reasoning']})"
            )

        tally: dict[str, int] = {}
        for target in ballots.values():
            tally[target] = tally.get(target, 0) + 1
        max_votes = max(tally.values())
        top = [aid for aid, count in tally.items() if count == max_votes]

        tie_break_used = len(top) > 1
        eliminated = self.rng.choice(top) if tie_break_used else top[0]

        self._broadcast_public(
            f"Vote result: {self.agents[eliminated].display_name} was "
            f"eliminated{' (tie broken randomly)' if tie_break_used else ''}."
        )

        return {
            "ballots": ballots,
            "justifications": justifications,
            "eliminated": eliminated,
            "tie_break_used": tie_break_used,
        }

    # -- helpers ---------------------------------------------------------

    def _find_living_by_role(self, role: str) -> Optional[Agent]:
        for aid in self.living_ids():
            if self.agents[aid].role == role:
                return self.agents[aid]
        return None

    def _check_winner(self, round_log: dict, round_num: int) -> bool:
        living = [self.agents[aid] for aid in self.living_ids()]
        mafia_alive = sum(1 for a in living if a.role == "Mafia")
        village_alive = sum(1 for a in living if a.role in VILLAGE_ROLES)

        if mafia_alive == 0:
            self.winner = "Village"
            self.end_reason = "All Mafia eliminated."
            self.ended_round = round_num
            return True
        if mafia_alive >= village_alive:
            self.winner = "Mafia"
            self.end_reason = "Mafia count reached or exceeded remaining Villagers."
            self.ended_round = round_num
            return True
        return False

    def _serialize(self) -> dict:
        return {
            "game_id": f"mafia_{time.strftime('%Y-%m-%d')}_{uuid.uuid4().hex[:6]}",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "config": {
                "roles": {
                    "Mafia": 1,
                    "Detective": 1,
                    "Doctor": 1,
                    "Villager": 3,
                },
                "round_cap": self.config.round_cap,
                "use_personas": self.config.use_personas,
            },
            "agents": {
                aid: {
                    "display_name": a.display_name,
                    "model": a.model,
                    "role": a.role,
                    "persona": a.persona,
                }
                for aid, a in self.agents.items()
            },
            "rounds": self.rounds,
            "outcome": {
                "winner": self.winner,
                "ended_round": self.ended_round,
                "reason": self.end_reason,
            },
            "meta": {
                "total_tokens": self.usage.total_tokens,
                "estimated_cost_usd": round(self.usage.estimated_cost_usd, 6),
            },
        }


def run_one_game(
    client: OpenAI, seed: Optional[int] = None, use_personas: bool = False
) -> dict:
    caller = LLMCaller(client)
    game = MafiaGame(caller, GameConfig(seed=seed, use_personas=use_personas))
    return game.run()
