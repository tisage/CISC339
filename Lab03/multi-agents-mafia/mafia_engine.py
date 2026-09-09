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
the Detective or Doctor) if it helps you survive the vote.""",
    "Detective": """\
You are the Detective, on the Village team. Each night you investigate one \
player and privately learn whether they are Mafia. Your goal is to use that \
private knowledge to guide the Village to vote out the Mafia - but revealing \
your role also makes you a target, so weigh when (or whether) to claim it \
publicly versus acting on your knowledge more subtly.""",
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


def build_system_prompt(agent_name: str, role: str, living_players: list[str]) -> str:
    return (
        f"You are playing a game of Mafia. You are {agent_name}.\n\n"
        f"Game rules: 6 players - 1 Mafia, 1 Detective, 1 Doctor, 3 Villagers. "
        f"Mafia wins if Mafia count >= remaining Villagers. Village wins if all "
        f"Mafia are eliminated. The game proceeds in rounds of: night actions, "
        f"a morning death announcement, a day discussion (each living player "
        f"speaks once), then a vote to eliminate one player.\n\n"
        f"Currently living players: {', '.join(living_players)}.\n\n"
        f"{ROLE_GOALS[role]}\n\n"
        f"{COMMITMENT_INSTRUCTION}\n\n"
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


@dataclass
class GameConfig:
    round_cap: int = ROUND_CAP
    seed: Optional[int] = None


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
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        raw_text = response.choices[0].message.content
        parsed = _safe_parse_json(raw_text)
        return {"parsed": parsed, "raw_text": raw_text, "usage": response.usage}


def _safe_parse_json(raw_text: str) -> dict:
    """Parse the model's structured-output JSON, tolerating common deviations
    (markdown code fences, leading/trailing prose) some OpenRouter upstreams
    add even when response_format=json_object is requested.

    Falls back to treating the raw text as the public output only as a last
    resort - and even then, if the raw text still looks like an embedded
    {"reasoning": ..., "output": ...} object (parse just failed to extract
    it), we must not let it leak whole into the public-facing "output" field,
    since that would leak an agent's private reasoning as if it were their
    public statement.
    """
    text = (raw_text or "").strip()

    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass

    # Strip ```json ... ``` or ``` ... ``` fences and retry.
    if text.startswith("```"):
        stripped = text.strip("`")
        stripped = stripped[4:] if stripped.startswith("json") else stripped
        try:
            return json.loads(stripped.strip())
        except (json.JSONDecodeError, TypeError):
            pass

    # Try to extract the first {...} block from surrounding prose.
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except (json.JSONDecodeError, TypeError):
            pass

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

        agents = {}
        for i, (name, model, role) in enumerate(zip(names, models, roles)):
            agent_id = f"agent_{i + 1}"
            agents[agent_id] = Agent(
                agent_id=agent_id, display_name=name, model=model, role=role
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
            agent.display_name, agent.role, self.living_names()
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

    # -- game loop ---------------------------------------------------------

    def run(self) -> dict:
        for round_num in range(1, self.config.round_cap + 1):
            round_log = {"round": round_num}

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

            if self._check_winner(round_log, round_num):
                self.rounds.append(round_log)
                break

            round_log["day_discussion"] = self._run_day_discussion()

            if self._check_winner(round_log, round_num):
                self.rounds.append(round_log)
                break

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

        eliminated_by_night = None
        if killed_id and killed_id != protected_id:
            eliminated_by_night = killed_id
        night["_eliminated_by_night"] = eliminated_by_night
        return night

    def _run_day_discussion(self) -> list[dict]:
        speaking_order = self.living_ids()
        self.rng.shuffle(speaking_order)
        transcript_so_far: list[str] = []
        discussion = []

        for aid in speaking_order:
            agent = self.agents[aid]
            history_block = (
                "\n".join(transcript_so_far)
                if transcript_so_far
                else "(No statements yet this round.)"
            )
            result = self._ask(
                agent,
                "It is the day discussion phase. Here is what has been said "
                f"so far this round:\n{history_block}\n\nMake your public "
                "statement now (accuse, defend, share a claim, or ask a "
                "question). Set \"output\" to that statement.",
            )
            statement = str(result["output"])
            discussion.append(
                {
                    "agent": aid,
                    "public_statement": statement,
                    "private_reasoning": result["reasoning"],
                }
            )
            transcript_so_far.append(f"{agent.display_name}: {statement}")

        return discussion

    def _run_vote(self) -> dict:
        living = self.living_ids()
        ballots: dict[str, str] = {}
        justifications: dict[str, str] = {}

        for aid in living:
            agent = self.agents[aid]
            others = ", ".join(self.agents[o].display_name for o in living if o != aid)
            result = self._ask(
                agent,
                "It is the vote phase. Cast your vote to eliminate one "
                f"living player (not yourself): {others}. Set \"output\" to "
                "the exact name of who you vote to eliminate, and put your "
                "justification in \"reasoning\".",
            )
            target_id = self.name_to_id(result["output"])
            if target_id not in living or target_id == aid:
                target_id = self.rng.choice([o for o in living if o != aid])
            ballots[aid] = target_id
            justifications[aid] = result["reasoning"]

        tally: dict[str, int] = {}
        for target in ballots.values():
            tally[target] = tally.get(target, 0) + 1
        max_votes = max(tally.values())
        top = [aid for aid, count in tally.items() if count == max_votes]

        tie_break_used = len(top) > 1
        eliminated = self.rng.choice(top) if tie_break_used else top[0]

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
            },
            "agents": {
                aid: {
                    "display_name": a.display_name,
                    "model": a.model,
                    "role": a.role,
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


def run_one_game(client: OpenAI, seed: Optional[int] = None) -> dict:
    caller = LLMCaller(client)
    game = MafiaGame(caller, GameConfig(seed=seed))
    return game.run()
