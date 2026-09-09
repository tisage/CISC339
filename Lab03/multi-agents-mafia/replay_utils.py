"""HTML replay rendering for the Multi-Agent LLM Mafia demo (Lab 03).

Renders a pre-generated games/*.json game log as HTML for display in a
Jupyter/Colab notebook via IPython.display.HTML. No API calls, no JS
frameworks, no external assets, no ipywidgets.Accordion (Colab's custom
widget manager has had compatibility issues with it) - just <details>/
<summary> and inline CSS so it renders identically in Colab, local Jupyter,
and a plain nbconvert --to html export.

See DESIGN.md Section 6 for the full design rationale, including why role/
model identity stays masked until a single end-of-game reveal block instead
of being revealed per elimination.

Two display modes:
    mode="full" (default) - renders the entire game as one linear stack of
        collapsed <details> blocks, ending in the reveal block. Use this to
        move through a game quickly while narrating over it.
    mode="step" - renders one round at a time with a "Next round" button,
        for the 1-2 games per class you want to linger on. Since this is
        pure local rendering of an already-generated file (no API calls,
        no latency, no cost), pausing is free to offer.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Optional

from IPython.display import HTML, display

# Stable per-agent color palette, assigned by agent_id order within a game
# (not by role - role stays masked until reveal).
AGENT_COLORS = [
    "#4C72B0",  # blue
    "#DD8452",  # orange
    "#55A868",  # green
    "#C44E52",  # red
    "#8172B2",  # purple
    "#937860",  # brown
]

ROLE_BADGE_COLORS = {
    "Mafia": "#C44E52",
    "Detective": "#4C72B0",
    "Doctor": "#55A868",
    "Villager": "#8C8C8C",
}


def load_game(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def _esc(text: object) -> str:
    return html.escape(str(text if text is not None else ""))


def _agent_color(game: dict, agent_id: str) -> str:
    ids = list(game["agents"].keys())
    idx = ids.index(agent_id) if agent_id in ids else 0
    return AGENT_COLORS[idx % len(AGENT_COLORS)]


def _agent_name(game: dict, agent_id: str) -> str:
    agent = game["agents"].get(agent_id)
    return agent["display_name"] if agent else agent_id


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def _style_block() -> str:
    return """
<style>
.mafia-demo { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
              max-width: 820px; }
.mafia-demo .agent-roster { display: flex; flex-wrap: wrap; gap: 8px; margin: 8px 0 16px; }
.mafia-demo .agent-chip { padding: 4px 10px; border-radius: 14px; color: white;
                          font-size: 0.85em; font-weight: 600; }
.mafia-demo details.round-block { border: 1px solid rgba(128,128,128,0.35);
                                   border-radius: 8px; margin: 10px 0; padding: 6px 12px; }
.mafia-demo details.round-block > summary { cursor: pointer; font-weight: 600;
                                             padding: 6px 2px; list-style: revert; }
.mafia-demo .announcement { font-style: italic; opacity: 0.85; margin: 8px 0; }
.mafia-demo .chat-bubble { border-left: 4px solid #999; background: rgba(128,128,128,0.08);
                           border-radius: 0 8px 8px 0; padding: 8px 12px; margin: 6px 0; }
.mafia-demo .chat-bubble .speaker { font-weight: 700; margin-right: 6px; }
.mafia-demo .vote-table { border-collapse: collapse; margin: 8px 0; font-size: 0.9em; }
.mafia-demo .vote-table td, .mafia-demo .vote-table th { border: 1px solid rgba(128,128,128,0.3);
                           padding: 4px 10px; text-align: left; }
.mafia-demo .reveal-block { border: 2px solid #C44E52; border-radius: 8px;
                            padding: 14px; margin-top: 18px; background: rgba(196,78,82,0.06); }
.mafia-demo .reveal-block h3 { margin-top: 0; }
.mafia-demo .role-badge { display: inline-block; padding: 2px 8px; border-radius: 10px;
                          color: white; font-size: 0.8em; font-weight: 700; margin-left: 6px; }
.mafia-demo .eliminated-tag { opacity: 0.6; text-decoration: line-through; }
</style>
"""


def _render_roster(game: dict) -> str:
    chips = []
    for aid, agent in game["agents"].items():
        color = _agent_color(game, aid)
        chips.append(
            f'<span class="agent-chip" style="background:{color}">'
            f'{_esc(agent["display_name"])}</span>'
        )
    n_players = len(game["agents"])
    return (
        f'<p>{n_players}-player game &middot; roles hidden until the end &mdash; '
        f"watch the public discussion and see if you can guess who's who.</p>"
        f'<div class="agent-roster">{"".join(chips)}</div>'
    )


def _render_night_summary(game: dict, night: dict) -> str:
    # Night actions are private; we only ever surface the public consequence
    # (the morning announcement), not who did what. This function exists in
    # case an instructor wants a debug/instructor-only view - not used by
    # the default student-facing render.
    return ""


def _render_day_discussion(game: dict, discussion: list[dict]) -> str:
    bubbles = []
    for entry in discussion:
        aid = entry["agent"]
        color = _agent_color(game, aid)
        name = _agent_name(game, aid)
        statement = _esc(entry["public_statement"])
        bubbles.append(
            f'<div class="chat-bubble" style="border-left-color:{color}">'
            f'<span class="speaker" style="color:{color}">{_esc(name)}:</span>'
            f"{statement}</div>"
        )
    return "".join(bubbles)


def _render_vote(game: dict, vote: dict) -> str:
    rows = []
    for voter_id, target_id in vote["ballots"].items():
        rows.append(
            f"<tr><td>{_esc(_agent_name(game, voter_id))}</td>"
            f"<td>&rarr; {_esc(_agent_name(game, target_id))}</td></tr>"
        )
    table = f'<table class="vote-table"><tr><th>Voter</th><th>Vote</th></tr>{"".join(rows)}</table>'

    eliminated_name = _agent_name(game, vote["eliminated"])
    tie_note = " (tie, broken randomly)" if vote.get("tie_break_used") else ""
    return (
        f"{table}"
        f"<p><strong>{_esc(eliminated_name)}</strong> was voted out{tie_note}.</p>"
    )


def _round_summary_line(game: dict, round_log: dict) -> str:
    n_statements = len(round_log.get("day_discussion", []))
    vote = round_log.get("vote")
    parts = [f"Round {round_log['round']}", round_log["morning_announcement"]]
    if n_statements:
        parts.append(f"{n_statements} statements")
    if vote:
        parts.append(f"voted out {_agent_name(game, vote['eliminated'])}")
    elif round_log.get("vote_skipped"):
        parts.append("no vote (Day 1 information-gathering only)")
    return " &mdash; ".join(_esc(p) if i else p for i, p in enumerate(parts))


def _render_round(game: dict, round_log: dict) -> str:
    summary = _round_summary_line(game, round_log)
    body = [f'<p class="announcement">{_esc(round_log["morning_announcement"])}</p>']

    if "day_discussion" in round_log:
        body.append(_render_day_discussion(game, round_log["day_discussion"]))
    if "vote" in round_log:
        body.append(_render_vote(game, round_log["vote"]))

    return (
        f'<details class="round-block" open>'
        f"<summary>{summary}</summary>"
        f'{"".join(body)}'
        f"</details>"
    )


def _render_reveal(game: dict) -> str:
    outcome = game["outcome"]
    rows = []
    for aid, agent in game["agents"].items():
        color = _agent_color(game, aid)
        role = agent["role"]
        badge_color = ROLE_BADGE_COLORS.get(role, "#8C8C8C")
        rows.append(
            f'<li><span class="agent-chip" style="background:{color}">'
            f'{_esc(agent["display_name"])}</span> '
            f'<span class="role-badge" style="background:{badge_color}">{_esc(role)}</span> '
            f'&mdash; <code>{_esc(agent["model"])}</code></li>'
        )

    meta = game.get("meta", {})
    cost_line = ""
    if "estimated_cost_usd" in meta:
        cost_line = f'<p style="opacity:0.7;font-size:0.85em;">Generation cost: ${meta["estimated_cost_usd"]:.4f} &middot; {meta.get("total_tokens", "?")} tokens</p>'

    return (
        '<div class="reveal-block">'
        f"<h3>Reveal &mdash; {_esc(outcome['winner'])} wins "
        f"(round {_esc(outcome['ended_round'])})</h3>"
        f"<p>{_esc(outcome['reason'])}</p>"
        f'<ul style="list-style:none;padding-left:0;">{"".join(rows)}</ul>'
        f"{cost_line}"
        "</div>"
    )


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def render_full(game: dict) -> str:
    """Render the entire game as one linear stack of collapsed round blocks,
    ending in the reveal block. Default mode - use for quick narration."""
    parts = [_style_block(), '<div class="mafia-demo">']
    parts.append(_render_roster(game))
    for round_log in game["rounds"]:
        parts.append(_render_round(game, round_log))
    parts.append(_render_reveal(game))
    parts.append("</div>")
    return "".join(parts)


def show_full(game_or_path: dict | str | Path) -> None:
    game = game_or_path if isinstance(game_or_path, dict) else load_game(game_or_path)
    display(HTML(render_full(game)))


class StepReplay:
    """Step-by-step replay controller for lingering on 1-2 games in class.

    Usage in a notebook cell:
        replay = StepReplay(game)
        replay.show()   # renders roster + an ipywidgets "Next round" button

    Each click renders one more round below the button, in place, ending
    with the reveal block on the final click. This is a pacing control for
    the instructor (pause where you want to talk), not a "guess now" prompt
    baked into the notebook - see DESIGN.md Section 6/9 for that distinction.
    """

    def __init__(self, game: dict):
        self.game = game
        self._round_idx = 0

    def show(self) -> None:
        import ipywidgets as widgets

        header = widgets.HTML(_style_block() + '<div class="mafia-demo">' + _render_roster(self.game) + "</div>")
        output = widgets.Output()
        button = widgets.Button(description="Next round ▶", button_style="primary")

        def on_click(_):
            with output:
                if self._round_idx < len(self.game["rounds"]):
                    round_log = self.game["rounds"][self._round_idx]
                    display(HTML('<div class="mafia-demo">' + _render_round(self.game, round_log) + "</div>"))
                    self._round_idx += 1
                    if self._round_idx == len(self.game["rounds"]):
                        button.description = "Reveal ▶"
                elif self._round_idx == len(self.game["rounds"]):
                    display(HTML('<div class="mafia-demo">' + _render_reveal(self.game) + "</div>"))
                    button.disabled = True
                    button.description = "Done"

        button.on_click(on_click)
        display(header, button, output)


# ---------------------------------------------------------------------------
# Game library browser
# ---------------------------------------------------------------------------


def list_games(games_dir: str | Path = "games") -> list[Path]:
    """List available game JSON files directly inside games_dir, sorted by
    filename (which sorts by generation date since game_id embeds a date).
    Does not recurse into subfolders like games/_discarded_*/ - those are
    intentionally excluded archives, not part of the active library."""
    games_dir = Path(games_dir)
    if not games_dir.exists():
        return []
    return sorted(p for p in games_dir.glob("*.json") if p.is_file())


def _game_summary_label(path: Path) -> str:
    """One-line label for a game, safe to show before the reveal - outcome
    (winner/round count/cost) is meta-game info, not a role/identity spoiler."""
    try:
        game = load_game(path)
    except (json.JSONDecodeError, OSError):
        return f"{path.name}  (could not read)"

    outcome = game.get("outcome", {})
    meta = game.get("meta", {})
    n_rounds = len(game.get("rounds", []))
    winner = outcome.get("winner", "?")
    cost = meta.get("estimated_cost_usd")
    cost_str = f"${cost:.2f}" if isinstance(cost, (int, float)) else "?"
    return f"{path.stem}  —  {n_rounds} round(s), {winner} won, {cost_str}"


class GameBrowser:
    """Dropdown picker over every game in games_dir - avoids hardcoding a
    single GAME_FILE path in the notebook as the library grows from a
    handful of games to 20-30. Selecting an entry re-renders in place.

    Usage in a notebook cell:
        browser = GameBrowser("games")
        browser.show()
    """

    def __init__(self, games_dir: str | Path = "games", mode: str = "full"):
        self.games_dir = Path(games_dir)
        self.mode = mode
        self.paths = list_games(self.games_dir)

    def show(self) -> None:
        import ipywidgets as widgets

        if not self.paths:
            display(HTML(f"<p><em>No games found in {self.games_dir}/.</em></p>"))
            return

        options = [(_game_summary_label(p), str(p)) for p in self.paths]
        dropdown = widgets.Dropdown(
            options=options, description="Game:", layout=widgets.Layout(width="600px")
        )
        output = widgets.Output()

        def on_change(change):
            if change["name"] != "value" or change["new"] is None:
                return
            output.clear_output()
            with output:
                game = load_game(change["new"])
                if self.mode == "step":
                    StepReplay(game).show()
                else:
                    display(HTML(render_full(game)))

        dropdown.observe(on_change, names="value")
        display(dropdown, output)
        on_change({"name": "value", "new": dropdown.value})
