"""HTML replay rendering for the Multi-Agent LLM Mafia demo (Lab 03).

Renders a pre-generated games/*.json game log as a card-based, stage-by-stage
replay for display in a Jupyter/Colab notebook via IPython.display.HTML and
ipywidgets. No JS frameworks, no external assets, no ipywidgets.Accordion
(Colab's custom widget manager has had compatibility issues with it) - just
plain HTML/CSS (a hidden checkbox + CSS sibling selector drives the
show/hide-roles toggle, so it works with zero JS) plus ipywidgets.Button for
stage advancement.

See DESIGN.md Section 6 for the full design rationale, including why role/
model identity stays masked until a single end-of-game reveal block instead
of being revealed per elimination.

Two display modes:
    mode="full" (default) - renders the entire game as one linear stack of
        stage cards, ending in the reveal block. Use this to move through a
        game quickly while narrating over it.
    mode="step" - reveals one stage at a time (night -> morning announcement
        -> day discussion -> vote/skip -> ... -> reveal) behind a "Next ▶"
        button, for the 1-2 games per class you want to linger on. Since
        this is pure local rendering of an already-generated file (no API
        calls, no latency, no cost), pausing is free to offer.
"""

from __future__ import annotations

import html
import json
import uuid
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


def _agent_initial(name: str) -> str:
    return (name[:1] or "?").upper()


def _dead_by_round_end(game: dict, round_idx: int) -> set[str]:
    """Agent ids eliminated (night or vote) at or before the given 0-based
    round index. Used so cards can show a "died this round" / "already
    dead" state as the replay advances round by round."""
    dead: set[str] = set()
    for r in game["rounds"][: round_idx + 1]:
        night_dead = r.get("night", {}).get("_eliminated_by_night")
        if night_dead:
            dead.add(night_dead)
        vote = r.get("vote")
        if vote and vote.get("eliminated"):
            dead.add(vote["eliminated"])
    return dead


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------


def _style_block(instance_id: str) -> str:
    # Roles are hidden by default via CSS; checking the "reveal" checkbox
    # (a sibling of everything else via the wrapper) flips a CSS selector
    # that shows every .role-badge and swaps each card's night-icon overlay
    # off. Pure CSS, no JS, so it survives a static HTML export too (the
    # checkbox will just be inert there, defaulting to hidden roles).
    return f"""
<style>
#{instance_id} {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
              max-width: 900px; }}
#{instance_id} .reveal-toggle-row {{ margin: 4px 0 14px; font-size: 0.9em; }}
#{instance_id} .reveal-toggle-row label {{ cursor: pointer; user-select: none; }}
#{instance_id} .role-reveal-checkbox {{ display: none; }}
#{instance_id} .role-badge {{ visibility: hidden; }}
#{instance_id} .role-reveal-checkbox:checked ~ * .role-badge {{ visibility: visible; }}
#{instance_id} .role-reveal-checkbox:checked ~ .reveal-toggle-row .toggle-label-hidden {{ display: none; }}
#{instance_id} .role-reveal-checkbox:not(:checked) ~ .reveal-toggle-row .toggle-label-shown {{ display: none; }}

#{instance_id} .player-circle {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 10px 0 18px;
                                  justify-content: center; }}
#{instance_id} .player-card {{ width: 108px; padding: 10px 8px; border-radius: 10px;
                                border: 2px solid rgba(128,128,128,0.25); text-align: center;
                                transition: opacity 0.2s, filter 0.2s; position: relative; }}
#{instance_id} .player-card.dead {{ opacity: 0.45; filter: grayscale(70%); }}
#{instance_id} .player-avatar {{ width: 44px; height: 44px; border-radius: 50%; color: white;
                                  display: flex; align-items: center; justify-content: center;
                                  font-weight: 700; font-size: 1.1em; margin: 0 auto 6px; }}
#{instance_id} .player-name {{ font-weight: 700; font-size: 0.9em; }}
#{instance_id} .player-card .role-badge {{ display: inline-block; margin-top: 4px; padding: 1px 7px;
                                            border-radius: 8px; color: white; font-size: 0.72em;
                                            font-weight: 700; }}
#{instance_id} .player-card .dead-tag {{ display: block; font-size: 0.72em; margin-top: 3px;
                                          opacity: 0.85; font-style: italic; }}

#{instance_id} .stage-card {{ border: 1px solid rgba(128,128,128,0.3); border-radius: 10px;
                               margin: 10px 0; padding: 12px 16px; }}
#{instance_id} .stage-card.night {{ background: rgba(76,114,176,0.07); }}
#{instance_id} .stage-card.morning {{ background: rgba(128,128,128,0.06); }}
#{instance_id} .stage-card.vote {{ background: rgba(196,78,82,0.05); }}
#{instance_id} .stage-title {{ font-weight: 700; margin: 0 0 8px; font-size: 0.95em;
                                text-transform: uppercase; letter-spacing: 0.03em; opacity: 0.7; }}
#{instance_id} .announcement {{ font-style: italic; margin: 4px 0; }}
#{instance_id} .chat-bubble {{ border-left: 4px solid #999; background: rgba(128,128,128,0.08);
                                border-radius: 0 8px 8px 0; padding: 8px 12px; margin: 6px 0; }}
#{instance_id} .chat-bubble .speaker {{ font-weight: 700; margin-right: 6px; }}
#{instance_id} .vote-table {{ border-collapse: collapse; margin: 8px 0; font-size: 0.9em; }}
#{instance_id} .vote-table td, #{instance_id} .vote-table th {{ border: 1px solid rgba(128,128,128,0.3);
                           padding: 4px 10px; text-align: left; }}
#{instance_id} .reveal-block {{ border: 2px solid #C44E52; border-radius: 10px;
                            padding: 14px 16px; margin-top: 18px; background: rgba(196,78,82,0.06); }}
#{instance_id} .reveal-block h3 {{ margin-top: 0; }}
#{instance_id} .night-action-row {{ margin: 3px 0; }}
#{instance_id} .round-heading {{ font-size: 1.05em; font-weight: 700; margin: 18px 0 6px;
                                  border-bottom: 2px solid rgba(128,128,128,0.25); padding-bottom: 4px; }}
</style>
"""


def _reveal_toggle_html(instance_id: str) -> str:
    checkbox_id = f"{instance_id}-reveal-toggle"
    return (
        f'<input type="checkbox" id="{checkbox_id}" class="role-reveal-checkbox">'
        f'<div class="reveal-toggle-row">'
        f'<label for="{checkbox_id}">'
        f'<span class="toggle-label-hidden">Roles hidden — click to reveal roles</span>'
        f'<span class="toggle-label-shown">Roles shown — click to hide roles</span>'
        f"</label></div>"
    )


# ---------------------------------------------------------------------------
# Player cards (the "seated in a circle" roster)
# ---------------------------------------------------------------------------


def _render_player_cards(game: dict, dead_ids: set[str]) -> str:
    cards = []
    for aid, agent in game["agents"].items():
        color = _agent_color(game, aid)
        role = agent["role"]
        badge_color = ROLE_BADGE_COLORS.get(role, "#8C8C8C")
        is_dead = aid in dead_ids
        dead_class = " dead" if is_dead else ""
        dead_tag = '<span class="dead-tag">eliminated</span>' if is_dead else ""
        cards.append(
            f'<div class="player-card{dead_class}">'
            f'<div class="player-avatar" style="background:{color}">'
            f'{_esc(_agent_initial(agent["display_name"]))}</div>'
            f'<div class="player-name">{_esc(agent["display_name"])}</div>'
            f'<span class="role-badge" style="background:{badge_color}">{_esc(role)}</span>'
            f"{dead_tag}"
            f"</div>"
        )
    return f'<div class="player-circle">{"".join(cards)}</div>'


# ---------------------------------------------------------------------------
# Stage renderers
# ---------------------------------------------------------------------------


def _render_night_stage(round_num: int, night: dict) -> str:
    """Night actions are shown as anonymous outcomes only ('someone was
    protected/attacked/investigated') - naming the actor here would leak
    their role before the end-of-game reveal, since only one role performs
    each action. The full attributed action log is shown later in the
    reveal stage alongside real identities."""
    lines = []
    if "doctor_protect" in night:
        lines.append('<div class="night-action-row">🛡️ Someone was protected by the Doctor.</div>')
    if "mafia_kill" in night:
        lines.append('<div class="night-action-row">🔪 The Mafia chose a target.</div>')
    if "detective_check" in night:
        lines.append('<div class="night-action-row">🔍 The Detective investigated someone.</div>')
    body = "".join(lines) or "<p>No night actions this round.</p>"
    return (
        f'<div class="stage-card night">'
        f'<p class="stage-title">Round {round_num} — Night</p>'
        f"{body}</div>"
    )


def _render_morning_stage(round_num: int, morning_announcement: str) -> str:
    return (
        f'<div class="stage-card morning">'
        f'<p class="stage-title">Round {round_num} — Morning</p>'
        f'<p class="announcement">{_esc(morning_announcement)}</p>'
        f"</div>"
    )


def _render_discussion_stage(game: dict, round_num: int, discussion: list[dict]) -> str:
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
    body = "".join(bubbles) or "<p>No statements this round.</p>"
    return (
        f'<div class="stage-card discussion">'
        f'<p class="stage-title">Round {round_num} — Day discussion</p>'
        f"{body}</div>"
    )


def _render_vote_stage(game: dict, round_num: int, round_log: dict) -> str:
    if round_log.get("vote_skipped"):
        return (
            f'<div class="stage-card vote">'
            f'<p class="stage-title">Round {round_num} — Vote</p>'
            f"<p><em>No vote today — the village agreed to gather information "
            f"on Day 1 before eliminating anyone.</em></p></div>"
        )

    vote = round_log["vote"]
    rows = []
    for voter_id, target_id in vote["ballots"].items():
        justification = vote.get("justifications", {}).get(voter_id, "")
        rows.append(
            f"<tr><td>{_esc(_agent_name(game, voter_id))}</td>"
            f"<td>&rarr; {_esc(_agent_name(game, target_id))}</td>"
            f"<td>{_esc(justification)}</td></tr>"
        )
    table = (
        '<table class="vote-table"><tr><th>Voter</th><th>Vote</th><th>Reason given</th></tr>'
        f'{"".join(rows)}</table>'
    )
    eliminated_name = _agent_name(game, vote["eliminated"])
    tie_note = " (tie, broken randomly)" if vote.get("tie_break_used") else ""
    return (
        f'<div class="stage-card vote">'
        f'<p class="stage-title">Round {round_num} — Vote</p>'
        f"{table}"
        f"<p><strong>{_esc(eliminated_name)}</strong> was voted out{tie_note}.</p>"
        f"</div>"
    )


def _render_reveal_stage(game: dict) -> str:
    outcome = game["outcome"]
    rows = []
    for aid, agent in game["agents"].items():
        color = _agent_color(game, aid)
        role = agent["role"]
        badge_color = ROLE_BADGE_COLORS.get(role, "#8C8C8C")
        rows.append(
            f'<li style="margin:4px 0;">'
            f'<span class="player-avatar" style="background:{color};width:26px;height:26px;'
            f'display:inline-flex;font-size:0.8em;border-radius:50%;color:white;'
            f'align-items:center;justify-content:center;vertical-align:middle;margin-right:6px;">'
            f'{_esc(_agent_initial(agent["display_name"]))}</span>'
            f'{_esc(agent["display_name"])} '
            f'<span class="role-badge" style="visibility:visible;background:{badge_color};'
            f'display:inline-block;padding:1px 7px;border-radius:8px;color:white;'
            f'font-size:0.8em;font-weight:700;">{_esc(role)}</span> '
            f'&mdash; <code>{_esc(agent["model"])}</code></li>'
        )

    action_lines = _render_attributed_night_actions(game)

    meta = game.get("meta", {})
    cost_line = ""
    if "estimated_cost_usd" in meta:
        cost_line = (
            f'<p style="opacity:0.7;font-size:0.85em;">Generation cost: '
            f'${meta["estimated_cost_usd"]:.4f} &middot; {meta.get("total_tokens", "?")} tokens</p>'
        )

    return (
        '<div class="reveal-block">'
        f"<h3>Reveal &mdash; {_esc(outcome['winner'])} wins "
        f"(round {_esc(outcome['ended_round'])})</h3>"
        f"<p>{_esc(outcome['reason'])}</p>"
        f'<ul style="list-style:none;padding-left:0;">{"".join(rows)}</ul>'
        f"<p style=\"margin-top:12px;font-weight:700;\">Full night-action log:</p>"
        f"{action_lines}"
        f"{cost_line}"
        "</div>"
    )


def _render_attributed_night_actions(game: dict) -> str:
    """Now that roles are revealed, show who actually did what each night -
    the identity-attributed counterpart to the anonymous per-round night
    stage shown during the live replay."""
    lines = []
    for r in game["rounds"]:
        night = r.get("night", {})
        parts = []
        if "doctor_protect" in night:
            v = night["doctor_protect"]
            parts.append(
                f'{_agent_name(game, v["agent"])} (Doctor) protected '
                f'{_agent_name(game, v["target"])}'
            )
        if "mafia_kill" in night:
            v = night["mafia_kill"]
            parts.append(
                f'{_agent_name(game, v["agent"])} (Mafia) attacked '
                f'{_agent_name(game, v["target"])}'
            )
        if "detective_check" in night:
            v = night["detective_check"]
            verdict = "IS Mafia" if v["result"] else "is NOT Mafia"
            parts.append(
                f'{_agent_name(game, v["agent"])} (Detective) investigated '
                f'{_agent_name(game, v["target"])} &rarr; {verdict}'
            )
        if parts:
            lines.append(f"<li>Round {r['round']}: " + "; ".join(_esc(p) for p in parts) + "</li>")
    return f'<ul style="font-size:0.9em;">{"".join(lines)}</ul>' if lines else "<p><em>No night actions recorded.</em></p>"


# ---------------------------------------------------------------------------
# Stage sequencing (shared by render_full and StepReplay)
# ---------------------------------------------------------------------------


def _build_stage_htmls(game: dict) -> list[str]:
    """Flatten the whole game into an ordered list of stage-card HTML
    snippets: for each round, night -> morning -> discussion -> vote, then
    a final reveal stage. This list is what both the linear "full" render
    and the stage-by-stage StepReplay walk through - one source of truth
    for sequencing so the two modes can't drift apart.

    IMPORTANT for StepReplay: each stage snippet here is displayed into a
    *separate* ipywidgets.Output widget, not inside the same DOM subtree as
    the reveal-toggle checkbox (which lives in the player-cards widget). The
    checkbox's CSS sibling-selector reveal trick therefore only affects
    .role-badge elements inside the player cards - it does NOT reach into
    per-stage HTML. This works today only because no stage before the final
    reveal stage contains a .role-badge; the reveal stage hardcodes its
    badges to visibility:visible instead of relying on the toggle. If you
    add a .role-badge to any earlier stage, it will not respect the
    show/hide toggle in StepReplay (it will in render_full, since there
    everything shares one DOM subtree)."""
    stages: list[str] = []
    for round_log in game["rounds"]:
        round_num = round_log["round"]
        stages.append(f'<div class="round-heading">Round {round_num}</div>')
        stages.append(_render_night_stage(round_num, round_log.get("night", {})))
        stages.append(_render_morning_stage(round_num, round_log["morning_announcement"]))
        if "day_discussion" in round_log:
            stages.append(_render_discussion_stage(game, round_num, round_log["day_discussion"]))
        if "vote" in round_log or round_log.get("vote_skipped"):
            stages.append(_render_vote_stage(game, round_num, round_log))
    stages.append(_render_reveal_stage(game))
    return stages


def _dead_ids_before_stage(game: dict, stage_idx: int) -> set[str]:
    """Which agents should render as 'eliminated' on the player cards once
    the replay has reached stage_idx (0-based, into the flattened stage
    list) - i.e. based on rounds fully resolved before this point, plus a
    round's own night-kill once its morning/discussion/vote stages are
    showing (a player doesn't un-die if you're mid-round)."""
    # Recompute by walking the same flattening logic, tracking round
    # boundaries, so this stays in sync with _build_stage_htmls by
    # construction rather than by parallel bookkeeping.
    dead: set[str] = set()
    idx = 0
    for round_log in game["rounds"]:
        round_num = round_log["round"]
        n_sub_stages = 1  # round heading
        n_sub_stages += 1  # night
        n_sub_stages += 1  # morning
        has_discussion = "day_discussion" in round_log
        has_vote_block = "vote" in round_log or round_log.get("vote_skipped")
        if has_discussion:
            n_sub_stages += 1
        if has_vote_block:
            n_sub_stages += 1

        round_start_idx = idx
        round_end_idx = idx + n_sub_stages - 1

        if stage_idx >= round_start_idx + 2:  # past the morning-announcement stage
            night_dead = round_log.get("night", {}).get("_eliminated_by_night")
            if night_dead:
                dead.add(night_dead)
        if stage_idx > round_end_idx:
            vote = round_log.get("vote")
            if vote and vote.get("eliminated"):
                dead.add(vote["eliminated"])

        idx += n_sub_stages

    return dead


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def render_full(game: dict) -> str:
    """Render the entire game as a linear stack of stage cards (night,
    morning, discussion, vote, per round) ending in the reveal block.
    Default mode - use for quick narration."""
    instance_id = f"mafia-{uuid.uuid4().hex[:8]}"
    stages = _build_stage_htmls(game)
    final_dead = _dead_by_round_end(game, len(game["rounds"]) - 1) if game["rounds"] else set()

    parts = [_style_block(instance_id), f'<div id="{instance_id}">']
    parts.append(_reveal_toggle_html(instance_id))
    parts.append(_render_player_cards(game, final_dead))
    parts.extend(stages)
    parts.append("</div>")
    return "".join(parts)


def show_full(game_or_path: dict | str | Path) -> None:
    game = game_or_path if isinstance(game_or_path, dict) else load_game(game_or_path)
    display(HTML(render_full(game)))


class StepReplay:
    """Stage-by-stage replay controller for lingering on 1-2 games in class.

    Usage in a notebook cell:
        replay = StepReplay(game)
        replay.show()   # renders player cards + a "Next ▶" button

    Each click reveals the next stage (night -> morning -> discussion ->
    vote, per round, then the final reveal) in place below the button. This
    is a pacing control for the instructor (pause where you want to talk),
    not a "guess now" prompt baked into the notebook - see DESIGN.md Section
    6/9 for that distinction. The role-reveal checkbox at the top is
    independent of stage progression - an instructor can peek at roles
    early for their own prep without that affecting what stage is showing.
    """

    def __init__(self, game: dict):
        self.game = game
        self.instance_id = f"mafia-step-{uuid.uuid4().hex[:8]}"
        self._stages = _build_stage_htmls(game)
        self._stage_idx = 0

    def show(self) -> None:
        import ipywidgets as widgets

        # The <style> block is scoped to #instance_id and only needs to be
        # injected once per widget tree - it lives in cards_area (the first
        # HTML widget shown) rather than being repeated into every stage
        # update or a separate, never-displayed header widget.
        def render_current_cards() -> str:
            dead = _dead_ids_before_stage(self.game, self._stage_idx)
            return (
                _style_block(self.instance_id)
                + f'<div id="{self.instance_id}">'
                + _reveal_toggle_html(self.instance_id)
                + _render_player_cards(self.game, dead)
                + "</div>"
            )

        cards_area = widgets.HTML(render_current_cards())
        output = widgets.Output()
        button = widgets.Button(description="Next ▶", button_style="primary")

        def on_click(_):
            # Show only the current stage - clear what was there before
            # rather than appending, so the display doesn't grow unbounded
            # as the instructor clicks through a game.
            output.clear_output(wait=True)
            with output:
                if self._stage_idx < len(self._stages):
                    display(HTML(f'<div id="{self.instance_id}">' + self._stages[self._stage_idx] + "</div>"))
                    self._stage_idx += 1
                    cards_area.value = render_current_cards()
                    if self._stage_idx == len(self._stages):
                        button.description = "Done"
                        button.disabled = True
                    elif "reveal-block" in self._stages[self._stage_idx]:
                        button.description = "Reveal ▶"

        button.on_click(on_click)
        display(cards_area, button, output)


# ---------------------------------------------------------------------------
# Game library browser
# ---------------------------------------------------------------------------


def list_games(games_dir: str | Path = "games") -> list[Path]:
    """List available game JSON files directly inside games_dir, sorted by
    filename (which sorts by generation date since game_id embeds a date).
    Does not recurse into subfolders like games/test/ - those are
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
            # wait=True avoids a visible flash of emptiness, but the key
            # fix is clearing at all: StepReplay.show() displays its own
            # cards/button/output widgets into this output area on every
            # call, so without clearing first, switching games N times
            # left N stacked sets of "Next ▶" buttons behind.
            output.clear_output(wait=True)
            with output:
                game = load_game(change["new"])
                if self.mode == "step":
                    StepReplay(game).show()
                else:
                    display(HTML(render_full(game)))

        dropdown.observe(on_change, names="value")
        display(dropdown, output)
        on_change({"name": "value", "new": dropdown.value})
