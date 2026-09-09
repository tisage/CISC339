# Design Doc: Multi-Agent LLM Mafia Demo

Status: draft for review
Owner: T. W (Lab 03 — Knowledge)
Last updated: 2026-09-09

## 1. Purpose & Placement in Lab 03

Lab 03 currently teaches **knowledge representation via propositional logic and model
checking** (`sympy.logic`), using the Cluedo game as the running example: rules are
hand-encoded, and a model checker proves what must be true.

This demo is the natural next step: **what happens when the "reasoner" is an LLM agent
instead of a symbolic model checker?** In Mafia, no one hand-codes the inference rules —
each agent must track "who knows what," detect contradictions, and reason about hidden
state using natural language alone. It's a concrete bridge from GOFAI (Good Old-Fashioned
AI) to modern LLM-based multi-agent reasoning.

Teaching goals surfaced by the demo:
- **AI agents**: each LLM instance is a persistent agent with a role, memory, and goal.
- **Knowledge inference**: agents must infer hidden roles from public statements + voting
  behavior — same underlying problem as the Cluedo section, no formal logic this time.
- **Deception**: the Mafia agent is explicitly prompted to lie; students watch it happen
  and compare an agent's private reasoning to its public statement.
- **Multi-agent collaboration**: Villagers/Detective/Doctor must coordinate through public
  channels only, with no shared ground truth.

Deliverable: a **standalone notebook**, separate from `Lab_03_Knowledge.ipynb`, so it can
be run as a self-contained in-class demo without touching the homework notebook.

## 2. Game Rules (deliberately simplified)

Standard references converge on 1 mafia per ~4 players, and the most commonly cited
balanced small setup is:

> **6 players: 1 Mafia, 1 Detective, 1 Doctor, 3 Villagers**

(This is the widely-cited "balanced 6-player Mafia setup" — see Sources.) We adopt this
exactly, with no house rules, no multiple mafia, no extra power roles (no Jester, no
Vigilante, etc.). Reasons:
- It's small enough that an LLM can track full game state in-context without external
  memory/RAG.
- It's simple enough that students can learn the rules in under a minute and focus on
  watching the *reasoning*, not decoding a rulebook.
- 1 Mafia keeps the "who is lying" question binary and legible for a first-time audience.

**Roles**
| Role | Team | Night action | Win condition (team) |
|---|---|---|---|
| Mafia (x1) | Mafia | Choose one player to eliminate | Mafia count ≥ remaining Villagers |
| Detective (x1) | Village | Investigate one player, learns `is_mafia: bool` | All Mafia eliminated |
| Doctor (x1) | Village | Protect one player from elimination (can self-protect) | All Mafia eliminated |
| Villager (x3) | Village | none | All Mafia eliminated |

**Round structure** (repeats until a win condition is met or a round cap is hit):
1. **Night phase** — private, sequential, simultaneous-in-theory but resolved in fixed
   order (Doctor protects → Mafia kills → Detective investigates). Only the acting agent
   sees the prompt; only that agent's action is written to their own private memory.
2. **Morning announcement** — who died last night (or "no one," if the Doctor saved the
   target). This is the only new public information of the round.
3. **Day discussion** — fixed number of statement turns (e.g., each living agent speaks
   once, in randomized order) where every agent sees all prior public statements this game
   and produces one public statement. Agents may lie, accuse, deflect, or share claimed
   information (e.g., a Detective may "claim" to be the Detective — true or false).
4. **Vote phase** — each living agent votes for one player to eliminate, with one short
   public justification. Majority (plurality with ties broken randomly, disclosed as such)
   is eliminated and their role is revealed publicly.
5. Check win condition; else increment round and repeat.

Round cap: hard-stop at e.g. 8 rounds → declare a draw if neither win condition triggers,
to bound API cost per game in generation.

## 3. Architecture: Generate/Replay Split

This is the core design decision and directly answers the "how do I demo live without API
flakiness/cost/latency" problem.

```
generate_games.py     → offline batch runner, calls OpenRouter, writes games/*.json
                         (run ahead of class, on your machine, at your pace)

Lab_03_Mafia_Demo.ipynb → in-class notebook, reads games/*.json, renders the replay
                         (zero API calls, zero latency, zero risk of a bad demo)
```

Why this over live calls in class:
- **Cost control**: with cheap models (gpt-5-mini, claude-haiku-4.5, gemini-flash-tier),
  a 6-agent, ~5–8 round game costs low tens of cents. Batch-generating 20–30 games ahead
  of time to build a curated library still fits comfortably in the ~$100 budget, with
  headroom for students who want to generate their own.
- **Reliability**: no risk of a rate limit, timeout, or a boring/incoherent game happening
  live in front of a class.
- **Curation**: you can generate many games, skim the outcomes, and hand-pick 2–3 that
  make good teaching material (e.g., one where the Mafia is caught by a contradiction, one
  where the Mafia wins by successfully framing the Detective).
- **Reproducibility**: the exact same game can be replayed in two different sections, or
  paused mid-lecture to ask students "who do you think is the Mafia right now?" before
  revealing the next round.

Students with their own OpenRouter key can run `generate_games.py` themselves to produce
new games — this is the natural optional/bonus extension mentioned in Section 6.

## 4. Data Model

One JSON file per game, saved to `games/<game_id>.json`. Design principle: **separate
private reasoning from public statements** — this is the single most important field
split for the teaching goal, since it lets us show students the public transcript first
(let them guess), then reveal the private reasoning/roles (the "aha").

```jsonc
{
  "game_id": "mafia_2026-09-09_001",
  "created_at": "2026-09-09T14:32:00Z",
  "config": {
    "roles": {"Mafia": 1, "Detective": 1, "Doctor": 1, "Villager": 3},
    "round_cap": 8
  },
  "agents": {
    "agent_1": {"display_name": "Alice", "model": "openai/gpt-5-mini", "role": "Villager", "persona": "..."},
    "agent_2": {"display_name": "Bruno", "model": "anthropic/claude-haiku-4.5", "role": "Mafia", "persona": "..."},
    "...": "..."
  },
  "rounds": [
    {
      "round": 1,
      "night": {
        "doctor_protect": {"agent": "agent_3", "target": "agent_3", "reasoning": "private text"},
        "mafia_kill": {"agent": "agent_2", "target": "agent_5", "reasoning": "private text"},
        "detective_check": {"agent": "agent_4", "target": "agent_2", "result": true, "reasoning": "private text"}
      },
      "morning_announcement": "agent_5 was found dead.",
      "day_discussion": [
        {"agent": "agent_1", "public_statement": "...", "private_reasoning": "..."},
        {"agent": "agent_4", "public_statement": "...", "private_reasoning": "..."}
      ],
      "vote": {
        "ballots": {"agent_1": "agent_2", "agent_3": "agent_2", "...": "..."},
        "eliminated": "agent_2",
        "tie_break_used": false
      }
    }
  ],
  "outcome": {
    "winner": "Village",
    "ended_round": 3,
    "reason": "All Mafia eliminated."
  },
  "meta": {
    "total_tokens": 18342,
    "estimated_cost_usd": 0.06,
    "generation_duration_sec": 47
  }
}
```

Notes:
- `private_reasoning` is never shown to other agents — enforced by the generation script
  only ever including other agents' `public_statement`/public actions in a given agent's
  prompt context, never their `private_reasoning` or true `role` (except the acting
  agent's own).
- Dead agents stop appearing in `day_discussion`/`vote` for subsequent rounds but their
  full history remains in the file.
- Per-round elimination does **not** carry a role reveal field — `role`/`model` live only
  in the top-level `agents` block, which the generation script always has full access to
  but `replay_utils.py` withholds from rendering until the final reveal (see §6). This is a
  rendering-layer decision, not a data-model one: the JSON always contains full ground
  truth; only the notebook's display logic staggers what it shows and when.
- `meta` block is for your own cost tracking across a batch of generated games — not shown
  to students by default, but harmless to show as a "here's what this cost" aside.

## 5. Prompting Design (brief)

Each agent gets a system prompt built from a shared template plus role-specific and
persona-specific fragments:
- **Shared rules block**: full game rules, current living players, its own role, its own
  private knowledge so far (night action results if any).
- **Role-specific goal**: Village roles are told to win by identifying the Mafia; the
  Mafia agent is explicitly instructed it may lie, deflect suspicion, and fabricate claims
  (e.g. false-claim being the Detective) — this is the "deception" teaching hook, made
  visible later via `private_reasoning` vs `public_statement`.
- **Persona fragment**: a short flavor string (e.g., "blunt and suspicious," "quiet and
  analytical," "overly trusting") randomized per game per agent, independent of role. This
  adds variety across games and avoids every Villager sounding identical — it also means
  persona is *not* a tell for role, which is intentional.
- **Output contract**: every agent call requests structured output (e.g., JSON with
  `reasoning` and `statement`/`action` fields) so the generation script can reliably split
  private vs public content. This also doubles as a mini teaching point about structured
  output / tool-call-style constraints on LLMs, if you want to mention it.

Model assignment: shuffle a pool of 6 cheap OpenRouter models across the 6 agent slots each
game — **openai/gpt-5-mini, anthropic/claude-haiku-4.5, google/gemini-flash, plus three
more from different vendors** (e.g. meta-llama/llama-3.3-70b-instruct or a Llama flash-tier
model, mistralai/mistral-small, and deepseek/deepseek-chat or qwen/qwen-2.5-72b-instruct —
pick whichever has the cheapest current OpenRouter pricing at generation time), so role and
model are decoupled — students see that "which AI plays the liar" changes game to game,
reinforcing that the deception is a *role-conditioned prompting* behavior, not something
specific to one vendor's model. Deliberately spanning vendors (not just picking a second
cheap tier from the same 3 labs) makes this cross-vendor point land harder.

## 6. Colab/Notebook Replay UI

Goal: rich enough to be engaging, without needing a real frontend. Plain
`IPython.display.HTML` + a bit of CSS is sufficient; avoid `ipywidgets.Accordion` as the
primary mechanism — it has had compatibility issues with Colab's custom widget manager —
and use native HTML `<details>/<summary>` instead, which renders identically in Colab,
Jupyter, and nbviewer with zero dependencies.

**Pacing**: this is a *demo*, not an interactive classroom game — Lab 03 has other
material to cover and the instructor should not need to solicit guesses from students each
round. But since replay is pure local rendering of an already-generated JSON file (no API
calls, no latency, no risk), a pause point costs nothing to offer — it's just "don't render
past round N yet," not a real wait. So both modes ship:
- **`mode="full"`** — renders the entire game as a single linear stack of collapsed
  `<details>` blocks in one call, ending in the reveal block. Default mode; use this to
  move through a game quickly while narrating over it.
- **`mode="step"`** — renders one round at a time; a "Next round ▶" `ipywidgets.Button`
  (or just re-running the cell) advances. This is a pacing control for the instructor, not
  a prompt for students — there's no built-in "guess now" text. Use it on the 1–2 games
  you want to linger on and narrate round-by-round; use `mode="full"` for the rest.

**Handling long conversations** (the specific concern raised):
- **Collapse by default, at round granularity.** Render each round as an HTML `<details>`
  block (closed by default) with a one-line `<summary>` (e.g. "Round 2 — agent_5 found
  dead — 4 statements, 1 vote"). This alone solves 90% of the length problem: a full game
  transcript becomes a stack of ~5–8 collapsed strips instead of a wall of text, which the
  instructor can open one at a time while narrating.
- **Role identity is masked until game end.** Public statements render under agent display
  names only throughout the game — elimination/death announcements name the agent but do
  *not* reveal their role or true model; all roles + models are revealed together in a
  final "reveal" block once a win condition (or the round cap) is hit. This keeps suspicion
  alive across the whole game (a dead agent might still turn out to have been a Villager,
  not the Mafia everyone assumed) and concentrates the "aha" into one satisfying reveal
  moment rather than spreading small reveals across rounds.
- **Visual encoding**: a colored left-border or badge per agent (stable color per agent
  for the duration of one game, assigned from a small fixed palette) makes it easy to
  visually track who's who across rounds without re-reading names every time — chat-bubble
  style rather than a raw text dump.
- **Non-negotiable for the medium**: no JS frameworks, no external CSS/JS assets, no
  widget-manager-dependent components — everything must render from a single `HTML()`
  call so it works identically in Colab, local Jupyter, and a plain `nbconvert --to html`
  export for students without a Colab account.

## 7. File Layout

```
Lab03/multi-agents-mafia/
├── DESIGN.md                    (this file)
├── Lab_03_Mafia_Demo.ipynb      (standalone in-class notebook; new deliverable)
├── generate_games.py            (offline batch generator; calls OpenRouter)
├── mafia_engine.py              (shared: game state machine, prompt templates, schema)
├── replay_utils.py              (shared: HTML rendering helpers imported by the notebook)
├── games/                       (generated .json game logs; curated subset checked in)
│   └── mafia_2026-xx-xx_00N.json
├── .env.example                 (OPENROUTER_API_KEY=...)
└── requirements.txt             (openai sdk, python-dotenv; no framework deps)
```

`mafia_engine.py` / `replay_utils.py` are factored out of the notebook so the notebook
stays readable as a *demo* (short cells, mostly calling into these modules) rather than a
wall of implementation code — consistent with the style of the other Lab teaching
notebooks (e.g. Lab 06/07 keep heavy logic in `.py` helper files alongside the notebook).

## 8. Cost & Scope Guardrails

- Target: 20–30 generated games for a curated library, well within the ~$100 OpenRouter
  budget at mini/haiku/flash-tier pricing (est. low tens of cents per game at 6 agents ×
  ~6 rounds × structured-output calls).
- `generate_games.py` logs running total spend estimate and supports a `--max-games` /
  `--budget-usd` stop condition so a batch run can't silently overrun.
- Round cap (8) bounds worst-case per-game cost regardless of model behavior.
- Student-facing bonus: point students at `generate_games.py --n 1` with their own key as
  an optional homework extension, not a requirement.

## 9. Open Questions — Resolved

1. **Persona variety**: short adjective-phrase persona (as drafted). Richer backstory
   personas (name, claimed occupation, etc.) risk leaking role information and would cost
   extra system-prompt tokens for no teaching benefit — the point is that persona is *not*
   a tell for role.
2. **"Predict, then reveal" interaction**: no built-in student-facing prompt text (no
   markdown "pause and guess" cells) — that stays verbal, at the instructor's discretion.
   A pacing control *is* built in, though: since replay is local rendering of a
   pre-generated JSON (no API calls, no cost, no latency), a "Next round ▶" pause point is
   free to offer — see `mode="step"` in the updated §6. The distinction is "instructor can
   pause if they want to" vs. "notebook tells students to guess," and only the former is
   in scope.
3. **Model pool**: fill out to 6 via vendor diversification, not more tiers from the same
   3 labs — see the updated §5 for the specific pool (OpenAI, Anthropic, Google, plus Llama/
   Mistral/DeepSeek-or-Qwen).
4. **Death reveal timing**: roles are masked until game end, revealed all at once in a
   final reveal block — not immediately on elimination. Keeps suspicion alive across the
   whole game (a dead agent isn't necessarily confirmed-Mafia) and concentrates the "aha"
   into one moment. §4 and §6 above have been updated to reflect this (no more
   `eliminated_role_revealed` per round; UI masks role/model until the end).

## Sources

- [Mafia Game for 6 Players — Roles, Setup & Rules](https://mafiarole.com/how-to-play/6-players)
- [Mafia Game for 7 Players — Roles, Setup & Rules](https://mafiarole.com/how-to-play/7-players)
- [Mafia Game Roles Explained: All 8 Roles & Powers](https://findtheimposter.com/blog/mafia-game-roles)
- [Mafia (party game): Strategies, Roles, and The Thrill of The Game](https://medium.com/@greenrighthands/mafia-party-game-strategies-roles-and-the-thrill-of-the-game-be477a15808f)
