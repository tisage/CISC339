"""Offline batch generator for the Multi-Agent LLM Mafia demo (Lab 03).

Runs one or more complete Mafia games against the fixed 6-model roster
(mafia_engine.ROSTER) and writes each as games/<game_id>.json. Meant to be
run ahead of class, at your own pace - see DESIGN.md Section 3.

Usage:
    uv run python generate_games.py --n 1                       # one game, sanity check
    uv run python generate_games.py --n 20 --budget-usd 5.0     # batch, with a stop condition
    uv run python generate_games.py --n 30 --max-games 30       # cap by count instead

Run test_models.py first - this script assumes the roster is reachable and
does not re-verify connectivity itself.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mafia_engine as me

load_dotenv(find_dotenv(usecwd=True))

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
GAMES_DIR = Path(__file__).resolve().parent / "games"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n", type=int, default=1, help="Number of games to attempt (default: 1)."
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=None,
        help="Hard cap on games generated this run, independent of --n.",
    )
    parser.add_argument(
        "--budget-usd",
        type=float,
        default=None,
        help="Stop starting new games once cumulative estimated cost reaches this.",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=None,
        help="If set, games are seeded seed_start, seed_start+1, ... for reproducibility.",
    )
    parser.add_argument(
        "--personas",
        action="store_true",
        help=(
            "Assign each agent a random flavor persona (voice/temperament, "
            "not correlated with role) for a more entertaining watch. "
            "Default is off, which keeps every agent's prompt identical "
            "except for role - the clean model-vs-model comparison mode."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set (check your .env file).")
        return 1

    GAMES_DIR.mkdir(exist_ok=True)

    # A hard per-request timeout matters here: with no timeout, a single slow
    # or hung upstream model response blocks the whole batch run indefinitely
    # (observed in practice - one call took well over 10x the typical latency
    # with no error and no progress). 90s comfortably covers normal reasoning
    # latency for this roster while still failing fast on a genuine hang.
    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key, timeout=90.0)

    n_games = args.n
    if args.max_games is not None:
        n_games = min(n_games, args.max_games)

    print(f"Generating up to {n_games} game(s)...")
    print(f"Personas: {'ON (flavor voices)' if args.personas else 'OFF (clean model comparison)'}")
    if args.budget_usd is not None:
        print(f"Budget cap: ${args.budget_usd:.2f}")

    total_cost = 0.0
    games_written = 0
    games_failed = 0

    for i in range(n_games):
        if args.budget_usd is not None and total_cost >= args.budget_usd:
            print(
                f"\nBudget cap (${args.budget_usd:.2f}) reached after "
                f"{games_written} game(s). Stopping."
            )
            break

        seed = (args.seed_start + i) if args.seed_start is not None else None
        print(f"\n--- Game {i + 1}/{n_games} (seed={seed}) ---")
        start = time.monotonic()

        try:
            result = me.run_one_game(client, seed=seed, use_personas=args.personas)
        except Exception:
            games_failed += 1
            print("FAILED with exception:")
            traceback.print_exc()
            print("Continuing to next game (this one is not written to disk).")
            continue

        elapsed = time.monotonic() - start
        result["meta"]["generation_duration_sec"] = round(elapsed, 1)

        cost = result["meta"]["estimated_cost_usd"]
        total_cost += cost

        out_path = GAMES_DIR / f"{result['game_id']}.json"
        out_path.write_text(_to_json(result))
        games_written += 1

        outcome = result["outcome"]
        print(
            f"  -> {out_path.name} | winner={outcome['winner']} "
            f"in {outcome['ended_round']} round(s) | "
            f"cost=${cost:.4f} | {elapsed:.1f}s | "
            f"running total=${total_cost:.4f}"
        )

    print(
        f"\nDone. {games_written} game(s) written to {GAMES_DIR}/, "
        f"{games_failed} failed. Total estimated cost: ${total_cost:.4f}"
    )
    return 0 if games_failed == 0 else 1


def _to_json(result: dict) -> str:
    import json

    return json.dumps(result, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    sys.exit(main())
