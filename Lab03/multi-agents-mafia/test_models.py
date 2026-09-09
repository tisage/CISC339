"""Connectivity + pricing sanity check for the fixed Mafia demo model roster.

Run this before generate_games.py. It sends one minimal request to each of the
6 roster models and reports whether the slug is reachable, how long it took,
and what it actually cost. Model slugs on OpenRouter drift; this script is the
source of truth, not any hardcoded assumption about what's currently live.

Usage:
    uv run python test_models.py
"""

import os
import sys
import time

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

# .env lives at the repo root (CISC339/.env), not in this subfolder; search upward.
load_dotenv(find_dotenv(usecwd=True))

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Fixed roster: 3 flagship + 3 budget, see DESIGN.md Section 5.
ROSTER = {
    "flagship_openai": "openai/gpt-6-astra",
    "flagship_anthropic_sonnet": "anthropic/claude-sonnet-5",
    "flagship_anthropic_opus": "anthropic/claude-opus-5",
    "budget_google": "google/gemini-3.8-flash",
    "budget_deepseek": "deepseek/deepseek-v4-flash-0731",
    "budget_openai": "openai/gpt-5.4-mini",
}

PING_PROMPT = "Reply with exactly one word: OK"


def ping_model(client: OpenAI, slug: str) -> dict:
    start = time.monotonic()
    try:
        response = client.chat.completions.create(
            model=slug,
            messages=[{"role": "user", "content": PING_PROMPT}],
            max_tokens=10,
        )
    except Exception as exc:  # noqa: BLE001 - want to report any failure, not crash
        return {
            "ok": False,
            "latency_sec": time.monotonic() - start,
            "error": f"{type(exc).__name__}: {exc}",
        }

    latency = time.monotonic() - start
    reply_text = ""
    if response.choices:
        reply_text = (response.choices[0].message.content or "").strip()

    usage = getattr(response, "usage", None)
    cost_usd = None
    if usage is not None:
        raw_usage = usage.model_dump() if hasattr(usage, "model_dump") else {}
        cost_usd = raw_usage.get("cost")

    return {
        "ok": True,
        "latency_sec": latency,
        "reply": reply_text,
        "cost_usd": cost_usd,
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
    }


def main() -> int:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set (check your .env file).")
        return 1

    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key, timeout=30.0)

    print(f"Testing {len(ROSTER)} roster models against OpenRouter...\n")

    results = {}
    any_failed = False
    total_cost = 0.0

    for label, slug in ROSTER.items():
        print(f"  [{label:28s}] {slug:35s} ... ", end="", flush=True)
        result = ping_model(client, slug)
        results[label] = {"slug": slug, **result}

        if not result["ok"]:
            any_failed = True
            print(f"FAILED ({result['error']})")
            continue

        cost = result.get("cost_usd")
        total_cost += cost or 0.0
        cost_str = f"${cost:.6f}" if cost is not None else "n/a"
        print(
            f"ok  reply={result['reply']!r}  "
            f"latency={result['latency_sec']:.2f}s  cost={cost_str}"
        )

    print()
    print(f"Total cost of this test run: ${total_cost:.6f}")

    if any_failed:
        print("\nOne or more roster models FAILED. Fix the slug(s) in this file")
        print("(and in mafia_engine.py once written) before running generate_games.py.")
        return 1

    print("\nAll 6 roster models reachable. Safe to proceed to generate_games.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
