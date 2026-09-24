"""Minimal generator for synthetic LogEntry records (Lab04).

Provisional / small-scale only - see log_scenarios.py's module docstring.
This validates the pipeline (schema, LLM call pattern, output quality),
not a finished dataset for a specific exercise, since the LLM chapter
that will consume this data isn't designed yet.

Usage:
    uv run --project . python Lab04/data_gen/generate_logs.py --n 40

Run from the repo root so the root .venv/pyproject.toml is used.
"""

from __future__ import annotations

import argparse
import asyncio
import random
import sys
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from schemas import LogEntry
from log_scenarios import LOG_SCENARIOS, LogScenario

load_dotenv(find_dotenv(usecwd=True))

MODEL = "gpt-5-mini"
BATCH_SIZE = 10
DEFAULT_OUT = Path(__file__).resolve().parent / "data" / "logs.jsonl"
MAX_RETRIES = 3


class LogBatch(BaseModel):
    logs: list[LogEntry]


def sample_scenarios(n: int, rng: random.Random) -> list[LogScenario]:
    weights = [s.weight for s in LOG_SCENARIOS]
    return rng.choices(LOG_SCENARIOS, weights=weights, k=n)


def build_prompt(scenarios: list[LogScenario], start_id: int) -> str:
    lines = [
        f"Generate exactly {len(scenarios)} synthetic SOC/security log "
        "lines for a cybersecurity teaching dataset (not real logs, no "
        "real IPs/hostnames/usernames). Each line below has ALREADY been "
        "assigned a scenario - do not change label or attack_type from "
        "what's specified. Write a realistic, varied raw_text for each "
        "(vary timestamp format, source, exact wording across records - "
        "do not reuse the same template).\n",
        "Records to generate:",
    ]
    for i, s in enumerate(scenarios):
        lines.append(
            f"{i + 1}. log_id=log_{start_id + i:06d} | scenario: {s.name} - "
            f"{s.description}\n"
            f"   label={s.label} attack_type={s.attack_type}"
        )
    return "\n".join(lines)


async def generate_one_batch(
    client: AsyncOpenAI, scenarios: list[LogScenario], start_id: int, semaphore: asyncio.Semaphore
) -> list[LogEntry] | None:
    prompt = build_prompt(scenarios, start_id)

    parsed = None
    async with semaphore:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = await client.chat.completions.parse(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    response_format=LogBatch,
                    max_completion_tokens=4000,
                    reasoning_effort="minimal",
                )
                parsed = resp.choices[0].message.parsed
                if parsed is None:
                    raise ValueError(
                        f"parse returned None, finish_reason="
                        f"{resp.choices[0].finish_reason}"
                    )
                break
            except Exception as e:
                print(f"  [batch start_id={start_id}] attempt {attempt}/"
                      f"{MAX_RETRIES} failed: {e}")
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(2 ** attempt)

    if parsed is None:
        print(f"  [batch start_id={start_id}] giving up after {MAX_RETRIES} attempts")
        return None
    return parsed.logs


async def run(n: int, out_path: Path, concurrency: int, seed: int) -> None:
    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(concurrency)
    rng = random.Random(seed)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_batches = (n + BATCH_SIZE - 1) // BATCH_SIZE
    tasks = []
    next_id = 1
    for i in range(n_batches):
        size = min(BATCH_SIZE, n - i * BATCH_SIZE)
        scenarios = sample_scenarios(size, rng)
        tasks.append(generate_one_batch(client, scenarios, next_id, semaphore))
        next_id += size

    total_written = 0
    with open(out_path, "w") as f:
        for coro in asyncio.as_completed(tasks):
            logs = await coro
            if logs is None:
                continue
            for log in logs:
                f.write(log.model_dump_json() + "\n")
            f.flush()
            total_written += len(logs)
            print(f"  progress: {total_written}/{n}")

    print(f"\nDone. Wrote {total_written} records to {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=40,
                        help="Number of log lines to generate (default: 40).")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help=f"Output JSONL path (default: {DEFAULT_OUT}).")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="Max concurrent API calls (default: 4).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for scenario sampling (default: 0).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run(args.n, args.out, args.concurrency, args.seed))
