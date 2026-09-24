"""Offline batch generator for synthetic NetworkFlow records (Lab04).

Statistical control (label ratio, firewall behavior, alert rates, which
attack patterns appear) is done in Python via scenarios.py BEFORE any LLM
call - each record's scenario, firewall_action, and alert_triggered are
pre-sampled, then handed to the LLM as constraints. The LLM's only job is
filling in realistic per-record detail (IPs, exact byte/packet counts,
timestamps) consistent with those constraints. This keeps the aggregate
distribution exactly what generate_flows.py asks for, regardless of what
the model would have produced left to its own judgment - see scenarios.py's
module docstring for why this split exists and how to extend it.

Concurrently calls the OpenAI API in small batches, validates/corrects
each batch with validators.py, and appends results to a JSONL file -
resumable if interrupted (re-running with the same --out just continues
past however many records are already there).

Usage:
    uv run --project . python Lab04/data_gen/generate_flows.py --n 100 --dry-run
    uv run --project . python Lab04/data_gen/generate_flows.py --n 40000 --concurrency 8

Run from the repo root so the root .venv/pyproject.toml is used. See
Lab04/data_gen/README.md for environment setup.
"""

from __future__ import annotations

import argparse
import asyncio
import random
import sys
import time
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from schemas import NetworkFlow
from scenarios import ScenarioInstance, sample_scenario_instance
from validators import validate_batch

load_dotenv(find_dotenv(usecwd=True))

MODEL = "gpt-5-mini"
BATCH_SIZE = 20            # flows requested per API call
ATTACK_RATIO = 0.05        # fraction of records that are attacks (overall,
                           # not per-scenario - see scenarios.py for that)
DEFAULT_OUT = Path(__file__).resolve().parent / "data" / "flows.jsonl"
MAX_RETRIES = 3            # a lost batch silently skews the scenario mix
                           # for the whole run, so retry transient
                           # failures (network blips, rate limits) rather
                           # than just dropping the batch


class FlowBatch(BaseModel):
    flows: list[NetworkFlow]


def _fmt_range(lo, hi) -> str:
    return f"{lo}-{hi}"


def build_prompt(instances: list[ScenarioInstance], start_id: int) -> str:
    """Renders each pre-sampled ScenarioInstance as one numbered
    constraint block. The LLM never chooses label/attack_type/
    firewall_action/alert_triggered/traffic_level - those are already
    decided; it only invents realistic IPs/ports/timestamps/exact numeric
    values within the given ranges.
    """
    lines = [
        f"Generate exactly {len(instances)} synthetic network flow records "
        "for a cybersecurity teaching dataset (not real traffic). Each "
        "record below has ALREADY been assigned a scenario and certain "
        "field values - do not change label, attack_type, firewall_action, "
        "or alert_triggered from what's specified. For traffic_level, use "
        "\"Abnormal\" only if the exact numeric values you choose for this "
        "record would genuinely look abnormal by volume/rate (very high "
        "packet rate with small packets, very large byte counts, or very "
        "long duration); otherwise use \"Normal\" - this may legitimately "
        "differ from whether the record is actually an attack, since "
        "stealthy attacks are designed to look numerically unremarkable. "
        "Your job is to invent realistic src_ip/dst_ip/dst_port/protocol/"
        "timestamp and exact numeric values for duration_ms/bytes_sent/"
        "bytes_recv/packet_count. The given ranges for these fields are "
        "HARD bounds - never generate a value outside the stated "
        "min-max range for a field, even if a slightly out-of-range value "
        "would seem more realistic to you.\n",
        "General rules:",
        "- src_ip is always a private-range IPv4 address (10.x.x.x, "
        "172.16-31.x.x, or 192.168.x.x) - the internal host.",
        "- dst_ip may be private-range (internal-to-internal) or a "
        "plausible public IPv4, as fits the scenario.",
        "- timestamps should be plausible ISO-8601 strings, varied across "
        "a few days.",
        "- flow_id must be sequential: "
        f"flow_{start_id:06d}, flow_{start_id + 1:06d}, ...\n",
        "Records to generate:",
    ]
    for i, inst in enumerate(instances):
        s = inst.scenario
        protocol_opts = ", ".join(
            f"{p} (weight {w})" for p, w in s.protocol_weights.items()
        )
        lines.append(
            f"{i + 1}. flow_id=flow_{start_id + i:06d} | scenario: {s.name} - "
            f"{s.description}\n"
            f"   label={s.label} attack_type={s.attack_type} "
            f"firewall_action={inst.firewall_action} "
            f"alert_triggered={inst.alert_triggered}\n"
            f"   protocol options: {protocol_opts} | "
            f"dst_port from: {s.dst_port_pool[:12]}"
            f"{'...' if len(s.dst_port_pool) > 12 else ''}\n"
            f"   duration_ms range: {_fmt_range(*s.duration_ms_range)} | "
            f"bytes_sent range: {_fmt_range(*s.bytes_sent_range)} | "
            f"bytes_recv range: {_fmt_range(*s.bytes_recv_range)} | "
            f"packet_count range: {_fmt_range(*s.packet_count_range)}"
        )
    return "\n".join(lines)


def sample_batch_instances(batch_size: int, rng: random.Random) -> list[ScenarioInstance]:
    n_attacks = max(1, round(batch_size * ATTACK_RATIO)) if batch_size >= round(1 / ATTACK_RATIO) else (
        1 if rng.random() < ATTACK_RATIO * batch_size else 0
    )
    n_benign = batch_size - n_attacks
    instances = [sample_scenario_instance(is_attack=False, rng=rng) for _ in range(n_benign)]
    instances += [sample_scenario_instance(is_attack=True, rng=rng) for _ in range(n_attacks)]
    rng.shuffle(instances)
    return instances


async def generate_one_batch(
    client: AsyncOpenAI,
    instances: list[ScenarioInstance],
    start_id: int,
    semaphore: asyncio.Semaphore,
) -> list[tuple[NetworkFlow, ScenarioInstance]] | None:
    """Returns (flow, scenario_instance) pairs, matched by flow_id (not
    position) so a reordered/dropped record from the model still pairs
    correctly with the scenario that constrained it - needed by
    validators.clamp_to_scenario_ranges().
    """
    prompt = build_prompt(instances, start_id)
    by_flow_id = {
        f"flow_{start_id + i:06d}": inst for i, inst in enumerate(instances)
    }

    parsed = None
    async with semaphore:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = await client.chat.completions.parse(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    response_format=FlowBatch,
                    max_completion_tokens=8000,
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
                    await asyncio.sleep(2 ** attempt)  # 2s, 4s

    if parsed is None:
        print(f"  [batch start_id={start_id}] giving up after {MAX_RETRIES} attempts")
        return None

    pairs = []
    for flow in parsed.flows:
        inst = by_flow_id.get(flow.flow_id)
        if inst is None:
            print(f"  [batch start_id={start_id}] dropping unrecognized "
                  f"flow_id={flow.flow_id!r} (not one we asked for)")
            continue
        pairs.append((flow, inst))
    return pairs


def load_existing_count(out_path: Path) -> int:
    if not out_path.exists():
        return 0
    with open(out_path) as f:
        return sum(1 for _ in f)


async def run(n: int, out_path: Path, concurrency: int, dry_run: bool, seed: int) -> None:
    already = load_existing_count(out_path)
    if already >= n:
        print(f"{out_path} already has {already} records >= target {n}. Nothing to do.")
        return
    remaining = n - already
    print(f"Resuming at {already} existing records; generating {remaining} more "
          f"(target {n}) with concurrency={concurrency}.")

    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(concurrency)
    rng = random.Random(seed + already)  # offset so resumed runs don't repeat scenario draws

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_mode = "a" if already > 0 else "w"

    n_batches = (remaining + BATCH_SIZE - 1) // BATCH_SIZE
    tasks = []
    next_id = already + 1
    for i in range(n_batches):
        size = min(BATCH_SIZE, remaining - i * BATCH_SIZE)
        instances = sample_batch_instances(size, rng)
        tasks.append(generate_one_batch(client, instances, next_id, semaphore))
        next_id += size

    start = time.time()
    total_written = 0
    total_clamped = 0
    total_traffic_corrected = 0

    with open(out_path, out_mode) as f:
        for coro in asyncio.as_completed(tasks):
            pairs = await coro
            if not pairs:
                continue
            corrected_flows, stats = validate_batch(pairs)
            total_clamped += stats["clamped"]
            total_traffic_corrected += stats["traffic_corrected"]
            if stats["clamped"] > 0 or stats["traffic_corrected"] > 0:
                print(f"  batch: {stats['clamped']}/{stats['total']} clamped to "
                      f"scenario range, {stats['traffic_corrected']}/{stats['total']} "
                      f"traffic_level corrected")
                for c in stats["sample_corrections"]:
                    print(f"    {c['flow_id']}: {c['old_traffic_level']} -> "
                          f"{c['new_traffic_level']} ({'; '.join(c['reasons']) or 'model over-flagged'})")

            if not dry_run:
                for flow in corrected_flows:
                    f.write(flow.model_dump_json() + "\n")
                f.flush()
            total_written += len(corrected_flows)
            elapsed = time.time() - start
            print(f"  progress: {total_written}/{remaining} new records "
                  f"({elapsed:.0f}s elapsed)")

    print(f"\nDone. Wrote {total_written} new records "
          f"({total_clamped} clamped, {total_traffic_corrected} traffic_level "
          f"corrections) to {out_path}"
          + (" [DRY RUN - nothing written]" if dry_run else ""))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100,
                        help="Target total number of records in --out (default: 100).")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help=f"Output JSONL path (default: {DEFAULT_OUT}).")
    parser.add_argument("--concurrency", type=int, default=8,
                        help="Max concurrent API calls (default: 8).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for scenario sampling (default: 0).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate and validate but don't write to --out.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run(args.n, args.out, args.concurrency, args.dry_run, args.seed))
