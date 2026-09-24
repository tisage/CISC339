"""Generator for synthetic Session records (Lab04 HMM chapter).

No LLM call - see session_scenarios.py's module docstring for why. Pure
Python sampling from the ground-truth transition/emission model, so this
runs in well under a second even for thousands of sessions and costs
nothing. Still resumable/appendable like generate_flows.py, for
consistency, though realistically you'd just regenerate the whole file
each time given the near-zero cost.

Usage:
    uv run --project . python Lab04/data_gen/generate_sessions.py --n 500

Run from the repo root so the root .venv/pyproject.toml is used.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from schemas import Session, SessionStep
from session_scenarios import generate_session_steps

DEFAULT_OUT = Path(__file__).resolve().parent / "data" / "sessions.jsonl"


def build_session(session_id: str, rng: random.Random) -> Session:
    steps = generate_session_steps(rng)
    return Session(
        session_id=session_id,
        steps=[
            SessionStep(step=i, hidden_state=s.hidden_state, observed_event=s.observed_event)
            for i, s in enumerate(steps)
        ],
    )


def run(n: int, out_path: Path, seed: int) -> None:
    rng = random.Random(seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sessions = [build_session(f"sess_{i + 1:05d}", rng) for i in range(n)]

    with open(out_path, "w") as f:
        for s in sessions:
            f.write(s.model_dump_json() + "\n")

    lengths = [len(s.steps) for s in sessions]
    n_ever_compromised = sum(
        1 for s in sessions if any(step.hidden_state == "compromised" for step in s.steps)
    )
    print(f"Wrote {n} sessions to {out_path}")
    print(f"  session length: min={min(lengths)} max={max(lengths)} "
          f"mean={sum(lengths)/len(lengths):.1f}")
    print(f"  sessions that ever become compromised: {n_ever_compromised}/{n} "
          f"({n_ever_compromised/n:.1%})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100,
                        help="Number of sessions to generate (default: 100).")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help=f"Output JSONL path (default: {DEFAULT_OUT}).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (default: 0).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.n, args.out, args.seed)
