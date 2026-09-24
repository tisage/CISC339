"""
Session generation model for the HMM chapter (Lab04).

Unlike flow generation (generate_flows.py), session data needs NO LLM
call: each observable event is one of a fixed 6-item vocabulary (see
schemas.ObservableEvent), so there's no free-text detail for an LLM to
usefully add - a session is fully described by its hidden-state sequence
and, at each step, a draw from that state's emission distribution. Both
of those are exactly what Python's `random` module is for. This keeps
session generation fast (no network calls) and its statistics exactly
what's specified here, not an approximation of what an LLM produced.

Design mirrors scenarios.py's separation of concerns: the *ground-truth
process* (state transitions, emission probabilities) lives here as plain
data, independent of how generate_sessions.py drives the sampling loop -
extending this to a 3rd hidden state (e.g. "compromised_lateral_movement")
or new event types means editing this file only.

Hidden-state model: two states, one-way transition (see module-level
constants) - `benign` can transition to `compromised`, never back. This
matches "a host doesn't spontaneously heal without remediation" and keeps
the transition matrix simple: only one free transition probability
(P(benign -> compromised)); everything else follows from row-sums to 1
and compromised being absorbing.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

HiddenState = str   # "benign" | "compromised" - kept as str, not an Enum,
                     # to match how schemas.HiddenState (a Literal) is used
EVENT_VOCAB = [
    "normal_traffic", "login_attempt", "failed_login",
    "port_scan", "data_transfer", "privilege_escalation",
]

# P(benign -> compromised) at each step, while in the benign state. Once
# compromised, the session stays compromised for the rest of its length
# (absorbing state - see module docstring).
#
# This is a per-step probability, but what matters for the dataset's
# overall balance is the *cumulative* probability of ever transitioning
# over a session's length: P(ever compromised) = 1 - (1-p)^length. Tuned
# so that with lengths in [10, 30] (mean ~20), roughly 15% of sessions end
# up compromised at some point - rarer than benign, but with enough
# compromised examples for hmmlearn to estimate a meaningful emission
# distribution for that state (unlike the flow data's ~5% attack ratio,
# which is fine there because each flow is independent, not part of a
# sequence a model needs to learn state dynamics from).
P_BECOME_COMPROMISED = 0.008

# Emission distribution while hidden_state == "benign": overwhelmingly
# normal traffic, with occasional login activity and rare noise (a benign
# host can still trigger a failed_login by fat-fingering a password, or a
# port_scan false-positive from a vulnerability scanner the admin ran).
EMISSION_BENIGN: dict[str, float] = {
    "normal_traffic": 0.80,
    "login_attempt": 0.12,
    "failed_login": 0.04,
    "port_scan": 0.02,
    "data_transfer": 0.02,
    "privilege_escalation": 0.00,
}

# Emission distribution while hidden_state == "compromised": attack-typical
# events dominate, but normal_traffic still appears sometimes - this is
# the whole reason HMM inference is more useful than "flag any suspicious
# event": a compromised host doesn't announce itself on every single step,
# it still produces ordinary-looking traffic much of the time.
EMISSION_COMPROMISED: dict[str, float] = {
    "normal_traffic": 0.35,
    "login_attempt": 0.10,
    "failed_login": 0.15,
    "port_scan": 0.15,
    "data_transfer": 0.15,
    "privilege_escalation": 0.10,
}

MIN_SESSION_LENGTH = 10
MAX_SESSION_LENGTH = 30


@dataclass
class GeneratedStep:
    hidden_state: HiddenState
    observed_event: str


def _sample_event(emission: dict[str, float], rng: random.Random) -> str:
    events = list(emission.keys())
    weights = list(emission.values())
    return rng.choices(events, weights=weights, k=1)[0]


def generate_session_steps(rng: random.Random, length: int | None = None) -> list[GeneratedStep]:
    """
    Simulates one session: starts benign, each step may irreversibly
    transition to compromised (see P_BECOME_COMPROMISED), and emits one
    observable event per step drawn from the current hidden state's
    emission distribution.
    """
    if length is None:
        length = rng.randint(MIN_SESSION_LENGTH, MAX_SESSION_LENGTH)

    steps = []
    state: HiddenState = "benign"
    for _ in range(length):
        if state == "benign" and rng.random() < P_BECOME_COMPROMISED:
            state = "compromised"
        emission = EMISSION_BENIGN if state == "benign" else EMISSION_COMPROMISED
        event = _sample_event(emission, rng)
        steps.append(GeneratedStep(hidden_state=state, observed_event=event))
    return steps


def true_transition_matrix() -> dict[str, dict[str, float]]:
    """The ground-truth transition matrix used by the generator above,
    for direct comparison against what hmmlearn estimates from the
    generated data (see the notebook's "hand-authored vs. data-estimated"
    comparison, mirroring the Bayesian Network chapter's same comparison)."""
    return {
        "benign": {"benign": 1 - P_BECOME_COMPROMISED, "compromised": P_BECOME_COMPROMISED},
        "compromised": {"benign": 0.0, "compromised": 1.0},
    }


def true_emission_matrix() -> dict[str, dict[str, float]]:
    """The ground-truth emission matrix, for the same comparison."""
    return {
        "benign": dict(EMISSION_BENIGN),
        "compromised": dict(EMISSION_COMPROMISED),
    }
