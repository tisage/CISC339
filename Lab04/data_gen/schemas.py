"""
Pydantic schemas for the synthetic network-security data used across
Lab04 (Bayesian Networks / HMM) and later labs (ML/DL/LLM chapters).

Three data layers, each independently generated and stored:
  - NetworkFlow: one row per network flow (tabular, for Bayesian Networks
    and classical ML). Mirrors the shape of NetFlow/CICIDS-style datasets.
  - Session: one JSON object per session, containing a time-ordered
    sequence of observed events plus the (hidden) true state at each step -
    directly usable as HMM training/eval data.
  - LogEntry: one JSON object per synthetic SOC log line, for later
    LLM-based classification/explanation exercises.

These are intentionally kept minimal and flat (no nested exotic types)
so that a `pandas.DataFrame(...)` or `pydantic.TypeAdapter(...).validate_json()`
round-trip is trivial for students in later labs.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Shared vocabularies - kept as plain Literals (not Enums) so pydantic
# serializes them as plain strings in JSON/Parquet, which is what pandas
# and pgmpy expect without extra conversion.
# ---------------------------------------------------------------------------

Protocol = Literal["TCP", "UDP", "ICMP"]
TrafficLevel = Literal["Normal", "Abnormal"]
FirewallAction = Literal["Block", "Non-block"]
AlertTriggered = Literal["Yes", "No"]
Label = Literal["benign", "attack"]
AttackType = Literal[
    "none", "DoS", "PortScan", "BruteForce", "Exfiltration", "Botnet"
]


# ---------------------------------------------------------------------------
# Layer 1: flow-level tabular data
# ---------------------------------------------------------------------------

class NetworkFlow(BaseModel):
    """One network flow record - the tabular layer used by the Bayesian
    Network / classical ML chapters. Field names deliberately match the
    node names already used in HW4-3's hand-built Bayesian Network
    (Traffic, Firewall, Alert, Intrusion) so that the same data can drive
    both the hand-authored CPTs and a data-estimated version side by side.
    """

    flow_id: str = Field(description="Unique id, e.g. 'flow_000001'.")
    timestamp: str = Field(description="ISO-8601 timestamp, synthetic.")

    src_ip: str = Field(description="Synthetic private-range source IP.")
    dst_ip: str = Field(description="Synthetic private-range destination IP.")
    dst_port: int = Field(ge=0, le=65535)
    protocol: Protocol

    duration_ms: float = Field(ge=0)
    bytes_sent: int = Field(ge=0)
    bytes_recv: int = Field(ge=0)
    packet_count: int = Field(ge=0)

    # Discrete fields matching the existing HW4-3 Bayesian Network nodes.
    traffic_level: TrafficLevel
    firewall_action: FirewallAction
    alert_triggered: AlertTriggered

    # Ground-truth label (what the BN/ML model is ultimately predicting).
    label: Label
    attack_type: AttackType = "none"

    @field_validator("attack_type")
    @classmethod
    def attack_type_consistent_with_label(cls, v: str, info):
        label = info.data.get("label")
        if label == "benign" and v != "none":
            raise ValueError("benign flows must have attack_type='none'")
        if label == "attack" and v == "none":
            raise ValueError("attack flows must have a non-'none' attack_type")
        return v


# ---------------------------------------------------------------------------
# Layer 2: session-level sequence data (for HMM)
# ---------------------------------------------------------------------------

ObservableEvent = Literal[
    "normal_traffic", "login_attempt", "failed_login",
    "port_scan", "data_transfer", "privilege_escalation",
]
HiddenState = Literal["benign", "compromised"]


class SessionStep(BaseModel):
    step: int = Field(ge=0, description="Index within the session, 0-based.")
    hidden_state: HiddenState = Field(
        description="Ground-truth hidden state at this step (not observed "
        "by the HMM at inference time - kept here for evaluation)."
    )
    observed_event: ObservableEvent


class Session(BaseModel):
    """One session: a time-ordered sequence of observable events, with the
    hidden ground-truth state at each step. Directly usable as HMM
    train/eval data - drop `hidden_state` at inference time and only feed
    `observed_event` sequences to the model.
    """

    session_id: str
    steps: list[SessionStep]

    @field_validator("steps")
    @classmethod
    def steps_are_contiguous(cls, v: list[SessionStep]):
        if [s.step for s in v] != list(range(len(v))):
            raise ValueError("steps must be contiguous starting at 0")
        return v


# ---------------------------------------------------------------------------
# Layer 3: raw log text (for later LLM chapters)
# ---------------------------------------------------------------------------

class LogEntry(BaseModel):
    """One synthetic SOC-style log line, for later LLM classification /
    explanation exercises. `label`/`attack_type` are the ground truth the
    LLM exercise will be scored against - not shown to the model at
    inference time.
    """

    log_id: str
    raw_text: str = Field(
        description="A single free-text log/alert line, e.g. from a "
        "firewall or IDS, written in a realistic but synthetic style."
    )
    label: Label
    attack_type: AttackType = "none"
