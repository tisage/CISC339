"""
Deterministic, rule-based post-generation checks for NetworkFlow records.

LLMs are good at producing plausible-looking varied records, but not
reliable at keeping every numeric field internally consistent with the
categorical labels they also assigned (e.g. a 10-minute, 450MB transfer
labeled traffic_level="Normal"). Rather than trust the model's own
judgment call on "is this normal or abnormal", we re-derive
`traffic_level` from simple, explainable thresholds on the numeric fields
and correct any mismatch - this keeps the dataset's Normal/Abnormal split
meaningful for the Bayesian Network chapter, where students are expected
to reason about Traffic -> Alert -> Intrusion using these very fields.

These thresholds are deliberately simple (not a real IDS rule engine) -
the goal is internal consistency for teaching, not detection accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass

from schemas import NetworkFlow
from scenarios import ScenarioInstance

# Thresholds used to decide whether a flow's *numeric* fields look
# abnormal, independent of what label the LLM assigned. Tuned loosely
# against typical benign traffic (web browsing, DNS, SSH sessions, and
# ordinary high-throughput transfers like downloads/streaming) so those
# fall well inside them.
#
# Absolute caps: even a legitimate transfer has some upper bound in a
# single flow record for teaching purposes.
MAX_BENIGN_BYTES = 50_000_000         # 50 MB either direction in one flow
MAX_BENIGN_DURATION_MS = 300_000      # 5 minutes

# Flood/scan signature: a flood (DoS) or scan sends many packets per
# second, but unlike a real high-throughput transfer (which sends
# near-MTU-sized packets, ~1400 bytes, to move data efficiently), a
# flood/scan's packets are typically small (SYN packets, probe packets)
# since the goal is volume/coverage, not payload delivery. So we only
# flag high packet rate when it's ALSO paired with small average packet
# size - that combination doesn't happen in ordinary traffic, whereas
# high rate alone (e.g. a fast download) does.
HIGH_PACKET_RATE_THRESHOLD = 200      # packets per second
SMALL_AVG_PACKET_SIZE_THRESHOLD = 100  # bytes/packet

# A rate computed from only a handful of packets is statistically
# meaningless (e.g. 2 packets in 9ms "looks like" 222 pkts/s, but that's
# just a fast, tiny, ordinary exchange like a DNS query - not a flood).
# Only apply the flood/scan rate check when there's enough packets for
# "rate" to mean anything.
MIN_PACKETS_FOR_RATE_CHECK = 20


@dataclass
class ValidationResult:
    flow: NetworkFlow
    was_corrected: bool
    reasons: list[str]


def clamp_to_scenario_ranges(flow: NetworkFlow, scenario_instance: ScenarioInstance) -> NetworkFlow:
    """
    Clamps duration_ms/bytes_sent/bytes_recv/packet_count into the ranges
    declared by the scenario that constrained this record's generation.
    The prompt already tells the model these are hard bounds, but LLMs
    don't reliably honor stated bounds - this guarantees it instead of
    hoping for it, which matters because scenario ranges are what keep,
    e.g., an "ssh_session" flow from silently drifting into
    "large_download" territory and confusing the aggregate distribution.
    """
    s = scenario_instance.scenario
    dur_lo, dur_hi = s.duration_ms_range
    sent_lo, sent_hi = s.bytes_sent_range
    recv_lo, recv_hi = s.bytes_recv_range
    pkt_lo, pkt_hi = s.packet_count_range

    return flow.model_copy(update={
        "duration_ms": min(max(flow.duration_ms, dur_lo), dur_hi),
        "bytes_sent": min(max(flow.bytes_sent, sent_lo), sent_hi),
        "bytes_recv": min(max(flow.bytes_recv, recv_lo), recv_hi),
        "packet_count": min(max(flow.packet_count, pkt_lo), pkt_hi),
    })


def _packet_rate(flow: NetworkFlow) -> float:
    if flow.duration_ms <= 0:
        return float("inf") if flow.packet_count > 0 else 0.0
    return flow.packet_count / (flow.duration_ms / 1000.0)


def _avg_packet_size(flow: NetworkFlow) -> float:
    if flow.packet_count <= 0:
        return 0.0
    return (flow.bytes_sent + flow.bytes_recv) / flow.packet_count


def looks_numerically_abnormal(flow: NetworkFlow) -> list[str]:
    """Returns a list of human-readable reasons the numeric fields look
    abnormal, or an empty list if they look like ordinary traffic
    (including ordinary *high-throughput* traffic, which is not the same
    as abnormal - see the flood/scan signature check below)."""
    reasons = []
    if flow.bytes_sent > MAX_BENIGN_BYTES:
        reasons.append(f"bytes_sent={flow.bytes_sent} exceeds {MAX_BENIGN_BYTES}")
    if flow.bytes_recv > MAX_BENIGN_BYTES:
        reasons.append(f"bytes_recv={flow.bytes_recv} exceeds {MAX_BENIGN_BYTES}")
    if flow.duration_ms > MAX_BENIGN_DURATION_MS:
        reasons.append(f"duration_ms={flow.duration_ms} exceeds {MAX_BENIGN_DURATION_MS}")

    rate = _packet_rate(flow)
    avg_size = _avg_packet_size(flow)
    if (
        flow.packet_count >= MIN_PACKETS_FOR_RATE_CHECK
        and rate > HIGH_PACKET_RATE_THRESHOLD
        and avg_size < SMALL_AVG_PACKET_SIZE_THRESHOLD
    ):
        reasons.append(
            f"flood/scan signature: packet_rate={rate:.0f}/s with small "
            f"avg_packet_size={avg_size:.0f}B"
        )
    return reasons


def reconcile_traffic_level(flow: NetworkFlow) -> ValidationResult:
    """
    Re-derives `traffic_level` from the numeric fields and overrides the
    LLM's own value if they disagree. Does NOT touch `label`/`attack_type`
    (those stay as the LLM's ground-truth intent for the record) - only
    the derived `traffic_level` field, which is meant to reflect what a
    simple monitoring rule would flag, consistent with the numeric fields.
    """
    reasons = looks_numerically_abnormal(flow)
    numerically_abnormal = len(reasons) > 0
    currently_abnormal = flow.traffic_level == "Abnormal"

    if numerically_abnormal == currently_abnormal:
        return ValidationResult(flow=flow, was_corrected=False, reasons=reasons)

    corrected = flow.model_copy(
        update={"traffic_level": "Abnormal" if numerically_abnormal else "Normal"}
    )
    return ValidationResult(flow=corrected, was_corrected=True, reasons=reasons)


def validate_batch(
    pairs: list[tuple[NetworkFlow, ScenarioInstance]]
) -> tuple[list[NetworkFlow], dict]:
    """
    For each (flow, scenario_instance) pair: clamps numeric fields to the
    scenario's declared ranges, then re-derives traffic_level from those
    (now-clamped) values. Returns (corrected_flows, stats) where stats
    summarizes how many records were touched and why - printed by the
    generator so corrections are visible, not silent.
    """
    corrected_flows = []
    n_clamped = 0
    n_traffic_corrected = 0
    sample_corrections = []

    for flow, scenario_instance in pairs:
        clamped = clamp_to_scenario_ranges(flow, scenario_instance)
        was_clamped = clamped != flow
        if was_clamped:
            n_clamped += 1

        result = reconcile_traffic_level(clamped)
        corrected_flows.append(result.flow)
        if result.was_corrected:
            n_traffic_corrected += 1
            if len(sample_corrections) < 5:
                sample_corrections.append(
                    {
                        "flow_id": flow.flow_id,
                        "old_traffic_level": clamped.traffic_level,
                        "new_traffic_level": result.flow.traffic_level,
                        "reasons": result.reasons,
                    }
                )

    stats = {
        "total": len(pairs),
        "clamped": n_clamped,
        "traffic_corrected": n_traffic_corrected,
        "sample_corrections": sample_corrections,
    }
    return corrected_flows, stats
