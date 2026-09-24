"""
Traffic scenario library for synthetic NetworkFlow generation.

Design intent: keep "what does realistic traffic/attack X look like" as
DATA (this file), separate from "how do we call the LLM and control
sampling proportions" (generate_flows.py). This split exists so that:

  1. Statistical control (label ratio, firewall behavior, alert rates)
     is enforced by Python before the LLM ever sees a prompt - the LLM
     is only asked to fill in realistic per-record detail (IPs, ports,
     exact byte counts) for a scenario that's already been decided. This
     is what keeps fields like `firewall_action` from silently collapsing
     to one value, since the split is chosen by `sample_scenario()` below,
     not left to the model to "remember" to vary.
  2. Extending the dataset to a new attack pattern (network threats change
     year to year) means adding one `Scenario` entry here - nothing in
     generate_flows.py or validators.py needs to change. A student or a
     future instructor can add a scenario without understanding the
     concurrency/prompting/validation plumbing at all.

A `Scenario` is a coarse behavioral template, not a fully-specified
record - it constrains the *ranges* an LLM should generate within, and
the *independent* probabilities for firewall_action/alert_triggered
(independent of label, matching how these actually behave: a
misconfigured firewall can let attacks through, or block benign traffic).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class Scenario:
    name: str
    label: str                    # "benign" or "attack"
    attack_type: str              # "none" or one of AttackType
    description: str              # human-readable behavioral summary, fed to the LLM
    protocol_weights: dict[str, float]       # {"TCP": 0.8, "UDP": 0.2, ...}
    dst_port_pool: list[int]                 # realistic ports for this scenario
    duration_ms_range: tuple[float, float]
    bytes_sent_range: tuple[int, int]
    bytes_recv_range: tuple[int, int]
    packet_count_range: tuple[int, int]
    # P(firewall_action == "Block"), independent of label - a scenario can
    # still specify this is usually not blocked (typical for stealthy
    # attacks) or usually blocked (typical for loud/obvious ones).
    p_firewall_block: float
    # P(alert_triggered == "Yes"), independent of label for the same reason.
    p_alert_yes: float
    # Relative weight for how often this scenario is picked within its
    # label group (benign scenarios compete with each other; same for
    # attack scenarios) - NOT the overall benign/attack ratio, which is
    # controlled separately by generate_flows.py's ATTACK_RATIO.
    weight: float = 1.0


# ---------------------------------------------------------------------------
# Benign scenarios
# ---------------------------------------------------------------------------

BENIGN_SCENARIOS = [
    Scenario(
        name="web_browsing",
        label="benign", attack_type="none",
        description="Ordinary HTTPS web browsing: short-to-medium requests, "
                     "moderate response sizes.",
        protocol_weights={"TCP": 1.0},
        dst_port_pool=[443, 80],
        duration_ms_range=(50, 5000),
        bytes_sent_range=(200, 20_000),
        bytes_recv_range=(2_000, 500_000),
        packet_count_range=(5, 300),
        p_firewall_block=0.02,
        p_alert_yes=0.01,
        weight=3.0,
    ),
    Scenario(
        name="large_download",
        label="benign", attack_type="none",
        description="Legitimate high-throughput transfer (large file "
                     "download, video streaming, backup sync): high byte "
                     "count and high packet rate, but near-MTU-sized "
                     "packets (large avg bytes/packet), not a flood.",
        protocol_weights={"TCP": 1.0},
        dst_port_pool=[443, 21, 990],
        duration_ms_range=(1_000, 60_000),
        bytes_sent_range=(50_000, 2_000_000),
        bytes_recv_range=(500_000, 40_000_000),
        packet_count_range=(300, 20_000),
        p_firewall_block=0.02,
        p_alert_yes=0.01,
        weight=1.5,
    ),
    Scenario(
        name="dns_query",
        label="benign", attack_type="none",
        description="Ordinary DNS lookup: tiny, near-instant.",
        protocol_weights={"UDP": 1.0},
        dst_port_pool=[53],
        duration_ms_range=(1, 200),
        bytes_sent_range=(40, 300),
        bytes_recv_range=(60, 800),
        packet_count_range=(2, 6),
        p_firewall_block=0.01,
        p_alert_yes=0.0,
        weight=2.0,
    ),
    Scenario(
        name="ssh_session",
        label="benign", attack_type="none",
        description="Legitimate SSH admin session: moderate duration, "
                     "modest byte counts.",
        protocol_weights={"TCP": 1.0},
        dst_port_pool=[22],
        duration_ms_range=(2_000, 300_000),
        bytes_sent_range=(2_000, 100_000),
        bytes_recv_range=(2_000, 100_000),
        packet_count_range=(20, 2_000),
        p_firewall_block=0.03,
        p_alert_yes=0.02,
        weight=1.5,
    ),
    Scenario(
        name="email",
        label="benign", attack_type="none",
        description="Ordinary outgoing/incoming email (SMTP/IMAPS).",
        protocol_weights={"TCP": 1.0},
        dst_port_pool=[25, 587, 993],
        duration_ms_range=(200, 10_000),
        bytes_sent_range=(500, 50_000),
        bytes_recv_range=(500, 50_000),
        packet_count_range=(5, 400),
        p_firewall_block=0.02,
        p_alert_yes=0.01,
        weight=1.5,
    ),
    Scenario(
        name="internal_file_share",
        label="benign", attack_type="none",
        description="Internal-to-internal file share access (SMB) between "
                     "two hosts on the same private network.",
        protocol_weights={"TCP": 1.0},
        dst_port_pool=[445, 139],
        duration_ms_range=(500, 30_000),
        bytes_sent_range=(1_000, 500_000),
        bytes_recv_range=(1_000, 2_000_000),
        packet_count_range=(10, 3_000),
        p_firewall_block=0.02,
        p_alert_yes=0.01,
        weight=1.0,
    ),
]

# ---------------------------------------------------------------------------
# Attack scenarios
# ---------------------------------------------------------------------------

ATTACK_SCENARIOS = [
    Scenario(
        name="dos_flood",
        label="attack", attack_type="DoS",
        description="Denial-of-service flood: very high packet rate, small "
                     "average packet size (SYN flood / UDP flood pattern), "
                     "short-to-moderate duration.",
        protocol_weights={"TCP": 0.6, "UDP": 0.4},
        dst_port_pool=[80, 443, 53],
        duration_ms_range=(500, 60_000),
        bytes_sent_range=(2_000, 200_000),
        bytes_recv_range=(0, 5_000),
        packet_count_range=(500, 50_000),
        p_firewall_block=0.55,
        p_alert_yes=0.75,
        weight=1.2,
    ),
    Scenario(
        name="port_scan",
        label="attack", attack_type="PortScan",
        description="Port scan: very short duration, tiny byte counts, "
                     "probing an unusual/uncommon destination port.",
        protocol_weights={"TCP": 0.8, "UDP": 0.2},
        dst_port_pool=list(range(1, 1024)),
        duration_ms_range=(1, 500),
        bytes_sent_range=(40, 200),
        bytes_recv_range=(0, 100),
        packet_count_range=(1, 5),
        p_firewall_block=0.35,
        p_alert_yes=0.5,
        weight=1.0,
    ),
    Scenario(
        name="brute_force",
        label="attack", attack_type="BruteForce",
        description="Credential brute-force: many short repeated "
                     "connection attempts to an auth service, moderate "
                     "packet count, low bytes per attempt.",
        protocol_weights={"TCP": 1.0},
        dst_port_pool=[22, 3389, 21, 23],
        duration_ms_range=(500, 20_000),
        bytes_sent_range=(500, 20_000),
        bytes_recv_range=(200, 10_000),
        packet_count_range=(50, 2_000),
        p_firewall_block=0.4,
        p_alert_yes=0.6,
        weight=1.0,
    ),
    Scenario(
        name="exfiltration",
        label="attack", attack_type="Exfiltration",
        description="Data exfiltration: large outbound byte count over a "
                     "longer duration, often to an unusual external port, "
                     "low inbound bytes (one-directional).",
        protocol_weights={"TCP": 1.0},
        dst_port_pool=[443, 8080, 4444, 9001],
        duration_ms_range=(10_000, 600_000),
        bytes_sent_range=(1_000_000, 100_000_000),
        bytes_recv_range=(0, 50_000),
        packet_count_range=(200, 30_000),
        p_firewall_block=0.3,
        p_alert_yes=0.45,
        weight=1.0,
    ),
    Scenario(
        name="botnet_beacon",
        label="attack", attack_type="Botnet",
        description="Botnet command-and-control beaconing: small, regular, "
                     "low-bandwidth connections to an external IP, short "
                     "duration, low bytes both ways.",
        protocol_weights={"TCP": 0.7, "UDP": 0.3},
        dst_port_pool=[443, 8080, 6667, 6697],
        duration_ms_range=(100, 5_000),
        bytes_sent_range=(100, 5_000),
        bytes_recv_range=(100, 5_000),
        packet_count_range=(2, 50),
        p_firewall_block=0.25,
        p_alert_yes=0.35,
        weight=1.0,
    ),
]

ALL_SCENARIOS = BENIGN_SCENARIOS + ATTACK_SCENARIOS


@dataclass
class ScenarioInstance:
    """One concrete, pre-decided draw: a Scenario plus the specific
    firewall_action/alert_triggered this particular record should have.
    This is what gets serialized into the LLM prompt - the LLM fills in
    realistic IPs/timestamps/exact numeric values consistent with these
    already-decided constraints; it does not decide label/attack_type/
    firewall_action/alert_triggered itself.

    Deliberately does NOT pre-decide `traffic_level` - that field is
    derived entirely from the record's own numeric fields by
    validators.reconcile_traffic_level() after generation, representing
    what a naive monitoring rule would flag. A stealthy attack scenario
    (BruteForce, Botnet, slow Exfiltration) is expected to often end up
    traffic_level="Normal" despite label="attack" - see schemas.py's
    NetworkFlow docstring for why that gap is intentional.
    """

    scenario: Scenario
    firewall_action: str    # "Block" or "Non-block", pre-sampled
    alert_triggered: str    # "Yes" or "No", pre-sampled


def _weighted_choice(scenarios: list[Scenario], rng: random.Random) -> Scenario:
    weights = [s.weight for s in scenarios]
    return rng.choices(scenarios, weights=weights, k=1)[0]


def sample_scenario_instance(is_attack: bool, rng: random.Random) -> ScenarioInstance:
    """Draws one scenario and pre-decides its firewall_action/alert_triggered
    independently via the scenario's own probabilities - this is the
    control point that keeps those fields from collapsing to a constant
    across a whole generation run, regardless of what the LLM would have
    done on its own. `traffic_level` is intentionally not decided here -
    see ScenarioInstance's docstring.
    """
    pool = ATTACK_SCENARIOS if is_attack else BENIGN_SCENARIOS
    scenario = _weighted_choice(pool, rng)
    firewall_action = "Block" if rng.random() < scenario.p_firewall_block else "Non-block"
    alert_triggered = "Yes" if rng.random() < scenario.p_alert_yes else "No"
    return ScenarioInstance(
        scenario=scenario,
        firewall_action=firewall_action,
        alert_triggered=alert_triggered,
    )
