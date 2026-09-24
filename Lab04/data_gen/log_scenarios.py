"""
Minimal scenario library for synthetic SOC log-line generation (Lab04,
for a not-yet-designed future LLM chapter).

Deliberately small and provisional: only enough scenario coverage to
validate the generation pipeline (schema, LLM call, small-batch output
quality) end to end. Expand this file with more log types/attack
patterns once the LLM chapter's actual exercise format is designed -
same "scenario as data" split as scenarios.py, so that expansion doesn't
require touching generate_logs.py.

Unlike NetworkFlow, a LogEntry has essentially one field with real
freedom (`raw_text`) - so unlike session_scenarios.py, this genuinely
needs an LLM to write plausible, varied natural-language lines. label/
attack_type are still pre-decided in Python, same reasoning as
scenarios.py: don't leave ground-truth-bearing fields for the model to
decide.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LogScenario:
    name: str
    label: str
    attack_type: str
    description: str
    weight: float = 1.0


LOG_SCENARIOS = [
    LogScenario(
        name="normal_access",
        label="benign", attack_type="none",
        description="An ordinary, successful access log line - e.g. a "
                     "successful login, a normal HTTP request, a routine "
                     "firewall pass-through. Written the way a real "
                     "firewall/auth/web server log line looks (include a "
                     "timestamp-like prefix, source IP, and a short "
                     "message), but must not literally match a template -"
                     " vary wording/format across records.",
        weight=3.0,
    ),
    LogScenario(
        name="failed_login_burst",
        label="attack", attack_type="BruteForce",
        description="A single log line representing one attempt within a "
                     "credential brute-force burst - e.g. a failed SSH/RDP "
                     "login from the same source IP, phrased the way a "
                     "real auth log would report it.",
        weight=1.0,
    ),
    LogScenario(
        name="firewall_block",
        label="attack", attack_type="PortScan",
        description="A firewall log line reporting a blocked connection "
                     "attempt to an unusual port, consistent with a port "
                     "scan probe.",
        weight=1.0,
    ),
    LogScenario(
        name="large_outbound_transfer",
        label="attack", attack_type="Exfiltration",
        description="A log line (e.g. from a DLP tool or proxy) flagging "
                     "an unusually large outbound data transfer to an "
                     "external destination.",
        weight=1.0,
    ),
]
