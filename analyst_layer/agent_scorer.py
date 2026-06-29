"""Format agent accuracy context for injection into the risk officer prompt.

After enough trades accumulate, each sub-agent's historical win-rate in the
current track and market regime is surfaced to the risk officer so it can
weight conflicting signals appropriately. An agent with 35% accuracy in
bearish regimes should carry less weight than one with 70% accuracy — this
makes that explicit in the LLM's context rather than leaving it implicit.
"""
from __future__ import annotations

_MIN_SAMPLE = 10  # minimum scored signals before surfacing an agent's track record


def format_accuracy_context(accuracy_rows: list[tuple[str, int, int]]) -> str:
    """Convert (agent_name, total, wins) rows into a prompt context block.

    Returns empty string when no agent has enough history — never cite a
    small sample as a meaningful track record.
    """
    if not accuracy_rows:
        return ""

    lines = []
    for agent_name, total, wins in accuracy_rows:
        if total < _MIN_SAMPLE:
            continue
        accuracy = wins / total
        lines.append(f"  • {agent_name}: {accuracy:.0%} win-rate ({wins}/{total} scored trades)")

    if not lines:
        return ""

    header = "[AGENT TRACK RECORD — historical accuracy in this track and market regime]"
    footer = "Weight signals accordingly when agents conflict."
    return "\n".join([header] + lines + [footer]) + "\n\n"
