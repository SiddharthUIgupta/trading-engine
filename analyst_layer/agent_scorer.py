"""Format agent accuracy context for injection into the risk officer prompt.

Two layers of historical signal:
  1. Flat per-agent win-rate table (from state_store.get_agent_accuracy) — simple,
     works immediately, surfaces obvious track records like "sentiment agent is
     wrong 70% of the time in bear regimes."
  2. VW bandit win-probability (from VWSignalBandit.predict_context) — contextual,
     learns the JOINT effect of track × regime × agent agreement patterns. Injected
     as a single calibrated probability estimate once the model has seen >= 20 examples.
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


def format_vw_context(win_probability: float | None, example_count: int = 0) -> str:
    """Format VW bandit win-probability as a prompt context block.

    Returns empty string when the model hasn't accumulated enough examples,
    or when win_probability is None (model not available / not ready).
    """
    if win_probability is None:
        return ""

    pct = win_probability * 100
    if pct >= 65:
        confidence_label = "STRONG HISTORICAL EDGE"
    elif pct >= 50:
        confidence_label = "SLIGHT HISTORICAL EDGE"
    elif pct >= 35:
        confidence_label = "SLIGHT HISTORICAL HEADWIND"
    else:
        confidence_label = "STRONG HISTORICAL HEADWIND"

    return (
        f"[VW BANDIT — CONTEXTUAL WIN PROBABILITY]\n"
        f"  Predicted win probability for this track × regime: {pct:.0f}% ({confidence_label})\n"
        f"  Based on {example_count} past trades with similar track and market regime.\n"
        f"  Factor this into your confidence weighting — it captures patterns the per-agent "
        f"accuracy table above cannot (joint effects and regime interactions).\n\n"
    )
