"""Per-million-token Anthropic pricing, used only to estimate spend for
StateStore.token_usage — this is a cost *estimate* for your own visibility,
not a billing-accurate ledger (Anthropic's invoice is authoritative).

Cache write is billed at 1.25x the base input rate (5-minute TTL, which is
what analyst_layer/agents/base.py uses) and cache read at 0.1x.
"""
from __future__ import annotations

CACHE_WRITE_MULTIPLIER = 1.25
CACHE_READ_MULTIPLIER = 0.1

PRICING_PER_MTOK: dict[str, dict[str, float]] = {
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
}


def estimate_cost_usd(model: str, usage) -> float:
    rates = PRICING_PER_MTOK.get(model)
    if rates is None:
        return 0.0  # unknown/unlisted model — don't guess at a rate

    input_tokens = getattr(usage, "input_tokens", 0) or 0
    output_tokens = getattr(usage, "output_tokens", 0) or 0
    cache_creation = getattr(usage, "cache_creation_input_tokens", 0) or 0
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0

    cost = (
        input_tokens * rates["input"]
        + output_tokens * rates["output"]
        + cache_creation * rates["input"] * CACHE_WRITE_MULTIPLIER
        + cache_read * rates["input"] * CACHE_READ_MULTIPLIER
    ) / 1_000_000
    return cost
