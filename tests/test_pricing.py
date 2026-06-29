from __future__ import annotations

from types import SimpleNamespace

import pytest

from analyst_layer.pricing import estimate_cost_usd


def _usage(input_tokens=0, output_tokens=0, cache_creation_input_tokens=0, cache_read_input_tokens=0):
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_input_tokens=cache_creation_input_tokens,
        cache_read_input_tokens=cache_read_input_tokens,
    )


def test_estimate_cost_sonnet_input_and_output():
    usage = _usage(input_tokens=1_000_000, output_tokens=1_000_000)
    cost = estimate_cost_usd("claude-sonnet-4-6", usage)
    assert cost == pytest.approx(3.00 + 15.00)


def test_estimate_cost_haiku_input_and_output():
    usage = _usage(input_tokens=1_000_000, output_tokens=1_000_000)
    cost = estimate_cost_usd("claude-haiku-4-5-20251001", usage)
    assert cost == pytest.approx(1.00 + 5.00)


def test_estimate_cost_cache_write_is_1_25x_input_rate():
    usage = _usage(cache_creation_input_tokens=1_000_000)
    cost = estimate_cost_usd("claude-sonnet-4-6", usage)
    assert cost == pytest.approx(3.00 * 1.25)


def test_estimate_cost_cache_read_is_0_1x_input_rate():
    usage = _usage(cache_read_input_tokens=1_000_000)
    cost = estimate_cost_usd("claude-sonnet-4-6", usage)
    assert cost == pytest.approx(3.00 * 0.1)


def test_estimate_cost_unknown_model_returns_zero_not_a_guess():
    usage = _usage(input_tokens=1_000_000, output_tokens=1_000_000)
    assert estimate_cost_usd("some-future-model-nobody-priced-yet", usage) == 0.0


def test_estimate_cost_zero_usage_is_zero():
    assert estimate_cost_usd("claude-sonnet-4-6", _usage()) == 0.0
