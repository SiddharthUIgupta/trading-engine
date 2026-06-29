from __future__ import annotations

from execution_layer.exit_rules import evaluate_exit

_KWARGS = dict(stop_loss_pct=0.02, take_profit_pct=0.03, trailing_stop_pct=0.015)


def test_holds_when_within_all_bands():
    decision = evaluate_exit(avg_entry_price=100.0, current_price=100.5, high_water_mark=100.5, **_KWARGS)
    assert decision.should_exit is False


def test_stop_loss_triggers():
    decision = evaluate_exit(avg_entry_price=100.0, current_price=97.9, high_water_mark=100.0, **_KWARGS)
    assert decision.should_exit is True
    assert "stop-loss" in decision.reason


def test_take_profit_triggers():
    decision = evaluate_exit(avg_entry_price=100.0, current_price=103.1, high_water_mark=103.1, **_KWARGS)
    assert decision.should_exit is True
    assert "target hit" in decision.reason


def test_trailing_stop_triggers_after_pullback_from_peak():
    # ran up to 110 (well past take-profit, but suppose take_profit_pct were
    # higher in this scenario — isolate trailing-stop by raising it)
    decision = evaluate_exit(
        avg_entry_price=100.0,
        current_price=107.0,
        high_water_mark=110.0,
        stop_loss_pct=0.02,
        take_profit_pct=0.50,  # disabled for this test
        trailing_stop_pct=0.02,
    )
    assert decision.should_exit is True
    assert "trailing stop" in decision.reason


def test_trailing_stop_does_not_trigger_before_any_pullback():
    decision = evaluate_exit(avg_entry_price=100.0, current_price=101.0, high_water_mark=101.0, **_KWARGS)
    assert decision.should_exit is False


def test_trailing_stop_ignored_when_high_water_mark_equals_entry():
    # never moved up from entry — no peak to trail from
    decision = evaluate_exit(avg_entry_price=100.0, current_price=99.0, high_water_mark=100.0, **_KWARGS)
    assert decision.should_exit is False


# ---- Thesis-track behavior: take_profit_pct=None, trailing activation gate ----

def test_none_take_profit_never_triggers_even_on_huge_gain():
    decision = evaluate_exit(
        avg_entry_price=100.0, current_price=250.0, high_water_mark=250.0,
        stop_loss_pct=0.18, take_profit_pct=None, trailing_stop_pct=0.10,
    )
    assert decision.should_exit is False


def test_trailing_stop_does_not_engage_before_activation_threshold():
    # up 10% from entry, but activation requires +20% before trailing applies
    decision = evaluate_exit(
        avg_entry_price=100.0, current_price=105.0, high_water_mark=110.0,
        stop_loss_pct=0.18, take_profit_pct=None, trailing_stop_pct=0.10,
        trailing_stop_activation_pct=0.20,
    )
    assert decision.should_exit is False


def test_trailing_stop_engages_once_activation_threshold_met():
    # peaked at +25% (past the +20% activation), pulled back 12% from that peak
    decision = evaluate_exit(
        avg_entry_price=100.0, current_price=110.0, high_water_mark=125.0,
        stop_loss_pct=0.18, take_profit_pct=None, trailing_stop_pct=0.10,
        trailing_stop_activation_pct=0.20,
    )
    assert decision.should_exit is True
    assert "trailing stop" in decision.reason


def test_wide_stop_loss_survives_normal_volatility_a_tight_stop_would_not():
    # down 10% from entry — would trip a 2% momentum stop, but not an 18% thesis stop
    decision = evaluate_exit(
        avg_entry_price=100.0, current_price=90.0, high_water_mark=100.0,
        stop_loss_pct=0.18, take_profit_pct=None, trailing_stop_pct=0.10,
        trailing_stop_activation_pct=0.20,
    )
    assert decision.should_exit is False
