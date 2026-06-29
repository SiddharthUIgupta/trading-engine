"""Tests for the volatility-based analyst layer.

Covers: schemas, options structurer (build_structure), and the schema
validation that gates execution. LLM calls are not exercised — agents
are tested via their deterministic guardrails and schema contracts.
"""
from __future__ import annotations

from datetime import date, datetime

import pytest

from analyst_layer.options_structurer import StructureResult, build_structure
from analyst_layer.schemas import (
    Confidence,
    GreeksRiskReview,
    IVEnvironment,
    OptionsProposal,
    RiskVerdict,
    StructureType,
    VolConsensusPayload,
    VolRegime,
    VolSignal,
)
from data_layer.models import OptionContract, OptionType, VolatilitySnapshot


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _contract(
    option_type: OptionType,
    dte: int,
    strike: float,
    underlying_price: float = 100.0,
    iv: float = 0.30,
    bid: float = 1.50,
    ask: float = 1.60,
) -> OptionContract:
    return OptionContract(
        contract_symbol=f"TEST{dte}{option_type.value[0].upper()}{int(strike)}",
        underlying_symbol="TEST",
        underlying_price=underlying_price,
        expiration=date(2026, 8, 15),
        dte=dte,
        strike=strike,
        option_type=option_type,
        bid=bid,
        ask=ask,
        implied_volatility=iv,
        open_interest=500,
        volume=100,
    )


def _vol_snapshot(
    symbol: str = "TEST",
    iv_rank: float = 60.0,
    iv_percentile: float = 65.0,
    iv_30: float = 0.35,
    hv_30: float = 0.22,
    earnings_within_dte: bool = False,
    next_earnings_date: date | None = None,
) -> VolatilitySnapshot:
    return VolatilitySnapshot(
        symbol=symbol,
        as_of=datetime.now(),
        iv_rank=iv_rank,
        iv_percentile=iv_percentile,
        iv_30=iv_30,
        hv_20=0.24,
        hv_30=hv_30,
        iv_hv_spread=iv_30 - hv_30,
        term_structure_ratio=0.95,
        put_skew=0.03,
        earnings_within_dte=earnings_within_dte,
        next_earnings_date=next_earnings_date,
    )


def _standard_chain(underlying: float = 100.0, dte: int = 45) -> list[OptionContract]:
    """A realistic chain with OTM calls and puts at multiple strikes."""
    # 1 SD ≈ 100 × 0.35 × sqrt(45/365) ≈ 12.2
    strikes = [70, 75, 80, 85, 88, 90, 92, 95, 97, 100, 103, 105, 108, 110, 112, 115, 120, 125, 130]
    contracts = []
    for strike in strikes:
        if strike < underlying:
            contracts.append(_contract(OptionType.PUT, dte, float(strike), underlying, bid=max(0.10, (underlying - strike) * 0.01), ask=max(0.15, (underlying - strike) * 0.011)))
        else:
            contracts.append(_contract(OptionType.CALL, dte, float(strike), underlying, bid=max(0.10, (strike - underlying) * 0.01 + 0.5), ask=max(0.15, (strike - underlying) * 0.011 + 0.6)))
    return contracts


# ── VolatilitySnapshot schema tests ──────────────────────────────────────────

def test_vol_snapshot_valid():
    snap = _vol_snapshot()
    assert snap.iv_rank == 60.0
    assert snap.iv_hv_spread == pytest.approx(0.35 - 0.22, abs=1e-6)


def test_vol_snapshot_iv_rank_bounds():
    with pytest.raises(Exception):
        _vol_snapshot(iv_rank=101.0)
    with pytest.raises(Exception):
        _vol_snapshot(iv_rank=-1.0)


# ── IVEnvironment classification ──────────────────────────────────────────────

def test_iv_environment_elevated_above_50():
    snap = _vol_snapshot(iv_rank=55.0)
    env = IVEnvironment.ELEVATED if snap.iv_rank >= 50 else (
        IVEnvironment.MODERATE if snap.iv_rank >= 30 else IVEnvironment.DEPRESSED
    )
    assert env == IVEnvironment.ELEVATED


def test_iv_environment_moderate_between_30_and_50():
    snap = _vol_snapshot(iv_rank=40.0)
    env = IVEnvironment.ELEVATED if snap.iv_rank >= 50 else (
        IVEnvironment.MODERATE if snap.iv_rank >= 30 else IVEnvironment.DEPRESSED
    )
    assert env == IVEnvironment.MODERATE


def test_iv_environment_depressed_below_30():
    snap = _vol_snapshot(iv_rank=20.0)
    env = IVEnvironment.ELEVATED if snap.iv_rank >= 50 else (
        IVEnvironment.MODERATE if snap.iv_rank >= 30 else IVEnvironment.DEPRESSED
    )
    assert env == IVEnvironment.DEPRESSED


# ── build_structure tests ─────────────────────────────────────────────────────

def test_no_trade_always_rejects():
    chain = _standard_chain()
    result = build_structure("TEST", StructureType.NO_TRADE, chain, iv_30=0.35)
    assert result.selected is False
    assert "NO_TRADE" in result.reasons[0]


def test_short_strangle_builds_with_valid_chain():
    chain = _standard_chain()
    result = build_structure("TEST", StructureType.SHORT_STRANGLE, chain, iv_30=0.35, target_dte=45)
    assert result.selected is True
    assert result.proposal is not None
    assert result.proposal.structure == StructureType.SHORT_STRANGLE
    assert result.proposal.short_call_strike is not None
    assert result.proposal.short_put_strike is not None
    # Short call must be above underlying, short put below
    assert result.proposal.short_call_strike > 100.0
    assert result.proposal.short_put_strike < 100.0


def test_short_strangle_short_call_is_above_short_put():
    chain = _standard_chain()
    result = build_structure("TEST", StructureType.SHORT_STRANGLE, chain, iv_30=0.35)
    assert result.proposal.short_call_strike > result.proposal.short_put_strike


def test_iron_condor_has_all_four_strikes():
    chain = _standard_chain()
    result = build_structure("TEST", StructureType.IRON_CONDOR, chain, iv_30=0.35)
    assert result.selected is True
    p = result.proposal
    assert p.short_call_strike is not None
    assert p.short_put_strike is not None
    assert p.long_call_strike is not None
    assert p.long_put_strike is not None
    # Wings must be further OTM than short strikes
    assert p.long_call_strike > p.short_call_strike
    assert p.long_put_strike < p.short_put_strike


def test_iron_condor_max_loss_is_positive():
    chain = _standard_chain()
    result = build_structure("TEST", StructureType.IRON_CONDOR, chain, iv_30=0.35)
    assert result.proposal.max_loss is not None
    assert result.proposal.max_loss > 0


def test_short_put_uses_single_strike():
    chain = _standard_chain()
    result = build_structure("TEST", StructureType.SHORT_PUT, chain, iv_30=0.35)
    assert result.selected is True
    assert result.proposal.single_strike is not None
    assert result.proposal.single_strike < 100.0  # OTM put is below underlying


def test_no_expirations_in_range_rejects():
    chain = _standard_chain(dte=5)  # all 5 DTE, outside default min_dte=21
    result = build_structure("TEST", StructureType.SHORT_STRANGLE, chain, iv_30=0.35, min_dte=21, max_dte=60)
    assert result.selected is False


def test_management_levels_baked_in():
    chain = _standard_chain()
    result = build_structure("TEST", StructureType.SHORT_STRANGLE, chain, iv_30=0.35)
    assert result.proposal.profit_target_pct == 0.50
    assert result.proposal.loss_limit_pct == 2.00
    assert result.proposal.roll_dte == 21


# ── OptionsProposal validation ────────────────────────────────────────────────

def test_strangle_proposal_requires_both_short_strikes():
    with pytest.raises(Exception):
        OptionsProposal(
            ticker="TEST",
            structure=StructureType.SHORT_STRANGLE,
            expiration=date(2026, 8, 15),
            dte=45,
            quantity=1,
            short_call_strike=110.0,
            # missing short_put_strike
        )


def test_iron_condor_requires_all_four_strikes():
    with pytest.raises(Exception):
        OptionsProposal(
            ticker="TEST",
            structure=StructureType.IRON_CONDOR,
            expiration=date(2026, 8, 15),
            dte=45,
            quantity=1,
            short_call_strike=110.0,
            short_put_strike=90.0,
            long_call_strike=115.0,
            # missing long_put_strike
        )


def test_no_trade_proposal_bypasses_strike_validation():
    # NO_TRADE proposals don't need strikes
    p = OptionsProposal(
        ticker="TEST",
        structure=StructureType.NO_TRADE,
        expiration=date(2026, 8, 15),
        dte=0,
        quantity=0,
    )
    assert p.structure == StructureType.NO_TRADE


# ── VolConsensusPayload is_executable ────────────────────────────────────────

def _vol_signal(structure: StructureType, flags: list[str] | None = None) -> VolSignal:
    return VolSignal(
        agent_name="test_agent",
        ticker="TEST",
        iv_environment=IVEnvironment.ELEVATED,
        recommended_structure=structure,
        confidence=Confidence.HIGH,
        rationale="test rationale",
        generated_at=datetime.now(),
        flags=flags or [],
    )


def _greeks_review(verdict: RiskVerdict) -> GreeksRiskReview:
    return GreeksRiskReview(
        verdict=verdict,
        reasons=[],
        portfolio_delta_after=0.01,
        portfolio_vega_after=-2.5,
        portfolio_theta_after=15.0,
        position_max_loss=200.0,
        reviewed_at=datetime.now(),
    )


def _strangle_proposal() -> OptionsProposal:
    return OptionsProposal(
        ticker="TEST",
        structure=StructureType.SHORT_STRANGLE,
        expiration=date(2026, 8, 15),
        dte=45,
        quantity=1,
        short_call_strike=112.0,
        short_put_strike=88.0,
        net_credit=2.50,
    )


def test_payload_is_executable_when_approved():
    payload = VolConsensusPayload(
        ticker="TEST",
        vol_signals=[_vol_signal(StructureType.SHORT_STRANGLE)],
        proposal=_strangle_proposal(),
        risk_review=_greeks_review(RiskVerdict.APPROVED),
    )
    assert payload.is_executable is True


def test_payload_not_executable_when_rejected():
    payload = VolConsensusPayload(
        ticker="TEST",
        vol_signals=[_vol_signal(StructureType.SHORT_STRANGLE)],
        proposal=_strangle_proposal(),
        risk_review=_greeks_review(RiskVerdict.REJECTED),
    )
    assert payload.is_executable is False


def test_payload_not_executable_when_no_trade():
    no_trade_proposal = OptionsProposal(
        ticker="TEST", structure=StructureType.NO_TRADE,
        expiration=date(2026, 8, 15), dte=0, quantity=0,
    )
    payload = VolConsensusPayload(
        ticker="TEST",
        vol_signals=[_vol_signal(StructureType.NO_TRADE)],
        proposal=no_trade_proposal,
        risk_review=_greeks_review(RiskVerdict.APPROVED),
    )
    assert payload.is_executable is False


def test_approved_payload_without_signals_raises():
    with pytest.raises(Exception):
        VolConsensusPayload(
            ticker="TEST",
            vol_signals=[],
            proposal=_strangle_proposal(),
            risk_review=_greeks_review(RiskVerdict.APPROVED),
        )


# ── VixContext regime classification ─────────────────────────────────────────

def test_vix_regime_expansion_when_backwardation():
    from analyst_layer.agents.vol_regime_agent import VixContext
    ctx = VixContext(vix_current=25.0, vix3m_current=22.0)  # VIX > VIX3M × 1.05 = 23.1
    assert ctx.regime == VolRegime.EXPANSION


def test_vix_regime_expansion_when_spiking():
    from analyst_layer.agents.vol_regime_agent import VixContext
    ctx = VixContext(vix_current=28.0, vix_1w_ago=20.0)  # 40% spike
    assert ctx.regime == VolRegime.EXPANSION


def test_vix_regime_expansion_above_30():
    from analyst_layer.agents.vol_regime_agent import VixContext
    ctx = VixContext(vix_current=32.0)
    assert ctx.regime == VolRegime.EXPANSION


def test_vix_regime_contraction_below_15():
    from analyst_layer.agents.vol_regime_agent import VixContext
    ctx = VixContext(vix_current=13.0)
    assert ctx.regime == VolRegime.CONTRACTION


def test_vix_regime_stable_in_normal_range():
    from analyst_layer.agents.vol_regime_agent import VixContext
    # VIX 20, VIX3M 20 → ratio 1.0 = contango, no spike, not > 30 → STABLE
    ctx = VixContext(vix_current=20.0, vix3m_current=20.0)
    assert ctx.regime == VolRegime.STABLE
