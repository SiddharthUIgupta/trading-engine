"""Options structure builder — deterministic, zero-LLM.

Given the vol agents' consensus on structure type and the live options chain,
picks the specific strikes and expiration that implement the chosen structure.

Strike selection follows tastylive's 16-delta rule: the short strikes are
placed approximately 1 standard deviation OTM. The standard deviation is
estimated from implied volatility and DTE using the Black-Scholes log-normal
approximation (no model calibration required — this is the same approximation
a floor trader would use in their head):

    1-SD move = underlying_price × IV_30 × sqrt(DTE / 365)

The 16-delta strike is roughly 1 SD OTM, which means roughly a 16% probability
(by the log-normal model) of expiring in-the-money. tastylive's mechanical
research on thousands of short strangles/iron condors shows this gives the
best combination of premium collected vs probability of profit.

Wings for iron condors are set 1 additional SD further OTM (so the spread
width is roughly 1 SD) — wide enough to collect meaningful premium, narrow
enough that the max loss stays within the 5% portfolio limit.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from analyst_layer.schemas import OptionsProposal, StructureType
from data_layer.models import OptionContract, OptionType


@dataclass(frozen=True)
class StructureResult:
    selected: bool
    reasons: list[str] = field(default_factory=list)
    proposal: OptionsProposal | None = None


def build_structure(
    ticker: str,
    structure_type: StructureType,
    chain: list[OptionContract],
    iv_30: float,
    target_dte: int = 45,
    min_dte: int = 21,
    max_dte: int = 60,
    target_delta: float = 0.16,
    wing_width_sd: float = 1.0,
    quantity: int = 1,
) -> StructureResult:
    """Build an OptionsProposal for the given structure type.

    Args:
        ticker: underlying symbol
        structure_type: the structure chosen by the vol consensus agents
        chain: full options chain for this underlying
        iv_30: 30-day implied vol (annualized decimal, e.g. 0.30)
        target_dte: preferred DTE (tastylive: 45)
        min_dte / max_dte: acceptable DTE range
        target_delta: OTM target delta (tastylive: 0.16 ≈ 1 SD)
        wing_width_sd: width of iron condor wings in SD units
        quantity: number of contracts
    """
    if structure_type == StructureType.NO_TRADE:
        return StructureResult(selected=False, reasons=["structure type is NO_TRADE"])

    # ── 1. Select expiration ──────────────────────────────────────────────────
    expirations = sorted({c.dte for c in chain if min_dte <= c.dte <= max_dte})
    if not expirations:
        return StructureResult(
            selected=False,
            reasons=[f"no expirations in DTE range [{min_dte}, {max_dte}]"],
        )
    # Pick the expiration closest to target_dte
    selected_dte = min(expirations, key=lambda d: abs(d - target_dte))
    exp_contracts = [c for c in chain if c.dte == selected_dte]

    if not exp_contracts:
        return StructureResult(selected=False, reasons=[f"no contracts at DTE {selected_dte}"])

    expiration = exp_contracts[0].expiration
    underlying = exp_contracts[0].underlying_price

    # ── 2. Compute 1-SD strike distance (Black-Scholes approximation) ─────────
    if iv_30 <= 0 or underlying <= 0:
        return StructureResult(selected=False, reasons=["invalid IV or underlying price for strike computation"])

    one_sd = underlying * iv_30 * math.sqrt(selected_dte / 365)

    # ── 3. Find the short strikes at ~1 SD OTM ───────────────────────────────
    call_target = underlying + one_sd
    put_target = underlying - one_sd

    liquid_calls = [
        c for c in exp_contracts
        if c.option_type == OptionType.CALL and c.bid > 0 and c.ask > 0 and c.strike > underlying
    ]
    liquid_puts = [
        c for c in exp_contracts
        if c.option_type == OptionType.PUT and c.bid > 0 and c.ask > 0 and c.strike < underlying
    ]

    if not liquid_calls or not liquid_puts:
        return StructureResult(
            selected=False,
            reasons=["insufficient liquid OTM contracts to build structure"],
        )

    short_call = min(liquid_calls, key=lambda c: abs(c.strike - call_target))
    short_put = min(liquid_puts, key=lambda c: abs(c.strike - put_target))

    short_call_mid = (short_call.bid + short_call.ask) / 2
    short_put_mid = (short_put.bid + short_put.ask) / 2
    total_credit = short_call_mid + short_put_mid

    # ── 4. Build structure-specific proposal ─────────────────────────────────

    if structure_type == StructureType.SHORT_STRANGLE:
        return StructureResult(
            selected=True,
            reasons=[
                f"short strangle: sell {short_call.strike}C + {short_put.strike}P "
                f"@ {selected_dte}d, credit ${total_credit:.2f}/share, "
                f"underlying ${underlying:.2f}, 1-SD = ${one_sd:.2f}"
            ],
            proposal=OptionsProposal(
                ticker=ticker,
                structure=StructureType.SHORT_STRANGLE,
                expiration=expiration,
                dte=selected_dte,
                quantity=quantity,
                short_call_strike=short_call.strike,
                short_put_strike=short_put.strike,
                net_credit=round(total_credit, 2),
                max_loss=None,  # undefined risk
            ),
        )

    if structure_type == StructureType.IRON_CONDOR:
        # Wings 1 additional SD further OTM
        long_call_target = underlying + one_sd * (1 + wing_width_sd)
        long_put_target = underlying - one_sd * (1 + wing_width_sd)

        wing_calls = [c for c in liquid_calls if c.strike > short_call.strike]
        wing_puts = [c for c in liquid_puts if c.strike < short_put.strike]

        if not wing_calls or not wing_puts:
            # Fall back to next available strike if no wing at exact target
            wing_calls = wing_calls or [c for c in liquid_calls if c.strike > short_call.strike]
            wing_puts = wing_puts or [c for c in liquid_puts if c.strike < short_put.strike]

        if not wing_calls or not wing_puts:
            return StructureResult(
                selected=False,
                reasons=["cannot build iron condor — no liquid wing strikes available"],
            )

        long_call = min(wing_calls, key=lambda c: abs(c.strike - long_call_target))
        long_put = min(wing_puts, key=lambda c: abs(c.strike - long_put_target))

        long_call_mid = (long_call.bid + long_call.ask) / 2
        long_put_mid = (long_put.bid + long_put.ask) / 2
        net_credit = total_credit - long_call_mid - long_put_mid
        call_spread_width = long_call.strike - short_call.strike
        put_spread_width = short_put.strike - long_put.strike
        max_loss = max(call_spread_width, put_spread_width) - net_credit

        return StructureResult(
            selected=True,
            reasons=[
                f"iron condor: sell {short_call.strike}C/{short_put.strike}P, "
                f"buy {long_call.strike}C/{long_put.strike}P @ {selected_dte}d, "
                f"net credit ${net_credit:.2f}/share, max loss ${max_loss:.2f}/share"
            ],
            proposal=OptionsProposal(
                ticker=ticker,
                structure=StructureType.IRON_CONDOR,
                expiration=expiration,
                dte=selected_dte,
                quantity=quantity,
                short_call_strike=short_call.strike,
                short_put_strike=short_put.strike,
                long_call_strike=long_call.strike,
                long_put_strike=long_put.strike,
                net_credit=round(max(net_credit, 0), 2),
                max_loss=round(max_loss, 2),
            ),
        )

    if structure_type in (StructureType.SHORT_PUT, StructureType.SHORT_PUT_SPREAD):
        credit = short_put_mid
        max_loss = short_put.strike - credit if structure_type == StructureType.SHORT_PUT else None

        if structure_type == StructureType.SHORT_PUT_SPREAD:
            long_put_target = underlying - one_sd * (1 + wing_width_sd)
            wing_puts = [c for c in liquid_puts if c.strike < short_put.strike]
            if not wing_puts:
                return StructureResult(selected=False, reasons=["no liquid wing puts for short put spread"])
            long_put = min(wing_puts, key=lambda c: abs(c.strike - long_put_target))
            long_put_mid = (long_put.bid + long_put.ask) / 2
            credit = short_put_mid - long_put_mid
            spread_width = short_put.strike - long_put.strike
            max_loss = spread_width - credit

        return StructureResult(
            selected=True,
            reasons=[
                f"{structure_type.value}: sell {short_put.strike}P @ {selected_dte}d, "
                f"credit ${credit:.2f}/share"
            ],
            proposal=OptionsProposal(
                ticker=ticker,
                structure=structure_type,
                expiration=expiration,
                dte=selected_dte,
                quantity=quantity,
                single_strike=short_put.strike,
                net_credit=round(max(credit, 0), 2),
                max_loss=round(max_loss, 2) if max_loss is not None else None,
            ),
        )

    if structure_type in (StructureType.SHORT_CALL, StructureType.SHORT_CALL_SPREAD):
        credit = short_call_mid
        max_loss = None  # naked call: theoretically unlimited

        if structure_type == StructureType.SHORT_CALL_SPREAD:
            long_call_target = underlying + one_sd * (1 + wing_width_sd)
            wing_calls = [c for c in liquid_calls if c.strike > short_call.strike]
            if not wing_calls:
                return StructureResult(selected=False, reasons=["no liquid wing calls for short call spread"])
            long_call = min(wing_calls, key=lambda c: abs(c.strike - long_call_target))
            long_call_mid = (long_call.bid + long_call.ask) / 2
            credit = short_call_mid - long_call_mid
            spread_width = long_call.strike - short_call.strike
            max_loss = spread_width - credit

        return StructureResult(
            selected=True,
            reasons=[
                f"{structure_type.value}: sell {short_call.strike}C @ {selected_dte}d, "
                f"credit ${credit:.2f}/share"
            ],
            proposal=OptionsProposal(
                ticker=ticker,
                structure=structure_type,
                expiration=expiration,
                dte=selected_dte,
                quantity=quantity,
                single_strike=short_call.strike,
                net_credit=round(max(credit, 0), 2),
                max_loss=round(max_loss, 2) if max_loss is not None else None,
            ),
        )

    if structure_type == StructureType.CALENDAR:
        # Calendar: sell front month (min_dte), buy back month (target_dte)
        front_dtes = sorted({c.dte for c in chain if min_dte <= c.dte <= 30})
        back_dtes = sorted({c.dte for c in chain if 30 < c.dte <= max_dte})
        if not front_dtes or not back_dtes:
            return StructureResult(selected=False, reasons=["insufficient expirations for calendar spread"])
        front_dte = front_dtes[0]
        back_dte_val = min(back_dtes, key=lambda d: abs(d - target_dte))
        front_calls = [c for c in chain if c.dte == front_dte and c.option_type == OptionType.CALL and c.bid > 0]
        back_calls = [c for c in chain if c.dte == back_dte_val and c.option_type == OptionType.CALL and c.bid > 0]
        if not front_calls or not back_calls:
            return StructureResult(selected=False, reasons=["insufficient liquid calls for calendar"])
        atm_front = min(front_calls, key=lambda c: abs(c.strike - underlying))
        atm_back = min(back_calls, key=lambda c: abs(c.strike - underlying))
        # Calendar debit: pay more for back, collect front
        debit = (atm_back.ask - atm_front.bid)
        return StructureResult(
            selected=True,
            reasons=[
                f"calendar: sell {atm_front.strike}C {front_dte}d, "
                f"buy {atm_back.strike}C {back_dte_val}d, debit ${debit:.2f}/share"
            ],
            proposal=OptionsProposal(
                ticker=ticker,
                structure=StructureType.CALENDAR,
                expiration=atm_back.expiration,
                dte=back_dte_val,
                quantity=quantity,
                single_strike=atm_front.strike,
                net_credit=round(-debit, 2),  # negative = debit paid
                max_loss=round(debit, 2),
            ),
        )

    return StructureResult(selected=False, reasons=[f"unhandled structure type: {structure_type.value}"])


# ── Legacy directional contract selector (kept for the existing options track) ──
# The new vol-based system uses build_structure(). select_contract() remains
# for the runtime's directional BUY/SELL → ATM call/put track.

from analyst_layer.schemas import Action  # noqa: E402 — after dataclasses to avoid circularity


@dataclass(frozen=True)
class ContractSelection:
    selected: bool
    reasons: list[str] = field(default_factory=list)
    contract: OptionContract | None = None


def select_contract(
    chain: list[OptionContract], direction: Action, min_dte: int, max_dte: int
) -> ContractSelection:
    """Pick the nearest ATM contract in the DTE window for a directional play."""
    if direction == Action.HOLD:
        return ContractSelection(selected=False, reasons=["HOLD carries no directional view to express"])

    option_type = OptionType.CALL if direction == Action.BUY else OptionType.PUT
    candidates = [
        c for c in chain
        if c.option_type == option_type and min_dte <= c.dte <= max_dte and c.bid > 0 and c.ask > 0
    ]
    if not candidates:
        return ContractSelection(
            selected=False,
            reasons=[f"no liquid {option_type.value} contracts with {min_dte} <= dte <= {max_dte}"],
        )

    nearest_dte = min(c.dte for c in candidates)
    same_expiration = [c for c in candidates if c.dte == nearest_dte]
    best = min(same_expiration, key=lambda c: abs(c.strike - c.underlying_price))

    return ContractSelection(
        selected=True,
        contract=best,
        reasons=[
            f"{option_type.value} expiring {best.expiration.isoformat()} ({best.dte}d out, "
            f">= {min_dte}d floor) at strike {best.strike:.2f} (underlying {best.underlying_price:.2f}, "
            f"nearest available to at-the-money)"
        ],
    )
