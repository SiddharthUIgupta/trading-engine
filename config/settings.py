"""Central configuration. Every other layer reads risk limits and environment
mode from here — nothing downstream is allowed to define its own copy of
MAX_POSITION_SIZE_PCT, MAX_DAILY_DRAWDOWN_PCT, or the paper/live switch.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LIVE_CONFIRM_TOKEN = "I_UNDERSTAND_THIS_IS_LIVE_CAPITAL"
UNCOVERED_CONFIRM_TOKEN = "I_UNDERSTAND_STRANGLES_HAVE_UNBOUNDED_RISK"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM provider ---
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    # Risk Officer only — the one call whose output actually clamps a trade.
    anthropic_model: str = Field(default="claude-sonnet-4-6", alias="ANTHROPIC_MODEL")
    # The 3 narrow sub-agents (sentiment/fundamental/technical) only need to
    # interpret an already-computed number — a cheaper model is sufficient
    # and cuts most of the per-cycle token cost.
    anthropic_subagent_model: str = Field(default="claude-haiku-4-5-20251001", alias="ANTHROPIC_SUBAGENT_MODEL")

    # --- OpenBB ---
    openbb_pat: str = Field(default="", alias="OPENBB_PAT")

    # --- Alpaca ---
    alpaca_api_key: str = Field(default="", alias="ALPACA_API_KEY")
    alpaca_secret_key: str = Field(default="", alias="ALPACA_SECRET_KEY")

    # --- Environment gate. Defaults to paper no matter what's missing. ---
    trading_env: Literal["paper", "live"] = Field(default="paper", alias="TRADING_ENV")
    trading_live_confirm: str = Field(default="", alias="TRADING_LIVE_CONFIRM")

    # --- Hard risk guardrails ---
    max_position_size_pct: float = Field(default=0.05, alias="MAX_POSITION_SIZE_PCT")
    max_daily_drawdown_pct: float = Field(default=0.02, alias="MAX_DAILY_DRAWDOWN_PCT")
    # Profit target: once today's gain reaches this threshold the engine stops
    # trading stocks and locks in the gain. Set DAILY_PROFIT_TARGET_PCT in .env
    # for a equity-scaled target (e.g. 0.005 = 0.5% of day-start equity).
    # If PCT is set it overrides the flat USD value at the start of each day.
    # Set to None / omit to disable the profit lock entirely.
    daily_profit_target_usd: float = Field(default=50.0, alias="DAILY_PROFIT_TARGET_USD")
    daily_profit_target_pct: float | None = Field(default=None, alias="DAILY_PROFIT_TARGET_PCT")

    # --- State persistence ---
    state_db_path: Path = Field(default=Path("./state/trading_engine.sqlite3"), alias="STATE_DB_PATH")

    # --- Pre-filter (deterministic, no LLM) ---
    # Gates which tickers are even worth the 4-agent consensus today. Keeps
    # Claude spend tied to opportunity count, not watchlist size.
    filter_rsi_period: int = Field(default=14, alias="FILTER_RSI_PERIOD")
    filter_rsi_oversold: float = Field(default=30.0, alias="FILTER_RSI_OVERSOLD")
    filter_rsi_overbought: float = Field(default=70.0, alias="FILTER_RSI_OVERBOUGHT")
    filter_sma_short_window: int = Field(default=10, alias="FILTER_SMA_SHORT_WINDOW")
    filter_sma_long_window: int = Field(default=30, alias="FILTER_SMA_LONG_WINDOW")
    filter_volume_spike_multiplier: float = Field(default=2.0, alias="FILTER_VOLUME_SPIKE_MULTIPLIER")
    filter_sentiment_abs_threshold: float = Field(default=0.3, alias="FILTER_SENTIMENT_ABS_THRESHOLD")
    filter_recent_filing_days: int = Field(default=3, alias="FILTER_RECENT_FILING_DAYS")

    # --- Dynamic universe (replaces a fixed watchlist) ---
    # Built daily from OpenBB's active/gainers/losers discovery screens
    # instead of a fixed ticker list. prerank_limit bounds how many raw
    # movers get the heavier intraday-bars + float fetch needed for the
    # momentum scan; max_daily_candidates bounds how many PASS that scan
    # and reach the (paid) 4-agent consensus, regardless of how many qualify.
    dynamic_universe_enabled: bool = Field(default=True, alias="DYNAMIC_UNIVERSE_ENABLED")
    # Kept small (~50, not 150) because each prerank candidate costs 3 OpenBB
    # calls (intraday bars + float + daily volume history) in the momentum
    # scan — at a 30-min scan cadence this keeps Yahoo Finance call volume
    # well under rate-limit risk.
    universe_prerank_limit: int = Field(default=50, alias="UNIVERSE_PRERANK_LIMIT")
    max_daily_candidates: int = Field(default=50, alias="MAX_DAILY_CANDIDATES")

    # --- Low-float momentum scanner (deterministic, no LLM) ---
    # The user's 7-criteria spec — all seven are conjunctive (a candidate
    # must clear every one). See analyst_layer/momentum_scanner.py.
    momentum_max_float_shares: int = Field(default=20_000_000, alias="MOMENTUM_MAX_FLOAT_SHARES")
    momentum_ema_short_period: int = Field(default=9, alias="MOMENTUM_EMA_SHORT_PERIOD")
    momentum_ema_long_period: int = Field(default=20, alias="MOMENTUM_EMA_LONG_PERIOD")
    momentum_min_daily_gain_pct: float = Field(default=0.05, alias="MOMENTUM_MIN_DAILY_GAIN_PCT")
    momentum_clean_body_dominance_threshold: float = Field(
        default=0.55, alias="MOMENTUM_CLEAN_BODY_DOMINANCE_THRESHOLD"
    )
    momentum_clean_lookback_bars: int = Field(default=12, alias="MOMENTUM_CLEAN_LOOKBACK_BARS")
    # Relative volume: today's volume vs. its own recent average — the most
    # commonly cited "necessary" criterion in this style of scanner, even
    # more fundamental than VWAP/EMA.
    momentum_min_relative_volume: float = Field(default=2.0, alias="MOMENTUM_MIN_RELATIVE_VOLUME")
    momentum_volume_lookback_days: int = Field(default=10, alias="MOMENTUM_VOLUME_LOOKBACK_DAYS")
    # Price band: float + daily-gain alone still admit sub-$1 penny stocks
    # and $200+ names that don't behave like the low-float setups this
    # strategy targets.
    momentum_price_min: float = Field(default=1.0, alias="MOMENTUM_PRICE_MIN")
    momentum_price_max: float = Field(default=20.0, alias="MOMENTUM_PRICE_MAX")

    # --- Intraday exit rules (deterministic, no LLM by default) ---
    exit_stop_loss_pct: float = Field(default=0.02, alias="EXIT_STOP_LOSS_PCT")
    # No hard profit cap — trailing stop rides winners instead of capping them.
    # exit_trailing_stop_activation_pct is when the trailing stop kicks in;
    # exit_trailing_stop_pct is how far behind the peak it trails.
    # A stock up 3% starts being trailed at 1.5% behind its peak — so a
    # continuation from +3% to +12% only stops out around +10.5%, not at +3%.
    exit_trailing_stop_pct: float = Field(default=0.015, alias="EXIT_TRAILING_STOP_PCT")
    exit_trailing_stop_activation_pct: float = Field(default=0.03, alias="EXIT_TRAILING_STOP_ACTIVATION_PCT")
    # LLM exit review only fires when the rule-based checks above all say
    # "hold" AND the position's regime has sharply reversed since entry —
    # and at most once per position per day, logged like any other agent call.
    intraday_llm_escalation_enabled: bool = Field(default=True, alias="INTRADAY_LLM_ESCALATION_ENABLED")

    # --- Thesis track (deterministic screen, no LLM) ---
    # Opposite shape of the momentum track: looks for quality-pool names
    # having a quiet pullback, not stocks already moving fast. Runs once
    # daily, not every 30 min — fundamentals don't change intraday.
    # See analyst_layer/thesis_scanner.py.
    thesis_track_enabled: bool = Field(default=True, alias="THESIS_TRACK_ENABLED")
    thesis_min_pullback_pct: float = Field(default=0.20, alias="THESIS_MIN_PULLBACK_PCT")
    # Ceiling, not just a floor — beyond this, a pullback is statistically
    # much more likely to be a genuinely impaired business than a quality
    # name temporarily out of favor (failed trial, accounting issue,
    # secular decline, vs. RDW-style dislocation).
    thesis_max_pullback_pct: float = Field(default=0.50, alias="THESIS_MAX_PULLBACK_PCT")
    thesis_max_daily_candidates: int = Field(default=10, alias="THESIS_MAX_DAILY_CANDIDATES")
    # Wide stop, no fixed take-profit, trailing stop only engages once the
    # position is up significantly — the point is to let a winner run
    # instead of capping it at a few percent like the momentum track does.
    thesis_stop_loss_pct: float = Field(default=0.18, alias="THESIS_STOP_LOSS_PCT")
    thesis_trailing_stop_pct: float = Field(default=0.10, alias="THESIS_TRAILING_STOP_PCT")
    thesis_trailing_stop_activation_pct: float = Field(default=0.20, alias="THESIS_TRAILING_STOP_ACTIVATION_PCT")

    # --- Options track (long calls/puts only — no spreads, no writing) ---
    # CAPABILITY FLAG: set True once when your broker account has options
    # trading enabled (Level 1+). The regime assessment then decides daily
    # whether market conditions suit running this track — you never touch
    # this flag again after initial setup.
    # Same momentum signal as the equity momentum track, expressed via a
    # leveraged, defined-risk instrument instead of shares. Deliberately
    # NOT 0-1 DTE: a fast-decaying near-term contract can be right about
    # direction and still expire worthless if the move is too slow. The
    # DTE floor below is what actually prevents that, not just habit.
    options_track_enabled: bool = Field(default=True, alias="OPTIONS_TRACK_ENABLED")
    options_min_dte: int = Field(default=5, alias="OPTIONS_MIN_DTE")
    options_max_dte: int = Field(default=10, alias="OPTIONS_MAX_DTE")
    # Much smaller than the 5% equity cap — sized off premium paid (max
    # loss on a long option), not share notional, and one contract already
    # carries embedded leverage the equity cap was never calibrated for.
    options_max_risk_pct: float = Field(default=0.01, alias="OPTIONS_MAX_RISK_PCT")
    options_stop_loss_pct: float = Field(default=0.40, alias="OPTIONS_STOP_LOSS_PCT")
    # Force-close regardless of P&L this many trading days before expiration
    # — avoids riding into the sharp theta/gamma acceleration in the final
    # days, which the stop-loss alone won't reliably catch in time.
    options_force_close_days_before_expiration: int = Field(default=2, alias="OPTIONS_FORCE_CLOSE_DAYS_BEFORE_EXPIRATION")

    # --- Vol options track (short premium — Natenberg/tastylive framework) ---
    # CAPABILITY FLAG: set True once when your broker account has options
    # writing approval (Level 2+ for spreads/iron condors, Level 3 for
    # covered straddles and multi-leg). The regime assessment then decides
    # daily whether VIX conditions suit premium selling — you never touch
    # this flag again after initial setup.
    # Structure selection is driven by IV Rank and vol regime consensus, not
    # direction. DTE targets are tastylive's 30-45 range; management rules
    # (50% profit, 2x loss, 21 DTE roll) are baked into the OptionsProposal.
    vol_options_track_enabled: bool = Field(default=True, alias="VOL_OPTIONS_TRACK_ENABLED")
    vol_options_target_dte: int = Field(default=45, alias="VOL_OPTIONS_TARGET_DTE")
    vol_options_min_dte: int = Field(default=21, alias="VOL_OPTIONS_MIN_DTE")
    vol_options_max_dte: int = Field(default=60, alias="VOL_OPTIONS_MAX_DTE")
    # Capital at risk per position — for defined-risk structures (iron condor,
    # spreads) this is the max loss; for undefined-risk (strangle) the Greeks
    # Risk Officer enforces a portfolio-level vega limit instead.
    vol_options_max_risk_pct: float = Field(default=0.02, alias="VOL_OPTIONS_MAX_RISK_PCT")
    # tastylive mechanical management thresholds
    vol_options_profit_target_pct: float = Field(default=0.50, alias="VOL_OPTIONS_PROFIT_TARGET_PCT")
    vol_options_loss_limit_multiplier: float = Field(default=2.00, alias="VOL_OPTIONS_LOSS_LIMIT_MULTIPLIER")
    vol_options_roll_dte: int = Field(default=21, alias="VOL_OPTIONS_ROLL_DTE")
    # --- Vol universe screener ---
    # Screens a dynamic candidate pool (market movers + seed watchlist) for
    # options liquidity before running the vol consensus. Names that don't clear
    # all three criteria are excluded so the system never tries to sell premium
    # into an illiquid market where mid-price fills and tight spreads won't happen.
    vol_universe_min_option_oi: int = Field(default=500, alias="VOL_UNIVERSE_MIN_OPTION_OI")
    # Max bid/ask spread as a fraction of mid for the ATM call and put.
    # 10% is permissive but catches names with truly broken quotes (bid=0, ask=3.00).
    # Tighten to 0.05 on a fast machine with real-time quotes; 0.10 is safe with yfinance.
    vol_universe_max_spread_pct: float = Field(default=0.10, alias="VOL_UNIVERSE_MAX_SPREAD_PCT")
    # Cap on the resulting universe. 20 names is enough diversification for a
    # short-premium book at this scale; beyond that, Greeks limits start binding.
    vol_universe_max_size: int = Field(default=20, alias="VOL_UNIVERSE_MAX_SIZE")

    # Enabling uncovered (naked) strangles requires TWO independent env vars to agree —
    # same ceremony as the live-trading gate. Naked short calls have theoretically
    # unbounded loss; this is a risk-profile change, not a config tweak.
    # Requires BOTH:
    #   VOL_OPTIONS_ALLOW_UNCOVERED=true
    #   VOL_OPTIONS_UNCOVERED_CONFIRM="I_UNDERSTAND_STRANGLES_HAVE_UNBOUNDED_RISK"
    # Any disagreement silently keeps iron condor (defined risk) — the correct
    # failure mode. If False (or the confirm token is missing), SHORT_STRANGLE is
    # downgraded to IRON_CONDOR in the greeks_risk_node before execution.
    vol_options_allow_uncovered: bool = Field(default=False, alias="VOL_OPTIONS_ALLOW_UNCOVERED")
    vol_options_uncovered_confirm: str = Field(default="", alias="VOL_OPTIONS_UNCOVERED_CONFIRM")

    @field_validator("max_position_size_pct", "max_daily_drawdown_pct", "options_max_risk_pct", "options_stop_loss_pct", "vol_options_max_risk_pct", "vol_options_profit_target_pct", "exit_trailing_stop_activation_pct", "exit_trailing_stop_pct", "exit_stop_loss_pct", "vol_universe_max_spread_pct")
    @classmethod
    def _fraction_in_range(cls, v: float) -> float:
        if not (0 < v <= 1):
            raise ValueError("risk limit fractions must be in (0, 1]")
        return v

    @model_validator(mode="after")
    def _options_dte_band_is_sane(self) -> "Settings":
        if self.options_min_dte < 1:
            raise ValueError("options_min_dte must be >= 1 — same-day expiration is out of scope for this track")
        if self.options_max_dte < self.options_min_dte:
            raise ValueError("options_max_dte must be >= options_min_dte")
        return self

    @field_validator("daily_profit_target_usd")
    @classmethod
    def _profit_target_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("daily_profit_target_usd must be > 0")
        return v

    @model_validator(mode="after")
    def _enforce_paper_default(self) -> "Settings":
        """The only way to flip live is both TRADING_ENV=live AND the literal
        confirmation token. Any other combination is forced back to paper.
        """
        if self.trading_env == "live" and self.trading_live_confirm != LIVE_CONFIRM_TOKEN:
            object.__setattr__(self, "trading_env", "paper")
        return self

    @model_validator(mode="after")
    def _enforce_uncovered_default(self) -> "Settings":
        """Naked strangles (undefined risk) require TWO independent signals —
        same ceremony as the live gate. A missing or wrong confirm token silently
        forces iron condor (defined risk). Never raises; always degrades safely.
        """
        if self.vol_options_allow_uncovered and self.vol_options_uncovered_confirm != UNCOVERED_CONFIRM_TOKEN:
            object.__setattr__(self, "vol_options_allow_uncovered", False)
        return self

    @property
    def is_live(self) -> bool:
        return self.trading_env == "live" and self.trading_live_confirm == LIVE_CONFIRM_TOKEN

    @property
    def is_uncovered_allowed(self) -> bool:
        return self.vol_options_allow_uncovered and self.vol_options_uncovered_confirm == UNCOVERED_CONFIRM_TOKEN


def get_settings() -> Settings:
    return Settings()
