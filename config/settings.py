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
    # Raised from 2% to 5%: swing trades need room to breathe intraday.
    # Individual per-position stops (EXIT_STOP_LOSS_PCT) are the primary
    # risk guard; this circuit breaker only stops NEW entries when the whole
    # portfolio is down 5% in a day, and no longer force-closes existing positions.
    max_daily_drawdown_pct: float = Field(default=0.05, alias="MAX_DAILY_DRAWDOWN_PCT")
    # Weekly drawdown limit — halts ALL strategies from Monday open until next
    # Monday if cumulative weekly loss exceeds this fraction of week-start equity.
    max_weekly_drawdown_pct: float = Field(default=0.08, alias="MAX_WEEKLY_DRAWDOWN_PCT")
    # Trailing drawdown limit — halts ALL strategies if account drops this far
    # from its all-time equity peak. Requires manual reset() to resume.
    max_trailing_drawdown_pct: float = Field(default=0.20, alias="MAX_TRAILING_DRAWDOWN_PCT")
    # Soft brake: after this many consecutive losses on a strategy, halve its
    # position sizing until the next win. Resets automatically on a winning trade.
    consecutive_loss_limit: int = Field(default=3, alias="CONSECUTIVE_LOSS_LIMIT")
    # Pre-market gap scanner (9:05 AM ET job)
    gap_scan_min_pct: float = Field(default=0.05, alias="GAP_SCAN_MIN_PCT")
    gap_scan_max_candidates: int = Field(default=5, alias="GAP_SCAN_MAX_CANDIDATES")
    # Profit target: once today's gain reaches this threshold the engine stops
    # trading stocks and locks in the gain. Set DAILY_PROFIT_TARGET_PCT in .env
    # for a equity-scaled target (e.g. 0.005 = 0.5% of day-start equity).
    # If PCT is set it overrides the flat USD value at the start of each day.
    # Set to None / omit to disable the profit lock entirely.
    # Raised from $50: a $50 ceiling halted new entries after any single
    # moderate win, throttling the thesis track before it could build a
    # full day's position. PCT-based target (2% of equity) scales correctly.
    daily_profit_target_usd: float = Field(default=500.0, alias="DAILY_PROFIT_TARGET_USD")
    daily_profit_target_pct: float | None = Field(default=0.02, alias="DAILY_PROFIT_TARGET_PCT")

    # --- Per-agent capital allocation (fractions of total equity, must sum ≤ 1) ---
    # Each agent can deploy up to its slice. The shared pool (equity - all deployed)
    # acts as a secondary cap — no single agent can crowd out the others.
    intraday_capital_pct: float = Field(default=0.35, alias="INTRADAY_CAPITAL_PCT")
    options_capital_pct: float = Field(default=0.35, alias="OPTIONS_CAPITAL_PCT")
    thesis_capital_pct: float = Field(default=0.10, alias="THESIS_CAPITAL_PCT")
    swing_capital_pct: float = Field(default=0.20, alias="SWING_CAPITAL_PCT")

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
    # Hard cap on concurrent equity positions. Prevents the scan from exhausting
    # buying power by placing limit orders on every ORB signal in one cycle.
    max_open_equity_positions: int = Field(default=15, alias="MAX_OPEN_EQUITY_POSITIONS")
    # Hard cap on concurrent options positions (directional calls/puts from the
    # options ORB track). Without this the track accumulates 30+ positions all
    # paying theta simultaneously, which bleeds premium on every tick.
    max_open_options_positions: int = Field(default=8, alias="MAX_OPEN_OPTIONS_POSITIONS")

    # --- ORB equity quality filters ---
    # Minimum gap from prior close for a valid ORB long setup. Stocks that
    # gapped up ≥2% already have pre-market conviction; flat opens produce
    # far more false breakouts because there's no directional catalyst.
    orb_min_gap_pct: float = Field(default=0.04, alias="ORB_MIN_GAP_PCT")
    # Volume confirmation: breakout bar must exceed this multiple of the
    # opening-range average volume. Raised from 1.5x — 2x is the threshold
    # where institutional participation is meaningfully confirmed vs. retail noise.
    orb_volume_confirmation_multiple: float = Field(default=2.0, alias="ORB_VOLUME_CONFIRMATION_MULTIPLE")
    # SPY direction gate: only trade long ORB when SPY is green on the day.
    # ORB long breakouts on red-tape days fail at a significantly higher rate
    # because sector rotations and macro selling pressure absorb the move.
    orb_require_spy_positive: bool = Field(default=True, alias="ORB_REQUIRE_SPY_POSITIVE")
    # Float cap: small-float stocks produce larger % moves on the same dollar volume.
    # 0 = disabled. At 30M the screen focuses on names where ORB squeeze is meaningful.
    orb_max_float_shares: int = Field(default=0, alias="ORB_MAX_FLOAT_SHARES")
    # Minimum gain % for an ORB winner to be converted to a swing hold at EOD
    # instead of closing. 3% means the breakout showed real conviction; below
    # that it's noise and should still close same-day per the intraday design.
    orb_swing_convert_pct: float = Field(default=0.03, alias="ORB_SWING_CONVERT_PCT")
    # On red SPY days, route confirmed short ORB breakdowns to put options instead
    # of discarding the signal. Higher-quality bearish entries than the general
    # options scan — each one is a specific breakdown confirmed by ORB + market direction.
    orb_spy_red_puts_enabled: bool = Field(default=True, alias="ORB_SPY_RED_PUTS_ENABLED")

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
    # Min price filters out true sub-$5 micro-caps. No max — quality stocks
    # trade at any price. Previously capped at $20 which excluded most quality names.
    momentum_price_min: float = Field(default=5.0, alias="MOMENTUM_PRICE_MIN")
    momentum_price_max: float = Field(default=0.0, alias="MOMENTUM_PRICE_MAX")  # 0 = no cap

    # --- Intraday exit rules (deterministic, no LLM by default) ---
    # Per-position stop-loss. Widened from 2% to 7% for swing trading —
    # 2% is intraday noise, not a meaningful signal that a trade is wrong.
    exit_stop_loss_pct: float = Field(default=0.07, alias="EXIT_STOP_LOSS_PCT")
    # Trailing stop activates at 15% gain, trails 7% behind peak.
    # Widened from 3%/1.5% to give swing trades room to develop and ride
    # multi-day trends without getting shaken out on normal pullbacks.
    exit_trailing_stop_pct: float = Field(default=0.07, alias="EXIT_TRAILING_STOP_PCT")
    exit_trailing_stop_activation_pct: float = Field(default=0.15, alias="EXIT_TRAILING_STOP_ACTIVATION_PCT")
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
    # Raised from 10: backtest shows 58% win rate and +6.27% avg return on thesis.
    # More candidates = more quality setups reviewed by the LLM each day.
    thesis_max_daily_candidates: int = Field(default=20, alias="THESIS_MAX_DAILY_CANDIDATES")
    # Wide stop, no fixed take-profit, trailing stop only engages once the
    # position is up significantly — the point is to let a winner run
    # instead of capping it at a few percent like the momentum track does.
    thesis_stop_loss_pct: float = Field(default=0.18, alias="THESIS_STOP_LOSS_PCT")
    thesis_trailing_stop_pct: float = Field(default=0.10, alias="THESIS_TRAILING_STOP_PCT")
    thesis_trailing_stop_activation_pct: float = Field(default=0.20, alias="THESIS_TRAILING_STOP_ACTIVATION_PCT")

    # --- Recovery scanner (market rebound / oversold bounce plays) ---
    # Deterministic screen: pulled back from 60-day high but 5d momentum
    # positive, above MA20, volume picking up. Feeds into the same consensus
    # pipeline as thesis (LLM agents vote; risk officer gates it). Runs
    # alongside thesis_scan_and_trade at 8:15am using the same universe.
    recovery_track_enabled: bool = Field(default=True, alias="RECOVERY_TRACK_ENABLED")
    recovery_min_pullback_pct: float = Field(default=0.15, alias="RECOVERY_MIN_PULLBACK_PCT")
    recovery_max_pullback_pct: float = Field(default=0.40, alias="RECOVERY_MAX_PULLBACK_PCT")
    recovery_volume_pickup_ratio: float = Field(default=1.20, alias="RECOVERY_VOLUME_PICKUP_RATIO")
    recovery_max_daily_candidates: int = Field(default=10, alias="RECOVERY_MAX_DAILY_CANDIDATES")

    # --- Swing trade track (3–6 week holds, trend-following with news/regime exits) ---
    # Capability flag: set False to disable this track entirely.
    swing_track_enabled: bool = Field(default=True, alias="SWING_TRACK_ENABLED")
    # Per-position hard stop — 8% from avg entry. Wider than intraday (7%) to
    # give multi-week trades room to breathe through normal intraday volatility.
    swing_stop_loss_pct: float = Field(default=0.08, alias="SWING_STOP_LOSS_PCT")
    # Trailing stop activates at 12% gain, trails 7% behind peak.
    swing_trailing_stop_pct: float = Field(default=0.07, alias="SWING_TRAILING_STOP_PCT")
    swing_trailing_stop_activation_pct: float = Field(default=0.12, alias="SWING_TRAILING_STOP_ACTIVATION_PCT")
    # Force-exit after this many calendar days regardless of P&L — prevents
    # swing positions from silently becoming thesis-style multi-month holds.
    swing_max_hold_days: int = Field(default=21, alias="SWING_MAX_HOLD_DAYS")
    # Hard cap on concurrent swing positions.
    swing_max_open_positions: int = Field(default=5, alias="SWING_MAX_OPEN_POSITIONS")

    # --- Macro news agent (pre-market, runs before regime assessment) ---
    # LLM generates search queries tuned to today's date, fetches headlines,
    # then scores overall US equity market sentiment. The result adjusts the
    # effective VIX used by assess_daily_regime, potentially arming tracks
    # that would otherwise sit just above a threshold. Uses the cheap
    # subagent model (Haiku). Two LLM calls per day.
    macro_news_enabled: bool = Field(default=True, alias="MACRO_NEWS_ENABLED")
    # Maximum VIX adjustment from news sentiment (in VIX points), scaled by
    # confidence. E.g. at 3.0 + confidence=0.8: effective VIX shifts by 2.4.
    macro_news_vix_adjustment: float = Field(default=3.0, alias="MACRO_NEWS_VIX_ADJUSTMENT")
    # Minimum confidence to apply any adjustment. Below this, news is too
    # uncertain to override pure technical data.
    macro_news_min_confidence: float = Field(default=0.6, alias="MACRO_NEWS_MIN_CONFIDENCE")

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
    # Disabled by default: ORB options accumulated 30 losing positions because
    # ORB is an intraday signal but we were holding 30-45 DTE contracts. The
    # intraday exit (OPTIONS_INTRADAY_STOP_PCT) partially fixes this, but the
    # core edge is weak. Re-enable only with a clear backtested thesis.
    options_track_enabled: bool = Field(default=False, alias="OPTIONS_TRACK_ENABLED")
    # 30-45 DTE for swing trading: gives the trade room to work over days
    # without being destroyed by theta in the final week. Previously 5-10
    # DTE which was essentially a same-week lottery ticket.
    options_min_dte: int = Field(default=30, alias="OPTIONS_MIN_DTE")
    options_max_dte: int = Field(default=45, alias="OPTIONS_MAX_DTE")
    # Much smaller than the 5% equity cap — sized off premium paid (max
    # loss on a long option), not share notional, and one contract already
    # carries embedded leverage the equity cap was never calibrated for.
    options_max_risk_pct: float = Field(default=0.01, alias="OPTIONS_MAX_RISK_PCT")
    options_stop_loss_pct: float = Field(default=0.50, alias="OPTIONS_STOP_LOSS_PCT")
    # Same-day exit: if an ORB option is down ≥ this % by 3pm ET, close it.
    # The ORB signal resolves intraday — a stalled breakout by late afternoon
    # should not be carried overnight as a multi-week theta bleed.
    options_intraday_stop_pct: float = Field(default=0.20, alias="OPTIONS_INTRADAY_STOP_PCT")
    # Force-close at 7 DTE to avoid gamma/theta spike in final week.
    options_force_close_days_before_expiration: int = Field(default=7, alias="OPTIONS_FORCE_CLOSE_DAYS_BEFORE_EXPIRATION")

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

    @field_validator("max_position_size_pct", "max_daily_drawdown_pct", "options_max_risk_pct", "options_stop_loss_pct", "vol_options_max_risk_pct", "vol_options_profit_target_pct", "exit_trailing_stop_activation_pct", "exit_trailing_stop_pct", "exit_stop_loss_pct", "vol_universe_max_spread_pct", "intraday_capital_pct", "options_capital_pct", "thesis_capital_pct", "swing_capital_pct", "swing_stop_loss_pct", "swing_trailing_stop_pct", "swing_trailing_stop_activation_pct")
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
