"""Post-trade reflection agent.

After every position closes, this agent examines the original agent reasoning
and the outcome, then extracts 1-3 ticker-agnostic lessons for future similar
setups. Lessons are stored and re-injected into future agent prompts so the
system accumulates genuine wisdom from experience rather than starting cold.

Critically, lessons are about SIGNALS and REASONING PATTERNS — never about
specific tickers. "AAPL went up → be bullish on AAPL" is the failure mode
this module is designed to prevent. "Gap-up breakouts with volume declining
after 30 min are likely exhausted" is the kind of insight we want.
"""
from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from analyst_layer.agents.base import BaseAgent

logger = logging.getLogger(__name__)

VALID_SETUP_TAGS: frozenset[str] = frozenset({
    "high_rsi", "low_rsi", "gap_up", "gap_down",
    "momentum", "thesis_pullback", "vol_short", "orb", "orb_options",
    "high_iv_rank", "low_iv_rank", "earnings_adjacent",
    "volume_spike", "low_volume",
    "bull_regime", "bear_regime", "neutral_regime",
    "low_float", "large_cap",
    "breakout", "breakdown", "failed_breakout",
    "late_entry", "false_signal", "oversold_bounce",
    "extended_market", "sector_rotation",
})


class LessonOutput(BaseModel):
    lesson: str = Field(
        description=(
            "One actionable sentence a future agent should internalize. "
            "Must be completely ticker-agnostic — about signals, conditions, and reasoning errors, "
            "never about a specific stock. "
            "Bad: 'AAPL tends to reverse at highs.' "
            "Good: 'Low-float breakouts with volume declining after the first 30 min are likely exhausted — "
            "wait for a higher-volume continuation candle before entering.'"
        )
    )
    setup_tags: list[str] = Field(
        description=(
            "Tags identifying which future setups this lesson applies to. "
            f"Use only values from: {sorted(VALID_SETUP_TAGS)}"
        )
    )


class ReflectionOutput(BaseModel):
    what_happened: str = Field(
        description="One sentence: describe the entry setup and what the market actually did."
    )
    root_cause: str = Field(
        description=(
            "The specific signal, missing context, or reasoning error that explains the outcome. "
            "For wins: what reasoning proved correct and why. "
            "For losses: what the agents failed to see or weighted incorrectly."
        )
    )
    lessons: list[LessonOutput] = Field(
        description=(
            "1-3 concise, ticker-agnostic, actionable lessons for future similar setups. "
            "Fewer strong lessons beat many weak ones. An empty list is better than a vague lesson."
        ),
        min_length=0,
        max_length=3,
    )
    outcome_was_noise: bool = Field(
        description=(
            "True if the outcome was driven by an unforeseeable binary event: "
            "surprise earnings beat/miss, unexpected FDA ruling, flash crash, "
            "geopolitical shock. If True, no agent reasoning could have anticipated it "
            "and lessons should be empty or very limited."
        )
    )


class ReflectionAgent(BaseAgent):
    """Examines completed trades and extracts generalizable lessons.

    Runs after every equity position close in a background thread — completely
    off the hot path and never blocking execution. Uses the cheap Haiku model;
    this is reading comprehension and extraction, not consequential decision-making.
    """

    name = "reflection_agent"

    @property
    def system_prompt(self) -> str:
        return (
            "You are a trading journal analyst who extracts generalizable lessons from completed trades.\n\n"
            "RULES — read carefully:\n"
            "1. NEVER mention specific ticker symbols in lessons. Ticker-specific conclusions "
            "do not generalize. Extract signal-level and reasoning-pattern insights only.\n"
            "2. Focus on what the agents should REASON DIFFERENTLY next time they see a "
            "similar setup — not on what the market did.\n"
            "3. If the outcome was clearly caused by an unforeseeable binary event "
            "(surprise news, FDA ruling, flash crash), set outcome_was_noise=True "
            "and keep lessons minimal or empty.\n"
            "4. Each lesson must be immediately actionable: a future agent reading it "
            "should know exactly what signal to weight differently or what check to add.\n"
            "5. Maximum 3 lessons. One sharp, specific lesson is worth more than three vague ones."
        )

    def reflect(
        self,
        strategy: str,
        agent_signals: list[dict],
        outcome_pnl: float,
        outcome_win: bool,
        market_context: dict,
    ) -> ReflectionOutput | None:
        """Run post-trade reflection and return extracted lessons.

        Parameters
        ----------
        strategy:
            Which track opened this position (e.g. "momentum", "thesis", "vol_short").
        agent_signals:
            List of dicts with keys: agent_name, stance, confidence, rationale.
            Pulled from the run_history ConsensusPayload that recommended entry.
        outcome_pnl:
            Realized P&L in dollars on this position.
        outcome_win:
            True if pnl > 0.
        market_context:
            Key-value pairs describing market conditions at entry (RSI, volume ratio,
            regime, etc.) for the reflection agent to reason about.
        """
        if not agent_signals:
            return None

        signals_text = "\n".join(
            f"  • {s.get('agent_name', 'unknown')}: {s.get('stance', '?')} "
            f"({s.get('confidence', '?')} confidence)\n"
            f"    Rationale: {str(s.get('rationale', ''))[:400]}"
            for s in agent_signals
        )
        context_text = (
            "\n".join(f"  {k}: {v}" for k, v in market_context.items())
            or "  (not recorded)"
        )

        prompt = (
            f"Strategy track: {strategy}\n"
            f"Outcome: {'WIN' if outcome_win else 'LOSS'} — realized P&L: ${outcome_pnl:+.2f}\n\n"
            f"Agent signals at entry:\n{signals_text}\n\n"
            f"Market context at entry:\n{context_text}\n\n"
            "Reflect on this trade. What should the agents reason differently "
            "next time they encounter a similar setup? Extract ticker-agnostic lessons only."
        )
        try:
            return self._call_structured(
                prompt, ReflectionOutput, tool_name="emit_reflection", max_tokens=1024
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("reflection agent failed: %s", exc)
            return None
