"""Tests for analyst_layer.agents.iv_surface_agent.IVSurfaceAgent's prompt construction."""
from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

from analyst_layer.agents.iv_surface_agent import IVSurfaceAgent
from analyst_layer.schemas import Confidence, IVEnvironment, StructureType, VolSignal
from data_layer.models import VolatilitySnapshot


def _snapshot(hv_20: float, hv_30: float, iv_30: float = 0.30) -> VolatilitySnapshot:
    return VolatilitySnapshot(
        symbol="AAPL",
        as_of=datetime.now(),
        iv_rank=60.0,
        iv_percentile=65.0,
        iv_30=iv_30,
        hv_20=hv_20,
        hv_30=hv_30,
        iv_hv_spread=iv_30 - hv_30,
        term_structure_ratio=0.95,
        put_skew=0.03,
        earnings_within_dte=False,
    )


# ── Regression: the prompt's "HV 20-day (realized)" line algebraically
# simplified to exactly hv_30 (iv_hv_spread + hv_30 - (iv_30 - hv_30) = hv_30,
# since iv_hv_spread is defined as iv_30 - hv_30) — the LLM was shown the
# 30-day HV number mislabeled as 20-day, and the real hv_20 field was never
# referenced anywhere in the file.

def test_prompt_shows_real_hv20_not_hv30_mislabeled():
    agent = IVSurfaceAgent(client=MagicMock(), model="claude-sonnet-4-6")
    snapshot = _snapshot(hv_20=0.22, hv_30=0.18)  # deliberately distinct values

    captured = {}

    def fake_call_structured(user_prompt, output_model, tool_name, max_tokens=1024):
        captured["prompt"] = user_prompt
        return VolSignal(
            agent_name="iv_surface_agent", ticker="AAPL",
            iv_environment=IVEnvironment.ELEVATED, recommended_structure=StructureType.IRON_CONDOR,
            confidence=Confidence.HIGH, rationale="test", generated_at=datetime.now(), flags=[],
        )

    with patch.object(agent, "_call_structured", side_effect=fake_call_structured):
        agent.analyze("AAPL", snapshot)

    assert "HV 20-day (realized): 22.0%" in captured["prompt"]
    assert "HV 30-day (realized): 18.0%" in captured["prompt"]
