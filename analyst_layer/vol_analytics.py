"""GARCH(1,1) realized volatility forecaster for the vol track.

Computes a forward-looking estimate of realized vol over the next N trading
days using the GARCH(1,1) filter with variance targeting. The result is in the
same annualized-decimal unit as IV30 so VRP = IV30 − GARCH_forecast is directly
comparable.

Why GARCH over a rolling historical vol (HV30)?
  HV30 looks backward — it tells you what vol was. GARCH estimates what vol is
  likely to be over the life of a trade by conditioning on the latest shock
  (today's return) and the recent variance level. The variance risk premium
  (VRP = IV − expected RV) is the structural edge in short-premium strategies;
  using a forward-looking RV estimate rather than a trailing one gives a more
  accurate read on how much edge is actually available.

Model:
  σ²_t = ω + α·r²_{t-1} + β·σ²_{t-1}    (GARCH(1,1), Bollerslev 1986)
  Variance targeting: ω = σ̄² · (1 − α − β)   (avoids MLE; σ̄² is the sample mean)
  h-step forecast:   σ²_{T+h} = VL + (α+β)^h · (σ²_T − VL)
    where VL = ω / (1 − α − β) = σ̄²  (the long-run unconditional variance)

Coefficients:
  α=0.10, β=0.85 → persistence=0.95. These fall within the range Bollerslev
  found for equity daily returns and imply moderate mean reversion: a vol spike
  today decays roughly to 1/e of its deviation from the long-run mean in about
  20 trading days — consistent with the 21-DTE tastylive roll rule.
"""
from __future__ import annotations

import math

from data_layer.models import PriceSeries


def estimate_garch_rv(
    price_series: PriceSeries,
    forecast_horizon: int = 30,
    alpha: float = 0.10,
    beta: float = 0.85,
) -> float | None:
    """GARCH(1,1) h-step-ahead annualized realized vol forecast.

    Returns None if the series has fewer than 31 bars (30 log-returns minimum).
    Returns the expected annualized vol over the next `forecast_horizon` trading
    days, in the same decimal unit as IV30 (e.g. 0.28 = 28%).

    Parameters
    ----------
    forecast_horizon:
        Number of trading days to forecast. Should match the DTE of the
        position being evaluated (e.g. 45 for a 45-DTE iron condor).
    alpha, beta:
        ARCH and GARCH coefficients. alpha+beta=0.95 is the persistence level —
        keep below 1.0 to ensure stationarity.
    """
    closes = [bar.close for bar in price_series.bars]
    if len(closes) < 31:
        return None

    returns = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
    n = len(returns)
    long_run_var = sum(r * r for r in returns) / n

    persistence = alpha + beta
    omega = long_run_var * (1.0 - persistence)

    # Forward filter: initialise at long-run variance, run through the return series
    sigma_sq = long_run_var
    for r in returns:
        sigma_sq = omega + alpha * r * r + beta * sigma_sq

    # Long-run (unconditional) variance — same as long_run_var under variance targeting
    vl = omega / (1.0 - persistence) if persistence < 1.0 else long_run_var

    # Sum of h-step-ahead variance forecasts:
    #   Σ_{i=1}^{h} [VL + (α+β)^i · (σ²_T − VL)]
    #   = h·VL + (σ²_T − VL) · persistence · (1 − persistence^h) / (1 − persistence)
    if 0.0 < persistence < 1.0:
        sum_var = (
            forecast_horizon * vl
            + (sigma_sq - vl) * persistence * (1.0 - persistence ** forecast_horizon) / (1.0 - persistence)
        )
    else:
        sum_var = forecast_horizon * sigma_sq

    avg_daily_var = sum_var / forecast_horizon
    return math.sqrt(max(avg_daily_var, 0.0) * 252)
