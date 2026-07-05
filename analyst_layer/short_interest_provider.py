"""Short-interest / squeeze-potential shadow signal.

Unlike Kronos (analyst_layer/kronos_provider.py), this provider does NOT use
pit_snapshot at all — short interest has no free historical time series, only
a "current" settlement snapshot (~20 days stale by FINRA's own bi-weekly
reporting cadence; see data_layer.models.ShortInterestSnapshot). That staleness
is tracked explicitly via get_metric_as_of() rather than assumed away, so
scripts/signal_uplift.py can report it next to every verdict. See CLAUDE.md
"Signal lifecycle" — current-snapshot-only signals need manual staleness
review before any PROMOTE-CANDIDATE, on top of the usual n>=300 gate.

Raw metrics only, no composite "squeeze_score" — short_percent_of_float and
days_to_cover can have opposite-signed relationships to forward return
depending on regime (rising short interest + rising price = squeeze building;
rising short interest + falling price = just bearish conviction), so a
composite would force an arbitrary weighting before any IC evidence justifies
one. Let scripts/signal_uplift.py's per-metric IC decide.
"""
from __future__ import annotations

import logging

from data_layer.alpaca_reference_client import AlpacaAssetReferenceClient
from data_layer.exceptions import DataLayerError
from data_layer.models import PriceSeries
from data_layer.openbb_client import OpenBBDataClient

logger = logging.getLogger(__name__)


class ShortInterestSignalProvider:
    name = "short_interest"
    # Bump on any change to which metrics are emitted or how they're computed —
    # signal_values rows are keyed on (candidate_id, signal_name, signal_version,
    # metric_name), so a version bump keeps old and new results from mixing.
    version = "short-interest-v1"

    def __init__(self, openbb_client: OpenBBDataClient, alpaca_client: AlpacaAssetReferenceClient) -> None:
        self._openbb = openbb_client
        self._alpaca = alpaca_client
        self._last_as_of: str | None = None

    def compute(self, ticker: str, pit_snapshot: PriceSeries) -> dict[str, float] | None:
        # pit_snapshot is unused — this provider's data doesn't come from
        # price history at all, unlike Kronos. Accepted only to satisfy the
        # SignalProvider protocol shape.
        try:
            short_interest = self._openbb.get_short_interest_snapshot(ticker)
        except DataLayerError as exc:
            logger.debug("%s: no short interest data available — Empty: %s", ticker, exc)
            return None

        shortable_status = None
        try:
            shortable_status = self._alpaca.get_shortable_status(ticker)
        except DataLayerError as exc:
            logger.debug("%s: shortable status fetch failed — proceeding without it: %s", ticker, exc)

        mom_change = None
        if short_interest.shares_short_prior_month and short_interest.shares_short_prior_month > 0:
            mom_change = (
                (short_interest.shares_short - short_interest.shares_short_prior_month)
                / short_interest.shares_short_prior_month
            )

        # Cached for get_metric_as_of() — safe because the harness calls
        # compute() then, only on success, immediately calls
        # get_metric_as_of() for the same candidate before moving on; no
        # concurrent compute() calls ever overlap for this instance.
        self._last_as_of = short_interest.as_of.isoformat()

        # All 5 keys always present, even when a given metric couldn't be
        # computed for this ticker (e.g. no prior-month data for mom_change,
        # or Alpaca doesn't recognize the symbol) — an explicit NULL row with
        # status='ok' records "we successfully queried this signal, this
        # particular metric just wasn't available," distinct from Empty
        # (nothing queryable at all) or Failed (the query itself broke).
        return {
            "short_percent_of_float": short_interest.short_percent_of_float,
            "days_to_cover": short_interest.days_to_cover,
            "short_interest_mom_change": mom_change,
            "shortable": (1.0 if shortable_status.shortable else 0.0) if shortable_status else None,
            "easy_to_borrow": (1.0 if shortable_status.easy_to_borrow else 0.0) if shortable_status else None,
        }

    def get_metric_as_of(self, ticker: str, candidate_date: str, result: dict[str, float]) -> str | None:
        return self._last_as_of
