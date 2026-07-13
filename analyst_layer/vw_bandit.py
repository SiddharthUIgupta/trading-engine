"""Vowpal Wabbit contextual bandit for agent signal quality estimation.

Replaces the flat per-agent win-rate table with an online logistic regression
model that learns the joint effect of:
  - market regime (bullish, bearish, neutral, ...)
  - strategy track (thesis, momentum, swing, ...)
  - each agent's stance (BUY/HOLD/SELL) and confidence (HIGH/MEDIUM/LOW)
  - how many agents agree

The model updates after every closed trade (online learning — no batch jobs,
no manual retraining step). It persists to disk so learning carries across
process restarts. The first 20 examples are used to warm up; predictions
are suppressed until then to avoid citing a meaningless small sample.

Usage:
    bandit = VWSignalBandit(model_path=Path("state/vw_bandit.model"))
    # On startup with no saved model, warm-start from DB:
    bandit.warm_start(state_store.get_scored_signal_logs())
    # After each trade closes:
    bandit.learn(track, regime, signals, pnl)
    # Before consensus, inject historical win-probability estimate:
    prob = bandit.predict_context(track, regime)
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

_MIN_EXAMPLES = 20  # suppress predictions below this sample size


try:
    import vowpalwabbit as _vw_lib
    _VW_AVAILABLE = True
except ImportError:  # pragma: no cover
    _VW_AVAILABLE = False
    logger.warning("vowpalwabbit not installed — VWSignalBandit will be a no-op")


class VWSignalBandit:
    """Online logistic regression over trade outcomes.

    Thread-safe: learn() and predict() both acquire a lock, since
    learn() runs inside the daemon reflection thread while predict()
    may run on the scheduler thread.

    Two prediction modes:
      predict_context(track, regime)
        — uses only track+regime features; available BEFORE consensus runs.
          Surfaced in the consensus prompt as historical win-rate context.

      predict_full(track, regime, signals)
        — adds per-agent stance+confidence features; available AFTER consensus.
          Used for logging / post-hoc analysis.
    """

    def __init__(self, model_path: Path) -> None:
        self._model_path = model_path
        self._count_path = model_path.with_suffix(".count")
        self._lock = threading.Lock()
        self._example_count = 0
        self._vw = None

        if not _VW_AVAILABLE:
            return

        vw_args = (
            "--quiet "
            "--loss_function logistic "
            "--link logistic "
            "--learning_rate 0.1 "
            "--bit_precision 18"
        )
        try:
            if model_path.exists():
                self._vw = _vw_lib.Workspace(f"{vw_args} -i {model_path}")
                self._example_count = int(self._count_path.read_text()) if self._count_path.exists() else 0
                logger.info(
                    "VWSignalBandit: loaded model from %s (%d examples)",
                    model_path, self._example_count,
                )
            else:
                model_path.parent.mkdir(parents=True, exist_ok=True)
                self._vw = _vw_lib.Workspace(vw_args)
                logger.info("VWSignalBandit: fresh model (no prior data at %s)", model_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("VWSignalBandit: failed to initialize VW — predictions disabled: %s", exc)
            self._vw = None

    def warm_start(self, scored_logs: list[dict]) -> int:
        """Replay historical scored logs so a fresh model bootstraps from all DB history.

        Call this ONCE on startup only when no saved model file exists — otherwise
        you'd re-learn already-incorporated examples and distort the model.

        Returns the number of examples replayed.
        """
        if self._vw is None or not scored_logs:
            return 0
        count = 0
        with self._lock:
            for log in scored_logs:
                pnl = log.get("outcome_pnl")
                if pnl is None:
                    continue
                self._learn_unlocked(
                    track=log.get("track", "unknown"),
                    regime=log.get("regime", "neutral"),
                    signals=log.get("signals", []),
                    pnl=pnl,
                )
                count += 1
            if count:
                self._save_unlocked()
        logger.info("VWSignalBandit: warm-started on %d historical examples", count)
        return count

    def learn(
        self, track: str, regime: str, signals: list[dict], pnl: float,
        ticker: str | None = None, promoted_factors: list[str] | None = None,
    ) -> None:
        """Update model from a just-closed trade. Thread-safe.

        ticker/promoted_factors are optional Vibe-Trading factor enrichment
        (CLAUDE.md Phase 2) — omit both to train on track+regime+signals only,
        exactly as before this was added.
        """
        if self._vw is None:
            return
        with self._lock:
            self._learn_unlocked(track, regime, signals, pnl, ticker, promoted_factors)
            self._save_unlocked()

    def predict_context(self, track: str, regime: str) -> float | None:
        """Win-probability estimate using only track+regime (pre-consensus).

        Returns None if the model hasn't seen enough examples yet.
        """
        if self._vw is None or self._example_count < _MIN_EXAMPLES:
            return None
        with self._lock:
            try:
                return float(self._vw.predict(_context_features(track, regime)))
            except Exception as exc:  # noqa: BLE001
                logger.debug("VWSignalBandit.predict_context failed: %s", exc)
                return None

    def predict_full(
        self, track: str, regime: str, signals: list[dict],
        ticker: str | None = None, promoted_factors: list[str] | None = None,
    ) -> float | None:
        """Win-probability estimate using track+regime+agent signals (post-consensus).

        Returns None if the model hasn't seen enough examples yet.
        """
        if self._vw is None or self._example_count < _MIN_EXAMPLES:
            return None
        with self._lock:
            try:
                features = _full_features(track, regime, signals, ticker, promoted_factors)
                return float(self._vw.predict(features))
            except Exception as exc:  # noqa: BLE001
                logger.debug("VWSignalBandit.predict_full failed: %s", exc)
                return None

    @property
    def example_count(self) -> int:
        return self._example_count

    def close(self) -> None:
        if self._vw is not None:
            try:
                self._vw.finish()
            except Exception:  # noqa: BLE001
                pass

    # ── private ──────────────────────────────────────────────────────────────

    def _learn_unlocked(
        self, track: str, regime: str, signals: list[dict], pnl: float,
        ticker: str | None = None, promoted_factors: list[str] | None = None,
    ) -> None:
        label = "1" if pnl > 0 else "-1"
        features = _full_features(track, regime, signals, ticker, promoted_factors)
        example_str = f"{label} {features}"
        try:
            self._vw.learn(example_str)
            self._example_count += 1
        except Exception as exc:  # noqa: BLE001
            logger.debug("VWSignalBandit.learn failed: %s", exc)

    def _save_unlocked(self) -> None:
        try:
            self._vw.save(str(self._model_path))
            self._count_path.write_text(str(self._example_count))
        except Exception as exc:  # noqa: BLE001
            logger.debug("VWSignalBandit._save failed: %s", exc)


# ── Feature builders ─────────────────────────────────────────────────────────

def _safe(s: str) -> str:
    """Strip characters that have special meaning in VW feature strings."""
    return s.replace(" ", "_").replace(":", "_").replace("|", "_").replace("=", "_")


def _context_features(track: str, regime: str) -> str:
    """Track+regime only — usable before consensus runs."""
    return f"| track={_safe(track)} regime={_safe(regime)}"


def _full_features(
    track: str, regime: str, signals: list[dict],
    ticker: str | None = None, promoted_factors: list[str] | None = None,
) -> str:
    """Full feature set including per-agent stance and confidence.

    ticker/promoted_factors add a separate |factors namespace of Vibe-Trading
    alpha values (CLAUDE.md Phase 2, shadow mode). Both default to None/empty,
    which reproduces the exact feature string from before this was added.
    """
    parts = [f"track={_safe(track)}", f"regime={_safe(regime)}"]

    buy_count = sum(1 for s in signals if s.get("stance", "").upper() == "BUY")
    high_conf = sum(1 for s in signals if s.get("confidence", "").upper() == "HIGH")
    med_conf = sum(1 for s in signals if s.get("confidence", "").upper() == "MEDIUM")

    parts += [
        f"agents_buy={buy_count}",
        f"conf_high={high_conf}",
        f"conf_med={med_conf}",
    ]
    if signals and buy_count == len(signals):
        parts.append("all_agree=1")

    for s in signals:
        name = _safe(s.get("agent_name", "unknown"))[:20]
        stance = s.get("stance", "HOLD").upper()
        conf = s.get("confidence", "LOW").upper()
        parts.append(f"{name}_{stance}_{conf}:1")

    feature_str = "| " + " ".join(parts)

    if ticker and promoted_factors:
        from analyst_layer.factor_provider import compute_factor_features
        factor_vals = compute_factor_features(ticker, promoted_factors)
        if factor_vals:
            factor_ns = " ".join(f"{_safe(k)}:{v:.4f}" for k, v in factor_vals.items())
            feature_str += f" |factors {factor_ns}"

    return feature_str
