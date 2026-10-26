"""Strategy stability checks and a persistent, one-time holdout vault."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bb.engine import run_backtest
from bb.metrics import annual_return, annual_turnover, deflated_sharpe_ratio, max_drawdown, pbo_cscv, sharpe
from bb.spec import StrategySpec


class HoldoutSealedError(RuntimeError):
    """Raised when a sealed holdout cannot be safely accessed."""


def default_holdout_start(index: pd.DatetimeIndex) -> date:
    """Return January 1 of the first of the last three available calendar years."""
    if len(index) == 0:
        raise ValueError("cannot determine a holdout from an empty index")
    return date(int(index.max().year) - 2, 1, 1)


def _new_spec(spec: StrategySpec, **updates: Any) -> StrategySpec:
    """Create a fully revalidated spec with selected overrides."""
    payload = spec.model_dump()
    payload.update(updates)
    return StrategySpec(**payload)


def _metric_for(spec: StrategySpec, returns: pd.DataFrame, rf: pd.Series) -> float:
    result = run_backtest(spec, returns, rf)
    return sharpe(result.daily_returns, rf.reindex(result.daily_returns.index), long_short=spec.long_short)


def walk_forward(
    spec: StrategySpec,
    returns: pd.DataFrame,
    rf: pd.Series,
    train_years: int = 5,
    test_years: int = 1,
) -> list[dict[str, Any]]:
    """Return fixed-parameter Sharpe checks over contiguous, non-overlapping test folds."""
    if train_years < 1 or test_years < 1:
        raise ValueError("train_years and test_years must be positive")
    available_end = min(spec.end, returns.index.max().date())
    train_end = pd.Timestamp(spec.start) + pd.DateOffset(years=train_years)
    test_start = (train_end + pd.Timedelta(days=1)).date()
    folds: list[dict[str, Any]] = []
    while test_start <= available_end:
        test_end = (pd.Timestamp(test_start) + pd.DateOffset(years=test_years) - pd.Timedelta(days=1)).date()
        if test_end > available_end:
            test_end = available_end
        fold_spec = _new_spec(spec, start=test_start, end=test_end)
        fold_returns = returns.loc[: pd.Timestamp(test_end)]
        fold_rf = rf.loc[: pd.Timestamp(test_end)]
        result = run_backtest(fold_spec, fold_returns, fold_rf)
        fold_rf = fold_rf.reindex(result.daily_returns.index)
        folds.append({
            "test_start": test_start.isoformat(),
            "test_end": test_end.isoformat(),
            "sharpe": sharpe(result.daily_returns, fold_rf, long_short=spec.long_short),
        })
        test_start = (pd.Timestamp(test_end) + pd.Timedelta(days=1)).date()
    return folds


def param_sensitivity(spec: StrategySpec, returns: pd.DataFrame, rf: pd.Series) -> dict[str, Any]:
    """Re-run at lookback and top_n perturbations of minus/plus 20 percent."""
    runs: dict[str, float] = {}
    for field, low, high in (("lookback_days", 5, 504), ("top_n", 1, 24)):
        original = getattr(spec, field)
        for label, factor in (("minus_20pct", 0.8), ("plus_20pct", 1.2)):
            value = int(round(original * factor))
            value = min(high, max(low, value))
            variant = _new_spec(spec, **{field: value})
            runs[f"{field}_{label}_{value}"] = _metric_for(variant, returns, rf)
    values = [value for value in runs.values() if np.isfinite(value)]
    return {"sharpe_by_variant": runs, "spread": float(max(values) - min(values)) if values else float("nan")}


def cost_stress(spec: StrategySpec, returns: pd.DataFrame, rf: pd.Series) -> dict[int, float]:
    """Return annualized Sharpe at fixed execution-cost assumptions in basis points."""
    return {bps: _metric_for(_new_spec(spec, cost_bps=float(bps)), returns, rf) for bps in (5, 10, 25, 50)}


def _holdout_digest(frame: pd.DataFrame) -> str:
    """Hash sorted columns, datetime index, and all holdout values."""
    canonical = frame.sort_index().sort_index(axis=1)
    digest = hashlib.sha256()
    digest.update(json.dumps([str(c) for c in canonical.columns], separators=(",", ":")).encode())
    digest.update(pd.util.hash_pandas_object(canonical, index=True).values.tobytes())
    return digest.hexdigest()


class HoldoutVault:
    """Persist a hash-sealed holdout and allow it to be evaluated only once."""

    def __init__(self, holdout_start: date, log_path: str | Path = "audit_log.jsonl") -> None:
        self.holdout_start = holdout_start
        self.log_path = Path(log_path)

    def _events(self) -> list[dict[str, Any]]:
        if not self.log_path.exists():
            return []
        events: list[dict[str, Any]] = []
        for line in self.log_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                events.append(json.loads(line))
        return events

    def _append(self, event: dict[str, Any]) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(event, sort_keys=True, default=str) + "\n")

    def _holdout(self, returns: pd.DataFrame) -> pd.DataFrame:
        return returns.loc[returns.index >= pd.Timestamp(self.holdout_start)]

    def seal(self, returns: pd.DataFrame) -> str:
        """Hash the holdout data and persist the sealed event."""
        events = self._events()
        if any(event.get("event") == "sealed" for event in events):
            raise HoldoutSealedError("holdout is already sealed")
        holdout = self._holdout(returns)
        if holdout.empty:
            raise ValueError("holdout slice is empty")
        digest = _holdout_digest(holdout)
        self._append({"event": "sealed", "holdout_start": self.holdout_start.isoformat(), "sha256": digest})
        return digest

    def in_sample(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Return data strictly before the sealed holdout boundary."""
        return returns.loc[returns.index < pd.Timestamp(self.holdout_start)].copy()

    def register_prediction(self, spec: StrategySpec, predicted_sharpe: float, rationale: str) -> None:
        """Persist the required prediction before the one permitted holdout run."""
        events = self._events()
        if not any(event.get("event") == "sealed" for event in events):
            raise HoldoutSealedError("seal the holdout before registering a prediction")
        if any(event.get("event") == "opened" for event in events):
            raise HoldoutSealedError("holdout has already been opened")
        self._append({
            "event": "prediction", "spec": json.loads(spec.to_json()),
            "predicted_sharpe": float(predicted_sharpe), "rationale": rationale,
        })

    def open_once(self, spec: StrategySpec, returns: pd.DataFrame, rf: pd.Series) -> dict[str, Any]:
        """Verify, run, and permanently record the single holdout evaluation."""
        events = self._events()
        sealed = [event for event in events if event.get("event") == "sealed"]
        if not sealed:
            raise HoldoutSealedError("holdout has not been sealed")
        if not any(event.get("event") == "prediction" for event in events):
            raise HoldoutSealedError("register a prediction before opening the holdout")
        if any(event.get("event") == "opened" for event in events):
            raise HoldoutSealedError("holdout has already been opened")
        holdout = self._holdout(returns)
        expected_hash = sealed[-1]["sha256"]
        if _holdout_digest(holdout) != expected_hash:
            raise HoldoutSealedError("holdout hash mismatch")
        start = self.holdout_start
        end = holdout.index.max().date()
        holdout_spec = _new_spec(spec, start=start, end=end)
        result = run_backtest(holdout_spec, returns.loc[:pd.Timestamp(end)], rf.loc[:pd.Timestamp(end)])
        holdout_daily = result.daily_returns.loc[holdout.index]
        holdout_rf = rf.reindex(holdout_daily.index)
        actual_sharpe = sharpe(holdout_daily, holdout_rf, long_short=spec.long_short)
        payload = {
            "sharpe": actual_sharpe,
            "annual_return": annual_return(holdout_daily),
            "max_drawdown": max_drawdown(holdout_daily),
            "annual_turnover": annual_turnover(result.turnover.loc[holdout.index]),
        }
        self._append({"event": "opened", "sha256": expected_hash, "result": payload})
        return payload


@dataclass
class AuditReport:
    """JSON-serializable output of the deterministic audit battery."""

    strategy_name: str
    sharpe: float
    max_drawdown: float
    annual_return: float
    annual_turnover: float
    walk_forward: list[dict[str, Any]]
    sensitivity: dict[str, Any]
    cost_stress: dict[int, float]
    deflated_sharpe: float
    pbo: float | None
    pbo_logits: list[float] | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe mapping."""
        return asdict(self)


def run_audit(
    spec: StrategySpec,
    returns: pd.DataFrame,
    rf: pd.Series,
    trials_history: list[float] | pd.DataFrame,
) -> AuditReport:
    """Run reproducible performance, stability, cost, and multiple-testing checks."""
    result = run_backtest(spec, returns, rf)
    aligned_rf = rf.reindex(result.daily_returns.index).fillna(0.0)
    observed_sharpe = sharpe(result.daily_returns, aligned_rf, long_short=spec.long_short)
    if isinstance(trials_history, pd.DataFrame):
        trials = [float(sharpe(trials_history[col], pd.Series(0.0, index=trials_history.index), long_short=True) / np.sqrt(252)) for col in trials_history]
        pbo, logits = pbo_cscv(trials_history)
    else:
        trials = [float(v) / np.sqrt(252) for v in trials_history if np.isfinite(v)]
        pbo, logits = None, None
    excess = result.daily_returns if spec.long_short else result.daily_returns - aligned_rf
    clean = excess.replace([np.inf, -np.inf], np.nan).dropna()
    dsr = deflated_sharpe_ratio(
        float(clean.mean() / clean.std(ddof=1)) if len(clean) > 1 and clean.std(ddof=1) > 0 else 0.0,
        trials,
        len(clean),
        float(clean.skew()) if len(clean) > 2 else 0.0,
        float(clean.kurt() + 3.0) if len(clean) > 3 else 3.0,
    )
    return AuditReport(
        strategy_name=spec.name,
        sharpe=observed_sharpe,
        max_drawdown=max_drawdown(result.daily_returns),
        annual_return=annual_return(result.daily_returns),
        annual_turnover=annual_turnover(result.turnover),
        walk_forward=walk_forward(spec, returns, rf),
        sensitivity=param_sensitivity(spec, returns, rf),
        cost_stress=cost_stress(spec, returns, rf),
        deflated_sharpe=dsr,
        pbo=pbo,
        pbo_logits=logits,
    )
