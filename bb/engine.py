"""Deterministic long-only and long-short cross-sectional backtest engine."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from bb.spec import StrategySpec


@dataclass(frozen=True)
class BacktestResult:
    """Daily portfolio results; returns and costs are decimal fractions."""

    daily_returns: pd.Series
    gross_returns: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    costs: pd.Series


def _signal_matrix(returns: pd.DataFrame, spec: StrategySpec) -> pd.DataFrame:
    """Calculate signals dated t using windows ending no later than t-1."""
    # Moving the source back by skip_days excludes the latest observations.
    source = returns.shift(spec.skip_days)
    valid = source.rolling(spec.lookback_days, min_periods=spec.lookback_days).count().eq(spec.lookback_days)
    if spec.signal_type == "low_vol":
        score = -source.rolling(spec.lookback_days, min_periods=spec.lookback_days).std(ddof=0)
    else:
        log_returns = np.log1p(source.where(source >= -1.0))
        cumulative = np.expm1(log_returns.rolling(spec.lookback_days, min_periods=spec.lookback_days).sum())
        if spec.signal_type == "reversal":
            score = -cumulative
        elif spec.signal_type == "trend":
            # Direction times cross-sectional percentile magnitude; ranking is within each date.
            ranks = cumulative.rank(axis=1, pct=True) - 0.5
            score = np.sign(cumulative) * ranks.abs()
        else:
            score = cumulative
    # The shift enforces that the portfolio for date t only sees scores known at t-1.
    return score.where(valid).shift(1)


def _rebalance_mask(index: pd.DatetimeIndex, frequency: str) -> np.ndarray:
    """Mark the first observation of each daily, weekly, or monthly period."""
    if frequency == "D":
        return np.ones(len(index), dtype=bool)
    periods = index.to_period("W-FRI" if frequency == "W" else "M")
    return np.r_[True, periods[1:] != periods[:-1]]


def _target_weights(scores: pd.Series, spec: StrategySpec) -> pd.Series:
    """Build equal-weight targets from eligible signal values."""
    eligible = scores.dropna()
    target = pd.Series(0.0, index=scores.index)
    n = min(spec.top_n, len(eligible))
    if n == 0:
        return target
    ranked = eligible.sort_values(kind="mergesort")
    if spec.long_short:
        if len(eligible) < 2 * n:
            n = len(eligible) // 2
        if n == 0:
            return target
        target.loc[ranked.index[-n:]] = 1.0 / n
        target.loc[ranked.index[:n]] = -1.0 / n
    else:
        target.loc[ranked.index[-n:]] = 1.0 / n
    return target


def run_backtest(spec: StrategySpec, returns: pd.DataFrame, rf: pd.Series) -> BacktestResult:
    """Run a daily marked portfolio, applying turnover costs on rebalance dates.

    ``rf`` is accepted for a stable interface; returns stay raw in the result so
    callers can explicitly form excess returns. Holdings drift between scheduled
    rebalances based on the previous day's asset returns.
    """
    del rf
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise TypeError("returns must have a DatetimeIndex")
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise TypeError("returns must have a DatetimeIndex")
    history = returns.sort_index().loc[: spec.end].astype(float)
    history = history.loc[~history.index.duplicated(keep="first")]
    scores = _signal_matrix(history, spec)
    evaluation = history.loc[spec.start : spec.end]
    if evaluation.empty:
        raise ValueError("no returns fall inside the strategy evaluation dates")
    scores = scores.loc[evaluation.index].to_numpy(dtype=float)
    asset_returns = evaluation.to_numpy(dtype=float)
    index = evaluation.index
    rebalance = _rebalance_mask(index, spec.rebalance)
    rebalance[0] = True
    rows, columns = asset_returns.shape
    weight_values = np.zeros((rows, columns), dtype=float)
    turnover_values = np.zeros(rows, dtype=float)
    costs_values = np.zeros(rows, dtype=float)
    previous = np.zeros(columns, dtype=float)
    for pos in range(rows):
        if pos:
            prior_returns = np.nan_to_num(asset_returns[pos - 1], nan=0.0)
            growth = 1.0 + np.dot(previous, prior_returns)
            if abs(growth) > 1e-12:
                previous = previous * (1.0 + prior_returns) / growth
        if rebalance[pos]:
            eligible = np.flatnonzero(np.isfinite(scores[pos]))
            target = np.zeros(columns, dtype=float)
            n = min(spec.top_n, len(eligible))
            if spec.long_short:
                n = min(n, len(eligible) // 2)
            if n:
                order = eligible[np.argsort(scores[pos, eligible], kind="stable")]
                if spec.long_short:
                    target[order[-n:]] = 1.0 / n
                    target[order[:n]] = -1.0 / n
                else:
                    target[order[-n:]] = 1.0 / n
            turnover_values[pos] = np.abs(target - previous).sum()
            costs_values[pos] = spec.cost_bps / 10_000.0 * turnover_values[pos]
            previous = target
        weight_values[pos] = previous

    weights = pd.DataFrame(weight_values, index=index, columns=evaluation.columns)
    turnover = pd.Series(turnover_values, index=index, name="turnover")
    costs = pd.Series(costs_values, index=index, name="costs")
    returns = evaluation
    gross = (weights * returns.fillna(0.0)).sum(axis=1).rename("gross_returns")
    net = (gross - costs).rename("daily_returns")
    return BacktestResult(net, gross, weights, turnover, costs)
