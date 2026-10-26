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
        cumulative = (1 + source).rolling(spec.lookback_days, min_periods=spec.lookback_days).apply(np.prod, raw=True) - 1
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
    returns = returns.sort_index().loc[spec.start : spec.end].astype(float)
    returns = returns.loc[~returns.index.duplicated(keep="first")]
    scores = _signal_matrix(returns, spec)
    rebalance = _rebalance_mask(returns.index, spec.rebalance)
    weights = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
    turnover = pd.Series(0.0, index=returns.index, name="turnover")
    costs = pd.Series(0.0, index=returns.index, name="costs")
    previous = pd.Series(0.0, index=returns.columns)

    for pos, day in enumerate(returns.index):
        if pos:
            prior_returns = returns.iloc[pos - 1].fillna(0.0)
            portfolio_growth = 1.0 + float((previous * prior_returns).sum())
            if abs(portfolio_growth) > 1e-12:
                previous = previous * (1.0 + prior_returns) / portfolio_growth
        if rebalance[pos]:
            target = _target_weights(scores.loc[day], spec)
            turnover.loc[day] = float((target - previous).abs().sum())
            costs.loc[day] = spec.cost_bps / 10_000.0 * turnover.loc[day]
            previous = target
        weights.loc[day] = previous

    gross = (weights * returns.fillna(0.0)).sum(axis=1).rename("gross_returns")
    net = (gross - costs).rename("daily_returns")
    return BacktestResult(net, gross, weights, turnover, costs)
