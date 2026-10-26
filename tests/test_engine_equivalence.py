"""NumPy engine agrees with the earlier pandas daily-loop implementation."""

from datetime import date

import numpy as np
import pandas as pd

from bb.engine import BacktestResult, _rebalance_mask, _signal_matrix, run_backtest
from bb.spec import StrategySpec


def _reference_run_backtest(spec: StrategySpec, returns: pd.DataFrame, rf: pd.Series) -> BacktestResult:
    """Reference implementation retaining pandas objects inside the daily loop."""
    del rf
    history = returns.sort_index().loc[: spec.end].astype(float)
    history = history.loc[~history.index.duplicated(keep="first")]
    scores = _signal_matrix(history, spec).loc[history.loc[spec.start : spec.end].index]
    sample = history.loc[spec.start : spec.end]
    rebalances = _rebalance_mask(sample.index, spec.rebalance)
    if len(rebalances):
        rebalances[0] = True
    weights = pd.DataFrame(0.0, index=sample.index, columns=sample.columns)
    turnover = pd.Series(0.0, index=sample.index, name="turnover")
    costs = pd.Series(0.0, index=sample.index, name="costs")
    previous = pd.Series(0.0, index=sample.columns)
    for pos, day in enumerate(sample.index):
        if pos:
            previous_returns = sample.iloc[pos - 1].fillna(0.0)
            growth = 1.0 + float((previous * previous_returns).sum())
            if abs(growth) > 1e-12:
                previous = previous * (1.0 + previous_returns) / growth
        if rebalances[pos]:
            eligible = scores.loc[day].dropna()
            target = pd.Series(0.0, index=sample.columns)
            n = min(spec.top_n, len(eligible))
            if spec.long_short:
                n = min(n, len(eligible) // 2)
            if n:
                ranked = eligible.sort_values(kind="mergesort")
                if spec.long_short:
                    target.loc[ranked.index[-n:]] = 1.0 / n
                    target.loc[ranked.index[:n]] = -1.0 / n
                else:
                    target.loc[ranked.index[-n:]] = 1.0 / n
            turnover.loc[day] = (target - previous).abs().sum()
            costs.loc[day] = spec.cost_bps / 10_000.0 * turnover.loc[day]
            previous = target
        weights.loc[day] = previous
    gross = (weights * sample.fillna(0.0)).sum(axis=1).rename("gross_returns")
    net = (gross - costs).rename("daily_returns")
    return BacktestResult(net, gross, weights, turnover, costs)


def test_numpy_engine_matches_reference_on_three_specs():
    """Long-only M, long-short W, and low-vol D agree field by field."""
    rng = np.random.default_rng(29)
    dates = pd.bdate_range("1998-01-01", periods=900)
    returns = pd.DataFrame(rng.normal(0.0002, 0.012, (len(dates), 12)), index=dates)
    rf = pd.Series(0.0001, index=dates)
    common = dict(lookback_days=30, skip_days=3, top_n=3, start=date(1999, 1, 1), end=date(2001, 1, 1))
    specs = [
        StrategySpec(name="long", signal_type="momentum", rebalance="M", long_short=False, **common),
        StrategySpec(name="ls", signal_type="reversal", rebalance="W", long_short=True, **common),
        StrategySpec(name="vol", signal_type="low_vol", rebalance="D", long_short=False, **common),
    ]
    for spec in specs:
        expected = _reference_run_backtest(spec, returns, rf)
        actual = run_backtest(spec, returns, rf)
        for field in ("daily_returns", "gross_returns", "weights", "turnover", "costs"):
            np.testing.assert_allclose(getattr(actual, field), getattr(expected, field), rtol=1e-11, atol=1e-12)
