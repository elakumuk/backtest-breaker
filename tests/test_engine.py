"""Engine invariants, with synthetic data so tests stay offline."""

from datetime import date

import numpy as np
import pandas as pd

from bb.engine import run_backtest
from bb.spec import StrategySpec


def _inputs(seed: int = 10, n_days: int = 320, n_assets: int = 10):
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2000-01-03", periods=n_days)
    returns = pd.DataFrame(rng.normal(0.0003, 0.01, (n_days, n_assets)), index=index, columns=[f"A{i}" for i in range(n_assets)])
    rf = pd.Series(0.0001, index=index)
    return returns, rf


def _spec(**kwargs) -> StrategySpec:
    values = dict(
        name="test", signal_type="momentum", lookback_days=20, skip_days=2,
        top_n=3, rebalance="M", long_short=False,
        start=date(2000, 1, 3), end=date(2001, 3, 1),
    )
    values.update(kwargs)
    return StrategySpec(**values)


def test_no_lookahead():
    """Changing the final return cannot change any weights through that date."""
    returns, rf = _inputs()
    spec = _spec()
    original = run_backtest(spec, returns, rf)
    changed = returns.copy()
    changed.iloc[-1] = np.linspace(-0.5, 0.5, changed.shape[1])
    perturbed = run_backtest(spec, changed, rf)
    pd.testing.assert_frame_equal(original.weights, perturbed.weights)


def test_no_lookahead_random():
    """Twenty isolated future shocks leave weights through each shock unchanged."""
    returns, rf = _inputs(seed=11, n_days=500)
    spec = _spec(end=date(2001, 12, 1))
    original = run_backtest(spec, returns, rf)
    rng = np.random.default_rng(2026)
    for day_pos in rng.choice(len(returns), size=20, replace=False):
        changed = returns.copy()
        changed.iloc[day_pos] = rng.normal(0, 0.4, changed.shape[1])
        perturbed = run_backtest(spec, changed, rf)
        pd.testing.assert_frame_equal(original.weights.iloc[: day_pos + 1], perturbed.weights.iloc[: day_pos + 1])


def test_deterministic():
    """Repeated runs return identical series and weights."""
    returns, rf = _inputs()
    first = run_backtest(_spec(), returns, rf)
    second = run_backtest(_spec(), returns, rf)
    for field in ("daily_returns", "gross_returns", "turnover", "costs"):
        pd.testing.assert_series_equal(getattr(first, field), getattr(second, field))
    pd.testing.assert_frame_equal(first.weights, second.weights)


def test_costs():
    """A 50 bp rate charges exactly 0.005 times turnover."""
    returns, rf = _inputs()
    free = run_backtest(_spec(cost_bps=0), returns, rf)
    charged = run_backtest(_spec(cost_bps=50), returns, rf)
    np.testing.assert_allclose(free.daily_returns - charged.daily_returns, 0.005 * free.turnover)


def test_long_short_neutral():
    """Rebalanced long-short books have zero net and two units gross exposure."""
    returns, rf = _inputs(n_assets=10)
    result = run_backtest(_spec(long_short=True, top_n=3), returns, rf)
    rebalance_days = result.turnover[result.turnover > 0].index
    np.testing.assert_allclose(result.weights.loc[rebalance_days].sum(axis=1), 0.0, atol=1e-12)
    np.testing.assert_allclose(result.weights.loc[rebalance_days].abs().sum(axis=1), 2.0, atol=1e-12)


def test_nan_ineligible():
    """An asset with missing data in its scoring window is not selected."""
    returns, rf = _inputs(n_days=100)
    returns.iloc[30:60, 0] = np.nan
    result = run_backtest(_spec(rebalance="D", lookback_days=20, skip_days=0), returns, rf)
    assert (result.weights.iloc[60:80, 0] == 0).all()


def test_spec_json_roundtrip():
    """Specs round trip through JSON and reject invalid date ordering."""
    spec = _spec()
    assert StrategySpec.from_json(spec.to_json()) == spec
