"""Statistical metric tests use deterministic synthetic returns."""

import numpy as np
import pandas as pd

from bb.metrics import deflated_sharpe_ratio, pbo_cscv, sharpe


def test_constant_positive_excess_sharpe_is_finite_and_large():
    """Zero volatility does not produce an infinite Sharpe ratio."""
    index = pd.bdate_range("2020-01-01", periods=20)
    result = sharpe(pd.Series(0.01, index=index), pd.Series(0.0, index=index))
    assert np.isfinite(result) and result > 100


def test_dsr_penalizes_trials_and_rewards_higher_observed_sharpe():
    """More trials lower confidence; higher observed SR raises it."""
    trials = np.linspace(-0.02, 0.02, 100)
    few = deflated_sharpe_ratio(0.02, trials[:5], 1000, 0.0, 3.0)
    many = deflated_sharpe_ratio(0.02, trials, 1000, 0.0, 3.0)
    higher = deflated_sharpe_ratio(0.04, trials, 1000, 0.0, 3.0)
    assert many < few
    assert higher > many


def test_pbo_is_near_half_for_noise_trials():
    """Pure noise has no systematic in-sample versus out-of-sample ranking."""
    rng = np.random.default_rng(311)
    frame = pd.DataFrame(rng.normal(0, 0.01, (512, 50)))
    probability, logits = pbo_cscv(frame)
    assert 0.3 <= probability <= 0.7
    assert len(logits) == 12870


def test_pbo_is_low_for_persistent_edge():
    """A durable winner remains near the top on complementary blocks."""
    rng = np.random.default_rng(733)
    values = rng.normal(0, 0.01, (512, 50))
    values[:, 0] += 0.002
    probability, _ = pbo_cscv(pd.DataFrame(values))
    assert probability < 0.2
