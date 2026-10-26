"""Audit-fold boundaries and persistent sealed holdout tests."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from bb.audit import HoldoutSealedError, HoldoutVault, walk_forward
from bb.spec import StrategySpec


def _data():
    rng = np.random.default_rng(503)
    index = pd.bdate_range("2000-01-03", "2020-12-31")
    returns = pd.DataFrame(rng.normal(0.0002, 0.01, (len(index), 8)), index=index)
    rf = pd.Series(0.0001, index=index)
    return returns, rf


def _spec(start=date(2000, 1, 3), end=date(2020, 12, 31)):
    return StrategySpec(
        name="audit test", signal_type="momentum", lookback_days=40, skip_days=2,
        top_n=2, rebalance="M", long_short=False, start=start, end=end,
    )


def test_vault_requires_prediction_and_allows_only_one_open(tmp_path):
    """Opening requires a prediction and is permanently one-time across instances."""
    returns, rf = _data()
    vault = HoldoutVault(date(2018, 1, 1), tmp_path / "audit.jsonl")
    vault.seal(returns)
    with pytest.raises(HoldoutSealedError):
        vault.open_once(_spec(), returns, rf)
    vault.register_prediction(_spec(), 0.5, "Expected modest positive Sharpe")
    result = vault.open_once(_spec(), returns, rf)
    assert "sharpe" in result
    with pytest.raises(HoldoutSealedError):
        HoldoutVault(date(2018, 1, 1), tmp_path / "audit.jsonl").open_once(_spec(), returns, rf)


def test_vault_rejects_holdout_hash_mismatch(tmp_path):
    """Changing a holdout value after sealing prevents evaluation."""
    returns, rf = _data()
    vault = HoldoutVault(date(2018, 1, 1), tmp_path / "audit.jsonl")
    vault.seal(returns)
    vault.register_prediction(_spec(), 0.5, "Prediction recorded before access")
    changed = returns.copy()
    changed.loc[changed.index[-1], 0] += 0.1
    with pytest.raises(HoldoutSealedError, match="hash mismatch"):
        vault.open_once(_spec(), changed, rf)


def test_in_sample_excludes_holdout_dates(tmp_path):
    """The vault's training accessor is strictly earlier than holdout_start."""
    returns, _ = _data()
    vault = HoldoutVault(date(2018, 1, 1), tmp_path / "audit.jsonl")
    sample = vault.in_sample(returns)
    assert (sample.index < pd.Timestamp("2018-01-01")).all()


def test_walk_forward_folds_are_non_overlapping_and_in_sample():
    """All out-of-sample test intervals are chronological, disjoint, pre-holdout."""
    returns, rf = _data()
    in_sample = returns.loc[returns.index < pd.Timestamp("2018-01-01")]
    in_rf = rf.reindex(in_sample.index)
    spec = _spec(end=date(2017, 12, 31))
    folds = walk_forward(spec, in_sample, in_rf, train_years=5, test_years=1)
    assert folds
    for i, fold in enumerate(folds):
        assert fold["test_end"] < "2018-01-01"
        if i:
            assert folds[i - 1]["test_end"] < fold["test_start"]
