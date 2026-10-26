"""Audit-fold boundaries and persistent sealed holdout tests."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from bb import data
from bb.audit import HoldoutSealedError, HoldoutVault, walk_forward
from bb.data import DataGate
from bb.spec import StrategySpec


def _data():
    rng = np.random.default_rng(503)
    index = pd.bdate_range("2000-01-03", "2020-12-31")
    returns = pd.DataFrame(rng.normal(0.0002, 0.01, (len(index), 8)), index=index)
    rf = pd.Series(0.0001, index=index)
    return returns, rf


@pytest.fixture
def fake_data(monkeypatch):
    """Expose synthetic source data through the same private loader as production."""
    returns, rf = _data()
    monkeypatch.setattr(data, "_load_universe", lambda: (returns, rf))
    return returns, rf


def _spec(start=date(2000, 1, 3), end=date(2020, 12, 31)):
    return StrategySpec(
        name="audit test", signal_type="momentum", lookback_days=40, skip_days=2,
        top_n=2, rebalance="M", long_short=False, start=start, end=end,
    )


def test_vault_requires_prediction_and_allows_only_one_open(tmp_path, fake_data):
    """Opening requires a prediction and is permanently one-time across instances."""
    vault = HoldoutVault(tmp_path / "audit.jsonl")
    vault.seal()
    in_sample_spec = _spec(end=date(2017, 12, 31))
    with pytest.raises(HoldoutSealedError):
        vault.open_once(in_sample_spec)
    vault.register_prediction(in_sample_spec, 0.5, "Expected modest positive Sharpe")
    result = vault.open_once(in_sample_spec)
    assert "sharpe" in result
    with pytest.raises(HoldoutSealedError):
        HoldoutVault(tmp_path / "audit.jsonl").open_once(in_sample_spec)


def test_vault_rejects_holdout_hash_mismatch(tmp_path, fake_data, monkeypatch):
    """Changing a holdout value after sealing prevents evaluation."""
    returns, rf = fake_data
    vault = HoldoutVault(tmp_path / "audit.jsonl")
    vault.seal()
    vault.register_prediction(_spec(end=date(2017, 12, 31)), 0.5, "Prediction recorded before access")
    changed = returns.copy()
    changed.loc[changed.index[-1], 0] += 0.1
    monkeypatch.setattr(data, "_load_universe", lambda: (changed, rf))
    with pytest.raises(HoldoutSealedError, match="hash mismatch"):
        vault.open_once(_spec(end=date(2017, 12, 31)))


def test_in_sample_excludes_holdout_dates(fake_data):
    """DataGate training data contains no dates on or after the boundary."""
    returns, _ = DataGate().in_sample()
    assert returns.index.max() < pd.Timestamp("2018-01-01")


def test_walk_forward_folds_are_non_overlapping_and_in_sample(fake_data):
    """All out-of-sample test intervals are chronological, disjoint, pre-holdout."""
    _, rf = fake_data
    returns, gated_rf = DataGate().in_sample()
    spec = _spec(end=date(2017, 12, 31))
    folds = walk_forward(spec, returns, gated_rf, train_years=5, test_years=1)
    assert folds
    for i, fold in enumerate(folds):
        assert fold["test_end"] < "2018-01-01"
        if i:
            assert folds[i - 1]["test_end"] < fold["test_start"]
