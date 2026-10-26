"""Structural tests for the public data gate and audit boundary checks."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from bb import audit, data
from bb.audit import HoldoutSealedError, HoldoutVault
from bb.spec import StrategySpec


def _fixture_data():
    rng = np.random.default_rng(404)
    index = pd.bdate_range("2010-01-04", "2020-12-31")
    returns = pd.DataFrame(rng.normal(0.0002, 0.01, (len(index), 8)), index=index)
    rf = pd.Series(0.0001, index=index)
    return returns, rf


def _spec(end: date = date(2017, 12, 31)) -> StrategySpec:
    return StrategySpec(
        name="gate test", signal_type="momentum", lookback_days=30, skip_days=2,
        top_n=2, rebalance="M", long_short=False, start=date(2010, 1, 4), end=end,
    )


@pytest.fixture
def source(monkeypatch):
    """Replace the private source loader with deterministic local data."""
    frame, rf = _fixture_data()
    monkeypatch.setattr(data, "_load_universe", lambda: (frame, rf))
    return frame, rf


def test_data_gate_returns_only_in_sample(source):
    """The public gate excludes all dates in the final three calendar years."""
    frame, _ = source
    gate = data.DataGate(holdout_years=3)
    returns, _ = gate.in_sample()
    assert gate.holdout_start == date(2018, 1, 1)
    assert returns.index.max() < pd.Timestamp(gate.holdout_start)


@pytest.mark.parametrize("function_name", ["walk_forward", "param_sensitivity", "cost_stress", "run_audit"])
def test_audit_functions_reject_spec_ending_in_holdout(source, function_name):
    """No audit entry point accepts a strategy whose configured end reaches holdout."""
    returns, rf = source
    function = getattr(audit, function_name)
    with pytest.raises(HoldoutSealedError):
        if function_name == "walk_forward":
            function(_spec(date(2019, 1, 1)), returns, rf)
        elif function_name in {"param_sensitivity", "cost_stress"}:
            function(_spec(date(2019, 1, 1)), returns, rf)
        else:
            function(_spec(date(2019, 1, 1)), returns, rf, [])


@pytest.mark.parametrize("function_name", ["walk_forward", "param_sensitivity", "cost_stress", "run_audit"])
def test_audit_functions_reject_returns_with_holdout_dates(source, function_name):
    """Every audit entry point rejects data containing any sealed date."""
    returns, rf = source
    function = getattr(audit, function_name)
    with pytest.raises(HoldoutSealedError):
        if function_name == "walk_forward":
            function(_spec(), returns, rf)
        elif function_name in {"param_sensitivity", "cost_stress"}:
            function(_spec(), returns, rf)
        else:
            function(_spec(), returns, rf, [])


def test_holdout_vault_reads_data_itself_and_opens_after_prediction(tmp_path, source):
    """The caller supplies no holdout data to seal or open it."""
    vault = HoldoutVault(log_path=tmp_path / "audit.jsonl")
    vault.seal()
    vault.register_prediction(_spec(), 0.25, "Mock registered prediction")
    actual = vault.open_once(_spec())
    assert "sharpe" in actual
    events = [line for line in (tmp_path / "audit.jsonl").read_text().splitlines()]
    assert [__import__("json").loads(line)["event"] for line in events] == ["sealed", "prediction", "opened"]
