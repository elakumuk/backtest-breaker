"""Offline research-agent logging, date protection, and seeded mock tests."""

import json
from datetime import date

import numpy as np
import pandas as pd

from bb import data
from bb.agents.research import run_research
from bb.llm import LLMResponse


def _synthetic_data():
    rng = np.random.default_rng(713)
    index = pd.bdate_range("1990-01-02", "2015-12-31")
    returns = pd.DataFrame(rng.normal(0.0003, 0.012, (len(index), 8)), index=index)
    rf = pd.Series(0.0001, index=index)
    return returns, rf


def test_research_logs_each_trial_and_registers_but_does_not_open(tmp_path, monkeypatch):
    """Every iteration has a returns column; predictions precede any opening."""
    returns, rf = _synthetic_data()
    monkeypatch.setattr(data, "_load_universe", lambda: (returns, rf))
    monkeypatch.setenv("PROVIDER", "mock")
    run = run_research(n_iters=5, seed=7, run_dir=tmp_path / "run", verbose=False)
    records = [json.loads(line) for line in (run.run_dir / "trials.jsonl").read_text().splitlines()]
    history = pd.read_parquet(run.run_dir / "trial_returns.parquet")
    assert len(records) == 5
    assert history.columns.tolist() == [record["trial_id"] for record in records]
    assert all(record["spec"] is None or date.fromisoformat(record["spec"]["end"]) < date(2013, 1, 1) for record in records)
    events = [json.loads(line)["event"] for line in (run.run_dir / "audit_log.jsonl").read_text().splitlines()]
    assert events == ["sealed", "prediction"]
    assert run.summary["n_trials"] == 5


def test_same_seed_produces_identical_trial_specs_and_metrics(tmp_path, monkeypatch):
    """The mock hill-climb is repeatable for a fixed seed and dataset."""
    returns, rf = _synthetic_data()
    monkeypatch.setattr(data, "_load_universe", lambda: (returns, rf))
    monkeypatch.setenv("PROVIDER", "mock")
    first = run_research(n_iters=6, seed=31, run_dir=tmp_path / "first", verbose=False)
    second = run_research(n_iters=6, seed=31, run_dir=tmp_path / "second", verbose=False)
    keys = ("spec", "metrics", "status", "error", "provider", "model", "token_usage")
    assert [[trial[key] for key in keys] for trial in first.trials] == [[trial[key] for key in keys] for trial in second.trials]


def test_invalid_candidates_are_retried_then_logged_as_failed(tmp_path, monkeypatch):
    """Validation failures consume one logged trial and retain a returns column."""
    returns, rf = _synthetic_data()
    monkeypatch.setattr(data, "_load_universe", lambda: (returns, rf))
    monkeypatch.setenv("PROVIDER", "mock")
    from bb.agents import research

    actual_chat = research.chat
    calls = {"count": 0}

    def invalid_first_candidate(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] <= 3:
            return LLMResponse("not json", "mock-v1", "mock", {"total_tokens": 0})
        return actual_chat(*args, **kwargs)

    monkeypatch.setattr(research, "chat", invalid_first_candidate)
    run = run_research(n_iters=2, seed=4, run_dir=tmp_path / "with-failure", verbose=False)
    assert [trial["status"] for trial in run.trials] == ["failed", "ok"]
    history = pd.read_parquet(run.run_dir / "trial_returns.parquet")
    assert history.columns.tolist() == ["trial_001", "trial_002"]
    assert history["trial_001"].isna().all()
