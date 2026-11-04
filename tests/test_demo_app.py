"""Demo replay: loaders match saved files, demo mode is offline, and every tab renders."""
from __future__ import annotations

import json
import socket
import statistics
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT / "scripts"))

import demo_data as dd  # noqa: E402

pytestmark = pytest.mark.skipif(not (ROOT / "demo" / "MANIFEST.json").exists(), reason="demo bundle not built")


@pytest.fixture(autouse=True)
def restore_env(monkeypatch):
    """The app sets BB_DEMO_MODE/PROVIDER in os.environ; register them so monkeypatch restores both."""
    monkeypatch.setenv("BB_DEMO_MODE", "0")
    monkeypatch.setenv("PROVIDER", "mock")


@pytest.fixture
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise RuntimeError("network blocked in demo test")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


def test_manifest_numbers_come_from_files():
    from build_manifest import build_manifest
    m = dd.manifest()
    assert build_manifest(dd.run_dir()) == m
    trials = dd.trials_frame()
    ok = trials[trials.status == "ok"]
    assert m["n_trials"] == len(trials)
    assert m["in_sample_best_sharpe"] == ok.sharpe.max()
    assert m["median_trial_sharpe"] == statistics.median(ok.sharpe)
    opened = next(e for e in dd.audit_events() if e["event"] == "opened")
    assert m["actual_holdout_sharpe"] == opened["result"]["sharpe"]
    assert m["final_verdict"] == dd.auditor_report()["final_verdict"]


def test_holdout_opened_exactly_once_after_prediction():
    events = [e["event"] for e in dd.audit_events()]
    assert events.count("opened") == 1
    assert events.index("prediction") < events.index("opened")


def test_demo_returns_are_our_own_strategy_returns_only():
    import pandas as pd
    frame = pd.read_parquet(dd.run_dir() / "trial_returns.parquet")
    assert all(c.startswith("trial_") for c in frame.columns)


def test_demo_loaders_work_offline(no_network):
    dd.trials_frame(); dd.auditor_report(); dd.best_equity_curve()
    for name in dd.notebook_names():
        dd.notebook_report(name); dd.notebook_cells(name)


def test_demo_mode_blocks_real_provider(monkeypatch):
    from bb.llm import chat
    monkeypatch.setenv("BB_DEMO_MODE", "1")
    monkeypatch.setenv("PROVIDER", "baseten")
    with pytest.raises(RuntimeError, match="offline"):
        chat([{"role": "user", "content": "hi"}])


def test_render_finding_handles_edge_cases():
    """No findings, dismissed-only, duplicate, and missing fix fields must not crash the card renderer."""
    from streamlit.testing.v1 import AppTest
    script = f"""
import sys; sys.path.insert(0, {str(ROOT / 'app')!r})
from cards import render_finding
cells = ['x = 1\\ny = 2']
for f in [{{}}, {{'status': 'DUPLICATE', 'cell_index': 0, 'line': 2}}, {{'status': 'DISMISSED', 'cell_index': 9}},
          {{'status': 'PROVEN', 'cell_index': 0, 'line': 1, 'fix_old': 'x = 1', 'fix_new': 'x = 0', 'test_code': 'def test(): pass'}}]:
    render_finding(f, cells)
"""
    at = AppTest.from_string(script, default_timeout=60).run()
    assert not at.exception


@pytest.mark.parametrize("tab", [0, 1, 2])
def test_app_renders_each_tab_in_demo_mode(tab, no_network):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=60).run()
    assert not at.exception
    assert len(at.tabs) == 3
    assert at.tabs[tab].children
    assert any("does not give investment advice" in c.value for c in at.caption)
