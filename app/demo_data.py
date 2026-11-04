"""Offline loaders for the demo replay. Reads only files under demo/; makes no network calls."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demo"


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def manifest(demo: Path = DEMO) -> dict[str, Any]:
    return json.loads((demo / "MANIFEST.json").read_text())


def run_dir(demo: Path = DEMO) -> Path:
    return demo / "runs" / manifest(demo)["run_id"]


def trials_frame(demo: Path = DEMO) -> pd.DataFrame:
    """One row per research trial: the spec Muse Spark proposed and its in-sample Sharpe."""
    rows = []
    for t in _jsonl(run_dir(demo) / "trials.jsonl"):
        spec = t.get("spec") or {}
        rows.append({
            "trial": t["iter"], "status": t.get("status"), "sharpe": (t.get("metrics") or {}).get("sharpe"),
            "cost_bps": spec.get("cost_bps"), "signal": spec.get("signal_type"), "lookback_days": spec.get("lookback_days"),
            "skip_days": spec.get("skip_days"), "top_n": spec.get("top_n"), "long_short": spec.get("long_short"),
            "rebalance": spec.get("rebalance"), "name": spec.get("name"), "latency_s": t.get("latency_seconds"),
        })
    frame = pd.DataFrame(rows).sort_values("trial").reset_index(drop=True)
    frame["best_so_far"] = frame["sharpe"].cummax()
    return frame


def auditor_report(demo: Path = DEMO) -> dict[str, Any]:
    return json.loads((run_dir(demo) / "auditor_report.json").read_text())


def audit_events(demo: Path = DEMO) -> list[dict[str, Any]]:
    return _jsonl(run_dir(demo) / "audit_events.jsonl")


def best_equity_curve(demo: Path = DEMO) -> pd.Series:
    """In-sample growth of $1 for the selected (best) trial."""
    m = manifest(demo)
    returns = pd.read_parquet(run_dir(demo) / "trial_returns.parquet")
    column = f"trial_{int(m['best_trial_iter']):03d}"
    return (1 + returns[column].fillna(0)).cumprod().rename(column)


def notebook_names(demo: Path = DEMO) -> list[str]:
    return sorted(p.name.removesuffix(".report.json") for p in (demo / "notebooks").glob("*.report.json"))


def notebook_report(name: str, demo: Path = DEMO) -> dict[str, Any]:
    return json.loads((demo / "notebooks" / f"{name}.report.json").read_text())


def notebook_cells(name: str, demo: Path = DEMO) -> list[str]:
    nb = json.loads((demo / "notebooks" / f"{name}.ipynb").read_text())
    return ["".join(c.get("source", "")) for c in nb.get("cells", [])]
