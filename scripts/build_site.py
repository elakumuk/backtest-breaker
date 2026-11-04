"""Bundle demo/ into docs/data.js for the static showcase page (GitHub Pages) (no network, no model calls)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
import demo_data as dd  # noqa: E402


def bundle() -> dict:
    m = dd.manifest()
    trials = dd.trials_frame()
    report = dd.auditor_report()
    events = dd.audit_events()
    curve = dd.best_equity_curve()
    weekly = curve.resample("W").last().dropna()
    wf = next(s["result"] for s in report["trace"] if s["tool"] == "walk_forward")
    notebooks = {}
    for name in dd.notebook_names():
        rep = dd.notebook_report(name)
        notebooks[name] = {"report": rep, "cells": dd.notebook_cells(name)}
    return {
        "manifest": m,
        "trials": json.loads(trials.to_json(orient="records")),
        "report": report,
        "events": [e for e in events if e["event"] in ("prediction", "opened")],
        "walk_forward": wf,
        "equity": [[d.strftime("%Y-%m-%d"), round(float(v), 4)] for d, v in weekly.items()],
        "notebooks": notebooks,
    }


if __name__ == "__main__":
    data = bundle()
    out = ROOT / "docs" / "data.js"
    out.write_text("window.BB_DATA = " + json.dumps(data, separators=(",", ":")) + ";\n")
    print(out, out.stat().st_size // 1024, "KB")
