"""Build demo/MANIFEST.json. Every number is read from saved run files, never typed."""
from __future__ import annotations

import json
import statistics
from pathlib import Path

DEMO = Path(__file__).resolve().parents[1] / "demo"


def build_manifest(run_dir: Path) -> dict:
    trials = [json.loads(line) for line in (run_dir / "trials.jsonl").read_text().splitlines() if line.strip()]
    summary = json.loads((run_dir / "summary.json").read_text())
    report = json.loads((run_dir / "auditor_report.json").read_text())
    events = [json.loads(line) for line in (run_dir / "audit_events.jsonl").read_text().splitlines() if line.strip()]
    ok = [t for t in trials if t.get("status") == "ok"]
    sharpes = [t["metrics"]["sharpe"] for t in ok]
    best = max(ok, key=lambda t: t["metrics"]["sharpe"])
    cost = next(step["result"] for step in report["trace"] if step["tool"] == "cost_stress")
    prediction = next(e for e in events if e["event"] == "prediction")
    opened = next(e for e in events if e["event"] == "opened")
    return {
        "run_id": run_dir.name,
        "model": summary["prediction_model"],
        "provider": summary["prediction_provider"],
        "date": trials[0]["timestamp"][:10],
        "n_trials": len(trials),
        "failed_trials": len(trials) - len(ok),
        "first_trial_sharpe": ok[0]["metrics"]["sharpe"],
        "median_trial_sharpe": statistics.median(sharpes),
        "in_sample_best_sharpe": best["metrics"]["sharpe"],
        "best_trial_iter": best["iter"],
        "best_trial_cost_bps": best["spec"]["cost_bps"],
        "best_sharpe_at_10bps": cost["10"],
        "predicted_holdout_sharpe": prediction["predicted_sharpe"],
        "prediction_rationale": prediction["rationale"],
        "actual_holdout_sharpe": opened["result"]["sharpe"],
        "holdout_start": summary["holdout_start"],
        "final_verdict": report["final_verdict"],
        "model_verdict": report["model_verdict"],
        "model_disagrees": report["model_disagrees"],
        "grounded": report["grounded"],
        "verdict_reasons": [r["rule"] for r in report["reasons"]],
        "research_total_tokens": sum((t.get("token_usage") or {}).get("total_tokens", 0) for t in trials),
        "research_total_latency_seconds": sum(t.get("latency_seconds") or 0 for t in trials),
        "auditor_total_tokens": report["total_tokens"],
    }


if __name__ == "__main__":
    run = sorted((DEMO / "runs").glob("live-*"))[-1]
    manifest = build_manifest(run)
    (DEMO / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
