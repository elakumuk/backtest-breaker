"""In-sample strategy research loop that deliberately searches for high Sharpe."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bb.audit import HoldoutVault
from bb.data import DataGate
from bb.engine import BacktestResult, run_backtest
from bb.llm import LLMResponse, chat
from bb.metrics import annual_return, annual_turnover, max_drawdown, sharpe
from bb.spec import StrategySpec

PREDICTION_SCHEMA = {
    "type": "object",
    "properties": {
        "predicted_holdout_sharpe": {"type": "number"},
        "rationale": {"type": "string"},
    },
    "required": ["predicted_holdout_sharpe", "rationale"],
    "additionalProperties": False,
}


@dataclass
class ResearchRun:
    """Files and winning result produced by :func:`run_research`."""

    run_dir: Path
    best_spec: StrategySpec
    trials: list[dict[str, Any]]
    summary: dict[str, Any]


def _run_directory(value: str | Path) -> Path:
    """Resolve the default timestamp placeholder and create a clean run folder."""
    if str(value) == "runs/<timestamp>":
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        value = Path("runs") / stamp
    folder = Path(value)
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _system_prompt(start: date, end: date) -> str:
    """Describe the in-sample overfitting objective and valid search space."""
    return (
        "You are an intentionally aggressive quant researcher. Your sole goal is to "
        "maximize in-sample Sharpe, even if this overfits. Reply ONLY with one JSON "
        "StrategySpec object matching the supplied schema. Never include markdown. "
        f"Research dates are fixed to {start.isoformat()} through {end.isoformat()}. "
        "Allowed signal_type: momentum, reversal, low_vol, trend; lookback_days 5..504; "
        "skip_days 0..42; top_n 1..24; rebalance D/W/M; cost_bps 0..200. "
        "These results are an audit input, not trading advice."
    )


def _clamp_spec(spec: StrategySpec, start: date, end: date, iteration: int) -> StrategySpec:
    """Force the agent's valid strategy dates to the authorized research interval."""
    payload = spec.model_dump()
    payload.update(name=f"{spec.name or 'candidate'} #{iteration}", start=start, end=end)
    return StrategySpec(**payload)


def _metrics(spec: StrategySpec, result: BacktestResult, rf: pd.Series) -> dict[str, float]:
    """Compute all reported trial values with deterministic Python metrics."""
    aligned_rf = rf.reindex(result.daily_returns.index).fillna(0.0)
    return {
        "sharpe": sharpe(result.daily_returns, aligned_rf, long_short=spec.long_short),
        "annual_return": annual_return(result.daily_returns),
        "max_drawdown": max_drawdown(result.daily_returns),
        "annual_turnover": annual_turnover(result.turnover),
    }


def _feedback_table(trials: list[dict[str, Any]]) -> str:
    """Format every prior trial compactly, ordered best-first."""
    ordered = sorted(
        trials,
        key=lambda item: (item["status"] == "ok", (item.get("metrics") or {}).get("sharpe", -math.inf)),
        reverse=True,
    )
    rows = ["iter | status | signal | lookback | top_n | rebalance | Sharpe"]
    for item in ordered:
        spec = item.get("spec") or {}
        score = (item.get("metrics") or {}).get("sharpe")
        score_text = f"{score:.4f}" if score is not None else "FAILED"
        rows.append(
            f"{item['iter']} | {item['status']} | {spec.get('signal_type', '-')} | "
            f"{spec.get('lookback_days', '-')} | {spec.get('top_n', '-')} | "
            f"{spec.get('rebalance', '-')} | {score_text}"
        )
    return "\n".join(rows)


def _candidate_for_iteration(
    messages: list[dict[str, str]],
    schema: dict[str, Any],
    seed: int,
    iteration: int,
    best_spec: StrategySpec | None,
    start: date,
    end: date,
) -> tuple[StrategySpec | None, LLMResponse, str | None]:
    """Ask for and validate a candidate, making two correction retries if needed."""
    response: LLMResponse | None = None
    error_text: str | None = None
    for attempt in range(3):
        response = chat(
            messages,
            json_schema=schema,
            mock_policy={
                "kind": "researcher", "seed": seed, "iteration": iteration,
                "best_spec": best_spec.model_dump(mode="json") if best_spec else None,
                "start": start.isoformat(), "end": end.isoformat(),
            },
        )
        try:
            parsed = StrategySpec.model_validate_json(response.content)
            return _clamp_spec(parsed, start, end, iteration), response, None
        except (ValueError, TypeError) as exc:
            error_text = f"Attempt {attempt + 1} validation error: {exc}"
            messages.extend([
                {"role": "assistant", "content": response.content},
                {"role": "user", "content": f"Your JSON did not validate. Correct this error and reply with JSON only: {error_text}"},
            ])
    assert response is not None
    return None, response, error_text


def run_research(
    n_iters: int = 25,
    seed: int = 0,
    run_dir: str | Path = "runs/<timestamp>",
    *,
    verbose: bool = True,
) -> ResearchRun:
    """Search only in-sample, log every attempt, and register a sealed prediction."""
    if n_iters < 1:
        raise ValueError("n_iters must be positive")
    gate = DataGate()
    returns, rf = gate.in_sample()
    research_start = date(1990, 1, 1)
    research_end = date.fromordinal(gate.holdout_start.toordinal() - 1)
    if research_end <= research_start:
        raise ValueError("available in-sample window ends before the research start")
    returns = returns.loc[:pd.Timestamp(research_end)]
    rf = rf.reindex(returns.index).fillna(0.0)
    folder = _run_directory(run_dir)
    schema = StrategySpec.json_schema()
    messages: list[dict[str, str]] = [
        {"role": "system", "content": _system_prompt(research_start, research_end)},
        {
            "role": "user",
            "content": (
                "Propose the first StrategySpec for the highest possible in-sample Sharpe. "
                "Use only the fixed research dates. Schema: " + json.dumps(schema, separators=(",", ":"))
            ),
        },
    ]
    trials: list[dict[str, Any]] = []
    return_columns: dict[str, pd.Series] = {}
    best_spec: StrategySpec | None = None
    best_sharpe = -math.inf
    trials_path = folder / "trials.jsonl"
    with trials_path.open("a", encoding="utf-8") as trial_log:
        for iteration in range(1, n_iters + 1):
            spec, response, error = _candidate_for_iteration(
                messages, schema, seed, iteration, best_spec, research_start, research_end,
            )
            trial_id = f"trial_{iteration:03d}"
            if spec is not None:
                result = run_backtest(spec, returns, rf)
                metrics = _metrics(spec, result, rf)
                status = "ok"
                trial_spec = json.loads(spec.to_json())
                return_columns[trial_id] = result.daily_returns.rename(trial_id)
                if metrics["sharpe"] > best_sharpe:
                    best_sharpe, best_spec = metrics["sharpe"], spec
            else:
                metrics = None
                status = "failed"
                trial_spec = None
                return_columns[trial_id] = pd.Series(np.nan, index=returns.index, name=trial_id)
            record = {
                "iter": iteration,
                "trial_id": trial_id,
                "spec": trial_spec,
                "metrics": metrics,
                "status": status,
                "error": error,
                "provider": response.provider,
                "model": response.model,
                "token_usage": response.usage,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            trials.append(record)
            trial_log.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            trial_log.flush()
            if verbose:
                if spec is None:
                    print(f"{iteration:>4} | FAILED | - | - | - | - | -")
                else:
                    print(
                        f"{iteration:>4} | {spec.signal_type:<9} | {spec.lookback_days:>3} | "
                        f"{spec.top_n:>2} | {spec.rebalance} | {metrics['sharpe']:.4f}"
                    )
            messages.extend([
                {"role": "assistant", "content": response.content},
                {
                    "role": "user",
                    "content": (
                        f"Trial {iteration} completed. Status={status}; metrics="
                        f"{json.dumps(metrics, sort_keys=True)}. All previous trials, best first:\n"
                        f"{_feedback_table(trials)}\nPropose a candidate intended to improve in-sample Sharpe."
                    ),
                },
            ])

    if best_spec is None:
        raise RuntimeError("all research trials failed validation; no prediction was registered")
    pd.DataFrame(return_columns, index=returns.index).to_parquet(folder / "trial_returns.parquet")
    (folder / "best_spec.json").write_text(best_spec.to_json() + "\n", encoding="utf-8")
    best_record = max((trial for trial in trials if trial["status"] == "ok"), key=lambda trial: trial["metrics"]["sharpe"])
    first_record = next((trial for trial in trials if trial["status"] == "ok"), None)
    prediction_messages = [
        {"role": "system", "content": "Predict holdout Sharpe from the in-sample results. Return only JSON matching the schema."},
        {"role": "user", "content": f"Best in-sample spec: {best_spec.to_json()}\nTrial history:\n{_feedback_table(trials)}"},
    ]
    prediction_response = chat(
        prediction_messages,
        json_schema=PREDICTION_SCHEMA,
        mock_policy={"kind": "prediction", "predicted_sharpe": best_record["metrics"]["sharpe"] * 0.5},
    )
    prediction = json.loads(prediction_response.content)
    holdout_path = folder / "audit_log.jsonl"
    vault = HoldoutVault(log_path=holdout_path)
    if not vault.is_sealed:
        vault.seal()
    vault.register_prediction(best_spec, prediction["predicted_holdout_sharpe"], prediction["rationale"])
    summary = {
        "best_spec": json.loads(best_spec.to_json()),
        "best_metrics": best_record["metrics"],
        "first_sharpe": first_record["metrics"]["sharpe"] if first_record else None,
        "best_sharpe": best_record["metrics"]["sharpe"],
        "n_trials": len(trials),
        "failed_trials": sum(trial["status"] == "failed" for trial in trials),
        "holdout_start": gate.holdout_start.isoformat(),
        "predicted_holdout_sharpe": prediction["predicted_holdout_sharpe"],
        "prediction_rationale": prediction["rationale"],
        "prediction_provider": prediction_response.provider,
        "prediction_model": prediction_response.model,
        "prediction_token_usage": prediction_response.usage,
    }
    (folder / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if verbose:
        print(f"Best spec: {best_spec.to_json()}")
        print(f"First vs best Sharpe: {summary['first_sharpe']:.4f} -> {summary['best_sharpe']:.4f}")
        print(f"n_trials: {len(trials)}")
        print(f"run_dir: {folder}")
    return ResearchRun(folder, best_spec, trials, summary)


def main() -> None:
    """CLI entry point for the in-sample overfitter demonstration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print("iter | signal    | lookback | top_n | rebalance | Sharpe")
    run_research(n_iters=args.iters, seed=args.seed, verbose=True)


if __name__ == "__main__":
    main()
