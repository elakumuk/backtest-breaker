"""Run the D3 audit battery on real Ken French industry data."""

import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from bb.audit import cost_stress, param_sensitivity, walk_forward
from bb.data import DataGate
from bb.engine import run_backtest
from bb.metrics import deflated_sharpe_ratio, sharpe
from bb.spec import StrategySpec


def main() -> None:
    """Load runtime source data and print stability / multiple-testing results."""
    started = time.perf_counter()
    returns, rf = DataGate().in_sample()
    spec = StrategySpec(
        name="12-1 momentum", signal_type="momentum", lookback_days=252,
        skip_days=21, top_n=5, rebalance="M", long_short=False,
        start=date(1990, 1, 1), end=date(2015, 12, 31),
    )
    folds = walk_forward(spec, returns, rf)
    sensitivity = param_sensitivity(spec, returns, rf)
    stress = cost_stress(spec, returns, rf)
    trial_sharpes = []
    rng = np.random.default_rng(3026)
    for trial in range(20):
        trial_spec = StrategySpec(
            name=f"random spec {trial + 1}", signal_type=rng.choice(["momentum", "reversal", "low_vol", "trend"]),
            lookback_days=int(rng.integers(30, 253)), skip_days=int(rng.integers(0, 22)),
            top_n=int(rng.integers(1, 12)), rebalance=str(rng.choice(["D", "W", "M"])),
            long_short=False, cost_bps=10, start=spec.start, end=spec.end,
        )
        trial_result = run_backtest(trial_spec, returns, rf)
        aligned_rf = rf.reindex(trial_result.daily_returns.index).fillna(0.0)
        excess = trial_result.daily_returns - aligned_rf
        std = excess.std(ddof=1)
        trial_sharpes.append(float(excess.mean() / std) if std > 0 else 0.0)

    result = run_backtest(spec, returns, rf)
    aligned_rf = rf.reindex(result.daily_returns.index).fillna(0.0)
    excess = result.daily_returns - aligned_rf
    sr_hat = float(excess.mean() / excess.std(ddof=1))
    dsr = deflated_sharpe_ratio(sr_hat, trial_sharpes, len(excess), float(excess.skew()), float(excess.kurt() + 3))
    print("Walk-forward folds:")
    for fold in folds:
        print(f"  {fold['test_start']} .. {fold['test_end']}: Sharpe {fold['sharpe']:.3f}")
    print(f"Sensitivity: {sensitivity}")
    print(f"Cost stress (bps -> Sharpe): {stress}")
    print(f"DSR (20 random specs): {dsr:.4f}")
    print(f"Elapsed: {time.perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
