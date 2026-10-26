"""Run a 12-1 momentum strategy on the real Ken French industry universe."""

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from bb.data import DataGate
from bb.engine import run_backtest
from bb.spec import StrategySpec


def main() -> None:
    """Load licensed data at runtime and print basic strategy diagnostics."""
    returns, rf = DataGate().in_sample()
    spec = StrategySpec(
        name="12-1 momentum", signal_type="momentum", lookback_days=252,
        skip_days=21, top_n=5, rebalance="M", long_short=False,
        start=date(1990, 1, 1), end=date(2015, 12, 31),
    )
    result = run_backtest(spec, returns, rf)
    excess = result.daily_returns - rf.reindex(result.daily_returns.index).fillna(0.0)
    sharpe = excess.mean() / excess.std(ddof=1) * np.sqrt(252) if excess.std(ddof=1) else float("nan")
    annual_return = (1 + result.daily_returns).prod() ** (252 / len(result.daily_returns)) - 1
    equity = (1 + result.daily_returns).cumprod()
    drawdown = equity / equity.cummax() - 1
    average_turnover = result.turnover.groupby(result.turnover.index.year).sum().mean()
    print(f"Annualized Sharpe (excess of RF): {sharpe:.3f}")
    print(f"Annual return: {annual_return:.2%}")
    print(f"Maximum drawdown: {drawdown.min():.2%}")
    print(f"Average annual turnover: {average_turnover:.2f}x")


if __name__ == "__main__":
    main()
