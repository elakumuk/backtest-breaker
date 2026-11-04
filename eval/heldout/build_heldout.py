"""Build the HELD-OUT notebook eval set for Backtest Breaker's Notebook Mode.

Written by Claude (not Codex) and kept OUTSIDE the repo so the detector can't be
tuned to these files. Commit them to the repo only at D8, after the final eval run.

Every notebook exposes the same interface the proof tests import:
    synthetic_prices() -> DataFrame, signal(prices) -> DataFrame, backtest(prices) -> Series

Layout per notebook: cell 0 markdown, cell 1 imports + data, cell 2 signal (bug lives here),
cell 3 backtest + reporting. Leaky notebooks come with a clean twin that differs only in
the buggy line(s). Decoys are clean but contain code that LOOKS suspicious.

    python build_heldout.py           # writes notebooks/ + labels.json
"""

from __future__ import annotations

import json
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

OUT = Path(__file__).resolve().parent / "notebooks"

DATA_CELL = '''import numpy as np
import pandas as pd

def synthetic_prices():
    rng = np.random.default_rng(7)
    returns = rng.normal(0.0005, 0.012, size=(120, 4))
    return pd.DataFrame(100.0 * (1.0 + returns).cumprod(axis=0), columns=list("ABCD"))
'''

BACKTEST_CELL = '''def backtest(prices, cost_bps=10):
    positions = signal(prices)
    asset_returns = prices.pct_change().fillna(0.0)
    gross = (positions * asset_returns).mean(axis=1)
    turnover = positions.diff().abs().sum(axis=1).fillna(0.0)
    return gross - turnover * cost_bps / 10_000

def sharpe(pnl):
    pnl_std = pnl.std()
    return float(pnl.mean() / pnl_std * np.sqrt(252)) if pnl_std > 0 else 0.0
'''

# name -> (title, data_cell, signal_leaky, signal_clean, accepted bug classes)
PAIRS: dict[str, tuple[str, str, str, str, list[str]]] = {
    "bfill_gaps": (
        "Momentum on a price panel with missing quotes",
        DATA_CELL.replace(
            '    return pd.DataFrame(100.0 * (1.0 + returns).cumprod(axis=0), columns=list("ABCD"))\n',
            '    prices = pd.DataFrame(100.0 * (1.0 + returns).cumprod(axis=0), columns=list("ABCD"))\n'
            '    prices.loc[prices.index % 5 >= 3, "C"] = np.nan  # stale / missing quotes\n'
            '    return prices\n',
        ),
        '''def signal(prices):
    """10-day momentum, lagged one bar. Missing quotes are filled first."""
    filled = prices.bfill()
    momentum = filled.pct_change(10).shift(1)
    return momentum.clip(-0.2, 0.2).fillna(0.0)
''',
        '''def signal(prices):
    """10-day momentum, lagged one bar. Missing quotes are filled first."""
    filled = prices.ffill()
    momentum = filled.pct_change(10).shift(1)
    return momentum.clip(-0.2, 0.2).fillna(0.0)
''',
        ["lookahead_shift", "fullsample_leak"],
    ),
    "centered_rolling": (
        "Smoothed short-term trend",
        DATA_CELL,
        '''def signal(prices):
    """Smooth daily returns over a 9-day window, then lag one bar."""
    smooth = prices.pct_change().rolling(9, center=True).mean()
    return (smooth.shift(1) * 100).clip(-1, 1).fillna(0.0)
''',
        '''def signal(prices):
    """Smooth daily returns over a 9-day window, then lag one bar."""
    smooth = prices.pct_change().rolling(9).mean()
    return (smooth.shift(1) * 100).clip(-1, 1).fillna(0.0)
''',
        ["lookahead_shift", "fullsample_leak"],
    ),
    "scaler_fit_all": (
        "Standardized multi-feature score",
        DATA_CELL,
        '''from sklearn.preprocessing import StandardScaler

def signal(prices):
    """Standardize lagged features with sklearn, then average them into a score."""
    rets = prices.pct_change().shift(1)
    feats = pd.concat({"r5": rets.rolling(5).mean(), "v20": rets.rolling(20).std()}, axis=1).fillna(0.0)
    scaled = pd.DataFrame(StandardScaler().fit_transform(feats), index=feats.index, columns=feats.columns)
    return (scaled["r5"] - scaled["v20"]).clip(-2, 2)
''',
        '''from sklearn.preprocessing import StandardScaler

def signal(prices):
    """Standardize lagged features with sklearn, then average them into a score."""
    rets = prices.pct_change().shift(1)
    feats = pd.concat({"r5": rets.rolling(5).mean(), "v20": rets.rolling(20).std()}, axis=1).fillna(0.0)
    mu = feats.expanding(min_periods=20).mean()
    sd = feats.expanding(min_periods=20).std().replace(0.0, np.nan)
    scaled = ((feats - mu) / sd).fillna(0.0)
    return (scaled["r5"] - scaled["v20"]).clip(-2, 2)
''',
        ["fullsample_leak"],
    ),
    "shuffled_split": (
        "Linear model forecasting next-day returns",
        DATA_CELL,
        '''from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def signal(prices):
    """Fit one pooled linear model on lagged returns; trade the sign of its forecast."""
    rets = prices.pct_change()
    X = pd.concat({"lag1": rets.shift(1).stack(), "lag5": rets.rolling(5).mean().shift(1).stack()}, axis=1)
    y = rets.stack().reindex(X.index)
    data = pd.concat([X, y.rename("y")], axis=1).dropna()
    X_train, X_test, y_train, y_test = train_test_split(data[["lag1", "lag5"]], data["y"], test_size=0.3, random_state=0)
    model = LinearRegression().fit(X_train, y_train)
    pred = pd.Series(model.predict(X.fillna(0.0)), index=X.index).unstack()
    return np.sign(pred).reindex(index=prices.index, columns=prices.columns).fillna(0.0)
''',
        '''from sklearn.linear_model import LinearRegression

def signal(prices):
    """Fit one pooled linear model on lagged returns; trade the sign of its forecast."""
    rets = prices.pct_change()
    X = pd.concat({"lag1": rets.shift(1).stack(), "lag5": rets.rolling(5).mean().shift(1).stack()}, axis=1)
    y = rets.stack().reindex(X.index)
    data = pd.concat([X, y.rename("y")], axis=1).dropna()
    train = data[data.index.get_level_values(0) < 40]
    model = LinearRegression().fit(train[["lag1", "lag5"]], train["y"])
    pred = pd.Series(model.predict(X.fillna(0.0)), index=X.index).unstack()
    pred.loc[pred.index < 40] = 0.0
    return np.sign(pred).reindex(index=prices.index, columns=prices.columns).fillna(0.0)
''',
        ["fullsample_leak"],
    ),
    "asof_forward": (
        "Event signal joined from an announcements table",
        DATA_CELL,
        '''def announcements(prices):
    """Every 10 bars an 'announcement' publishes the trailing 10-bar return."""
    dates = prices.index[10::10]
    trailing = prices.pct_change(10).loc[dates]
    return trailing.reset_index().rename(columns={"index": "t"})

def signal(prices):
    """Attach the relevant announcement to each bar and trade its sign."""
    ann = announcements(prices)
    bars = pd.DataFrame({"t": prices.index})
    merged = pd.merge_asof(bars, ann, on="t", direction="forward")
    return (merged.set_index("t")[prices.columns].shift(1) * 10).clip(-1, 1).fillna(0.0)
''',
        '''def announcements(prices):
    """Every 10 bars an 'announcement' publishes the trailing 10-bar return."""
    dates = prices.index[10::10]
    trailing = prices.pct_change(10).loc[dates]
    return trailing.reset_index().rename(columns={"index": "t"})

def signal(prices):
    """Attach the relevant announcement to each bar and trade its sign."""
    ann = announcements(prices)
    bars = pd.DataFrame({"t": prices.index})
    merged = pd.merge_asof(bars, ann, on="t", direction="backward")
    return (merged.set_index("t")[prices.columns].shift(1) * 10).clip(-1, 1).fillna(0.0)
''',
        ["lookahead_shift", "fullsample_leak"],
    ),
}

# Clean notebooks with code that looks suspicious. Any PROVEN finding here is a false positive.
DECOYS: dict[str, tuple[str, str]] = {
    "decoy_forward_label": (
        "Momentum with an information-coefficient diagnostic",
        '''def signal(prices):
    """20-day momentum known at the close of t-1."""
    return np.sign(prices.pct_change(20).shift(1)).fillna(0.0)

def information_coefficient(prices):
    """Diagnostic only: correlate today's signal with the NEXT day's return (standard IC)."""
    fwd_ret = prices.pct_change().shift(-1)
    return signal(prices).stack().corr(fwd_ret.stack())
''',
    ),
    "decoy_fixed_params": (
        "Volatility-scaled trend with a fixed, pre-set target",
        '''VOL_TARGET = 0.10          # chosen before looking at this data
LOOKBACK = 60

def signal(prices):
    """Trend sign scaled by trailing realized vol (all lagged)."""
    rets = prices.pct_change()
    trend = np.sign(rets.rolling(LOOKBACK).mean().shift(1))
    realized_std = rets.rolling(20).std().shift(1) * np.sqrt(252)
    scale = (VOL_TARGET / realized_std).clip(upper=3.0)
    return (trend * scale).fillna(0.0)
''',
    ),
}


def notebook(title: str, data_cell: str, signal_cell: str) -> nbformat.NotebookNode:
    nb = new_notebook()
    nb.cells = [
        new_markdown_cell(f"# {title}"),
        new_code_cell(data_cell),
        new_code_cell(signal_cell),
        new_code_cell(BACKTEST_CELL + "\nprint(round(sharpe(backtest(synthetic_prices())), 3))\n"),
    ]
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    return nb


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name, (title, data_cell, leaky, clean, accept) in PAIRS.items():
        nbformat.write(notebook(title, data_cell, leaky), OUT / f"{name}.ipynb")
        nbformat.write(notebook(title, data_cell, clean), OUT / f"{name}__clean.ipynb")
        manifest.append({"notebook": f"{name}.ipynb", "control": f"{name}__clean.ipynb",
                         "labels": [{"cell_index": 2, "accept": accept}]})
        manifest.append({"notebook": f"{name}__clean.ipynb", "control": "decoy_fixed_params.ipynb", "labels": []})
    for name, (title, signal_cell) in DECOYS.items():
        nbformat.write(notebook(title, DATA_CELL, signal_cell), OUT / f"{name}.ipynb")
        # Control is a DIFFERENT clean notebook, so a non-specific test can still be falsely PROVEN.
        manifest.append({"notebook": f"{name}.ipynb", "control": "centered_rolling__clean.ipynb", "labels": []})
    (OUT.parent / "labels.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(list(OUT.glob('*.ipynb')))} notebooks + labels.json")


if __name__ == "__main__":
    main()
