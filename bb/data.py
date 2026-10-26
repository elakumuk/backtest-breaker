"""Runtime loader and parquet cache for Ken French daily portfolio data."""

from __future__ import annotations

import csv
import io
import re
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
RETURNS_ZIP = "49_Industry_Portfolios_daily_CSV.zip"
FACTORS_ZIP = "F-F_Research_Data_Factors_daily_CSV.zip"
CACHE_DIR = Path("data")


def _download_csv(zip_name: str) -> list[str]:
    """Download and read the first CSV member of a Ken French archive."""
    request = urllib.request.Request(BASE_URL + zip_name, headers={"User-Agent": "BacktestBreaker/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    member = next(name for name in archive.namelist() if name.lower().endswith(".csv"))
    return archive.read(member).decode("latin-1").splitlines()


def _read_daily_table(lines: list[str], n_values: int) -> pd.DataFrame:
    """Extract the daily table, stopping at its first non-date row."""
    start = next(i for i, line in enumerate(lines) if re.match(r"^\s*\d{8}\s*,", line))
    header_idx = start - 1
    while header_idx >= 0 and not lines[header_idx].strip():
        header_idx -= 1
    headers = next(csv.reader([lines[header_idx]]))[1:]
    records: list[list[float]] = []
    dates: list[pd.Timestamp] = []
    for line in lines[start:]:
        if not re.match(r"^\s*\d{8}\s*,", line):
            if dates:
                break
            continue
        row = next(csv.reader([line]))
        dates.append(pd.to_datetime(row[0].strip(), format="%Y%m%d"))
        values = [float(value.strip()) if value.strip() else np.nan for value in row[1 : n_values + 1]]
        records.append(values)
    frame = pd.DataFrame(records, index=pd.DatetimeIndex(dates), columns=headers[:n_values])
    frame = frame.replace([-99.99, -999.0], np.nan).div(100.0)
    return frame.sort_index()


def load_universe() -> tuple[pd.DataFrame, pd.Series]:
    """Load 49 value-weighted industry returns and daily risk-free rates."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    returns_path = CACHE_DIR / "industry_returns.parquet"
    rf_path = CACHE_DIR / "risk_free.parquet"
    if returns_path.exists() and rf_path.exists():
        return pd.read_parquet(returns_path), pd.read_parquet(rf_path).iloc[:, 0].rename("RF")

    returns_lines = _download_csv(RETURNS_ZIP)
    marker = next(i for i, line in enumerate(returns_lines) if "Average Value Weighted Returns" in line)
    returns = _read_daily_table(returns_lines[marker + 1 :], 49)
    factors = _read_daily_table(_download_csv(FACTORS_ZIP), 4)
    rf = factors["RF"].reindex(returns.index).rename("RF")
    returns.to_parquet(returns_path)
    rf.to_frame().to_parquet(rf_path)
    return returns, rf
