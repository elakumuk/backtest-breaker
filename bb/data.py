"""Runtime loader and parquet cache for Ken French daily portfolio data."""

from __future__ import annotations

import csv
import io
import re
import urllib.request
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
RETURNS_ZIP = "49_Industry_Portfolios_daily_CSV.zip"
FACTORS_ZIP = "F-F_Research_Data_Factors_daily_CSV.zip"
CACHE_DIR = Path(__file__).resolve().parents[1] / "data"


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


def _read_weighted_section(lines: list[str], n_values: int) -> pd.DataFrame:
    """Select and parse a value-weighted section with the requested width."""
    marker = next(i for i, line in enumerate(lines) if "Average Value Weighted Returns -- Daily" in line)
    return _read_daily_table(lines[marker + 1 :], n_values)


def _read_value_weighted_daily(lines: list[str]) -> pd.DataFrame:
    """Select and parse the 49-industry value-weighted daily section."""
    return _read_weighted_section(lines, 49)


def _load_universe() -> tuple[pd.DataFrame, pd.Series]:
    """Private loader for 49 value-weighted industry returns and daily risk-free rates."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    returns_path = CACHE_DIR / "industry_returns.parquet"
    rf_path = CACHE_DIR / "risk_free.parquet"
    if returns_path.exists() and rf_path.exists():
        return pd.read_parquet(returns_path), pd.read_parquet(rf_path).iloc[:, 0].rename("RF")

    returns_lines = _download_csv(RETURNS_ZIP)
    returns = _read_value_weighted_daily(returns_lines)
    factors = _read_daily_table(_download_csv(FACTORS_ZIP), 4)
    rf = factors["RF"].reindex(returns.index).rename("RF")
    returns.to_parquet(returns_path)
    rf.to_frame().to_parquet(rf_path)
    return returns, rf


class DataGate:
    """Public access point that exposes only pre-holdout returns and risk-free rates."""

    def __init__(self, holdout_years: int = 3) -> None:
        if holdout_years < 1:
            raise ValueError("holdout_years must be positive")
        self.holdout_years = holdout_years
        self._returns, self._rf = _load_universe()
        if self._returns.empty:
            raise ValueError("Ken French returns are empty")
        self._holdout_start = pd.Timestamp(
            year=int(self._returns.index.max().year) - holdout_years + 1,
            month=1,
            day=1,
        ).date()

    @property
    def holdout_start(self) -> date:
        """The first calendar date reserved for the holdout period."""
        return self._holdout_start

    def in_sample(self) -> tuple[pd.DataFrame, pd.Series]:
        """Return copies containing dates strictly before ``holdout_start``."""
        mask = self._returns.index < pd.Timestamp(self.holdout_start)
        return self._returns.loc[mask].copy(), self._rf.loc[self._rf.index < pd.Timestamp(self.holdout_start)].copy()
