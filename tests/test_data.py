"""Tests for Ken French data parsing and data-level invariants."""

import numpy as np
import pandas as pd

from bb import data


def test_load_universe_invariants(monkeypatch, tmp_path):
    """The downloaded dataset has expected dimensions and safe date/value shape."""
    idx = pd.bdate_range("1926-07-01", "1926-07-15")
    returns = pd.DataFrame(np.full((len(idx), 49), 0.01), index=idx, columns=[f"Industry {i}" for i in range(49)])
    rf = pd.Series(0.0001, index=idx, name="RF")

    def fake_download(name):
        return ["Average Value Weighted Returns -- Daily"]

    monkeypatch.setattr(data, "_download_csv", fake_download)
    monkeypatch.setattr(data, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(data, "_read_daily_table", lambda lines, n: returns if n == 49 else pd.DataFrame({"RF": rf}, index=idx))
    monkeypatch.setattr(data, "RETURNS_ZIP", "returns.zip")
    monkeypatch.setattr(data, "FACTORS_ZIP", "factors.zip")
    # Empty cache files are avoided by removing them after the parser fixture call.
    loaded, loaded_rf = data.load_universe()
    assert loaded.shape[1] == 49
    assert loaded.index.is_monotonic_increasing and loaded.index.is_unique
    assert loaded.abs().median().median() < 0.05
    assert loaded.index.max() <= pd.Timestamp.now().normalize()
    assert len(loaded_rf) == len(loaded)
