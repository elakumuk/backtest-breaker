"""Tests for Ken French data parsing and data-level invariants."""

import numpy as np
import pandas as pd

from bb.data import _read_weighted_section


def test_value_weighted_csv_section_parser():
    """Only the weighted table is parsed and sentinel values become missing."""
    lines = [
        "Ken French daily portfolio data",
        "Average Value Weighted Returns -- Daily",
        "",
        " ,Alpha,Beta",
        "20200103,3.00,-99.99",
        "20200102,1.00,2.00",
        "20200106,5.00,6.00",
        "Average Equal Weighted Returns -- Daily",
        " ,Alpha,Beta",
        "20200102,99.00,99.00",
    ]
    frame = _read_weighted_section(lines, 2)
    assert len(frame) == 3
    assert frame.index.is_monotonic_increasing and frame.index.is_unique
    assert frame.loc["2020-01-02", "Alpha"] == 0.01
    assert np.isnan(frame.loc["2020-01-03", "Beta"])
    assert frame.loc["2020-01-06", "Beta"] == 0.06
