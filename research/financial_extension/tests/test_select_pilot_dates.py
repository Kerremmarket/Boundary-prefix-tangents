from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE))

from select_pilot_dates import (  # noqa: E402
    FEATURE_COLUMNS,
    compute_market_features,
    select_diverse_dates,
)


def test_market_features_use_declared_surface_cells_and_decimal_rate() -> None:
    date = pd.Timestamp("2025-01-31")
    rows = []
    for days, call50, put50 in [(182, 0.20, 0.22), (365, 0.21, 0.23), (730, 0.24, 0.26)]:
        rows.extend(
            [
                (date, days, "C", 50.0, call50),
                (date, days, "P", -50.0, put50),
            ]
        )
    rows.extend(
        [
            (date, 365, "C", 25.0, 0.18),
            (date, 365, "P", -25.0, 0.30),
        ]
    )
    surface = pd.DataFrame(
        rows, columns=["date", "days", "cp_flag", "delta", "impl_volatility"]
    )
    zero = pd.DataFrame(
        {
            "date": [date, date],
            "days": [180, 730],
            "zero_rate": [4.0, 6.0],
        }
    )
    features = compute_market_features(surface, zero).iloc[0]
    assert np.isclose(features["atm_iv_1y"], 0.22)
    assert np.isclose(features["skew_25d_1y"], 0.12)
    assert np.isclose(features["iv_term_slope"], 0.04)
    expected_rate = np.interp(365, [180, 730], [4.0, 6.0]) / 100.0
    assert np.isclose(features["zero_rate_1y"], expected_rate)


def test_diverse_selection_is_deterministic_and_unique() -> None:
    rng = np.random.default_rng(19)
    features = pd.DataFrame(rng.normal(size=(30, 4)), columns=FEATURE_COLUMNS)
    features.insert(0, "date", pd.date_range("2000-01-31", periods=30, freq="ME"))
    first = select_diverse_dates(features, 8)
    second = select_diverse_dates(features, 8)
    assert first.equals(second)
    assert len(first) == 8
    assert first["date"].is_unique
    assert first["selection_order"].tolist() == list(range(1, 9))
