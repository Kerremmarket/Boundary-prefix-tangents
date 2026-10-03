from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE))

from arbitrage_repair import repair_call_slice  # noqa: E402


def quote_slice(mids, half_spreads):
    mids = np.asarray(mids, dtype=float)
    half_spreads = np.asarray(half_spreads, dtype=float)
    return pd.DataFrame(
        {
            "strike": [90.0, 100.0, 110.0],
            "call_equivalent_bid": mids - half_spreads,
            "call_equivalent_offer": mids + half_spreads,
            "call_equivalent_mid": mids,
            "discount_factor": [1.0, 1.0, 1.0],
            "matched_forward_price": [100.0, 100.0, 100.0],
        }
    )


def slopes(frame: pd.DataFrame) -> np.ndarray:
    return np.diff(frame["repaired_call_price"]) / np.diff(frame["strike"])


def test_arbitrage_free_curve_is_unchanged() -> None:
    result = repair_call_slice(quote_slice([15.0, 8.0, 3.0], [0.5, 0.5, 0.5]))
    assert result.feasible_within_spread is True
    assert result.rows_changed == 0
    assert np.allclose(result.quotes["repaired_call_price"], [15.0, 8.0, 3.0])


def test_convexity_violation_is_repaired_inside_wide_spreads() -> None:
    result = repair_call_slice(quote_slice([15.0, 10.0, 3.0], [2.0, 2.0, 2.0]))
    repaired_slopes = slopes(result.quotes)
    assert result.feasible_within_spread is True
    assert repaired_slopes[0] <= repaired_slopes[1] + 1e-10
    assert np.all(repaired_slopes <= 1e-10)
    assert np.all(repaired_slopes >= -1.0 - 1e-10)
    assert not result.quotes["repair_outside_spread"].any()


def test_tight_inconsistent_spreads_are_flagged_as_outside_repair() -> None:
    result = repair_call_slice(quote_slice([15.0, 10.0, 3.0], [0.05, 0.05, 0.05]))
    assert result.feasible_within_spread is False
    assert result.quotes["repair_outside_spread"].any()
    repaired_slopes = slopes(result.quotes)
    assert repaired_slopes[0] <= repaired_slopes[1] + 1e-10


def test_duplicate_strikes_fail_closed() -> None:
    frame = quote_slice([15.0, 8.0, 3.0], [0.5, 0.5, 0.5])
    frame.loc[1, "strike"] = 90.0
    try:
        repair_call_slice(frame)
    except ValueError as error:
        assert "one selected quote per strike" in str(error)
    else:
        raise AssertionError("duplicate strikes were not rejected")
