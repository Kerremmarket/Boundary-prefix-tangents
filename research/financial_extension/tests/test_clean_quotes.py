from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SOURCE = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE))

from clean_quotes import prepare_calibration_quotes  # noqa: E402


def data_frames():
    date = pd.Timestamp("2025-01-31")
    expiry = pd.Timestamp("2026-01-31")
    strikes = np.asarray([80.0, 90.0, 100.0, 110.0, 120.0])
    discount = np.exp(-0.05)
    forward = 100.0
    rows = []
    optionid = 1
    for strike in strikes:
        put_mid = max(strike - forward, 0.0) * discount + 2.0
        call_mid = put_mid + discount * (forward - strike)
        for cp_flag, mid in (("C", call_mid), ("P", put_mid)):
            rows.append(
                {
                    "date": date,
                    "exdate": expiry,
                    "optionid": optionid,
                    "symbol": f"SPXW 260131{cp_flag}{int(strike * 1000):08d}",
                    "cp_flag": cp_flag,
                    "strike": strike,
                    "best_bid": mid - 0.1,
                    "best_offer": mid + 0.1,
                    "dte": 365,
                }
            )
            optionid += 1
    options = pd.DataFrame(rows)
    forwards = pd.DataFrame(
        {
            "date": [date, date],
            "expiration": [expiry, expiry],
            "am_settlement": [0, 1],
            "forward_price": [100.01, 101.0],
        }
    )
    zero = pd.DataFrame(
        {
            "date": [date, date],
            "days": [180, 730],
            "zero_rate": [5.0, 5.0],
        }
    )
    spot = pd.DataFrame({"date": [date], "spot": [99.0]})
    return options, forwards, zero, spot


def test_cleaning_infers_parity_forward_and_constructs_otm_equivalents() -> None:
    options, forwards, zero, spot = data_frames()
    result = prepare_calibration_quotes(options, forwards, zero, spot)
    assert result.attrition["input"] == 10
    assert result.attrition["parity_identified_forward"] == 10
    assert len(result.quotes) == 5
    assert result.quotes["strike"].nunique() == 5
    assert np.allclose(result.quotes["zero_rate"], 0.05)
    assert np.allclose(result.quotes["matched_forward_price"], 100.0)
    assert np.allclose(result.quotes["vendor_forward_price"], 100.01)
    assert set(result.quotes["contract_family"]) == {"SPXW"}

    put = result.quotes.loc[result.quotes["strike"] == 90.0].iloc[0]
    parity = put["discount_factor"] * (
        put["matched_forward_price"] - put["strike"]
    )
    assert put["cp_flag"] == "P"
    assert np.isclose(put["call_equivalent_mid"], put["mid"] + parity)
    assert put["log_forward_moneyness"] < 0


def test_current_extraction_does_not_require_optional_contract_metadata() -> None:
    options, forwards, zero, spot = data_frames()
    result = prepare_calibration_quotes(options, forwards, zero, spot)
    assert len(result.quotes) == 5


def test_missing_observed_symbol_fails_closed() -> None:
    options, forwards, zero, spot = data_frames()
    with pytest.raises(ValueError, match="symbol"):
        prepare_calibration_quotes(options.drop(columns=["symbol"]), forwards, zero, spot)


def test_duplicate_settlement_forward_key_fails_closed() -> None:
    options, forwards, zero, spot = data_frames()
    duplicated = pd.concat([forwards, forwards.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="not unique"):
        prepare_calibration_quotes(options, duplicated, zero, spot)


def test_legacy_roots_are_pooled_before_parity_estimation() -> None:
    options, forwards, zero, spot = data_frames()
    calls = options["cp_flag"].eq("C")
    options.loc[calls, "symbol"] = "SXB.FE"
    options.loc[~calls, "symbol"] = "SZT.RE"
    result = prepare_calibration_quotes(options, forwards, zero, spot)
    assert set(result.quotes["contract_family"]) == {"LEGACY"}
    assert np.allclose(result.quotes["matched_forward_price"], 100.0)
