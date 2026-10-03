from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


SOURCE = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE))

from audit_optionmetrics import forward_audit, option_audit, surface_audit  # noqa: E402


def base_options() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-31", "2025-01-31"]),
            "exdate": pd.to_datetime(["2026-01-31", "2026-01-31"]),
            "optionid": [1, 2],
            "symbol": ["SPXW 260131C00100000", "SPXW 260131P00110000"],
            "best_bid": [1.0, 0.0],
            "best_offer": [1.2, 0.1],
            "strike": [100.0, 110.0],
            "dte": [365, 365],
            "cp_flag": ["C", "P"],
            "volume": [0, 2],
            "open_interest": [4, 0],
            "impl_volatility": [0.2, None],
        }
    )


def test_option_audit_accepts_existing_calibration_fields() -> None:
    options = base_options()
    spot = pd.DataFrame({"date": pd.to_datetime(["2025-01-31"]), "spot": [100.0]})
    audit = option_audit(options, spot)
    assert audit["calibration_fields_ready"] is True
    assert "am_settlement" in audit["optional_metadata_missing"]
    assert audit["pre_forward_valid_quote_rows"] == 1
    assert audit["bid_le_zero"] == 1
    assert audit["missing_vendor_iv"] == 1


def test_option_audit_passes_metadata_presence() -> None:
    options = base_options()
    options["am_settlement"] = 0
    options["contract_size"] = 100
    options["special_settlement"] = "0"
    options["forward_price"] = 101.0
    options["expiry_indicator"] = "m"
    options["root"] = "SPXW"
    spot = pd.DataFrame({"date": pd.to_datetime(["2025-01-31"]), "spot": [100.0]})
    audit = option_audit(options, spot)
    assert audit["calibration_fields_ready"] is True
    assert audit["optional_metadata_missing"] == []
    assert audit["root_counts"] == {"SPXW": 2}


def test_forward_audit_requires_settlement_to_disambiguate() -> None:
    forwards = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-31", "2025-01-31"]),
            "expiration": pd.to_datetime(["2026-01-31", "2026-01-31"]),
            "am_settlement": [0, 1],
        }
    )
    audit = forward_audit(forwards)
    assert audit["rows_in_duplicate_date_expiration_groups"] == 2
    assert audit["rows_in_duplicate_date_expiration_settlement_groups"] == 0


def test_surface_missingness_is_reported_by_maturity() -> None:
    surface = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-31", "2025-01-31"]),
            "days": [365, 730],
            "impl_volatility": [0.2, None],
        }
    )
    audit = surface_audit(surface)
    assert audit["missing_iv"] == 1
    assert audit["missing_iv_by_maturity"] == {"730": 1}
