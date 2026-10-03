from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


SOURCE = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE))

from wrds_optionmetrics_extract import normalize_frame, resolve_columns  # noqa: E402


def test_option_schema_requires_settlement_and_contract_metadata() -> None:
    columns = [
        "secid",
        "date",
        "optionid",
        "symbol",
        "exdate",
        "cp_flag",
        "strike_price",
        "best_bid",
        "best_offer",
        "volume",
        "open_interest",
        "impl_volatility",
        "delta",
        "gamma",
        "vega",
        "theta",
        "am_settlement",
        "contract_size",
        "ss_flag",
        "forward_price",
        "expiry_indicator",
        "root",
        "suffix",
    ]
    mapping = resolve_columns("option_prices", columns)
    assert mapping is not None
    assert mapping["special_settlement"] == "ss_flag"
    assert mapping["am_settlement"] == "am_settlement"
    assert mapping["root"] == "root"


def test_normalized_option_frame_keeps_enrichment_fields() -> None:
    frame = pd.DataFrame(
        {
            "secid": [108105],
            "date": ["2025-01-31"],
            "optionid": [1],
            "symbol": ["SPXW TEST"],
            "exdate": ["2026-01-31"],
            "cp_flag": ["C"],
            "strike_price": [5000000.0],
            "strike": [5000.0],
            "best_bid": [10.0],
            "best_offer": [11.0],
            "volume": [0],
            "open_interest": [10],
            "impl_volatility": [0.2],
            "delta": [0.5],
            "gamma": [0.001],
            "vega": [1.0],
            "theta": [-1.0],
            "am_settlement": [0],
            "contract_size": [100],
            "special_settlement": ["0"],
            "forward_price": [5100.0],
            "expiry_indicator": ["m"],
            "root": ["SPXW"],
            "suffix": [""],
            "dte": [365],
        }
    )
    normalized = normalize_frame("option_prices", frame)
    assert normalized.loc[0, "am_settlement"] == 0
    assert normalized.loc[0, "root"] == "SPXW"
    assert normalized.loc[0, "special_settlement"] == "0"
    assert normalized.loc[0, "forward_price"] == 5100.0
