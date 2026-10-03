from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE))

from market_models import black_option_price  # noqa: E402
from run_market_pilot import (  # noqa: E402
    repair_target_slices,
    select_target_maturity_slices,
    select_unique_quote_per_strike,
)


def synthetic_clean_quotes() -> pd.DataFrame:
    date = pd.Timestamp("2025-01-31")
    frames = []
    optionid = 1
    for dte in (170, 190, 360, 550, 720):
        strikes = np.linspace(70.0, 130.0, 15)
        discount = np.exp(-0.04 * dte / 365.0)
        prices = black_option_price(
            forward=100.0,
            strike=strikes,
            discount=discount,
            maturity=dte / 365.0,
            volatility=0.20,
            cp_flag=["C"] * len(strikes),
        )
        frames.append(
            pd.DataFrame(
                {
                    "date": date,
                    "exdate": date + pd.to_timedelta(dte, unit="D"),
                    "contract_family": "SPXW",
                    "vendor_forward_price": 100.1,
                    "vendor_forward_relative_difference": 0.001,
                    "parity_pair_count": 15,
                    "parity_forward_relative_mad": 0.0001,
                    "dte": dte,
                    "strike": strikes,
                    "call_equivalent_mid": prices,
                    "call_equivalent_bid": prices - 0.10,
                    "call_equivalent_offer": prices + 0.10,
                    "half_spread": 0.10,
                    "discount_factor": discount,
                    "matched_forward_price": 100.0,
                    "open_interest": 100,
                    "optionid": np.arange(optionid, optionid + len(strikes)),
                }
            )
        )
        optionid += len(strikes)
    return pd.concat(frames, ignore_index=True)


def test_maturity_assignment_is_distinct_and_nearest_jointly() -> None:
    slices = select_target_maturity_slices(synthetic_clean_quotes(), min_strikes=12)
    assert set(slices) == {182, 365, 547, 730}
    actual = {target: int(frame["dte"].median()) for target, frame in slices.items()}
    assert actual == {182: 190, 365: 360, 547: 550, 730: 720}
    assert len(set(actual.values())) == 4


def test_repair_orchestrator_preserves_arbitrage_free_slices() -> None:
    slices = select_target_maturity_slices(synthetic_clean_quotes(), min_strikes=12)
    repaired, diagnostics = repair_target_slices(slices)
    assert len(repaired) == 4 * 15
    assert all(record["feasible_within_spread"] for record in diagnostics)
    assert all(record["rows_changed"] == 0 for record in diagnostics)
    assert {record["contract_family"] for record in diagnostics} == {"SPXW"}
    assert all(record["parity_pair_count"] == 15 for record in diagnostics)


def test_duplicate_strike_selection_prefers_tight_then_open_interest() -> None:
    frame = synthetic_clean_quotes().query("dte == 360").iloc[:2].copy()
    duplicate = frame.iloc[[0]].copy()
    duplicate["optionid"] = 999
    duplicate["half_spread"] = 0.05
    duplicate["call_equivalent_bid"] = duplicate["call_equivalent_mid"] - 0.05
    duplicate["call_equivalent_offer"] = duplicate["call_equivalent_mid"] + 0.05
    selected = select_unique_quote_per_strike(
        pd.concat([frame, duplicate], ignore_index=True)
    )
    chosen = selected.loc[selected["strike"] == duplicate.iloc[0]["strike"]].iloc[0]
    assert chosen["optionid"] == 999
