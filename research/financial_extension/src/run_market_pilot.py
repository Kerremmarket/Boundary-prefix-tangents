#!/usr/bin/env python3
"""Run the predeclared parity-consistent OptionMetrics calibration pilot."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from arbitrage_repair import repair_call_slice
from calibration import fit_black, fit_kou, fit_lognormal_mixture
from clean_quotes import prepare_calibration_quotes


TARGET_DTES = (182, 365, 547, 730)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--pilot-dates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-strikes", type=int, default=12)
    parser.add_argument("--kou-integration-nodes", type=int, default=192)
    return parser.parse_args()


def select_unique_quote_per_strike(slice_quotes: pd.DataFrame) -> pd.DataFrame:
    """Select the tightest interval per economically identical strike key."""

    frame = slice_quotes.copy()
    if "half_spread" not in frame:
        frame["half_spread"] = (
            frame["call_equivalent_offer"] - frame["call_equivalent_bid"]
        ) / 2.0
    open_interest = (
        frame["open_interest"]
        if "open_interest" in frame
        else pd.Series(0.0, index=frame.index)
    )
    optionid = (
        frame["optionid"]
        if "optionid" in frame
        else pd.Series(np.arange(len(frame)), index=frame.index)
    )
    frame["_open_interest_sort"] = -pd.to_numeric(
        open_interest, errors="coerce"
    ).fillna(0)
    frame["_optionid_sort"] = pd.to_numeric(
        optionid, errors="coerce"
    ).fillna(np.inf)
    frame = frame.sort_values(
        ["strike", "half_spread", "_open_interest_sort", "_optionid_sort"]
    )
    selected = frame.drop_duplicates("strike", keep="first").copy()
    return selected.drop(columns=["_open_interest_sort", "_optionid_sort"])


def select_target_maturity_slices(
    date_quotes: pd.DataFrame,
    *,
    target_dtes=TARGET_DTES,
    min_strikes: int = 12,
) -> dict[int, pd.DataFrame]:
    """Assign distinct listed expiries to target DTEs without interpolation."""

    if min_strikes < 5:
        raise ValueError("min_strikes must be at least five")
    keys = ["date", "exdate", "contract_family"]
    candidates = []
    selected_groups = {}
    for key, group in date_quotes.groupby(keys, sort=True):
        unique = select_unique_quote_per_strike(group)
        if unique["strike"].nunique() < min_strikes:
            continue
        median_dte = float(unique["dte"].median())
        candidate_index = len(candidates)
        candidates.append((key, median_dte, len(unique)))
        selected_groups[candidate_index] = unique
    if len(candidates) < len(target_dtes):
        raise ValueError(
            f"only {len(candidates)} eligible expiry slices for {len(target_dtes)} targets"
        )

    cost = np.asarray(
        [
            [abs(candidate[1] - target) for candidate in candidates]
            for target in target_dtes
        ],
        dtype=float,
    )
    target_indices, candidate_indices = linear_sum_assignment(cost)
    if len(target_indices) != len(target_dtes):
        raise RuntimeError("maturity assignment did not cover every target")
    result = {}
    for target_index, candidate_index in zip(target_indices, candidate_indices):
        target = int(target_dtes[target_index])
        frame = selected_groups[int(candidate_index)].copy()
        frame["target_dte"] = target
        frame["target_dte_distance"] = abs(frame["dte"].median() - target)
        result[target] = frame
    return result


def repair_target_slices(slices: dict[int, pd.DataFrame]) -> tuple[pd.DataFrame, list[dict]]:
    repaired = []
    diagnostics = []
    for target, frame in sorted(slices.items()):
        result = repair_call_slice(frame)
        output = result.quotes.copy()
        output["target_dte"] = target
        repaired.append(output)
        diagnostics.append(
            {
                "target_dte": target,
                "actual_dte": float(frame["dte"].median()),
                "exdate": pd.Timestamp(frame["exdate"].iloc[0]).strftime("%Y-%m-%d"),
                "contract_family": str(frame["contract_family"].iloc[0]),
                "parity_forward": float(frame["matched_forward_price"].iloc[0]),
                "vendor_forward": float(frame["vendor_forward_price"].iloc[0]),
                "vendor_forward_relative_difference": float(
                    frame["vendor_forward_relative_difference"].iloc[0]
                ),
                "parity_pair_count": int(frame["parity_pair_count"].iloc[0]),
                "parity_forward_relative_mad": float(
                    frame["parity_forward_relative_mad"].iloc[0]
                ),
                "quotes": int(len(frame)),
                "feasible_within_spread": result.feasible_within_spread,
                "weighted_l1_objective": result.weighted_l1_objective,
                "max_absolute_change": result.max_absolute_change,
                "rows_changed": result.rows_changed,
                "rows_outside_spread": int(
                    result.quotes["repair_outside_spread"].sum()
                ),
            }
        )
    return pd.concat(repaired, ignore_index=True), diagnostics


def calibration_record(result) -> dict:
    payload = asdict(result)
    payload.pop("fitted_prices")
    return payload


def calibrate_market_date(
    repaired_quotes: pd.DataFrame,
    *,
    kou_integration_nodes: int,
) -> tuple[list[dict], pd.DataFrame]:
    one_year = repaired_quotes.loc[repaired_quotes["target_dte"] == 365].copy()
    if len(one_year) < 5:
        raise ValueError("one-year target slice is unavailable")
    results = [
        fit_black(repaired_quotes),
        fit_lognormal_mixture(one_year),
        fit_kou(repaired_quotes, integration_nodes=kou_integration_nodes),
    ]
    fitted = repaired_quotes.copy()
    # BS and Kou cover all target maturities; the annual mixture covers only
    # the one-year slice and is stored separately in its result record.
    fitted["black_fitted_price"] = results[0].fitted_prices
    fitted["kou_fitted_price"] = results[2].fitted_prices
    return [calibration_record(result) for result in results], fitted


def _read_selected(path: Path) -> list[pd.Timestamp]:
    selected = pd.read_csv(path, parse_dates=["date"])
    if "selection_order" in selected:
        selected = selected.sort_values("selection_order")
    if selected["date"].duplicated().any():
        raise ValueError("pilot date file contains duplicates")
    return [pd.Timestamp(value).normalize() for value in selected["date"]]


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    selected_dates = _read_selected(args.pilot_dates.expanduser().resolve())

    options = pd.read_parquet(
        data_dir / "spx_option_prices_monthly_2005_2025.parquet"
    )
    options["date"] = pd.to_datetime(options["date"]).dt.normalize()
    options = options.loc[options["date"].isin(selected_dates)].copy()
    forwards = pd.read_parquet(
        data_dir / "spx_forward_prices_monthly_2005_2025.parquet"
    )
    zero = pd.read_parquet(data_dir / "spx_zero_curve_monthly_2005_2025.parquet")
    spot = pd.read_parquet(data_dir / "spx_spot_monthly_2005_2025.parquet")
    for frame in (forwards, zero, spot):
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    forwards = forwards.loc[forwards["date"].isin(selected_dates)].copy()
    zero = zero.loc[zero["date"].isin(selected_dates)].copy()
    spot = spot.loc[spot["date"].isin(selected_dates)].copy()

    cleaned = prepare_calibration_quotes(options, forwards, zero, spot)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "cleaning_attrition.json").write_text(
        json.dumps(
            {"attrition": cleaned.attrition, "diagnostics": cleaned.diagnostics},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    records = []
    repair_records = []
    fitted_frames = []
    for date in selected_dates:
        date_text = date.strftime("%Y-%m-%d")
        date_quotes = cleaned.quotes.loc[cleaned.quotes["date"] == date]
        try:
            slices = select_target_maturity_slices(
                date_quotes, min_strikes=args.min_strikes
            )
            repaired, repair_diagnostics = repair_target_slices(slices)
            calibration_results, fitted = calibrate_market_date(
                repaired,
                kou_integration_nodes=args.kou_integration_nodes,
            )
            for record in calibration_results:
                records.append({"date": date_text, "status": "ok", **record})
            for record in repair_diagnostics:
                repair_records.append({"date": date_text, **record})
            fitted["date"] = date
            fitted_frames.append(fitted)
        except Exception as error:
            records.append(
                {
                    "date": date_text,
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )

    pd.DataFrame(records).to_json(
        output_dir / "calibration_results.json", orient="records", indent=2
    )
    pd.DataFrame(repair_records).to_csv(
        output_dir / "repair_diagnostics.csv", index=False
    )
    if fitted_frames:
        pd.concat(fitted_frames, ignore_index=True).to_parquet(
            output_dir / "pilot_repaired_quotes_and_fits.parquet", index=False
        )
    failures = sum(record.get("status") == "failed" for record in records)
    print(f"pilot dates={len(selected_dates)} failure records={failures}")
    print(output_dir)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
