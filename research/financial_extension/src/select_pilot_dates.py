#!/usr/bin/env python3
"""Select market-state-diverse pilot dates without using regret outcomes."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_COLUMNS = ["atm_iv_1y", "skew_25d_1y", "iv_term_slope", "zero_rate_1y"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=12)
    return parser.parse_args()


def _surface_value(
    surface: pd.DataFrame, *, days: int, cp_flag: str, delta: float
) -> pd.Series:
    subset = surface.loc[
        (surface["days"] == days)
        & (surface["cp_flag"] == cp_flag)
        & (surface["delta"] == delta),
        ["date", "impl_volatility"],
    ]
    if subset.duplicated("date", keep=False).any():
        raise ValueError("surface feature cell is not unique by date")
    return subset.set_index("date")["impl_volatility"]


def compute_market_features(
    surface: pd.DataFrame, zero_curve: pd.DataFrame
) -> pd.DataFrame:
    surface = surface.copy()
    surface["date"] = pd.to_datetime(surface["date"], errors="coerce").dt.normalize()
    surface["days"] = pd.to_numeric(surface["days"], errors="coerce")
    surface["delta"] = pd.to_numeric(surface["delta"], errors="coerce")
    surface["impl_volatility"] = pd.to_numeric(
        surface["impl_volatility"], errors="coerce"
    )
    surface["cp_flag"] = surface["cp_flag"].astype("string").str.upper()

    call_50_365 = _surface_value(surface, days=365, cp_flag="C", delta=50.0)
    put_50_365 = _surface_value(surface, days=365, cp_flag="P", delta=-50.0)
    put_25_365 = _surface_value(surface, days=365, cp_flag="P", delta=-25.0)
    call_25_365 = _surface_value(surface, days=365, cp_flag="C", delta=25.0)
    call_50_182 = _surface_value(surface, days=182, cp_flag="C", delta=50.0)
    put_50_182 = _surface_value(surface, days=182, cp_flag="P", delta=-50.0)
    call_50_730 = _surface_value(surface, days=730, cp_flag="C", delta=50.0)
    put_50_730 = _surface_value(surface, days=730, cp_flag="P", delta=-50.0)

    features = pd.DataFrame(
        {
            "atm_iv_1y": 0.5 * (call_50_365 + put_50_365),
            "skew_25d_1y": put_25_365 - call_25_365,
            "iv_term_slope": 0.5 * (call_50_730 + put_50_730)
            - 0.5 * (call_50_182 + put_50_182),
        }
    )

    zero = zero_curve.copy()
    zero["date"] = pd.to_datetime(zero["date"], errors="coerce").dt.normalize()
    zero["days"] = pd.to_numeric(zero["days"], errors="coerce")
    zero["zero_rate"] = pd.to_numeric(zero["zero_rate"], errors="coerce")
    rates = {}
    for date, group in zero.dropna().groupby("date"):
        ordered = group.sort_values("days")
        if ordered["days"].min() <= 365 <= ordered["days"].max():
            rates[date] = float(
                np.interp(365, ordered["days"], ordered["zero_rate"])
            ) / 100.0
    features["zero_rate_1y"] = pd.Series(rates)
    features.index.name = "date"
    return features.reset_index().sort_values("date").reset_index(drop=True)


def select_diverse_dates(features: pd.DataFrame, count: int) -> pd.DataFrame:
    if count < 2:
        raise ValueError("pilot count must be at least two")
    complete = features.dropna(subset=FEATURE_COLUMNS).copy().reset_index(drop=True)
    if len(complete) < count:
        raise ValueError("not enough complete market dates for requested pilot")
    values = complete[FEATURE_COLUMNS].to_numpy(dtype=float)
    center = np.median(values, axis=0)
    mad = np.median(np.abs(values - center), axis=0)
    fallback = np.std(values, axis=0, ddof=1)
    scale = np.where(mad > 0, 1.4826 * mad, fallback)
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("market features have degenerate robust scale")
    standardized = (values - center) / scale

    distance_to_center = np.sum(standardized**2, axis=1)
    selected = [int(np.argmin(distance_to_center))]
    minimum_distance = np.sum(
        (standardized - standardized[selected[0]]) ** 2, axis=1
    )
    while len(selected) < count:
        minimum_distance[selected] = -np.inf
        next_index = int(np.argmax(minimum_distance))
        selected.append(next_index)
        distance_to_new = np.sum(
            (standardized - standardized[next_index]) ** 2, axis=1
        )
        minimum_distance = np.minimum(minimum_distance, distance_to_new)

    result = complete.iloc[selected].copy()
    result.insert(0, "selection_order", np.arange(1, len(result) + 1))
    return result.reset_index(drop=True)


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    surface = pd.read_parquet(
        data_dir / "spx_vol_surface_monthly_2005_2025.parquet"
    )
    zero = pd.read_parquet(data_dir / "spx_zero_curve_monthly_2005_2025.parquet")
    features = compute_market_features(surface, zero)
    selected = select_diverse_dates(features, args.count)
    output_dir.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_dir / "all_market_features.csv", index=False)
    selected.to_csv(output_dir / "selected_pilot_dates.csv", index=False)

    lines = [
        "# Predeclared OptionMetrics pilot dates",
        "",
        "Dates are selected before any boundary-regret calculation by deterministic farthest-point sampling in robustly standardized market-state features: one-year ATM IV, one-year 25-delta skew, the 182-to-730-day ATM IV slope, and the interpolated one-year zero rate.",
        "",
        f"Complete feature dates: {features[FEATURE_COLUMNS].notna().all(axis=1).sum()} of {len(features)}.",
        "",
        "| Order | Date | 1y ATM IV | 1y 25d skew | IV term slope | 1y zero rate |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in selected.itertuples(index=False):
        lines.append(
            f"| {row.selection_order} | {row.date.strftime('%Y-%m-%d')} | "
            f"{row.atm_iv_1y:.6f} | {row.skew_25d_1y:.6f} | "
            f"{row.iv_term_slope:.6f} | {row.zero_rate_1y:.6f} |"
        )
    lines.extend(
        [
            "",
            "The selection uses the standardized surface only to stratify market states. Raw settlement-consistent quotes remain the calibration target. No regret outcome or contract boundary enters the selection.",
            "",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
