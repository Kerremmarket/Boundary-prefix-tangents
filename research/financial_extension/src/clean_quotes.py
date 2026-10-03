"""Parity-consistent construction of calibration-ready SPX quote slices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_OPTION_COLUMNS = {
    "date",
    "exdate",
    "optionid",
    "symbol",
    "cp_flag",
    "strike",
    "best_bid",
    "best_offer",
    "dte",
}


@dataclass(frozen=True)
class QuoteCleaningResult:
    quotes: pd.DataFrame
    attrition: dict[str, int]
    diagnostics: dict[str, Any]


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _normalize_dates(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        result[column] = pd.to_datetime(result[column], errors="coerce").dt.normalize()
    return result


def interpolate_zero_rates(
    quote_dates: pd.Series,
    dte: pd.Series,
    zero_curve: pd.DataFrame,
) -> np.ndarray:
    """Interpolate OptionMetrics continuously compounded rates in decimal units."""

    _require_columns(zero_curve, {"date", "days", "zero_rate"}, "zero_curve")
    curve = _normalize_dates(zero_curve, ("date",))
    curve["days"] = pd.to_numeric(curve["days"], errors="coerce")
    curve["zero_rate"] = pd.to_numeric(curve["zero_rate"], errors="coerce")
    grouped = {
        date: group.sort_values("days")
        for date, group in curve.dropna(subset=["date", "days", "zero_rate"]).groupby("date")
    }
    result = np.full(len(quote_dates), np.nan)
    normalized_dates = pd.to_datetime(quote_dates, errors="coerce").dt.normalize()
    maturity_days = pd.to_numeric(dte, errors="coerce").to_numpy(dtype=float)
    for date, indices in normalized_dates.groupby(normalized_dates).groups.items():
        group = grouped.get(date)
        if group is None or len(group) < 2:
            continue
        targets = maturity_days[np.asarray(indices, dtype=int)]
        minimum = float(group["days"].min())
        maximum = float(group["days"].max())
        inside = np.isfinite(targets) & (targets >= minimum) & (targets <= maximum)
        interpolated = np.full(len(targets), np.nan)
        interpolated[inside] = np.interp(
            targets[inside],
            group["days"].to_numpy(dtype=float),
            group["zero_rate"].to_numpy(dtype=float) / 100.0,
        )
        result[np.asarray(indices, dtype=int)] = interpolated
    return result


def _add_contract_family(frame: pd.DataFrame) -> pd.DataFrame:
    """Recover a conservative contract-family key from observed symbols.

    OSI-era symbols expose their root directly.  In the legacy dot format the
    apparent roots partition strike ranges, so they must not be treated as
    distinct economic contracts; those rows are pooled within expiration.
    """

    result = frame.copy()
    symbols = result["symbol"].astype("string").str.strip()
    parsed_root = symbols.str.extract(r"^([A-Za-z]+)", expand=False).str.upper()
    osi = symbols.str.match(r"^[A-Za-z]+\s+\d{6}[CP]\d+$", na=False)
    result["parsed_root"] = parsed_root
    result["symbol_format"] = np.where(osi, "OSI", "legacy")
    result["contract_family"] = np.where(osi, parsed_root, "LEGACY")
    return result


def _parity_forward_table(
    frame: pd.DataFrame,
    forwards: pd.DataFrame,
    *,
    minimum_pairs: int = 5,
) -> pd.DataFrame:
    """Estimate slice forwards from paired mids and cross-check vendor rows.

    For each observed contract family, put--call parity gives
    ``F_i = K_i + (C_i-P_i)/D``.  The median is used as a robust slice
    estimate.  The closest same-expiration vendor forward is retained only as
    a diagnostic together with its observed settlement flag; neither value is
    substituted for the parity estimate.
    """

    forward = _normalize_dates(forwards, ("date", "expiration"))
    _require_columns(
        forward,
        {"date", "expiration", "am_settlement", "forward_price"},
        "forwards",
    )
    forward["am_settlement"] = pd.to_numeric(
        forward["am_settlement"], errors="coerce"
    )
    forward["forward_price"] = pd.to_numeric(
        forward["forward_price"], errors="coerce"
    )
    if forward.duplicated(
        ["date", "expiration", "am_settlement"], keep=False
    ).any():
        raise ValueError("forwards are not unique by date, expiration, and settlement")

    records: list[dict[str, Any]] = []
    keys = ["date", "exdate", "contract_family"]
    for key, group in frame.groupby(keys, sort=True, dropna=False):
        pairs = group.pivot_table(
            index="strike", columns="cp_flag", values="mid", aggfunc="median"
        )
        if "C" not in pairs or "P" not in pairs:
            continue
        pairs = pairs.dropna(subset=["C", "P"])
        if len(pairs) < minimum_pairs:
            continue
        discount = float(group["discount_factor"].median())
        if not np.isfinite(discount) or discount <= 0:
            continue
        parity_forwards = (
            pairs.index.to_numpy(dtype=float)
            + (pairs["C"] - pairs["P"]).to_numpy(dtype=float) / discount
        )
        parity_forwards = parity_forwards[
            np.isfinite(parity_forwards) & (parity_forwards > 0)
        ]
        if len(parity_forwards) < minimum_pairs:
            continue
        parity_forward = float(np.median(parity_forwards))
        parity_mad = float(np.median(np.abs(parity_forwards - parity_forward)))

        date, exdate, contract_family = key
        candidates = forward.loc[
            forward["date"].eq(date)
            & forward["expiration"].eq(exdate)
            & forward["forward_price"].gt(0)
        ].copy()
        if candidates.empty:
            continue
        candidates["relative_difference"] = (
            candidates["forward_price"].sub(parity_forward).abs() / parity_forward
        )
        best = candidates.loc[candidates["relative_difference"].idxmin()]
        records.append(
            {
                "date": date,
                "exdate": exdate,
                "contract_family": contract_family,
                "matched_forward_price": parity_forward,
                "parity_pair_count": int(len(parity_forwards)),
                "parity_forward_mad": parity_mad,
                "parity_forward_relative_mad": parity_mad / parity_forward,
                "vendor_forward_price": float(best["forward_price"]),
                "vendor_forward_relative_difference": float(
                    best["relative_difference"]
                ),
                "forward_candidate_count": int(len(candidates)),
                "nearest_vendor_am_settlement": int(best["am_settlement"])
                if np.isfinite(best["am_settlement"])
                else np.nan,
            }
        )
    return pd.DataFrame.from_records(records)


def prepare_calibration_quotes(
    options: pd.DataFrame,
    forwards: pd.DataFrame,
    zero_curve: pd.DataFrame,
    spot: pd.DataFrame,
) -> QuoteCleaningResult:
    """Apply mechanical validity and parity-consistent OTM selection.

    This function does not perform cross-strike arbitrage repair.  It produces
    the quote intervals and call-equivalent prices required by that next step.
    """

    _require_columns(options, REQUIRED_OPTION_COLUMNS, "options")
    _require_columns(spot, {"date", "spot"}, "spot")

    frame = _add_contract_family(_normalize_dates(options, ("date", "exdate")))
    frame = frame.reset_index(drop=True)
    attrition: dict[str, int] = {"input": int(len(frame))}

    numeric_columns = (
        "strike",
        "best_bid",
        "best_offer",
        "dte",
    )
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["cp_flag"] = frame["cp_flag"].astype("string").str.upper().str.strip()

    basic = (
        frame["date"].notna()
        & frame["exdate"].notna()
        & frame["cp_flag"].isin(["C", "P"])
        & frame["strike"].gt(0)
        & frame["dte"].between(30, 730)
        & frame["symbol"].notna()
    )
    frame = frame.loc[basic].copy()
    attrition["basic_contract_fields"] = int(len(frame))

    spot_frame = _normalize_dates(spot, ("date",))
    spot_frame["spot"] = pd.to_numeric(spot_frame["spot"], errors="coerce")
    if spot_frame.duplicated("date", keep=False).any():
        raise ValueError("spot is not unique by date")
    frame = frame.merge(
        spot_frame[["date", "spot"]],
        on="date",
        how="inner",
        validate="many_to_one",
    )
    attrition["spot_matched"] = int(len(frame))

    frame = frame.reset_index(drop=True)
    frame["zero_rate"] = interpolate_zero_rates(
        frame["date"], frame["dte"], zero_curve
    )
    frame["year_fraction"] = frame["dte"] / 365.0
    frame["discount_factor"] = np.exp(
        -frame["zero_rate"] * frame["year_fraction"]
    )
    rate_valid = (
        np.isfinite(frame["zero_rate"])
        & np.isfinite(frame["discount_factor"])
        & frame["discount_factor"].gt(0)
        & frame["spot"].gt(0)
    )
    frame = frame.loc[rate_valid].copy()
    attrition["rate_and_spot_valid"] = int(len(frame))

    frame["mid"] = (frame["best_bid"] + frame["best_offer"]) / 2.0
    frame["half_spread"] = (frame["best_offer"] - frame["best_bid"]) / 2.0
    frame["relative_spread"] = (
        frame["best_offer"] - frame["best_bid"]
    ) / frame["mid"].replace(0.0, np.nan)
    quote_valid = (
        frame["best_bid"].gt(0)
        & frame["best_offer"].gt(frame["best_bid"])
        & np.isfinite(frame["mid"])
    )
    frame = frame.loc[quote_valid].copy()
    attrition["positive_ordered_quotes"] = int(len(frame))

    parity_forwards = _parity_forward_table(frame, forwards)
    if parity_forwards.empty:
        raise ValueError("no contract family has at least five usable call--put pairs")
    frame = frame.merge(
        parity_forwards,
        on=["date", "exdate", "contract_family"],
        how="inner",
        validate="many_to_one",
    )
    attrition["parity_identified_forward"] = int(len(frame))

    discount = frame["discount_factor"]
    forward_value = frame["matched_forward_price"]
    strike = frame["strike"]
    is_call = frame["cp_flag"] == "C"
    frame["elementary_lower"] = np.where(
        is_call,
        discount * np.maximum(forward_value - strike, 0.0),
        discount * np.maximum(strike - forward_value, 0.0),
    )
    frame["elementary_upper"] = np.where(
        is_call,
        discount * forward_value,
        discount * strike,
    )
    interval_feasible = (
        frame["best_offer"] >= frame["elementary_lower"]
    ) & (frame["best_bid"] <= frame["elementary_upper"])
    frame = frame.loc[interval_feasible].copy()
    attrition["quote_interval_intersects_elementary_bounds"] = int(len(frame))

    otm = np.where(
        frame["cp_flag"] == "C",
        frame["strike"] >= frame["matched_forward_price"],
        frame["strike"] < frame["matched_forward_price"],
    )
    frame = frame.loc[otm].copy()
    attrition["otm_quotes"] = int(len(frame))

    parity = frame["discount_factor"] * (
        frame["matched_forward_price"] - frame["strike"]
    )
    put = frame["cp_flag"] == "P"
    frame["call_equivalent_bid"] = frame["best_bid"] + np.where(put, parity, 0.0)
    frame["call_equivalent_offer"] = frame["best_offer"] + np.where(put, parity, 0.0)
    frame["call_equivalent_mid"] = frame["mid"] + np.where(put, parity, 0.0)
    frame["log_forward_moneyness"] = np.log(
        frame["strike"] / frame["matched_forward_price"]
    )
    frame["strict_liquidity_sensitivity"] = (
        frame["mid"] >= 0.125
    ) & (frame["relative_spread"] <= 0.50)

    diagnostics = {
        "forward_relative_difference_max": float(
            frame["vendor_forward_relative_difference"].max()
        )
        if len(frame)
        else None,
        "forward_relative_difference_p99": float(
            frame["vendor_forward_relative_difference"].quantile(0.99)
        )
        if len(frame)
        else None,
        "strict_liquidity_rows": int(frame["strict_liquidity_sensitivity"].sum()),
        "dates": int(frame["date"].nunique()),
        "parity_forward_relative_mad_p99": float(
            frame["parity_forward_relative_mad"].quantile(0.99)
        )
        if len(frame)
        else None,
        "contract_slices": int(
            frame[["date", "exdate", "contract_family"]]
            .drop_duplicates()
            .shape[0]
        ),
    }
    return QuoteCleaningResult(
        quotes=frame.sort_values(
            ["date", "exdate", "contract_family", "strike", "cp_flag"]
        ).reset_index(drop=True),
        attrition=attrition,
        diagnostics=diagnostics,
    )
