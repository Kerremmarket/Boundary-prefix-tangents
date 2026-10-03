#!/usr/bin/env python3
"""Create a read-only, machine-verifiable audit of the SPX OptionMetrics files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


FILES = {
    "option_prices": "spx_option_prices_monthly_2005_2025.parquet",
    "vol_surface": "spx_vol_surface_monthly_2005_2025.parquet",
    "forward_prices": "spx_forward_prices_monthly_2005_2025.parquet",
    "zero_curve": "spx_zero_curve_monthly_2005_2025.parquet",
    "spot": "spx_spot_monthly_2005_2025.parquet",
}

OPTIONAL_OPTION_METADATA = {
    "am_settlement",
    "contract_size",
    "special_settlement",
    "forward_price",
    "expiry_indicator",
    "root",
}

CALIBRATION_OPTION_FIELDS = {
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parquet_metadata(path: Path) -> dict[str, Any]:
    parquet = pq.ParquetFile(path)
    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "rows": parquet.metadata.num_rows,
        "columns": parquet.schema_arrow.names,
        "row_groups": parquet.metadata.num_row_groups,
    }


def scalar(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.strftime("%Y-%m-%d")
    return value


def date_summary(frame: pd.DataFrame) -> dict[str, Any]:
    dates = pd.to_datetime(frame["date"], errors="coerce")
    counts = frame.assign(_date=dates).groupby("_date", dropna=False).size()
    valid = dates.dropna()
    return {
        "unique_dates": int(valid.nunique()),
        "date_min": scalar(valid.min()) if len(valid) else None,
        "date_max": scalar(valid.max()) if len(valid) else None,
        "rows_per_date_min": int(counts.min()) if len(counts) else 0,
        "rows_per_date_median": float(counts.median()) if len(counts) else 0.0,
        "rows_per_date_max": int(counts.max()) if len(counts) else 0,
    }


def option_audit(options: pd.DataFrame, spot: pd.DataFrame) -> dict[str, Any]:
    merged = options.merge(
        spot[["date", "spot"]], on="date", how="left", validate="many_to_one"
    )
    bid = pd.to_numeric(merged["best_bid"], errors="coerce")
    offer = pd.to_numeric(merged["best_offer"], errors="coerce")
    mid = (bid + offer) / 2.0
    spread = offer - bid
    relative_spread = spread / mid.replace(0.0, np.nan)
    strike = pd.to_numeric(merged["strike"], errors="coerce")
    spot_value = pd.to_numeric(merged["spot"], errors="coerce")
    dte = pd.to_numeric(merged["dte"], errors="coerce")
    valid_quote = (
        np.isfinite(bid)
        & np.isfinite(offer)
        & (bid > 0)
        & (offer > bid)
        & np.isfinite(strike)
        & (strike > 0)
        & dte.between(30, 730)
        & merged["cp_flag"].isin(["C", "P"])
    )
    strict_liquidity = valid_quote & (mid >= 0.125) & (relative_spread <= 0.50)
    k_over_s = (strike / spot_value).replace([np.inf, -np.inf], np.nan).dropna()
    quantile_levels = [0.0, 0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0]
    quantile_names = ["min", "p001", "p01", "p05", "p50", "p95", "p99", "max"]

    available = set(options.columns)
    missing_metadata = sorted(OPTIONAL_OPTION_METADATA - available)
    missing_calibration_fields = sorted(CALIBRATION_OPTION_FIELDS - available)
    symbols = options.get("symbol", pd.Series(dtype="string")).astype("string").str.strip()
    parsed_roots = symbols.str.extract(r"^([A-Za-z]+)", expand=False).str.upper()
    result: dict[str, Any] = {
        **date_summary(options),
        "bid_le_zero": int((bid <= 0).sum()),
        "offer_le_zero": int((offer <= 0).sum()),
        "offer_le_bid": int((offer <= bid).sum()),
        "negative_spread": int((spread < 0).sum()),
        "mid_below_0_125": int((mid < 0.125).sum()),
        "relative_spread_above_0_50": int((relative_spread > 0.50).sum()),
        "zero_volume": int((pd.to_numeric(merged["volume"], errors="coerce") == 0).sum()),
        "zero_open_interest": int(
            (pd.to_numeric(merged["open_interest"], errors="coerce") == 0).sum()
        ),
        "missing_vendor_iv": int(merged["impl_volatility"].isna().sum()),
        "pre_2008_03_05_rows": int(
            (pd.to_datetime(merged["date"]) < pd.Timestamp("2008-03-05")).sum()
        ),
        "dte_min": int(dte.min()),
        "dte_median": float(dte.median()),
        "dte_max": int(dte.max()),
        "k_over_spot_quantiles": {
            name: float(value)
            for name, value in zip(quantile_names, k_over_s.quantile(quantile_levels))
        },
        "pre_forward_valid_quote_rows": int(valid_quote.sum()),
        "strict_liquidity_sensitivity_rows": int(strict_liquidity.sum()),
        "optional_metadata_missing": missing_metadata,
        "calibration_fields_missing": missing_calibration_fields,
        "calibration_fields_ready": not missing_calibration_fields,
        "parsed_symbol_root_counts": {
            str(key): int(value)
            for key, value in parsed_roots.fillna("<NA>").value_counts().items()
        },
    }

    if not missing_metadata:
        result["metadata_null_counts"] = {
            name: int(options[name].isna().sum()) for name in sorted(OPTIONAL_OPTION_METADATA)
        }
        result["root_counts"] = {
            str(key): int(value)
            for key, value in options["root"].fillna("<NA>").value_counts().items()
        }
        result["special_settlement_counts"] = {
            str(key): int(value)
            for key, value in options["special_settlement"]
            .fillna("<NA>")
            .value_counts()
            .items()
        }
    return result


def surface_audit(surface: pd.DataFrame) -> dict[str, Any]:
    missing_by_maturity = (
        surface.groupby("days")["impl_volatility"].apply(lambda values: values.isna().sum())
    )
    return {
        **date_summary(surface),
        "maturities": sorted(int(value) for value in surface["days"].dropna().unique()),
        "missing_iv": int(surface["impl_volatility"].isna().sum()),
        "missing_iv_by_maturity": {
            str(int(key)): int(value)
            for key, value in missing_by_maturity.items()
            if value
        },
    }


def forward_audit(forwards: pd.DataFrame) -> dict[str, Any]:
    duplicated_pair = forwards.duplicated(["date", "expiration"], keep=False)
    duplicated_full = forwards.duplicated(
        ["date", "expiration", "am_settlement"], keep=False
    )
    return {
        **date_summary(forwards),
        "rows_in_duplicate_date_expiration_groups": int(duplicated_pair.sum()),
        "rows_in_duplicate_date_expiration_settlement_groups": int(duplicated_full.sum()),
        "missing_settlement_flag": int(forwards["am_settlement"].isna().sum()),
    }


def markdown(audit: dict[str, Any]) -> str:
    files = audit["files"]
    option = audit["option_prices"]
    surface = audit["vol_surface"]
    forward = audit["forward_prices"]
    ready = "PASS" if audit["stage0_ready"] else "BLOCKED"
    missing = option["optional_metadata_missing"]
    lines = [
        "# OptionMetrics Stage-0 audit",
        "",
        f"**Stage-0 readiness:** {ready}",
        "",
        "The audit is read-only. Hashes identify the exact licensed source files; no raw data are copied into Git.",
        "",
        "## File inventory",
        "",
        "| Dataset | Rows | Columns | Row groups | SHA-256 |",
        "|---|---:|---:|---:|---|",
    ]
    for name in FILES:
        item = files[name]
        lines.append(
            f"| {name} | {item['rows']:,} | {len(item['columns'])} | "
            f"{item['row_groups']} | `{item['sha256']}` |"
        )
    lines.extend(
        [
            "",
            "## Coverage",
            "",
            f"- Option dates: {option['unique_dates']} ({option['date_min']} to {option['date_max']}).",
            f"- Surface dates: {surface['unique_dates']} ({surface['date_min']} to {surface['date_max']}).",
            f"- Forward dates: {forward['unique_dates']} ({forward['date_min']} to {forward['date_max']}).",
            "",
            "## Raw-option diagnostics",
            "",
            "| Diagnostic | Rows |",
            "|---|---:|",
            f"| Bid <= 0 | {option['bid_le_zero']:,} |",
            f"| Offer <= bid | {option['offer_le_bid']:,} |",
            f"| Midquote < 0.125 | {option['mid_below_0_125']:,} |",
            f"| Relative spread > 50% | {option['relative_spread_above_0_50']:,} |",
            f"| Zero volume | {option['zero_volume']:,} |",
            f"| Zero open interest | {option['zero_open_interest']:,} |",
            f"| Missing vendor IV | {option['missing_vendor_iv']:,} |",
            f"| Pre-forward valid quote rows | {option['pre_forward_valid_quote_rows']:,} |",
            f"| Strict liquidity sensitivity rows | {option['strict_liquidity_sensitivity_rows']:,} |",
            "",
            "## Forward-identification check",
            "",
            f"- Rows in duplicated `(date, expiration)` forward groups: {forward['rows_in_duplicate_date_expiration_groups']:,}.",
            f"- Rows duplicated after adding settlement type: {forward['rows_in_duplicate_date_expiration_settlement_groups']:,}.",
        ]
    )
    if missing:
        lines.extend(
            [
                f"- Optional option-level metadata absent: {', '.join(f'`{name}`' for name in missing)}.",
                "- Result: this does not block calibration. Observed symbols define contract families; paired call/put quotes and the zero curve identify slice forwards, while the forward file supplies an external cross-check.",
            ]
        )
    else:
        lines.extend(
            [
                "- Optional option-level contract metadata are present.",
                "- Result: parity identification and quote-to-forward cross-checks remain part of the cleaning stage.",
            ]
        )
    lines.extend(
        [
            "",
            "## Standardized surface",
            "",
            f"- Maturities: {surface['maturities']}.",
            f"- Missing implied-volatility cells: {surface['missing_iv']:,}.",
            "- The standardized surface is an initializer/diagnostic, not the primary calibration target.",
            "",
            "## Interpretation",
            "",
            "The pre-forward quote count is not a final cleaned sample. It applies only mechanical quote, strike, call/put, and DTE checks. Parity-implied forwards, vendor-forward diagnostics, discounted bounds, OTM selection, and static-arbitrage repair remain subsequent explicit steps.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    paths = {name: data_dir / filename for name, filename in FILES.items()}
    missing_files = [str(path) for path in paths.values() if not path.exists()]
    if missing_files:
        raise FileNotFoundError(f"Missing required Parquets: {missing_files}")

    frames = {name: pd.read_parquet(path) for name, path in paths.items()}
    file_inventory = {name: parquet_metadata(path) for name, path in paths.items()}
    audit = {
        "schema_version": 1,
        "data_dir": str(data_dir),
        "files": file_inventory,
        "option_prices": option_audit(frames["option_prices"], frames["spot"]),
        "vol_surface": surface_audit(frames["vol_surface"]),
        "forward_prices": forward_audit(frames["forward_prices"]),
        "zero_curve": date_summary(frames["zero_curve"]),
        "spot": date_summary(frames["spot"]),
    }
    audit["stage0_ready"] = bool(
        audit["option_prices"]["calibration_fields_ready"]
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "optionmetrics_stage0_audit.json"
    markdown_path = output_dir / "optionmetrics_stage0_audit.md"
    json_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(markdown(audit), encoding="utf-8")
    print(markdown_path)
    print(json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
