#!/usr/bin/env python3
"""Resumable WRDS OptionMetrics IvyDB US extraction for monthly SPX snapshots.

This pipeline intentionally discovers the live ``optionm_all`` schema before it
builds any data query.  It never stores credentials and it automatically declines
the optional prompt from the WRDS package to create a .pgpass file.
"""

from __future__ import annotations

import argparse
import builtins
import getpass
import json
import os
import re
import sys
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
try:  # Optional: the current pilot uses the validated local Parquets.
    import wrds
except ModuleNotFoundError:  # pragma: no cover - exercised only by the extractor
    wrds = None


LIBRARY = "optionm_all"
SECID = 108105
START_YEAR = 2005
END_YEAR = 2025
END_DATE = pd.Timestamp("2025-08-31")
PIPELINE_VERSION = "2026-08-26.2"
DATASET_SCHEMA_VERSIONS = {
    "option_prices": 2,
    "vol_surface": 1,
    "forward_prices": 2,
    "zero_curve": 1,
    "spot": 1,
}
# Use a new output tree by default.  The licensed source Parquets are not
# overwritten unless the operator explicitly points this variable at them.
OUTPUT_DIR = Path(
    os.environ.get(
        "PAPER1_OPTIONMETRICS_OUTPUT_DIR",
        str(Path.cwd() / "output" / "optionmetrics_stage0_v2"),
    )
).expanduser().resolve()
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
PROGRESS_LOG = OUTPUT_DIR / "extraction_progress.log"
SCHEMA_INVENTORY = OUTPUT_DIR / "wrds_schema_inventory.txt"

FINAL_FILES = {
    "option_prices": OUTPUT_DIR / "spx_option_prices_monthly_2005_2025.parquet",
    "vol_surface": OUTPUT_DIR / "spx_vol_surface_monthly_2005_2025.parquet",
    "forward_prices": OUTPUT_DIR / "spx_forward_prices_monthly_2005_2025.parquet",
    "zero_curve": OUTPUT_DIR / "spx_zero_curve_monthly_2005_2025.parquet",
    "spot": OUTPUT_DIR / "spx_spot_monthly_2005_2025.parquet",
}

YEAR_RE = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(message: str) -> None:
    line = f"[{utc_now()}] {message}"
    print(line, flush=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with PROGRESS_LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(tmp, index=False, engine="pyarrow", compression="zstd")
    os.replace(tmp, path)


def guarded_wrds_connection() -> wrds.Connection:
    """Call wrds.Connection() without echoing credentials or saving a .pgpass."""
    if wrds is None:
        raise RuntimeError(
            "the optional 'wrds' package is required only for a new extraction"
        )
    original_input = builtins.input

    def guarded_input(prompt: str = "") -> str:
        lower = prompt.lower()
        if "create .pgpass" in lower:
            print(f"{prompt}n", flush=True)
            return "n"
        if "wrds username" in lower:
            # Treat the username as a credential too: getpass prevents terminal echo.
            return getpass.getpass(prompt)
        return original_input(prompt)

    builtins.input = guarded_input
    try:
        # Required connection method; no credential arguments are supplied or stored.
        db = wrds.Connection()
    finally:
        builtins.input = original_input
    return db


def qident(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def qtable(table: str) -> str:
    return f"{qident(LIBRARY)}.{qident(table)}"


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def choose_column(columns: Sequence[str], candidates: Sequence[str]) -> str | None:
    by_normalized = {normalize_name(column): column for column in columns}
    for candidate in candidates:
        found = by_normalized.get(normalize_name(candidate))
        if found is not None:
            return found
    return None


ROLE_SPECS: dict[str, dict[str, tuple[Sequence[str], bool]]] = {
    "option_prices": {
        "secid": (["secid"], True),
        "date": (["date"], True),
        "optionid": (["optionid", "option_id"], True),
        "symbol": (["symbol", "option_symbol"], False),
        "exdate": (["exdate", "expiration", "expiration_date"], True),
        "cp_flag": (["cp_flag", "cpflag", "put_call", "call_put"], True),
        "strike_price": (["strike_price", "strikeprice", "strike"], True),
        "best_bid": (["best_bid", "bestbid", "bid"], True),
        "best_offer": (["best_offer", "bestoffer", "best_ask", "ask", "offer"], True),
        "volume": (["volume"], True),
        "open_interest": (["open_interest", "openinterest"], True),
        "impl_volatility": (["impl_volatility", "implied_volatility", "implvolatility", "iv"], True),
        "delta": (["delta"], True),
        "gamma": (["gamma"], True),
        "vega": (["vega"], True),
        "theta": (["theta"], True),
        "am_settlement": (["am_settlement", "amsettlement", "am_set_flag"], True),
        "contract_size": (["contract_size", "contractsize"], True),
        "special_settlement": (["ss_flag", "special_settlement", "specialsettlement"], True),
        "forward_price": (["forward_price", "forwardprice", "forward"], True),
        "expiry_indicator": (["expiry_indicator", "expiryindicator"], True),
        "root": (["root"], True),
        "suffix": (["suffix"], False),
    },
    "vol_surface": {
        "secid": (["secid"], True),
        "date": (["date"], True),
        "days": (["days", "maturity_days", "maturitydays"], True),
        "cp_flag": (["cp_flag", "cpflag", "put_call", "call_put"], True),
        "delta": (["delta"], True),
        "impl_volatility": (["impl_volatility", "implied_volatility", "implvolatility", "iv"], True),
    },
    "forward_prices": {
        "secid": (["secid"], True),
        "date": (["date"], True),
        "maturity": (["expiration", "exdate", "maturity", "maturity_date", "maturity_days", "days"], True),
        "am_settlement": (["amsettlement", "am_settlement", "am_set_flag"], False),
        "forward_price": (["forwardprice", "forward_price", "forward"], True),
    },
    "zero_curve": {
        "date": (["date"], True),
        "days": (["days", "maturity_days", "maturitydays", "maturity"], True),
        "zero_rate": (["rate", "zero_rate", "zerorate"], True),
    },
    "spot": {
        "secid": (["secid"], True),
        "date": (["date"], True),
        "spot": (["close", "close_price", "closeprice", "index_level", "indexlevel", "price"], True),
    },
}


def resolve_columns(role: str, columns: Sequence[str], strict: bool = True) -> dict[str, str | None] | None:
    resolved: dict[str, str | None] = {}
    for canonical, (candidates, required) in ROLE_SPECS[role].items():
        actual = choose_column(columns, candidates)
        if required and actual is None:
            if strict:
                raise RuntimeError(f"Missing required {role} column {canonical!r}; live columns={list(columns)}")
            return None
        resolved[canonical] = actual
    return resolved


@dataclass
class TableFamily:
    pattern: str
    tables_by_year: dict[int, str]
    shared_table: str | None
    representative: str
    columns: list[str]
    description: pd.DataFrame

    def table_for_year(self, year: int) -> str:
        if self.shared_table is not None:
            return self.shared_table
        if year not in self.tables_by_year:
            raise RuntimeError(f"No live table in family {self.pattern!r} for {year}")
        return self.tables_by_year[year]

    @property
    def covers_all_years(self) -> bool:
        return self.shared_table is not None or all(year in self.tables_by_year for year in range(START_YEAR, END_YEAR + 1))


def group_table_families(tables: Sequence[str]) -> list[tuple[str, dict[int, str], str | None]]:
    groups: dict[str, dict[int, str]] = defaultdict(dict)
    shared: dict[str, str] = {}
    for table in sorted(set(tables)):
        matches = list(YEAR_RE.finditer(table))
        target_matches = [match for match in matches if START_YEAR <= int(match.group()) <= END_YEAR]
        if target_matches:
            # Annual OptionMetrics names have one year token. Replacing every year token
            # makes the family construction independent of a hard-coded table prefix.
            pattern = YEAR_RE.sub("{year}", table)
            year = int(target_matches[-1].group())
            groups[pattern][year] = table
        elif not matches:
            shared[table] = table

    result: list[tuple[str, dict[int, str], str | None]] = []
    for pattern, mapping in sorted(groups.items()):
        result.append((pattern, mapping, None))
    for table in sorted(shared):
        result.append((table, {}, table))
    return result


def description_columns(description: pd.DataFrame) -> list[str]:
    for candidate in ("name", "column_name", "column"):
        if candidate in description.columns:
            return description[candidate].astype(str).tolist()
    raise RuntimeError(f"Unexpected describe_table output columns: {list(description.columns)}")


def family_score(role: str, family: TableFamily) -> int | None:
    mapping = resolve_columns(role, family.columns, strict=False)
    if mapping is None:
        return None
    normalized = {normalize_name(column) for column in family.columns}
    score = 100 + len(family.columns)
    if family.covers_all_years:
        score += 50
    # Column-based exclusions keep overlapping price schemas from being mistaken
    # for another requested dataset. No table-name assumption is used here.
    if role == "vol_surface" and ("optionid" in normalized or "strikeprice" in normalized):
        return None
    if role == "forward_prices" and (
        {
            "optionid", "strikeprice", "premium", "cpflag", "implvolatility",
            "delta", "gamma", "theta", "vega",
        }
        & normalized
    ):
        # Standardized-options tables also carry a calculated forward-price
        # field. They are not the requested Forward Prices source, whose live
        # schema is a pure (secid, date, maturity, forward price) relation.
        return None
    if role == "zero_curve" and "secid" in normalized:
        return None
    if role == "spot" and ({"optionid", "cpflag", "forwardprice"} & normalized):
        return None
    if role == "option_prices" and family.shared_table is not None:
        # The requested Option Price source must be an annual family.
        return None
    return score


def discover_schema(db: wrds.Connection) -> tuple[dict[str, TableFamily], dict[str, dict[str, str | None]]]:
    log(f"Discovering live WRDS schema with db.list_tables(library={LIBRARY!r})")
    tables = sorted(db.list_tables(library=LIBRARY))
    if not tables:
        raise RuntimeError(f"No tables visible in WRDS library {LIBRARY!r}")

    inventory: list[str] = [
        "WRDS OptionMetrics live schema inventory",
        f"Generated: {utc_now()}",
        f"Library: {LIBRARY}",
        f"Visible table count: {len(tables)}",
        "",
        "VISIBLE TABLES",
        *[f"- {table}" for table in tables],
        "",
        "DESCRIBED TABLE-FAMILY REPRESENTATIVES",
    ]

    families: list[TableFamily] = []
    failures: list[str] = []
    for pattern, mapping, shared_table in group_table_families(tables):
        if mapping:
            representative = mapping.get(START_YEAR) or mapping[min(mapping)]
        else:
            assert shared_table is not None
            representative = shared_table
        try:
            description = db.describe_table(library=LIBRARY, table=representative)
            columns = description_columns(description)
        except Exception as exc:
            failures.append(f"{representative}: {type(exc).__name__}: {exc}")
            continue
        family = TableFamily(pattern, mapping, shared_table, representative, columns, description)
        families.append(family)
        inventory.extend(
            [
                "",
                f"Family: {pattern}",
                f"Representative described via db.describe_table: {representative}",
                f"Coverage: {'shared table' if shared_table else sorted(mapping)}",
                description.to_string(index=False),
            ]
        )

    if failures:
        inventory.extend(["", "DESCRIPTION FAILURES", *[f"- {failure}" for failure in failures]])

    selected: dict[str, TableFamily] = {}
    selected_columns: dict[str, dict[str, str | None]] = {}
    inventory.extend(["", "RESOLVED MAPPING"])
    for role in ROLE_SPECS:
        ranked = sorted(
            ((family_score(role, family), family) for family in families),
            key=lambda item: (-1 if item[0] is None else item[0]),
            reverse=True,
        )
        ranked = [(score, family) for score, family in ranked if score is not None]
        if not ranked:
            raise RuntimeError(f"Could not identify the {role} table from live column descriptions")
        top_score, chosen = ranked[0]
        if not chosen.covers_all_years:
            raise RuntimeError(
                f"Best live {role} family {chosen.pattern!r} does not cover every requested year: "
                f"{sorted(chosen.tables_by_year)}"
            )
        mapping = resolve_columns(role, chosen.columns, strict=True)
        assert mapping is not None
        selected[role] = chosen
        selected_columns[role] = mapping
        inventory.extend(
            [
                "",
                f"{role}:",
                f"  table family: {chosen.pattern}",
                f"  representative: {chosen.representative}",
                f"  annual tables: {json.dumps(chosen.tables_by_year, sort_keys=True)}" if chosen.shared_table is None else f"  shared table: {chosen.shared_table}",
                f"  resolved columns: {json.dumps(mapping, sort_keys=True)}",
                "  next candidates: " + ", ".join(f"{family.pattern} (score={score})" for score, family in ranked[1:4]),
            ]
        )

    SCHEMA_INVENTORY.write_text("\n".join(inventory) + "\n", encoding="utf-8")
    printable = {
        role: {
            "table_family": family.pattern,
            "table_or_years": family.shared_table or family.tables_by_year,
            "columns": selected_columns[role],
        }
        for role, family in selected.items()
    }
    print("\nRESOLVED LIVE OPTIONMETRICS MAPPING (before extraction)", flush=True)
    print(json.dumps(printable, indent=2, sort_keys=True, default=str), flush=True)
    print(f"Full inventory: {SCHEMA_INVENTORY}", flush=True)
    return selected, selected_columns


def get_live_columns(db: wrds.Connection, table: str) -> list[str]:
    description = db.describe_table(library=LIBRARY, table=table)
    return description_columns(description)


def sql_date_list(dates: Sequence[pd.Timestamp]) -> str:
    if not dates:
        raise ValueError("Cannot construct an empty monthly-date predicate")
    return ", ".join(f"DATE '{pd.Timestamp(date).strftime('%Y-%m-%d')}'" for date in dates)


def checkpoint_paths(dataset: str, year: int) -> tuple[Path, Path]:
    base = CHECKPOINT_DIR / dataset / str(year)
    return base.with_suffix(".parquet"), base.with_suffix(".json")


def checkpoint_valid(dataset: str, year: int, source_table: str, dates: Sequence[pd.Timestamp]) -> bool:
    parquet_path, manifest_path = checkpoint_paths(dataset, year)
    if not parquet_path.exists() or not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_dates = [pd.Timestamp(date).strftime("%Y-%m-%d") for date in dates]
        return (
            manifest.get("pipeline_version") == PIPELINE_VERSION
            and manifest.get("dataset_schema_version", 1) == DATASET_SCHEMA_VERSIONS[dataset]
            and manifest.get("dataset") == dataset
            and manifest.get("year") == year
            and manifest.get("source_table") == source_table
            and manifest.get("dates") == expected_dates
            and manifest.get("secid") == (None if dataset == "zero_curve" else SECID)
        )
    except Exception:
        return False


def save_checkpoint(
    dataset: str,
    year: int,
    source_table: str,
    dates: Sequence[pd.Timestamp],
    frame: pd.DataFrame,
) -> None:
    parquet_path, manifest_path = checkpoint_paths(dataset, year)
    atomic_parquet(frame, parquet_path)
    atomic_json(
        manifest_path,
        {
            "pipeline_version": PIPELINE_VERSION,
            "dataset_schema_version": DATASET_SCHEMA_VERSIONS[dataset],
            "dataset": dataset,
            "year": year,
            "source_table": source_table,
            "dates": [pd.Timestamp(date).strftime("%Y-%m-%d") for date in dates],
            "secid": None if dataset == "zero_curve" else SECID,
            "rows": int(len(frame)),
            "columns": list(frame.columns),
            "created_utc": utc_now(),
        },
    )


def normalize_frame(dataset: str, frame: pd.DataFrame) -> pd.DataFrame:
    schemas = {
        "option_prices": [
            "secid", "date", "optionid", "symbol", "exdate", "cp_flag",
            "strike_price", "strike", "best_bid", "best_offer", "volume",
            "open_interest", "impl_volatility", "delta", "gamma", "vega", "theta",
            "am_settlement", "contract_size", "special_settlement", "forward_price",
            "expiry_indicator", "root", "suffix", "dte",
        ],
        "vol_surface": ["secid", "date", "days", "cp_flag", "delta", "impl_volatility"],
        "zero_curve": ["date", "days", "zero_rate"],
        "spot": ["secid", "date", "spot"],
    }
    if dataset in schemas:
        expected = schemas[dataset]
    elif dataset == "forward_prices":
        maturity_col = "expiration" if "expiration" in frame.columns else "maturity_days"
        expected = ["secid", "date", maturity_col, "am_settlement", "forward_price"]
    else:
        raise ValueError(dataset)

    for column in expected:
        if column not in frame:
            frame[column] = pd.NA
    frame = frame[expected].copy()

    date_columns = [column for column in ("date", "exdate", "expiration") if column in frame]
    for column in date_columns:
        frame[column] = pd.to_datetime(frame[column], errors="coerce").dt.normalize()

    integer_columns = [column for column in ("secid", "optionid", "dte", "days", "maturity_days", "am_settlement") if column in frame]
    for column in integer_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")

    string_columns = [
        column
        for column in (
            "symbol", "cp_flag", "special_settlement", "expiry_indicator", "root", "suffix"
        )
        if column in frame
    ]
    for column in string_columns:
        frame[column] = frame[column].astype("string")

    excluded = set(date_columns + integer_columns + string_columns)
    for column in frame.columns:
        if column not in excluded:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
    return frame


def year_dates(month_dates: pd.DataFrame, year: int) -> list[pd.Timestamp]:
    subset = month_dates.loc[month_dates["date"].dt.year == year, "date"]
    return [pd.Timestamp(value) for value in subset.sort_values().tolist()]


def construct_month_end_dates(db: wrds.Connection, family: TableFamily) -> pd.DataFrame:
    all_parts: list[pd.DataFrame] = []
    for year in range(START_YEAR, END_YEAR + 1):
        table = family.table_for_year(year)
        checkpoint = CHECKPOINT_DIR / "month_end_dates" / f"{year}.csv"
        manifest = checkpoint.with_suffix(".json")
        expected_meta = {
            "pipeline_version": PIPELINE_VERSION,
            "year": year,
            "source_table": table,
            "secid": SECID,
        }
        reusable = False
        if checkpoint.exists() and manifest.exists():
            try:
                reusable = all(json.loads(manifest.read_text(encoding="utf-8")).get(k) == v for k, v in expected_meta.items())
            except Exception:
                reusable = False
        if reusable:
            part = pd.read_csv(checkpoint, parse_dates=["date"])
            log(f"Month-end dates {year}: reused checkpoint ({len(part)} dates)")
        else:
            columns = get_live_columns(db, table)
            mapping = resolve_columns("option_prices", columns, strict=True)
            assert mapping is not None
            date_col = qident(str(mapping["date"]))
            secid_col = qident(str(mapping["secid"]))
            sql = f"""
                SELECT MAX({date_col}) AS date
                FROM {qtable(table)}
                WHERE {secid_col} = {SECID}
                  AND {date_col} >= DATE '{year}-01-01'
                  AND {date_col} < DATE '{year + 1}-01-01'
                GROUP BY DATE_TRUNC('month', {date_col})
                ORDER BY DATE_TRUNC('month', {date_col})
            """
            log(f"Month-end dates {year}: querying MAX(date) by calendar month from {table}")
            part = db.raw_sql(sql, date_cols=["date"])
            part["date"] = pd.to_datetime(part["date"], errors="coerce").dt.normalize()
            part = part.dropna(subset=["date"]).drop_duplicates().sort_values("date").reset_index(drop=True)
            atomic_csv(part, checkpoint)
            atomic_json(manifest, {**expected_meta, "rows": len(part), "created_utc": utc_now()})
        all_parts.append(part[["date"]])

    combined = pd.concat(all_parts, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"]).dt.normalize()
    combined = combined.loc[combined["date"] <= END_DATE]
    combined = combined.drop_duplicates().sort_values("date").reset_index(drop=True)
    combined["calendar_month"] = combined["date"].dt.to_period("M").astype(str)
    atomic_csv(combined, OUTPUT_DIR / "month_end_dates.csv")
    return combined


def select_expressions(mapping: dict[str, str | None], ordered_aliases: Sequence[str]) -> list[str]:
    expressions: list[str] = []
    for alias in ordered_aliases:
        actual = mapping.get(alias)
        if actual is None:
            expressions.append(f"CAST(NULL AS TEXT) AS {qident(alias)}")
        else:
            expressions.append(f"{qident(str(actual))} AS {qident(alias)}")
    return expressions


def extract_option_year(db: wrds.Connection, table: str, dates: Sequence[pd.Timestamp]) -> pd.DataFrame:
    columns = get_live_columns(db, table)
    mapping = resolve_columns("option_prices", columns, strict=True)
    assert mapping is not None
    aliases = [
        "secid", "date", "optionid", "symbol", "exdate", "cp_flag", "strike_price",
        "best_bid", "best_offer", "volume", "open_interest", "impl_volatility",
        "delta", "gamma", "vega", "theta", "am_settlement", "contract_size",
        "special_settlement", "forward_price", "expiry_indicator", "root", "suffix",
    ]
    expressions = select_expressions(mapping, aliases)
    date_col = qident(str(mapping["date"]))
    exdate_col = qident(str(mapping["exdate"]))
    secid_col = qident(str(mapping["secid"]))
    expressions.append(f"({exdate_col} - {date_col}) AS dte")
    sql = f"""
        SELECT {', '.join(expressions)}
        FROM {qtable(table)}
        WHERE {secid_col} = {SECID}
          AND {date_col} IN ({sql_date_list(dates)})
          AND ({exdate_col} - {date_col}) BETWEEN 30 AND 730
        ORDER BY {date_col}, {qident(str(mapping['optionid']))}
    """
    frame = db.raw_sql(sql, date_cols=["date", "exdate"])
    raw_strike = pd.to_numeric(frame.get("strike_price"), errors="coerce")
    nonnull = raw_strike.dropna()
    # IvyDB US convention is strike*1000. This data-driven guard keeps the script
    # safe if WRDS ever exposes an already-normalized variant.
    scale = 1000.0 if len(nonnull) and float(nonnull.median()) > 10000.0 else 1.0
    frame["strike"] = raw_strike / scale
    frame.attrs["strike_scale"] = scale
    return normalize_frame("option_prices", frame)


def extract_surface_year(db: wrds.Connection, table: str, dates: Sequence[pd.Timestamp]) -> pd.DataFrame:
    columns = get_live_columns(db, table)
    mapping = resolve_columns("vol_surface", columns, strict=True)
    assert mapping is not None
    expressions = select_expressions(mapping, ["secid", "date", "days", "cp_flag", "delta", "impl_volatility"])
    date_col, secid_col = qident(str(mapping["date"])), qident(str(mapping["secid"]))
    sql = f"""
        SELECT {', '.join(expressions)}
        FROM {qtable(table)}
        WHERE {secid_col} = {SECID}
          AND {date_col} IN ({sql_date_list(dates)})
        ORDER BY {date_col}, {qident(str(mapping['days']))}, {qident(str(mapping['cp_flag']))}, {qident(str(mapping['delta']))}
    """
    return normalize_frame("vol_surface", db.raw_sql(sql, date_cols=["date"]))


def extract_forward_year(db: wrds.Connection, table: str, dates: Sequence[pd.Timestamp]) -> pd.DataFrame:
    columns = get_live_columns(db, table)
    mapping = resolve_columns("forward_prices", columns, strict=True)
    assert mapping is not None
    maturity_actual = str(mapping["maturity"])
    maturity_normalized = normalize_name(maturity_actual)
    maturity_alias = "maturity_days" if "day" in maturity_normalized else "expiration"
    expressions = select_expressions(mapping, ["secid", "date"])
    expressions.extend(
        [
            f"{qident(maturity_actual)} AS {qident(maturity_alias)}",
        ]
    )
    if mapping["am_settlement"] is None:
        expressions.append(f"CAST(NULL AS DOUBLE PRECISION) AS {qident('am_settlement')}")
    else:
        expressions.append(f"{qident(str(mapping['am_settlement']))} AS {qident('am_settlement')}")
    expressions.append(f"{qident(str(mapping['forward_price']))} AS {qident('forward_price')}")
    date_col, secid_col = qident(str(mapping["date"])), qident(str(mapping["secid"]))
    sql = f"""
        SELECT {', '.join(expressions)}
        FROM {qtable(table)}
        WHERE {secid_col} = {SECID}
          AND {date_col} IN ({sql_date_list(dates)})
        ORDER BY {date_col}, {qident(maturity_actual)}
    """
    date_cols = ["date", "expiration"] if maturity_alias == "expiration" else ["date"]
    return normalize_frame("forward_prices", db.raw_sql(sql, date_cols=date_cols))


def extract_zero_year(db: wrds.Connection, table: str, dates: Sequence[pd.Timestamp]) -> pd.DataFrame:
    columns = get_live_columns(db, table)
    mapping = resolve_columns("zero_curve", columns, strict=True)
    assert mapping is not None
    expressions = select_expressions(mapping, ["date", "days", "zero_rate"])
    date_col = qident(str(mapping["date"]))
    sql = f"""
        SELECT {', '.join(expressions)}
        FROM {qtable(table)}
        WHERE {date_col} IN ({sql_date_list(dates)})
        ORDER BY {date_col}, {qident(str(mapping['days']))}
    """
    return normalize_frame("zero_curve", db.raw_sql(sql, date_cols=["date"]))


def extract_spot_year(db: wrds.Connection, table: str, dates: Sequence[pd.Timestamp]) -> pd.DataFrame:
    columns = get_live_columns(db, table)
    mapping = resolve_columns("spot", columns, strict=True)
    assert mapping is not None
    expressions = select_expressions(mapping, ["secid", "date", "spot"])
    date_col, secid_col = qident(str(mapping["date"])), qident(str(mapping["secid"]))
    sql = f"""
        SELECT {', '.join(expressions)}
        FROM {qtable(table)}
        WHERE {secid_col} = {SECID}
          AND {date_col} IN ({sql_date_list(dates)})
        ORDER BY {date_col}
    """
    return normalize_frame("spot", db.raw_sql(sql, date_cols=["date"]))


EXTRACTORS = {
    "option_prices": extract_option_year,
    "vol_surface": extract_surface_year,
    "forward_prices": extract_forward_year,
    "zero_curve": extract_zero_year,
    "spot": extract_spot_year,
}


def extract_all(db: wrds.Connection, families: dict[str, TableFamily], month_dates: pd.DataFrame, force: bool) -> None:
    # Annual Option Prices are deliberately processed sequentially. The same annual
    # batching pattern keeps all other monthly snapshot queries bounded and resumable.
    for dataset in ("option_prices", "vol_surface", "forward_prices", "zero_curve", "spot"):
        log(f"Starting {dataset} extraction")
        for year in range(START_YEAR, END_YEAR + 1):
            dates = year_dates(month_dates, year)
            table = families[dataset].table_for_year(year)
            parquet_path, _ = checkpoint_paths(dataset, year)
            if not force and checkpoint_valid(dataset, year, table, dates):
                rows = pq.ParquetFile(parquet_path).metadata.num_rows
                log(f"{dataset} {year}: reused {rows:,}-row checkpoint from {table}")
                continue
            if not dates:
                log(f"{dataset} {year}: no master monthly dates; writing an explicit empty checkpoint")
                empty = normalize_frame(dataset, pd.DataFrame())
                save_checkpoint(dataset, year, table, dates, empty)
                continue
            log(f"{dataset} {year}: querying {table} for {len(dates)} monthly dates")
            frame = EXTRACTORS[dataset](db, table, dates)
            save_checkpoint(dataset, year, table, dates, frame)
            log(f"{dataset} {year}: saved {len(frame):,} rows")


def assemble_final(dataset: str) -> None:
    destination = FINAL_FILES[dataset]
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    writer: pq.ParquetWriter | None = None
    schema: pa.Schema | None = None
    total_rows = 0
    try:
        for year in range(START_YEAR, END_YEAR + 1):
            path, _ = checkpoint_paths(dataset, year)
            if not path.exists():
                raise RuntimeError(f"Missing checkpoint: {path}")
            table = pq.read_table(path)
            if schema is None:
                schema = table.schema
                writer = pq.ParquetWriter(tmp, schema, compression="zstd")
            elif not table.schema.equals(schema, check_metadata=False):
                table = table.cast(schema)
            assert writer is not None
            writer.write_table(table)
            total_rows += table.num_rows
    finally:
        if writer is not None:
            writer.close()
    if schema is None:
        raise RuntimeError(f"No checkpoints available to assemble {dataset}")
    os.replace(tmp, destination)
    log(f"Assembled {destination.name}: {total_rows:,} rows")


def expected_months() -> list[str]:
    return pd.period_range(f"{START_YEAR}-01", "2025-08", freq="M").astype(str).tolist()


def fmt_pct(value: float) -> str:
    return f"{value:.4f}%"


def checkpoint_frames(dataset: str) -> Iterable[pd.DataFrame]:
    for year in range(START_YEAR, END_YEAR + 1):
        path, _ = checkpoint_paths(dataset, year)
        yield pd.read_parquet(path)


def summarize_dataset(dataset: str, keys: Sequence[str]) -> dict[str, Any]:
    rows = 0
    date_values: set[pd.Timestamp] = set()
    nulls: Counter[str] = Counter()
    columns: list[str] = []
    duplicates = 0
    cp_counts: Counter[str] = Counter()
    minima: dict[str, Any] = {}
    maxima: dict[str, Any] = {}
    coverage_counts: Counter[str] = Counter()
    maturity_values_by_date: dict[str, set[str]] = defaultdict(set)
    for frame in checkpoint_frames(dataset):
        rows += len(frame)
        if not columns:
            columns = list(frame.columns)
        for column in frame.columns:
            nulls[column] += int(frame[column].isna().sum())
        if "date" in frame:
            normalized_dates = pd.to_datetime(frame["date"], errors="coerce").dropna().dt.normalize()
            date_values.update(normalized_dates.tolist())
            coverage_counts.update(normalized_dates.dt.strftime("%Y-%m-%d").tolist())
            maturity_column = "expiration" if "expiration" in frame else ("maturity_days" if "maturity_days" in frame else None)
            if maturity_column is not None:
                maturity_frame = frame[["date", maturity_column]].dropna().copy()
                maturity_frame["date"] = pd.to_datetime(maturity_frame["date"]).dt.strftime("%Y-%m-%d")
                for date_value, group in maturity_frame.groupby("date"):
                    maturity_values_by_date[str(date_value)].update(group[maturity_column].astype(str).tolist())
        usable_keys = [key for key in keys if key in frame.columns]
        if usable_keys and len(frame):
            duplicates += int(frame.duplicated(usable_keys, keep=False).sum())
        if "cp_flag" in frame:
            cp_counts.update(frame["cp_flag"].dropna().astype(str).tolist())
        for column in ("dte", "days", "maturity_days", "expiration"):
            if column in frame and frame[column].notna().any():
                current_min, current_max = frame[column].min(), frame[column].max()
                minima[column] = current_min if column not in minima else min(minima[column], current_min)
                maxima[column] = current_max if column not in maxima else max(maxima[column], current_max)
    sorted_dates = sorted(date_values)
    return {
        "dataset": dataset,
        "rows": rows,
        "unique_dates": len(sorted_dates),
        "min_date": sorted_dates[0].strftime("%Y-%m-%d") if sorted_dates else None,
        "max_date": sorted_dates[-1].strftime("%Y-%m-%d") if sorted_dates else None,
        "dates": {value.strftime("%Y-%m-%d") for value in sorted_dates},
        "duplicate_rows_by_key": duplicates,
        "null_percentages": {column: (100.0 * nulls[column] / rows if rows else float("nan")) for column in columns},
        "cp_counts": dict(sorted(cp_counts.items())),
        "minima": {key: str(value) for key, value in minima.items()},
        "maxima": {key: str(value) for key, value in maxima.items()},
        "coverage_counts": dict(sorted(coverage_counts.items())),
        "unique_maturities_by_date": {
            date_value: len(values) for date_value, values in sorted(maturity_values_by_date.items())
        },
    }


def validate(month_dates: pd.DataFrame) -> str:
    master_date_strings = set(month_dates["date"].dt.strftime("%Y-%m-%d"))
    actual_months = set(month_dates["date"].dt.to_period("M").astype(str))
    missing_months = sorted(set(expected_months()) - actual_months)
    extra_months = sorted(actual_months - set(expected_months()))
    duplicate_month_rows = int(month_dates.duplicated("calendar_month", keep=False).sum())

    keys = {
        "option_prices": ["secid", "date", "optionid"],
        "vol_surface": ["secid", "date", "days", "cp_flag", "delta"],
        "forward_prices": ["secid", "date", "expiration", "maturity_days", "am_settlement"],
        "zero_curve": ["date", "days"],
        "spot": ["secid", "date"],
    }
    summaries = {dataset: summarize_dataset(dataset, key) for dataset, key in keys.items()}

    lines = [
        "WRDS OptionMetrics SPX monthly extraction report",
        f"Generated: {utc_now()}",
        f"Pipeline version: {PIPELINE_VERSION}",
        f"Library: {LIBRARY}",
        f"SPX secid: {SECID}",
        f"Requested period: {START_YEAR}-01 through 2025-08",
        "",
        "MASTER MONTHLY DATES",
        f"- Monthly dates: {len(month_dates)}",
        f"- Expected calendar months: {len(expected_months())}",
        f"- Date range: {month_dates['date'].min().date() if len(month_dates) else None} to {month_dates['date'].max().date() if len(month_dates) else None}",
        f"- Missing calendar months: {missing_months if missing_months else 'none'}",
        f"- Extra calendar months: {extra_months if extra_months else 'none'}",
        f"- Duplicate calendar-month rows: {duplicate_month_rows}",
        "",
        "DATASET VALIDATION",
    ]

    for dataset, summary in summaries.items():
        missing_dates = sorted(master_date_strings - summary["dates"])
        extra_dates = sorted(summary["dates"] - master_date_strings)
        lines.extend(
            [
                "",
                f"[{dataset}]",
                f"- File: {FINAL_FILES[dataset].name}",
                f"- Rows: {summary['rows']}",
                f"- Unique dates: {summary['unique_dates']}",
                f"- Min/max dates: {summary['min_date']} / {summary['max_date']}",
                f"- Missing master dates: {missing_dates if missing_dates else 'none'}",
                f"- Extra dates: {extra_dates if extra_dates else 'none'}",
                f"- Duplicate rows participating by declared key: {summary['duplicate_rows_by_key']}",
                f"- Null percentages: {json.dumps({k: fmt_pct(v) for k, v in summary['null_percentages'].items()}, sort_keys=True)}",
            ]
        )
        if summary["cp_counts"]:
            lines.append(f"- Call/put counts: {json.dumps(summary['cp_counts'], sort_keys=True)}")
        if summary["minima"]:
            lines.append(f"- Maturity/DTE minima: {json.dumps(summary['minima'], sort_keys=True)}")
            lines.append(f"- Maturity/DTE maxima: {json.dumps(summary['maxima'], sort_keys=True)}")

    spot_missing = sorted(master_date_strings - summaries["spot"]["dates"])
    zero_missing = sorted(master_date_strings - summaries["zero_curve"]["dates"])
    surface_coverage = summaries["vol_surface"]["coverage_counts"]
    forward_coverage = summaries["forward_prices"]["coverage_counts"]
    forward_maturity_coverage = summaries["forward_prices"]["unique_maturities_by_date"]
    lines.extend(
        [
            "",
            "MANDATORY COVERAGE CHECKS",
            f"- Every monthly date has SPX spot: {not spot_missing}",
            f"- SPX spot missing dates: {spot_missing if spot_missing else 'none'}",
            f"- Every monthly date has zero-curve data: {not zero_missing}",
            f"- Zero-curve missing dates: {zero_missing if zero_missing else 'none'}",
            f"- Volatility-surface rows per date: {json.dumps(surface_coverage, sort_keys=True)}",
            f"- Forward-price observations per date (settlement-specific): {json.dumps(forward_coverage, sort_keys=True)}",
            f"- Forward-price unique maturities per date: {json.dumps(forward_maturity_coverage, sort_keys=True)}",
            "",
            "READINESS",
        ]
    )
    mandatory_ready = not missing_months and not spot_missing and not zero_missing
    lines.append(f"- Mandatory monthly coverage complete: {mandatory_ready}")
    if summaries["option_prices"]["minima"].get("dte") != "30" or summaries["option_prices"]["maxima"].get("dte") != "730":
        lines.append("- Note: observed DTE endpoints need not equal filters; every retained DTE is checked below.")

    # Explicitly check every option DTE, without loading all years simultaneously.
    dte_violations = 0
    strike_scales: set[float] = set()
    for year in range(START_YEAR, END_YEAR + 1):
        frame = next(iter([pd.read_parquet(checkpoint_paths("option_prices", year)[0])]))
        if "dte" in frame:
            dte = pd.to_numeric(frame["dte"], errors="coerce")
            dte_violations += int((dte.notna() & ~dte.between(30, 730)).sum())
        if "strike_price" in frame and "strike" in frame:
            valid = frame[["strike_price", "strike"]].dropna()
            valid = valid[valid["strike"] != 0]
            if len(valid):
                ratios = (valid["strike_price"] / valid["strike"]).round(6)
                strike_scales.update(float(value) for value in ratios.unique())
    lines.extend(
        [
            f"- Option DTE rows outside [30, 730]: {dte_violations}",
            f"- Raw-to-normalized strike scale(s): {sorted(strike_scales)}",
        ]
    )

    report = "\n".join(lines) + "\n"
    (OUTPUT_DIR / "extraction_report.txt").write_text(report, encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema-only", action="store_true", help="Discover and print schema mapping, then stop")
    parser.add_argument("--force", action="store_true", help="Re-query otherwise valid yearly checkpoints")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    log(f"Starting pipeline version {PIPELINE_VERSION}")
    db: wrds.Connection | None = None
    try:
        log("Opening WRDS connection; interactive authentication may appear in this terminal")
        db = guarded_wrds_connection()
        log("WRDS connection established")
        families, _ = discover_schema(db)
        if args.schema_only:
            log("Schema-only run complete")
            return 0
        month_dates = construct_month_end_dates(db, families["option_prices"])
        log(f"Master monthly-date list contains {len(month_dates)} dates")
        extract_all(db, families, month_dates, force=args.force)
        for dataset in FINAL_FILES:
            assemble_final(dataset)
        report = validate(month_dates)
        print("\n" + report, flush=True)
        log("Extraction and validation completed successfully")
        return 0
    except KeyboardInterrupt:
        log("Execution interrupted safely; completed yearly checkpoints remain resumable")
        return 130
    except EOFError:
        log("Execution paused during interactive input; rerun in a terminal to complete WRDS authentication")
        return 130
    except Exception as exc:
        # Never include connection object details or credentials in our log.
        log(f"Pipeline failed: {type(exc).__name__}: {exc}")
        if db is not None:
            traceback.print_exc()
        return 1
    finally:
        if db is not None:
            try:
                db.close()
                log("WRDS connection closed safely")
            except Exception:
                log("WRDS connection cleanup encountered a non-fatal error")


if __name__ == "__main__":
    sys.exit(main())
