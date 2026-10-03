#!/usr/bin/env python3
"""Prepare licensed inputs locally and run the paper's numerical stages."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from reproduction_support import digest, register_local_files, validate_source_manifest

ROOT = Path(__file__).resolve().parent
FINANCE = "research/financial_extension"
PANEL = f"{FINANCE}/evidence/full_market_panel"
SELECTION = f"{FINANCE}/evidence/pilot_selection"
CLIQ = "research/cliquet_feasibility"
AUDIT = "research/cliquet_integration"
PREFIX = "research/prefix_comparator_revision"
DATES = ["2016-10-31", "2008-11-28", "2020-02-28", "2009-04-30", "2006-03-31"]


def run(*args: str, paths=()) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), *(str(ROOT / p) for p in paths)])
    subprocess.run([sys.executable, *args], cwd=ROOT, env=env, check=True)


def export_calibrated_inputs() -> None:
    import pandas as pd
    import numpy as np
    panel = pd.read_csv(ROOT / PANEL / "date_status.csv")
    selection = pd.read_csv(ROOT / SELECTION / "selected_pilot_dates.csv")
    selected = selection.sort_values("selection_order").head(5)["date"].tolist()
    if selected != DATES:
        raise RuntimeError(f"Selection differs from the paper: {selected}. Preserve the download and investigate; do not silently substitute dates.")
    market = panel[panel.date.isin(DATES)].copy()
    if len(market) != 5 or market.date.nunique() != 5:
        raise RuntimeError("Expected exactly five calibrated market snapshots")
    market["selection_order"] = market.date.map({d: i+1 for i,d in enumerate(DATES)})
    market["black_scholes_dividend_yield"] = market.zero_rate - np.log(market.annual_gross_forward)
    columns = ["selection_order", "date", "spot", "one_year_forward", "actual_maturity",
               "annual_gross_forward", "zero_rate", "black_scholes_dividend_yield",
               "atm_iv_1y", "skew_25d_1y", "iv_term_slope"]
    market = market.sort_values("selection_order")[columns]
    if market.isna().any().any():
        raise RuntimeError("Missing market input values")
    calibrations = pd.read_csv(ROOT / PANEL / "annual_calibrations.csv")
    mixture = calibrations[calibrations.date.isin(DATES)].copy()
    if len(mixture) != 5 or mixture.date.nunique() != 5 or not (mixture.status.eq("ok") & mixture.success.eq(True)).all():
        raise RuntimeError("A selected calibration failed")
    inputs = ROOT / CLIQ / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    market.to_csv(inputs / "market_snapshots.csv", index=False)
    mixture.to_csv(inputs / "mixture_calibrations.csv", index=False)
    files = [*inputs.glob("*.csv"), *(ROOT / PANEL).glob("*.csv"), *(ROOT / SELECTION).glob("*.csv")]
    register_local_files(ROOT, files, stage="prepare")
    config = json.loads((ROOT / CLIQ / "config.json").read_text())
    comparison = {name: {"historical_sha256": expected,
                        "local_sha256": digest((ROOT / CLIQ / name).resolve()),
                        "byte_identical": digest((ROOT / CLIQ / name).resolve()) == expected}
                  for name, expected in config["input_hashes"].items()}
    (ROOT / "input-comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")


def prepare(data_dir: Path) -> None:
    if data_dir is None:
        raise SystemExit("prepare requires --data-dir pointing to your licensed extraction")
    source = f"{FINANCE}/src"
    run(f"{source}/select_pilot_dates.py", "--data-dir", str(data_dir.resolve()),
        "--output-dir", SELECTION, "--count", "12")
    run(f"{source}/run_full_annual_panel.py", "--data-dir", str(data_dir.resolve()),
        "--market-features", f"{SELECTION}/all_market_features.csv", "--output-dir", PANEL,
        paths=[source])
    export_calibrated_inputs()


def register_outputs(stage: str, package: str) -> None:
    register_local_files(ROOT, [p for directory in ["results", "tables", "figures"]
                               for p in (ROOT/package/directory).rglob("*") if p.is_file()], stage=stage)


def cliquet() -> None:
    paths = [f"{CLIQ}/src", f"{AUDIT}/src"]
    run(f"{CLIQ}/src/run_pilot.py", paths=paths)
    run(f"{CLIQ}/src/analyze_results.py", paths=paths)
    run(f"{CLIQ}/src/validate_results.py", paths=paths)
    run("-m", "pytest", "-q", f"{CLIQ}/tests/test_analysis.py", paths=paths)
    register_outputs("cliquet-pilot", CLIQ)
    run(f"{AUDIT}/src/run_uncapped_audit.py", paths=paths)
    run(f"{AUDIT}/src/run_refinement_audit.py", paths=paths)
    run(f"{AUDIT}/src/analyze_audit.py", paths=paths)
    register_outputs("cliquet-audit", AUDIT)


def prefix() -> None:
    paths = [f"{CLIQ}/src", f"{AUDIT}/src", f"{PREFIX}/src"]
    run(f"{PREFIX}/src/run_experiments.py", paths=paths)
    run(f"{PREFIX}/src/analyze_results.py", paths=paths)
    figures = ROOT / "paper/figures"
    figures.mkdir(parents=True, exist_ok=True)
    for p in (ROOT / PREFIX / "figures").glob("*.pdf"):
        shutil.copy2(p, figures/p.name)
    run(f"{PREFIX}/src/validate_outputs.py", paths=paths)
    register_outputs("prefix", PREFIX)


def bermudan() -> None:
    paths = [f"{FINANCE}/src"]
    run(f"{FINANCE}/src/run_bermudan_diagnostic.py", paths=paths)
    run(f"{FINANCE}/src/validate_financial_diagnostic.py", paths=paths)
    run("research/finance_integration/certify_tiny_losses.py", paths=paths)


def annuity() -> None:
    source = f"{FINANCE}/src"
    evidence = f"{FINANCE}/phase2/evidence/phase2a"
    for script in ["run_phase2a.py", "run_phase2a_joint_refinement.py"]:
        run(f"{source}/{script}", "--panel-dir", PANEL, "--output-dir", evidence, paths=[source])
    run(f"{source}/analyze_phase2a.py", "--evidence-dir", evidence,
        "--output-dir", f"{FINANCE}/phase2/publication", paths=[source])
    run(f"{source}/validate_phase2a.py", "--evidence-dir", evidence,
        "--output-dir", f"{FINANCE}/phase2/audit", paths=[source])


def tests() -> None:
    run("-m", "pytest", "-q", "tests")
    run("-m", "unittest", "discover", "-s", "reproducibility/tests", "-v", paths=["reproducibility"])
    run("-m", "pytest", "-q", f"{FINANCE}/tests", paths=[f"{FINANCE}/src"])
    run("-m", "pytest", "-q", f"{CLIQ}/tests/test_cliquet.py", paths=[f"{CLIQ}/src"])
    run("-m", "pytest", "-q", f"{AUDIT}/tests", paths=[f"{AUDIT}/src"])
    run("-m", "pytest", "-q", f"{PREFIX}/tests")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["tests", "theory", "prepare", "cliquet", "prefix", "bermudan", "annuity"])
    parser.add_argument("--data-dir", type=Path)
    args = parser.parse_args()
    validate_source_manifest(ROOT)
    if args.stage == "prepare":
        prepare(args.data_dir)
    elif args.stage == "theory":
        run("reproducibility/run_all.py")
    else:
        globals()[args.stage]()


if __name__ == "__main__":
    main()
