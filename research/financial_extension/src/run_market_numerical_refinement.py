#!/usr/bin/env python3
"""Refine credited-law and regret quadrature on the pilot boundary dates."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from indexed_annuity import (
    ContractSpec,
    CreditingSpec,
    MixtureCreditedDistribution,
    log_state_grid,
    solve_stationary_bellman,
    upper_regular_boundary,
)
from market_models import LognormalMixtureParams
from prefix_regret import (
    market_two_date_actual_regret,
    market_two_date_coefficients,
)


LAW_NODES = (32, 48, 64, 96, 192)
INTEGRATION_NODES = (24, 32, 48, 64, 96)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market-pilot-dir", type=Path, required=True)
    parser.add_argument("--regret-pilot-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _single(frame: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(frame[column], errors="coerce").dropna().unique()
    if len(values) != 1:
        raise ValueError(f"{column} is not constant")
    return float(values[0])


def main() -> int:
    args = parse_args()
    pilot_dir = args.market_pilot_dir.expanduser().resolve()
    regret_dir = args.regret_pilot_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration = pd.read_json(pilot_dir / "calibration_results.json")
    quotes = pd.read_parquet(
        pilot_dir / "pilot_repaired_quotes_and_fits.parquet"
    )
    quotes = quotes.loc[quotes["target_dte"].eq(365)].copy()
    status = pd.read_csv(regret_dir / "date_status.csv")
    boundary_dates = pd.to_datetime(
        status.loc[status["status"] == "regular_upper_boundary", "date"]
    ).dt.normalize()
    crediting = CreditingSpec(0.80, 0.08)
    law_records: list[dict] = []
    integration_records: list[dict] = []

    for date in boundary_dates:
        date_quotes = quotes.loc[
            pd.to_datetime(quotes["date"]).dt.normalize().eq(date)
        ]
        spot = _single(date_quotes, "spot")
        forward = _single(date_quotes, "matched_forward_price")
        maturity = _single(date_quotes, "year_fraction")
        zero_rate = _single(date_quotes, "zero_rate")
        gross_forward = float(np.exp(np.log(forward / spot) / maturity))
        row = calibration.loc[
            calibration["model"].eq("annual_lognormal_mixture")
            & pd.to_datetime(calibration["date"]).dt.normalize().eq(date)
        ].iloc[0]
        payload = row["parameters"]
        distribution = MixtureCreditedDistribution(
            gross_forward=gross_forward,
            mixture=LognormalMixtureParams(
                payload["low_weight"],
                payload["low_forward_multiplier"],
                payload["low_volatility"],
                payload["high_volatility"],
            ),
            crediting=crediting,
        )
        contract = ContractSpec(
            discount_factor=float(np.exp(-zero_rate)),
            termination_probability=0.03,
            surrender_haircut=0.08,
            death_guarantee=1.0,
        )
        for nodes in LAW_NODES:
            law = distribution.discretize(nodes)
            solution = solve_stationary_bellman(
                grid=log_state_grid(0.03, 20.0, 60_000),
                law=law,
                contract=contract,
                tolerance=1e-12,
            )
            boundary, slope = upper_regular_boundary(solution)
            law_records.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "law_nodes_per_component": nodes,
                    "support_points": len(law.factors),
                    "expected_credited_factor": law.expected_factor,
                    "floor_probability": law.floor_probability,
                    "cap_probability": law.cap_probability,
                    "boundary": boundary,
                    "stopping_side_gap_slope": slope,
                }
            )

        law = distribution.discretize(96)
        solution = solve_stationary_bellman(
            grid=log_state_grid(0.03, 20.0, 120_000),
            law=law,
            contract=contract,
            tolerance=1e-12,
        )
        boundary, slope = upper_regular_boundary(solution)
        initial_state = 0.96 * boundary
        for nodes in INTEGRATION_NODES:
            coefficients = market_two_date_coefficients(
                boundary=boundary,
                stopping_side_gap_slope=slope,
                h1=boundary,
                h2=boundary,
                initial_state=initial_state,
                distribution=distribution,
                contract=contract,
                integration_nodes=nodes,
            )
            regret = market_two_date_actual_regret(
                epsilon=0.005,
                boundary=boundary,
                h1=boundary,
                h2=boundary,
                initial_state=initial_state,
                distribution=distribution,
                solution=solution,
                contract=contract,
                integration_nodes=nodes,
            )
            integration_records.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "integration_nodes": nodes,
                    "epsilon": 0.005,
                    "diagonal_coefficient": coefficients.diagonal,
                    "copied_prefix": coefficients.copied_prefix,
                    "prefix_coefficient": coefficients.prefix,
                    "actual_scaled_loss": regret.total / 0.005**2,
                    "absolute_prefix_error": abs(
                        regret.total / 0.005**2 - coefficients.prefix
                    ),
                }
            )

    law_frame = pd.DataFrame(law_records)
    integration_frame = pd.DataFrame(integration_records)
    law_frame.to_csv(output_dir / "law_quadrature_refinement.csv", index=False)
    integration_frame.to_csv(
        output_dir / "regret_integration_refinement.csv", index=False
    )
    law_spread = law_frame.groupby("date").agg(
        boundary_range=("boundary", lambda values: float(values.max() - values.min())),
        slope_range=(
            "stopping_side_gap_slope",
            lambda values: float(values.max() - values.min()),
        ),
    )
    integration_spread = integration_frame.groupby("date").agg(
        prefix_range=(
            "prefix_coefficient",
            lambda values: float(values.max() - values.min()),
        ),
        actual_scaled_range=(
            "actual_scaled_loss",
            lambda values: float(values.max() - values.min()),
        ),
    )
    (output_dir / "summary.md").write_text(
        f"""# Market numerical refinement

- Boundary dates: {len(boundary_dates)}.
- Credited-law nodes per mixture component: {list(LAW_NODES)}.
- Regret integration nodes: {list(INTEGRATION_NODES)}.
- Maximum boundary range across law quadratures: {law_spread['boundary_range'].max():.9g}.
- Maximum stopping-slope range across law quadratures: {law_spread['slope_range'].max():.9g}.
- Maximum prefix-coefficient range across integration rules: {integration_spread['prefix_range'].max():.9g}.
- Maximum actual-scaled-loss range across integration rules: {integration_spread['actual_scaled_range'].max():.9g}.

Bellman fixed-point tolerance is `1e-12`. Grid refinement is reported separately
in the main market-regret pilot.
""",
        encoding="utf-8",
    )
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
