#!/usr/bin/env python3
"""Run the 12-date market-calibrated indexed-annuity regret pilot."""

from __future__ import annotations

import argparse
import json
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


GRID_REFINEMENT = (30_000, 60_000, 120_000)
EPSILONS = (0.02, 0.01, 0.005, 0.0025, 0.00125)
MONEYNESS_RATIOS = (0.94, 0.96, 0.98)
DIRECTIONS = {
    "joint_equal": (1.0, 1.0),
    "later_larger": (0.8, 1.1),
    "earlier_larger": (1.1, 0.8),
    "earlier_only": (1.0, 0.0),
    "later_only": (0.0, 1.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market-pilot-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--participation", type=float, default=0.80)
    parser.add_argument("--cap", type=float, default=0.08)
    parser.add_argument("--termination-probability", type=float, default=0.03)
    parser.add_argument("--surrender-haircut", type=float, default=0.08)
    parser.add_argument("--law-nodes", type=int, default=96)
    parser.add_argument("--integration-nodes", type=int, default=64)
    return parser.parse_args()


def _single_value(frame: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(frame[column], errors="coerce").dropna().unique()
    if len(values) != 1:
        raise ValueError(f"{column} is not constant within the selected slice")
    return float(values[0])


def _mixture_params(payload: dict) -> LognormalMixtureParams:
    return LognormalMixtureParams(
        low_weight=float(payload["low_weight"]),
        low_forward_multiplier=float(payload["low_forward_multiplier"]),
        low_volatility=float(payload["low_volatility"]),
        high_volatility=float(payload["high_volatility"]),
    )


def _predecision_value(
    *,
    initial_state: float,
    solution,
    law,
    contract: ContractSpec,
) -> float:
    next_state = initial_state * law.factors
    continuation_values = np.interp(
        next_state, solution.grid, solution.value
    )
    termination_values = contract.termination_value(next_state)
    return float(
        contract.discount_factor
        * np.dot(
            law.weights,
            contract.termination_probability * termination_values
            + (1.0 - contract.termination_probability) * continuation_values,
        )
    )


def _markdown_summary(
    *,
    status: pd.DataFrame,
    coefficients: pd.DataFrame,
    convergence: pd.DataFrame,
    parameters: dict,
) -> str:
    regular = status.loc[status["status"] == "regular_upper_boundary"]
    absent = status.loc[status["status"] == "no_upper_stopping_tail"]
    joint = coefficients.loc[
        coefficients["direction"].isin(
            ["joint_equal", "later_larger", "earlier_larger"]
        )
    ]
    smallest = convergence.loc[
        convergence["epsilon"] == min(EPSILONS)
    ]
    joint_smallest = smallest.loc[
        smallest["direction"].isin(
            ["joint_equal", "later_larger", "earlier_larger"]
        )
    ]
    prefix_wins = int(
        (
            joint_smallest["absolute_prefix_error"]
            < joint_smallest["absolute_diagonal_error"]
        ).sum()
    )
    comparisons = int(len(joint_smallest))
    return f"""# Market-calibrated indexed-annuity regret pilot

This bounded experiment uses the annual two-lognormal fallback calibrated on
the 12 predeclared OptionMetrics dates. The licensed quote-level panel remains
local; only compact diagnostics are retained in Git.

## Contract and numerical design

- Participation: {parameters['participation']:.2f}; annual cap: {parameters['cap']:.2f}.
- Surrender haircut: {parameters['surrender_haircut']:.2f}; annual termination probability: {parameters['termination_probability']:.2f}.
- Grid refinement: {', '.join(f'{value:,}' for value in GRID_REFINEMENT)} points.
- Credited-law quadrature: {parameters['law_nodes']} nodes per mixture component.
- Initial guarantee-state ratios: {', '.join(f'{value:.2f}' for value in MONEYNESS_RATIOS)} times the endogenous boundary.
- Perturbations are relative boundary shifts, with epsilon in {list(EPSILONS)}.

## Boundary gate

- Regular upper surrender boundary: {len(regular)} of {len(status)} market dates.
- No asymptotic upper stopping tail: {len(absent)} dates.
- Boundary dates: {', '.join(pd.to_datetime(regular['date']).dt.strftime('%Y-%m-%d')) if len(regular) else 'none'}.

Absence of a boundary is retained as an economic result rather than repaired by
tuning the contract. On those dates the credited guarantee is too valuable
relative to the market discount rate for high-state surrender to be optimal
under this fixed benchmark contract.

## Prefix comparison

- Positive copied-prefix coefficients among joint directions: {int((joint['copied_prefix'] > 0).sum())} of {len(joint)}.
- At the smallest reported epsilon, the prefix approximation is closer than the date-diagonal approximation in {prefix_wins} of {comparisons} joint comparisons.
- Median copied share of the full prefix coefficient across joint directions: {joint['copied_share_of_prefix'].median():.4f}.
- Median prefix uplift relative to the diagonal coefficient: {joint['prefix_uplift_over_diagonal'].median():.4f}.

The finite-epsilon table must be read together with the grid-refinement table:
once a boundary displacement approaches the Bellman grid spacing, interpolation
error can dominate the asymptotic remainder. No economic-materiality conclusion
is based solely on the smallest epsilon.
"""


def main() -> int:
    args = parse_args()
    pilot_dir = args.market_pilot_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    calibration = pd.read_json(pilot_dir / "calibration_results.json")
    mixture_rows = calibration.loc[
        calibration["model"].eq("annual_lognormal_mixture")
        & calibration["status"].eq("ok")
        & calibration["success"].eq(True)  # noqa: E712
    ].copy()
    quotes = pd.read_parquet(
        pilot_dir / "pilot_repaired_quotes_and_fits.parquet"
    )
    one_year = quotes.loc[quotes["target_dte"].eq(365)].copy()
    dates = sorted(pd.to_datetime(mixture_rows["date"]).dt.normalize().unique())
    if len(dates) != 12:
        raise ValueError(f"expected 12 successful mixture dates, found {len(dates)}")

    crediting = CreditingSpec(args.participation, args.cap)
    status_records: list[dict] = []
    grid_records: list[dict] = []
    coefficient_records: list[dict] = []
    epsilon_records: list[dict] = []

    for date_value in dates:
        date = pd.Timestamp(date_value).normalize()
        calibration_row = mixture_rows.loc[
            pd.to_datetime(mixture_rows["date"]).dt.normalize().eq(date)
        ].iloc[0]
        slice_quotes = one_year.loc[
            pd.to_datetime(one_year["date"]).dt.normalize().eq(date)
        ]
        if slice_quotes.empty:
            raise ValueError(f"missing one-year quote slice for {date.date()}")
        spot = _single_value(slice_quotes, "spot")
        forward = _single_value(slice_quotes, "matched_forward_price")
        maturity = _single_value(slice_quotes, "year_fraction")
        zero_rate = _single_value(slice_quotes, "zero_rate")
        if maturity <= 0 or spot <= 0 or forward <= 0:
            raise ValueError(f"invalid market inputs for {date.date()}")
        annual_gross_forward = float(np.exp(np.log(forward / spot) / maturity))
        mixture = _mixture_params(calibration_row["parameters"])
        distribution = MixtureCreditedDistribution(
            gross_forward=annual_gross_forward,
            mixture=mixture,
            crediting=crediting,
        )
        law = distribution.discretize(args.law_nodes)
        contract = ContractSpec(
            discount_factor=float(np.exp(-zero_rate)),
            termination_probability=args.termination_probability,
            surrender_haircut=args.surrender_haircut,
            death_guarantee=1.0,
        )
        liquidation_slope = 1.0 - contract.surrender_haircut
        continuation_if_next_stop = (
            contract.discount_factor
            * law.expected_factor
            * (
                contract.termination_probability
                + (1.0 - contract.termination_probability) * liquidation_slope
            )
        )
        high_state_margin = liquidation_slope - continuation_if_next_stop
        base_record = {
            "date": date.strftime("%Y-%m-%d"),
            "spot": spot,
            "one_year_forward": forward,
            "actual_maturity": maturity,
            "annual_gross_forward": annual_gross_forward,
            "zero_rate": zero_rate,
            "discount_factor": contract.discount_factor,
            "mixture_standardized_rmse": float(
                calibration_row["standardized_rmse"]
            ),
            "mixture_within_spread_share": float(
                calibration_row["within_spread_share"]
            ),
            "low_weight": mixture.low_weight,
            "low_forward_multiplier": mixture.low_forward_multiplier,
            "high_forward_multiplier": mixture.high_forward_multiplier,
            "low_volatility": mixture.low_volatility,
            "high_volatility": mixture.high_volatility,
            "floor_probability": distribution.floor_probability,
            "cap_probability": distribution.cap_probability,
            "expected_credited_factor": law.expected_factor,
            "high_state_surrender_margin": high_state_margin,
        }
        if high_state_margin <= 0:
            status_records.append(
                {**base_record, "status": "no_upper_stopping_tail"}
            )
            continue

        fine_solution = None
        fine_boundary = None
        fine_slope = None
        for grid_points in GRID_REFINEMENT:
            solution = solve_stationary_bellman(
                grid=log_state_grid(0.03, 20.0, grid_points),
                law=law,
                contract=contract,
                tolerance=1e-12,
            )
            boundary, slope = upper_regular_boundary(solution)
            local_index = int(np.searchsorted(solution.grid, boundary))
            local_spacing = float(
                solution.grid[local_index] - solution.grid[local_index - 1]
            )
            grid_records.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "grid_points": grid_points,
                    "boundary": boundary,
                    "stopping_side_gap_slope": slope,
                    "local_grid_spacing": local_spacing,
                    "iterations": solution.iterations,
                    "fixed_point_sup_error": solution.sup_error,
                }
            )
            fine_solution = solution
            fine_boundary = boundary
            fine_slope = slope
        assert fine_solution is not None
        assert fine_boundary is not None
        assert fine_slope is not None
        status_records.append(
            {
                **base_record,
                "status": "regular_upper_boundary",
                "boundary": fine_boundary,
                "stopping_side_gap_slope": fine_slope,
                "bellman_iterations": fine_solution.iterations,
                "bellman_sup_error": fine_solution.sup_error,
            }
        )
        fine_spacing = grid_records[-1]["local_grid_spacing"]

        for moneyness_ratio in MONEYNESS_RATIOS:
            initial_state = moneyness_ratio * fine_boundary
            source_density_at_boundary = float(
                distribution.state_continuous_density(
                    np.array([fine_boundary]), initial_state
                )[0]
            )
            contract_value = _predecision_value(
                initial_state=initial_state,
                solution=fine_solution,
                law=law,
                contract=contract,
            )
            for direction_name, (relative_h1, relative_h2) in DIRECTIONS.items():
                h1 = relative_h1 * fine_boundary
                h2 = relative_h2 * fine_boundary
                coefficients = market_two_date_coefficients(
                    boundary=fine_boundary,
                    stopping_side_gap_slope=fine_slope,
                    h1=h1,
                    h2=h2,
                    initial_state=initial_state,
                    distribution=distribution,
                    contract=contract,
                    integration_nodes=max(args.integration_nodes, 64),
                )
                coefficient_records.append(
                    {
                        "date": date.strftime("%Y-%m-%d"),
                        "moneyness_ratio": moneyness_ratio,
                        "initial_state": initial_state,
                        "direction": direction_name,
                        "relative_h1": relative_h1,
                        "relative_h2": relative_h2,
                        "h1": h1,
                        "h2": h2,
                        "source_density_at_boundary": source_density_at_boundary,
                        "predecision_contract_value": contract_value,
                        "date1_diagonal": coefficients.date1_diagonal,
                        "date2_diagonal": coefficients.date2_diagonal,
                        "diagonal": coefficients.diagonal,
                        "copied_prefix": coefficients.copied_prefix,
                        "prefix": coefficients.prefix,
                        "copied_share_of_prefix": (
                            coefficients.copied_prefix / coefficients.prefix
                            if coefficients.prefix > 0
                            else 0.0
                        ),
                        "prefix_uplift_over_diagonal": (
                            coefficients.copied_prefix / coefficients.diagonal
                            if coefficients.diagonal > 0
                            else 0.0
                        ),
                    }
                )
                for epsilon in EPSILONS:
                    regret = market_two_date_actual_regret(
                        epsilon=epsilon,
                        boundary=fine_boundary,
                        h1=h1,
                        h2=h2,
                        initial_state=initial_state,
                        distribution=distribution,
                        solution=fine_solution,
                        contract=contract,
                        integration_nodes=args.integration_nodes,
                    )
                    scaled = regret.total / epsilon**2
                    positive_shifts = [
                        value
                        for value in (epsilon * h1, epsilon * h2)
                        if value > 0
                    ]
                    cells = (
                        min(positive_shifts) / fine_spacing
                        if positive_shifts
                        else np.nan
                    )
                    epsilon_records.append(
                        {
                            "date": date.strftime("%Y-%m-%d"),
                            "moneyness_ratio": moneyness_ratio,
                            "direction": direction_name,
                            "epsilon": epsilon,
                            "minimum_shift_grid_cells": cells,
                            "date1_loss": regret.date1,
                            "date2_loss": regret.date2,
                            "actual_loss": regret.total,
                            "actual_loss_per_100k_guarantee": regret.total * 100_000,
                            "actual_loss_basis_points_of_contract_value": (
                                regret.total / contract_value * 10_000
                            ),
                            "actual_scaled_loss": scaled,
                            "diagonal_coefficient": coefficients.diagonal,
                            "prefix_coefficient": coefficients.prefix,
                            "absolute_diagonal_error": abs(
                                scaled - coefficients.diagonal
                            ),
                            "absolute_prefix_error": abs(
                                scaled - coefficients.prefix
                            ),
                        }
                    )

    status = pd.DataFrame(status_records)
    grid = pd.DataFrame(grid_records)
    coefficients = pd.DataFrame(coefficient_records)
    convergence = pd.DataFrame(epsilon_records)
    status.to_csv(output_dir / "date_status.csv", index=False)
    grid.to_csv(output_dir / "grid_refinement.csv", index=False)
    coefficients.to_csv(output_dir / "coefficient_results.csv", index=False)
    convergence.to_csv(output_dir / "epsilon_convergence.csv", index=False)
    parameters = {
        "participation": args.participation,
        "cap": args.cap,
        "termination_probability": args.termination_probability,
        "surrender_haircut": args.surrender_haircut,
        "law_nodes": args.law_nodes,
        "integration_nodes": args.integration_nodes,
        "grid_refinement": list(GRID_REFINEMENT),
        "epsilons": list(EPSILONS),
        "moneyness_ratios": list(MONEYNESS_RATIOS),
        "directions": {name: list(value) for name, value in DIRECTIONS.items()},
    }
    (output_dir / "parameters.json").write_text(
        json.dumps(parameters, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(
        _markdown_summary(
            status=status,
            coefficients=coefficients,
            convergence=convergence,
            parameters=parameters,
        ),
        encoding="utf-8",
    )
    print(f"dates={len(status)} regular_boundaries={(status.status == 'regular_upper_boundary').sum()}")
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
