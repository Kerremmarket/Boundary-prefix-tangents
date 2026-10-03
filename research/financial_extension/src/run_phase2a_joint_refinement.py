#!/usr/bin/env python3
"""Joint grid/epsilon and carrier-reconciliation audit for Phase II-A."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from indexed_annuity import ContractSpec, log_state_grid, upper_regular_boundary
from phase2_multidate import (
    component_history_coefficient,
    direct_multidate_regret,
    reference_forward_audit,
    solve_stationary_with_tail,
)
from run_phase2a import DIRECTIONS, EPSILONS, PRODUCTS, REGULAR_DATES, _distribution


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--grid-points", nargs="+", type=int, default=[30_000, 60_000, 120_000]
    )
    parser.add_argument("--law-nodes", type=int, default=64)
    parser.add_argument("--integration-nodes", type=int, default=64)
    return parser.parse_args()


def _state_density(log_grid: np.ndarray, log_density: np.ndarray, state: float) -> float:
    return float(
        np.interp(
            np.log(state), log_grid, log_density, left=0.0, right=0.0
        )
        / state
    )


def main() -> int:
    args = parse_args()
    panel_dir = args.panel_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    calibrations = pd.read_csv(panel_dir / "annual_calibrations.csv").set_index(
        "date"
    )
    statuses = pd.read_csv(panel_dir / "date_status.csv").set_index("date")
    refinement: list[dict] = []
    reconciliation: list[dict] = []

    for grid_points in sorted(set(args.grid_points)):
        grid = log_state_grid(0.03, 20.0, grid_points)
        log_grid = np.log(grid)
        for date in REGULAR_DATES:
            calibration = calibrations.loc[date]
            status = statuses.loc[date]
            beta = float(status["discount_factor"])
            discount_survival = beta * 0.97
            for product in PRODUCTS:
                distribution = _distribution(calibration, status, product)
                law = distribution.discretize(args.law_nodes)
                contract = ContractSpec(
                    discount_factor=beta,
                    termination_probability=0.03,
                    surrender_haircut=0.0,
                    death_guarantee=1.0,
                )
                solution, tail_slope, margin = solve_stationary_with_tail(
                    grid=grid,
                    law=law,
                    contract=contract,
                    tolerance=1e-11,
                )
                if margin <= 0:
                    refinement.append(
                        {
                            "grid_points": grid_points,
                            "date": date,
                            "product": product.code,
                            "status": "no_upper_stopping_tail",
                            "high_state_margin": margin,
                        }
                    )
                    continue
                boundary, slope = upper_regular_boundary(solution)
                initial_state = 0.96 * boundary
                boundaries = (boundary,) * 5
                slopes = (slope,) * 5
                audit = reference_forward_audit(
                    log_grid=log_grid,
                    initial_state=initial_state,
                    initial_distribution=distribution,
                    transition_distributions=[distribution] * 4,
                    boundaries=boundaries,
                )
                trace_index = int(np.searchsorted(log_grid, np.log(boundary))) - 2
                if trace_index < 0:
                    raise ValueError("boundary is too close to the lower grid edge")
                trace_state = float(grid[trace_index])
                for target in range(5):
                    aggregate_density = float(
                        audit.predecision[target].continuous_density[trace_index]
                        / trace_state
                    )
                    component_density = sum(
                        float(
                            audit.fresh_arrival_log_densities[source][trace_index]
                            / trace_state
                        )
                        * distribution.floor_probability ** (target - source)
                        for source in range(target + 1)
                    )
                    reconciliation.append(
                        {
                            "grid_points": grid_points,
                            "date": date,
                            "product": product.code,
                            "target_date": target + 1,
                            "left_trace_state": trace_state,
                            "aggregate_boundary_density": aggregate_density,
                            "component_sum_boundary_density": component_density,
                            "absolute_difference": abs(
                                aggregate_density - component_density
                            ),
                            "relative_difference": (
                                abs(aggregate_density - component_density)
                                / aggregate_density
                                if aggregate_density > 0
                                else 0.0
                            ),
                        }
                    )

                relative = DIRECTIONS["common"]
                shifts = tuple(boundary * item for item in relative)
                coefficient = component_history_coefficient(
                    audit=audit,
                    boundaries=boundaries,
                    slopes=slopes,
                    shifts=shifts,
                    discount_survival=discount_survival,
                    floor_probabilities=[distribution.floor_probability] * 4,
                    alignment_groups=["stationary"] * 5,
                )
                for epsilon in EPSILONS:
                    direct = direct_multidate_regret(
                        log_grid=log_grid,
                        state_grid=grid,
                        initial_state=initial_state,
                        initial_distribution=distribution,
                        transition_distributions=[distribution] * 4,
                        boundaries=boundaries,
                        gaps=[solution.gap] * 5,
                        shifts=shifts,
                        epsilon=epsilon,
                        discount_survival=discount_survival,
                        integration_nodes=args.integration_nodes,
                    )
                    scaled = direct.total / epsilon**2
                    refinement.append(
                        {
                            "grid_points": grid_points,
                            "date": date,
                            "product": product.code,
                            "status": "regular_upper_boundary",
                            "high_state_margin": margin,
                            "boundary": boundary,
                            "slope": slope,
                            "epsilon": epsilon,
                            "actual_loss": direct.total,
                            "actual_scaled_loss": scaled,
                            "full_coefficient": coefficient.full,
                            "relative_full_error": abs(
                                scaled - coefficient.full
                            )
                            / coefficient.full,
                        }
                    )

    refinement_frame = pd.DataFrame(refinement)
    reconciliation_frame = pd.DataFrame(reconciliation)
    refinement_frame.to_csv(
        output_dir / "joint_grid_epsilon_refinement.csv", index=False
    )
    reconciliation_frame.to_csv(
        output_dir / "carrier_density_reconciliation.csv", index=False
    )
    regular = refinement_frame.loc[
        refinement_frame["status"].eq("regular_upper_boundary")
    ]
    smallest = regular.loc[regular["epsilon"].eq(min(EPSILONS))]
    pivot = smallest.pivot_table(
        index=["date", "product"],
        columns="grid_points",
        values="actual_scaled_loss",
    )
    grid_range = (pivot.max(axis=1) - pivot.min(axis=1)).max()
    finest_grids = sorted(set(args.grid_points))[-2:]
    finest_range = (
        pivot[finest_grids].max(axis=1) - pivot[finest_grids].min(axis=1)
    ).max()
    maximum_reconciliation = reconciliation_frame["relative_difference"].max()
    print(
        f"regular rows={len(regular)}; max smallest-epsilon scaled-loss grid "
        f"range={grid_range:.8g}; finest-two range={finest_range:.8g}; "
        f"max carrier relative difference="
        f"{maximum_reconciliation:.8g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
