#!/usr/bin/env python3
"""Regenerate the synthetic Bellman/prefix feasibility evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from indexed_annuity import (
    ContractSpec,
    CreditingSpec,
    black_scholes_credited_law,
    log_state_grid,
    solve_stationary_bellman,
    upper_regular_boundary,
)
from prefix_regret import (
    lognormal_source_density,
    two_date_actual_regret,
    two_date_coefficients,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    market = {
        "annual_rate": 0.05,
        "dividend_yield": 0.02,
        "volatility": 0.20,
    }
    crediting = CreditingSpec(participation=0.80, cap=0.08)
    contract = ContractSpec(
        discount_factor=float(np.exp(-0.05)),
        termination_probability=0.10,
        surrender_haircut=0.08,
        death_guarantee=1.0,
    )
    law = black_scholes_credited_law(
        **market,
        crediting=crediting,
        quadrature_nodes=32,
    )

    grid_rows = []
    fine_solution = None
    for points in (15_000, 30_000, 60_000):
        solution = solve_stationary_bellman(
            grid=log_state_grid(0.05, 8.0, points),
            law=law,
            contract=contract,
            tolerance=1e-12,
        )
        boundary, slope = upper_regular_boundary(solution)
        grid_rows.append(
            {
                "grid_points": points,
                "boundary": boundary,
                "stopping_side_gap_slope": slope,
                "iterations": solution.iterations,
                "fixed_point_sup_error": solution.sup_error,
            }
        )
        fine_solution = solution
    assert fine_solution is not None
    boundary, slope = upper_regular_boundary(fine_solution)

    h1, h2 = 0.8, 1.1
    source_density = lognormal_source_density(median=0.82, log_sigma=0.16)
    coefficients = two_date_coefficients(
        boundary=boundary,
        stopping_side_gap_slope=slope,
        h1=h1,
        h2=h2,
        source_density=source_density,
        law=law,
        contract=contract,
    )
    epsilon_rows = []
    for epsilon in (0.01, 0.005, 0.0025, 0.00125, 0.000625):
        regret = two_date_actual_regret(
            epsilon=epsilon,
            boundary=boundary,
            h1=h1,
            h2=h2,
            source_density=source_density,
            solution=fine_solution,
            law=law,
            contract=contract,
            integration_nodes=96,
        )
        scaled = regret.total / epsilon**2
        epsilon_rows.append(
            {
                "epsilon": epsilon,
                "date1_loss": regret.date1,
                "date2_loss": regret.date2,
                "actual_scaled_loss": scaled,
                "diagonal_coefficient": coefficients.diagonal,
                "prefix_coefficient": coefficients.prefix,
                "absolute_diagonal_error": abs(scaled - coefficients.diagonal),
                "absolute_prefix_error": abs(scaled - coefficients.prefix),
            }
        )

    grid_frame = pd.DataFrame(grid_rows)
    epsilon_frame = pd.DataFrame(epsilon_rows)
    grid_frame.to_csv(output_dir / "grid_refinement.csv", index=False)
    epsilon_frame.to_csv(output_dir / "epsilon_convergence.csv", index=False)
    payload = {
        "status": "synthetic_feasibility_only_not_market_calibrated",
        "market": market,
        "crediting": {
            "participation": crediting.participation,
            "cap": crediting.cap,
        },
        "contract": {
            "discount_factor": contract.discount_factor,
            "termination_probability": contract.termination_probability,
            "surrender_haircut": contract.surrender_haircut,
            "death_guarantee": contract.death_guarantee,
            "minimum_surrender_value": contract.minimum_surrender_value,
        },
        "source_density": {"family": "lognormal", "median": 0.82, "log_sigma": 0.16},
        "directions": {"h1": h1, "h2": h2},
        "credited_law": {
            "floor_probability": law.floor_probability,
            "cap_probability": law.cap_probability,
            "expected_factor": law.expected_factor,
        },
        "boundary": boundary,
        "stopping_side_gap_slope": slope,
        "coefficients": {
            "date1_diagonal": coefficients.date1_diagonal,
            "date2_diagonal": coefficients.date2_diagonal,
            "diagonal": coefficients.diagonal,
            "copied_prefix": coefficients.copied_prefix,
            "prefix": coefficients.prefix,
        },
    }
    (output_dir / "parameters_and_coefficients.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    best = epsilon_frame.loc[epsilon_frame["absolute_prefix_error"].idxmin()]
    summary = f"""# Synthetic indexed-annuity feasibility check

**Status:** numerical architecture check only; not calibrated to OptionMetrics.

- Credited-floor probability: {law.floor_probability:.6f}.
- Credited-cap probability: {law.cap_probability:.6f}.
- Fine-grid upper surrender boundary: {boundary:.9f}.
- Stopping-side gap slope: {slope:.9f}.
- Date-diagonal coefficient: {coefficients.diagonal:.9f}.
- Copied-prefix increment: {coefficients.copied_prefix:.9f}.
- Full prefix coefficient: {coefficients.prefix:.9f}.
- At the best-resolved epsilon={best['epsilon']:.6f}, actual scaled loss: {best['actual_scaled_loss']:.9f}.
- Absolute diagonal error at that epsilon: {best['absolute_diagonal_error']:.9f}.
- Absolute prefix error at that epsilon: {best['absolute_prefix_error']:.9f}.

The synthetic check establishes that the contract architecture can have a regular
stationary surrender boundary and a positive floor-copy prefix term. It does not
establish empirical magnitude or market-model adequacy. Those claims remain gated
by settlement-consistent OptionMetrics extraction and calibration.
"""
    (output_dir / "summary.md").write_text(summary, encoding="utf-8")
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
