#!/usr/bin/env python3
"""Vary declared contract terms around the fixed annual-annuity benchmark."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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
from prefix_regret import market_two_date_actual_regret, market_two_date_coefficients


@dataclass(frozen=True)
class SensitivityCase:
    name: str
    varied_term: str
    varied_value: float
    participation: float = 0.80
    cap: float = 0.08
    surrender_haircut: float = 0.08
    termination_probability: float = 0.03


CASES = (
    SensitivityCase("benchmark", "benchmark", np.nan),
    SensitivityCase("participation_60", "participation", 0.60, participation=0.60),
    SensitivityCase("participation_100", "participation", 1.00, participation=1.00),
    SensitivityCase("cap_04", "cap", 0.04, cap=0.04),
    SensitivityCase("cap_12", "cap", 0.12, cap=0.12),
    SensitivityCase("haircut_04", "surrender_haircut", 0.04, surrender_haircut=0.04),
    SensitivityCase("haircut_12", "surrender_haircut", 0.12, surrender_haircut=0.12),
    SensitivityCase(
        "termination_01",
        "termination_probability",
        0.01,
        termination_probability=0.01,
    ),
    SensitivityCase(
        "termination_05",
        "termination_probability",
        0.05,
        termination_probability=0.05,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-panel-dir", type=Path, required=True)
    parser.add_argument("--pilot-dates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--grid-points", type=int, default=30_000)
    parser.add_argument("--law-nodes", type=int, default=64)
    parser.add_argument("--integration-nodes", type=int, default=48)
    parser.add_argument("--epsilon", type=float, default=0.005)
    return parser.parse_args()


def _distribution(row, case: SensitivityCase) -> MixtureCreditedDistribution:
    mixture = LognormalMixtureParams(
        low_weight=float(row.low_weight),
        low_forward_multiplier=float(row.low_forward_multiplier),
        low_volatility=float(row.low_volatility),
        high_volatility=float(row.high_volatility),
    )
    return MixtureCreditedDistribution(
        gross_forward=float(row.annual_gross_forward),
        mixture=mixture,
        crediting=CreditingSpec(case.participation, case.cap),
    )


def _contract(row, case: SensitivityCase) -> ContractSpec:
    return ContractSpec(
        discount_factor=float(np.exp(-row.zero_rate)),
        termination_probability=case.termination_probability,
        surrender_haircut=case.surrender_haircut,
        death_guarantee=1.0,
    )


def _high_state_margin(expected_factor: float, contract: ContractSpec) -> float:
    liquidation_slope = 1.0 - contract.surrender_haircut
    return float(
        liquidation_slope
        - contract.discount_factor
        * expected_factor
        * (
            contract.termination_probability
            + (1.0 - contract.termination_probability) * liquidation_slope
        )
    )


def main() -> int:
    args = parse_args()
    full_dir = args.full_panel_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    calibrations = pd.read_csv(
        full_dir / "annual_calibrations.csv", parse_dates=["date"]
    )
    status = pd.read_csv(full_dir / "date_status.csv", parse_dates=["date"])
    required_status = status[
        ["date", "annual_gross_forward", "zero_rate", "status"]
    ].rename(columns={"status": "benchmark_panel_status"})
    panel = calibrations.merge(required_status, on="date", validate="one_to_one")
    panel = panel.loc[panel["status"].eq("ok")].copy()
    if len(panel) != len(status):
        raise ValueError("contract sensitivity requires one successful fit per market date")
    pilot_dates = set(
        pd.read_csv(args.pilot_dates.expanduser().resolve(), parse_dates=["date"])[
            "date"
        ].dt.normalize()
    )

    condition_records: list[dict] = []
    pilot_records: list[dict] = []
    grid = log_state_grid(0.03, 20.0, args.grid_points)
    for row in panel.itertuples(index=False):
        date = pd.Timestamp(row.date).normalize()
        for case in CASES:
            distribution = _distribution(row, case)
            law = distribution.discretize(args.law_nodes)
            contract = _contract(row, case)
            margin = _high_state_margin(law.expected_factor, contract)
            positive_margin = margin > 0.0
            condition_records.append(
                {
                    "date": date,
                    "case": case.name,
                    "varied_term": case.varied_term,
                    "varied_value": case.varied_value,
                    "participation": case.participation,
                    "cap": case.cap,
                    "surrender_haircut": case.surrender_haircut,
                    "termination_probability": case.termination_probability,
                    "zero_rate": row.zero_rate,
                    "floor_probability": distribution.floor_probability,
                    "cap_probability": distribution.cap_probability,
                    "expected_credited_factor": law.expected_factor,
                    "high_state_surrender_margin": margin,
                    "positive_asymptotic_surrender_margin": positive_margin,
                    "predeclared_pilot_date": date in pilot_dates,
                }
            )
            if date not in pilot_dates:
                continue
            base_record = {
                "date": date,
                "case": case.name,
                "varied_term": case.varied_term,
                "varied_value": case.varied_value,
                "high_state_surrender_margin": margin,
                "status": "no_upper_stopping_tail",
            }
            if not positive_margin:
                pilot_records.append(base_record)
                continue
            try:
                solution = solve_stationary_bellman(
                    grid=grid,
                    law=law,
                    contract=contract,
                    tolerance=1e-11,
                )
                boundary, slope = upper_regular_boundary(solution)
                initial_state = 0.96 * boundary
                if initial_state * distribution.cap_factor <= boundary:
                    base_record.update(
                        {
                            "status": "regular_boundary_zero_source_density",
                            "boundary": boundary,
                            "stopping_side_gap_slope": slope,
                            "floor_probability": distribution.floor_probability,
                            "reason": (
                                "the maximum credited factor cannot carry x0=0.96b "
                                "to the first decision boundary"
                            ),
                        }
                    )
                    pilot_records.append(base_record)
                    continue
                coefficients = market_two_date_coefficients(
                    boundary=boundary,
                    stopping_side_gap_slope=slope,
                    h1=boundary,
                    h2=boundary,
                    initial_state=initial_state,
                    distribution=distribution,
                    contract=contract,
                    integration_nodes=args.integration_nodes,
                )
                regret = market_two_date_actual_regret(
                    epsilon=args.epsilon,
                    boundary=boundary,
                    h1=boundary,
                    h2=boundary,
                    initial_state=initial_state,
                    distribution=distribution,
                    solution=solution,
                    contract=contract,
                    integration_nodes=args.integration_nodes,
                )
                scaled = regret.total / args.epsilon**2
                base_record.update(
                    {
                        "status": "regular_upper_boundary",
                        "boundary": boundary,
                        "stopping_side_gap_slope": slope,
                        "floor_probability": distribution.floor_probability,
                        "diagonal_coefficient": coefficients.diagonal,
                        "copied_prefix": coefficients.copied_prefix,
                        "prefix_coefficient": coefficients.prefix,
                        "copied_share_of_prefix": (
                            coefficients.copied_prefix / coefficients.prefix
                        ),
                        "prefix_uplift_over_diagonal": (
                            coefficients.copied_prefix / coefficients.diagonal
                        ),
                        "epsilon": args.epsilon,
                        "actual_scaled_loss": scaled,
                        "actual_loss_per_100k_guarantee": regret.total * 100_000,
                        "absolute_diagonal_error": abs(
                            scaled - coefficients.diagonal
                        ),
                        "absolute_prefix_error": abs(scaled - coefficients.prefix),
                    }
                )
            except Exception as error:
                base_record.update(
                    {
                        "status": "failed",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
            pilot_records.append(base_record)

    conditions = pd.DataFrame(condition_records)
    pilots = pd.DataFrame(pilot_records)
    conditions.to_csv(output_dir / "full_panel_contract_conditions.csv", index=False)
    pilots.to_csv(output_dir / "pilot_contract_results.csv", index=False)

    benchmark_conditions = conditions.loc[conditions["case"].eq("benchmark")]
    benchmark_positive = set(
        benchmark_conditions.loc[
            benchmark_conditions["positive_asymptotic_surrender_margin"], "date"
        ]
    )
    benchmark_regular = set(
        status.loc[status["status"].eq("regular_upper_boundary"), "date"]
    )
    if benchmark_positive != benchmark_regular:
        raise AssertionError(
            "benchmark asymptotic-margin classification does not reproduce the "
            "validated full-panel boundary dates"
        )

    summary_rows: list[dict] = []
    for case in CASES:
        condition_group = conditions.loc[conditions["case"].eq(case.name)]
        pilot_group = pilots.loc[pilots["case"].eq(case.name)]
        regular = pilot_group.loc[pilot_group["status"].eq("regular_upper_boundary")]
        regular_boundary_count = int(
            pilot_group["status"].str.startswith("regular_").sum()
        )
        summary_rows.append(
            {
                "case": case.name,
                "varied_term": case.varied_term,
                "varied_value": case.varied_value,
                "full_panel_positive_margin_dates": int(
                    condition_group["positive_asymptotic_surrender_margin"].sum()
                ),
                "pilot_regular_boundary_dates": regular_boundary_count,
                "pilot_coefficient_ready_dates": len(regular),
                "pilot_zero_source_density_dates": int(
                    pilot_group["status"].eq(
                        "regular_boundary_zero_source_density"
                    ).sum()
                ),
                "pilot_failed_dates": int(pilot_group["status"].eq("failed").sum()),
                "median_pilot_copied_share": regular[
                    "copied_share_of_prefix"
                ].median(),
                "median_pilot_prefix_uplift": regular[
                    "prefix_uplift_over_diagonal"
                ].median(),
                "median_pilot_loss_per_100k": regular[
                    "actual_loss_per_100k_guarantee"
                ].median(),
                "prefix_closer_share": (
                    regular["absolute_prefix_error"]
                    < regular["absolute_diagonal_error"]
                ).mean(),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "table_contract_sensitivity.csv", index=False)
    (output_dir / "table_contract_sensitivity.tex").write_text(
        summary.to_latex(index=False, float_format=lambda value: f"{value:.4f}"),
        encoding="utf-8",
    )

    regular_pilots = pilots.loc[pilots["status"].eq("regular_upper_boundary")]
    report = f"""# Contract-parameter sensitivity

- One-at-a-time cases: {len(CASES)} (benchmark plus two values for participation, cap, surrender haircut, and annual termination probability).
- Market dates per case: {len(panel)}; predeclared pilot dates per case: {len(pilot_dates)}.
- Benchmark positive-margin dates: {int(benchmark_conditions['positive_asymptotic_surrender_margin'].sum())}; this exactly reproduces the 60 validated regular-boundary dates.
- Pilot Bellman failures: {int(pilots['status'].eq('failed').sum())}.
- Regular pilot case/date solves: {int(pilots['status'].str.startswith('regular_').sum())}; coefficient-ready at the fixed `x0=0.96b`: {len(regular_pilots)}.
- Regular cases with zero first-date boundary density at `x0=0.96b`: {int(pilots['status'].eq('regular_boundary_zero_source_density').sum())}.
- Prefix closer than diagonal at epsilon={args.epsilon}: {int((regular_pilots['absolute_prefix_error'] < regular_pilots['absolute_diagonal_error']).sum())} of {len(regular_pilots)}.
- Copied share over regular pilot cases: median {regular_pilots['copied_share_of_prefix'].median():.6f}, range [{regular_pilots['copied_share_of_prefix'].min():.6f}, {regular_pilots['copied_share_of_prefix'].max():.6f}].

The full-panel column is an analytic high-state surrender-margin diagnostic, not
a substitute for solving every counterfactual Bellman problem. The benchmark
identity with the validated 60-date classification checks this diagnostic in the
central specification. Bellman solutions and finite-perturbation losses are
recomputed only on the 12 market-state pilot dates selected before viewing any
regret outcome. Contract maturity is not varied because the experiment is an
infinite/life-contingent stationary contract with an exact stationary tail, not
an ordinary finite-maturity product.
"""
    (output_dir / "summary.md").write_text(report, encoding="utf-8")
    print(output_dir)
    return 1 if pilots["status"].eq("failed").any() else 0


if __name__ == "__main__":
    raise SystemExit(main())
