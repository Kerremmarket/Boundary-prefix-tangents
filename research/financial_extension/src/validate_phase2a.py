#!/usr/bin/env python3
"""Validate the frozen Phase II-A evidence and gated classification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    evidence = args.evidence_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    schedule = pd.read_csv(evidence / "schedule_grid_refinement.csv")
    coefficients = pd.read_csv(evidence / "multidate_coefficients.csv")
    convergence = pd.read_csv(evidence / "epsilon_convergence.csv")
    occupancy = pd.read_csv(evidence / "issue_occupancy.csv")
    lifetime = pd.read_csv(evidence / "lifetime_sequence.csv")
    renewal = pd.read_csv(evidence / "renewal_cap_control.csv")
    joint = pd.read_csv(evidence / "joint_grid_epsilon_refinement.csv")
    carrier = pd.read_csv(evidence / "carrier_density_reconciliation.csv")
    parameters = json.loads((evidence / "parameters.json").read_text(encoding="utf-8"))

    assert len(schedule) == 648
    assert set(schedule["grid_points"]) == {30_000, 60_000, 120_000}
    assert len(coefficients) == 168
    assert len(convergence) == 840
    assert len(occupancy) == 35
    assert len(lifetime) == 350
    assert len(renewal) == 35
    assert len(joint.loc[joint["status"].eq("regular_upper_boundary")]) == 140
    assert set(joint["grid_points"]) == {30_000, 60_000, 120_000, 240_000}

    coefficient_keys = [
        "date",
        "product",
        "initialization",
        "horizon",
        "direction",
    ]
    convergence_keys = coefficient_keys + ["epsilon"]
    assert not coefficients.duplicated(coefficient_keys).any()
    assert not convergence.duplicated(convergence_keys).any()

    local = coefficients.loc[
        coefficients["initialization"].eq("local_rho_boundary")
    ]
    issue = coefficients.loc[coefficients["initialization"].eq("issue_state")]
    assert len(local) == 84 and (local["full_coefficient"] > 0).all()
    assert len(issue) == 84 and (issue["full_coefficient"] == 0).all()
    assert (
        convergence.loc[
            convergence["initialization"].eq("issue_state"), "actual_loss"
        ]
        == 0
    ).all()
    assert (occupancy["continuous_density_at_boundary"] == 0).all()
    assert (occupancy["survivor_mass_from_issue"] == 0).all()

    coherent_error = float(
        np.max(np.abs(coefficients["coherent_minus_full"]))
    )
    assert coherent_error < 1e-14
    additive_overcounts = int(
        (coefficients["additive_minus_full"] > 1e-12).sum()
    )
    assert additive_overcounts == 72
    maximum_carrier_error = float(carrier["relative_difference"].max())
    assert maximum_carrier_error < 1e-12

    smallest_120 = convergence.loc[
        convergence["initialization"].eq("local_rho_boundary")
        & convergence["epsilon"].eq(0.00125)
    ]
    maximum_120_error = float(smallest_120["relative_full_error"].max())
    assert maximum_120_error < 0.02
    smallest_joint = joint.loc[
        joint["status"].eq("regular_upper_boundary")
        & joint["epsilon"].eq(0.00125)
    ]
    smallest_240 = smallest_joint.loc[smallest_joint["grid_points"].eq(240_000)]
    maximum_240_error = float(smallest_240["relative_full_error"].max())
    assert maximum_240_error < 0.01
    scaled = smallest_joint.pivot_table(
        index=["date", "product"],
        columns="grid_points",
        values="actual_scaled_loss",
    )
    finest_relative_range = float(
        ((scaled[240_000] - scaled[120_000]).abs() / scaled[240_000].abs()).max()
    )
    assert finest_relative_range < 0.02

    final_lifetime = lifetime.loc[lifetime["decision"].eq(50)]
    maximum_final_increment = float(final_lifetime["coefficient_increment"].max())
    assert maximum_final_increment < 1e-18
    minimum_cap_boundary_move = float(renewal["boundary_change"].dropna().abs().min())
    assert minimum_cap_boundary_move > 0.02

    assert parameters["gate_classification"] == "APPLICATION REMAINS ILLUSTRATIVE"
    summary = (evidence / "summary.md").read_text(encoding="utf-8")
    assert "**APPLICATION REMAINS ILLUSTRATIVE**" in summary
    assert "Phase II-B is not authorized" in summary

    metrics = {
        "schedule_rows": len(schedule),
        "coefficient_rows": len(coefficients),
        "convergence_rows": len(convergence),
        "local_positive_rows": int((local["full_coefficient"] > 0).sum()),
        "issue_positive_rows": int((issue["full_coefficient"] > 0).sum()),
        "issue_maximum_loss": float(
            convergence.loc[
                convergence["initialization"].eq("issue_state"), "actual_loss"
            ].max()
        ),
        "coherent_maximum_error": coherent_error,
        "additive_overcount_rows": additive_overcounts,
        "carrier_maximum_relative_error": maximum_carrier_error,
        "maximum_120k_small_epsilon_error": maximum_120_error,
        "maximum_240k_small_epsilon_error": maximum_240_error,
        "maximum_120k_to_240k_scaled_loss_relative_range": finest_relative_range,
        "maximum_date50_increment": maximum_final_increment,
        "minimum_renewal_cap_boundary_move": minimum_cap_boundary_move,
        "gate_classification": parameters["gate_classification"],
    }
    (output / "validation_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = f"""# Phase II-A validation report

## Status

All encoded evidence invariants pass.

## Evidence shape and keys

- Schedule rows: {metrics['schedule_rows']} on 30k, 60k, and 120k grids.
- Coefficient rows: {metrics['coefficient_rows']}; unique product/date,
  initialization, horizon, and direction keys.
- Direct-repricing rows: {metrics['convergence_rows']}; unique coefficient keys
  plus epsilon.
- Focused joint refinement includes 30k, 60k, 120k, and 240k grids.

## Mathematical and numerical invariants

- Nonzero local rows: {metrics['local_positive_rows']} of 84.
- Nonzero issue rows: {metrics['issue_positive_rows']} of 84.
- Maximum issue loss: {metrics['issue_maximum_loss']:.3g}.
- Maximum coherent-pairwise/full difference:
  {metrics['coherent_maximum_error']:.3g}.
- Additive-pairwise overcount rows: {metrics['additive_overcount_rows']}.
- Maximum labelled-carrier/aggregate left-trace relative difference:
  {metrics['carrier_maximum_relative_error']:.3g}.
- Maximum 120k relative error at epsilon 0.00125:
  {metrics['maximum_120k_small_epsilon_error']:.4%}.
- Maximum focused 240k relative error at epsilon 0.00125:
  {metrics['maximum_240k_small_epsilon_error']:.4%}.
- Maximum 120k-to-240k scaled-loss relative range:
  {metrics['maximum_120k_to_240k_scaled_loss_relative_range']:.4%}.
- Maximum date-50 coefficient increment:
  {metrics['maximum_date50_increment']:.3g}.

## Gate

The evidence summary and frozen parameters both classify Phase II-A as:

**APPLICATION REMAINS ILLUSTRATIVE**

Phase II-B and Phase II-C are intentionally absent because the frozen
issue-occupancy gate failed.
"""
    (output / "validation_report.md").write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
