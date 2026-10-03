"""Validate the frozen cliquet pilot outputs and baseline isolation."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
RESULTS = ROOT / "results"
TABLES = ROOT / "tables"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def check(name: str, condition: bool, detail: object) -> dict[str, object]:
    return {"name": name, "pass": bool(condition), "detail": detail}


def run() -> None:
    config = json.loads((ROOT / "config.json").read_text())
    metadata = json.loads((RESULTS / "run_metadata.json").read_text())
    summary = json.loads((RESULTS / "analysis_summary.json").read_text())
    boundaries = pd.read_csv(RESULTS / "boundaries.csv")
    traces = pd.read_csv(RESULTS / "trace_audit.csv")
    mechanism = pd.read_csv(RESULTS / "mechanism_results.csv")
    mechanism_dates = pd.read_csv(RESULTS / "mechanism_by_date.csv")
    policy_occupancy = pd.read_csv(RESULTS / "policy_occupancy.csv")
    independent = pd.read_csv(RESULTS / "independent_checks.csv")
    finance = pd.read_csv(RESULTS / "finance_valuation.csv")
    certified_mechanism = pd.read_csv(TABLES / "mechanism_certified.csv")
    certified_finance = pd.read_csv(TABLES / "finance_certified.csv")

    validations: list[dict[str, object]] = []
    warnings: list[dict[str, object]] = []

    observed_hashes = {
        relative: digest((ROOT / relative).resolve())
        for relative in config["input_hashes"]
    }
    validations.append(
        check(
            "input_hashes",
            observed_hashes == metadata["observed_input_hashes"],
            observed_hashes,
        )
    )
    validations.append(
        check(
            "run_config_hash",
            metadata["config_sha256"] == digest(ROOT / "config.json"),
            metadata["config_sha256"],
        )
    )
    validations.append(
        check(
            "frozen_case_count",
            metadata["case_count"] == 30,
            metadata["case_count"],
        )
    )
    validations.append(
        check(
            "deterministic_only",
            metadata["monte_carlo_used"] is False,
            metadata["monte_carlo_used"],
        )
    )
    validations.append(
        check(
            "wall_clock_limit",
            metadata["wall_seconds"]
            <= 60 * float(config["numerics"]["wall_clock_minutes"]),
            metadata["wall_seconds"],
        )
    )

    from reproduction_support import validate_source_manifest, validate_reproduction_file
    validate_source_manifest(REPO)
    for relative, expected in config["input_hashes"].items():
        validate_reproduction_file((ROOT / relative).resolve(), expected, REPO)
    validations.append(check("public_source_integrity", True, "SOURCE_MANIFEST.sha256"))

    expected_counts = metadata["outcome_row_counts"]
    observed_counts = {
        "boundaries.csv": len(boundaries),
        "trace_audit.csv": len(traces),
        "mechanism_results.csv": len(mechanism),
        "mechanism_by_date.csv": len(mechanism_dates),
        "policy_occupancy.csv": len(policy_occupancy),
        "independent_checks.csv": len(independent),
        "finance_valuation.csv": len(finance),
        "coefficient_components.csv": len(
            pd.read_csv(RESULTS / "coefficient_components.csv")
        ),
        "trace_components.csv": len(
            pd.read_csv(RESULTS / "trace_components.csv")
        ),
        "timings.csv": len(pd.read_csv(RESULTS / "timings.csv")),
    }
    validations.append(
        check(
            "raw_row_counts",
            observed_counts == expected_counts,
            observed_counts,
        )
    )

    grid_width = boundaries["local_cap"] / boundaries[
        "intervals_per_local_cap"
    ]
    max_lattice_widths = float(
        (boundaries["lattice_boundary_error"].abs() / grid_width).max()
    )
    max_quadrature_widths = float(
        (boundaries["quadrature_boundary_error"].abs() / grid_width).max()
    )
    width_tolerance = float(
        config["numerics"]["boundary_grid_widths_tolerance"]
    )
    validations.append(
        check(
            "analytic_boundary_reconciliation",
            max(max_lattice_widths, max_quadrature_widths) <= width_tolerance,
            {
                "lattice_max_grid_widths": max_lattice_widths,
                "quadrature_max_grid_widths": max_quadrature_widths,
                "tolerance": width_tolerance,
            },
        )
    )
    expectation_error = float(boundaries["expectation_error"].abs().max())
    validations.append(
        check(
            "analytic_credit_moment",
            expectation_error < 1e-12,
            expectation_error,
        )
    )
    validations.append(
        check(
            "all_observed_rates_imply_beta_below_one",
            bool((boundaries["beta"] < 1).all()),
            {
                "minimum_beta": float(boundaries["beta"].min()),
                "maximum_beta": float(boundaries["beta"].max()),
            },
        )
    )
    minimum_slope = float(
        boundaries[["right_gap_slope", "left_gap_slope"]].min().min()
    )
    validations.append(
        check("positive_one_sided_slopes", minimum_slope > 0, minimum_slope)
    )
    minimum_collision_distance = float(
        boundaries["accumulator_atom_distance"].min()
    )
    validations.append(
        check(
            "no_boundary_atom_collision",
            minimum_collision_distance
            > float(config["numerics"]["atom_collision_tolerance"]),
            minimum_collision_distance,
        )
    )
    trace_error = float(traces["reconciliation_error"].max())
    validations.append(
        check(
            "trace_component_reconciliation",
            trace_error
            <= float(
                config["numerics"]["component_reconciliation_absolute_tolerance"]
            ),
            trace_error,
        )
    )
    validations.append(
        check(
            "trace_densities_nonnegative",
            bool(
                (
                    traces[
                        [
                            "total_density",
                            "fresh_density",
                            "combinatorial_density",
                        ]
                    ]
                    >= -1e-12
                ).all().all()
            ),
            float(
                traces[
                    ["total_density", "fresh_density", "combinatorial_density"]
                ].min().min()
            ),
        )
    )

    identity_error = float(
        mechanism["internal_reconciliation_error"].abs().max()
    )
    validations.append(
        check(
            "performance_difference_identity",
            identity_error < 1e-12,
            identity_error,
        )
    )
    occupancy_groups = [
        "date",
        "contract_id",
        "model",
        "level",
        "policy_id",
    ]
    ordered = mechanism_dates.sort_values(occupancy_groups + ["decision_date"])
    first_mass_error = float(
        ordered.groupby(occupancy_groups)["predecision_mass"].first().sub(1.0).abs().max()
    )
    prior_survival = ordered.groupby(occupancy_groups)[
        "perturbed_survival_mass"
    ].shift(1)
    later = ordered["decision_date"] > 1
    transition_mass_error = float(
        (
            ordered.loc[later, "predecision_mass"]
            - prior_survival.loc[later]
        ).abs().max()
    )
    survival_ordered = bool(
        (
            ordered["perturbed_survival_mass"]
            <= ordered["predecision_mass"] + 2e-12
        ).all()
    )
    validations.append(
        check(
            "forward_mass_conservation",
            max(first_mass_error, transition_mass_error) < 2e-12
            and survival_ordered,
            {
                "first_date_error": first_mass_error,
                "transition_error": transition_mass_error,
                "survival_not_above_predecision": survival_ordered,
            },
        )
    )
    occupancy_keys = ["date", "contract_id", "model", "level", "policy_role"]
    policy_ordered = policy_occupancy.sort_values(
        occupancy_keys + ["decision_date"]
    )
    policy_prior_survival = policy_ordered.groupby(occupancy_keys)[
        "survival_mass"
    ].shift(1)
    policy_later = policy_ordered["decision_date"] > 1
    policy_transition_error = float(
        (
            policy_ordered.loc[policy_later, "predecision_mass"]
            - policy_prior_survival.loc[policy_later]
        ).abs().max()
    )
    reference_rows = policy_ordered[
        policy_ordered["policy_role"] == "reference_native"
    ]
    validations.append(
        check(
            "reference_and_perturbed_survival_exported",
            set(policy_ordered["policy_role"])
            == {"reference_native", "perturbed_transferred"}
            and policy_transition_error < 2e-12
            and float(reference_rows["mismatch_mass"].abs().max()) < 2e-12,
            {
                "rows": int(len(policy_ordered)),
                "transition_error": policy_transition_error,
                "reference_maximum_mismatch_mass": float(
                    reference_rows["mismatch_mass"].abs().max()
                ),
            },
        )
    )

    independent_fine = independent[independent["level"] == "fine"]
    mechanism_cross_max = float(
        independent_fine["cross_evaluator_error"].abs().max()
    )
    reconciliation_tolerance = float(
        config["numerics"]["value_reconciliation_absolute_tolerance"]
    )
    validations.append(
        check(
            "mechanism_independent_evaluator_target",
            mechanism_cross_max <= reconciliation_tolerance,
            {
                "maximum": mechanism_cross_max,
                "tolerance": reconciliation_tolerance,
            },
        )
    )
    finance_fine = finance[finance["level"] == "fine"].copy()
    finance_fine["cross_abs"] = finance_fine[
        "transfer_cross_evaluator_difference"
    ].abs()
    finance_cross_failures = finance_fine[
        finance_fine["cross_abs"] > reconciliation_tolerance
    ]
    warnings.append(
        {
            "name": "finance_independent_evaluator_target",
            "pass": len(finance_cross_failures) == 0,
            "failure_count": int(len(finance_cross_failures)),
            "maximum": float(finance_fine["cross_abs"].max()),
            "tolerance": reconciliation_tolerance,
            "failed_cases": finance_cross_failures[
                ["date", "contract_id", "reference_model", "cross_abs"]
            ].to_dict(orient="records"),
            "disposition": "retained and absorbed into the pre-specified certification floor",
        }
    )
    cancellation_max = float(
        independent["quadrature_cancellation_diagnostic"].abs().max()
    )
    validations.append(
        check(
            "quadrature_price_subtraction_not_cancellation_limited",
            cancellation_max < 1e-12,
            cancellation_max,
        )
    )
    validations.append(
        check(
            "relative_ratios_suppressed_below_floor",
            bool(
                certified_mechanism.loc[
                    ~certified_mechanism["resolved"],
                    [
                        "full_relative_error",
                        "additive_relative_error",
                        "fresh_relative_error",
                    ],
                ].isna().all().all()
                and certified_finance.loc[
                    ~certified_finance["resolved"],
                    [
                        "loss_fraction_of_premium",
                        "loss_fraction_of_early_exercise_value",
                        "full_approximation_relative_error",
                    ],
                ].isna().all().all()
            ),
            {
                "unresolved_mechanism_rows": int(
                    (~certified_mechanism["resolved"]).sum()
                ),
                "unresolved_finance_rows": int(
                    (~certified_finance["resolved"]).sum()
                ),
            },
        )
    )
    out_of_local_domain = int(
        certified_mechanism["threshold_above_global_cap"].sum()
        + certified_mechanism["threshold_below_zero"].sum()
    )
    validations.append(
        check(
            "finite_scale_failures_retained",
            out_of_local_domain > 0,
            out_of_local_domain,
        )
    )
    validations.append(
        check(
            "frozen_verdict_gates",
            summary["verdict"] == "LIMITED VALUE"
            and summary["occupancy_alignment_gate"]["pass"]
            and summary["mechanism_gate"]["pass"]
            and not summary["finance_gate"]["adopt_pass"],
            summary,
        )
    )

    mandatory_pass = all(item["pass"] for item in validations)
    report = {
        "mandatory_pass": mandatory_pass,
        "warning_count": sum(not item["pass"] for item in warnings),
        "validations": validations,
        "warnings": warnings,
    }
    (RESULTS / "validation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    if not mandatory_pass:
        failed = [item["name"] for item in validations if not item["pass"]]
        raise SystemExit(f"mandatory validation failures: {failed}")


if __name__ == "__main__":
    run()
