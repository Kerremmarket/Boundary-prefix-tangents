#!/usr/bin/env python3
"""Run the frozen Phase II-A multidate and documented-schedule gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from indexed_annuity import (
    ContractSpec,
    CreditingSpec,
    MixtureCreditedDistribution,
    log_state_grid,
)
from market_models import LognormalMixtureParams
from phase2_multidate import (
    ForwardAudit,
    advance_issue_measure,
    component_history_coefficient,
    direct_multidate_regret,
    reference_forward_audit,
    reference_forward_audit_from_measure,
    solve_charge_schedule,
    solve_stationary_with_tail,
    stationary_lifetime_sequence,
)


REGULAR_DATES = ("2006-03-31", "2022-10-31", "2023-05-31")
CONTROL_DATES = ("2008-11-28", "2016-10-31", "2020-03-31")
ALL_DATES = REGULAR_DATES + CONTROL_DATES
GRID_POINTS = (30_000, 60_000, 120_000)
EPSILONS = (0.02, 0.01, 0.005, 0.0025, 0.00125)
DIRECTIONS = {
    "common": (1.0, 1.0, 1.0, 1.0, 1.0),
    "front_loaded": (1.2, 1.1, 1.0, 0.9, 0.8),
    "back_loaded": (0.8, 0.9, 1.0, 1.1, 1.2),
    "alternating": (1.0, 0.6, 1.2, 0.8, 1.1),
}


@dataclass(frozen=True)
class ProductProjection:
    code: str
    name: str
    participation: float
    cap: float
    decision_charges: tuple[float, ...]
    charged_decisions: int

    @property
    def varying_caps(self) -> tuple[float, ...]:
        if self.code in {"A10", "AA7"}:
            return tuple(
                self.cap * (0.75 if index % 2 == 0 else 1.0)
                for index in range(5)
            )
        return (0.04, 0.03, 0.04, 0.03, 0.04)


PRODUCTS = (
    ProductProjection(
        "A10",
        "Athene Accumulator 10",
        1.0,
        0.10,
        (0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.0)
        + (0.0,) * 5,
        9,
    ),
    ProductProjection(
        "AA7",
        "Allianz Accumulation Advantage 7",
        1.0,
        0.08,
        (0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.0) + (0.0,) * 5,
        6,
    ),
    ProductProjection(
        "IP4",
        "MassMutual Ascend Index Protector 4",
        1.0,
        0.03,
        (0.056, 0.056, 0.056, 0.0) + (0.0,) * 5,
        3,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--grid-points",
        type=int,
        nargs="+",
        default=list(GRID_POINTS),
    )
    parser.add_argument("--law-nodes", type=int, default=64)
    parser.add_argument("--integration-nodes", type=int, default=64)
    parser.add_argument("--lifetime-dates", type=int, default=50)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _distribution(
    calibration: pd.Series,
    status: pd.Series,
    product: ProductProjection,
    cap: float | None = None,
) -> MixtureCreditedDistribution:
    mixture = LognormalMixtureParams(
        float(calibration["low_weight"]),
        float(calibration["low_forward_multiplier"]),
        float(calibration["low_volatility"]),
        float(calibration["high_volatility"]),
    )
    return MixtureCreditedDistribution(
        gross_forward=float(status["annual_gross_forward"]),
        mixture=mixture,
        crediting=CreditingSpec(
            participation=product.participation,
            cap=product.cap if cap is None else float(cap),
        ),
    )


def _slice_audit(audit: ForwardAudit, horizon: int) -> ForwardAudit:
    return ForwardAudit(
        predecision=audit.predecision[:horizon],
        survivors=audit.survivors[:horizon],
        fresh_arrival_log_densities=audit.fresh_arrival_log_densities[:horizon],
        propagation_mass_errors=audit.propagation_mass_errors[: max(0, horizon - 1)],
        atom_boundary_collisions=tuple(
            item for item in audit.atom_boundary_collisions if item[0] <= horizon
        ),
    )


def _status_for_slice(item) -> str:
    if item.high_state_margin <= 0:
        return "no_upper_stopping_tail"
    if item.boundary is None:
        return "positive_margin_no_regular_root"
    return "regular_upper_boundary"


def main() -> int:
    args = parse_args()
    panel_dir = args.panel_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration_path = panel_dir / "annual_calibrations.csv"
    status_path = panel_dir / "date_status.csv"
    calibrations = pd.read_csv(calibration_path).set_index("date")
    statuses = pd.read_csv(status_path).set_index("date")
    missing = sorted(set(ALL_DATES) - set(calibrations.index))
    if missing:
        raise ValueError(f"frozen dates are absent from calibrations: {missing}")

    grid_sizes = tuple(sorted(set(args.grid_points)))
    if max(grid_sizes) < 120_000:
        raise ValueError("the final Phase II-A run requires at least 120,000 points")
    schedule_records: list[dict] = []
    coefficient_records: list[dict] = []
    convergence_records: list[dict] = []
    occupancy_records: list[dict] = []
    lifetime_records: list[dict] = []
    varying_cap_records: list[dict] = []
    reference_objects: dict[tuple[str, str], tuple] = {}

    for grid_points in grid_sizes:
        grid = log_state_grid(0.03, 20.0, grid_points)
        for date in ALL_DATES:
            calibration = calibrations.loc[date]
            status = statuses.loc[date]
            beta = float(status["discount_factor"])
            for product in PRODUCTS:
                distribution = _distribution(calibration, status, product)
                law = distribution.discretize(args.law_nodes)
                tail_margin_at_stop = 1.0 - beta * law.expected_factor
                discounted_survival_growth = (
                    beta * (1.0 - 0.03) * law.expected_factor
                )
                if tail_margin_at_stop < 0 and discounted_survival_growth >= 1.0:
                    for decision, charge in enumerate(
                        product.decision_charges, start=1
                    ):
                        schedule_records.append(
                            {
                                "grid_points": grid_points,
                                "date": date,
                                "date_class": (
                                    "regular_pilot"
                                    if date in REGULAR_DATES
                                    else "control"
                                ),
                                "product": product.code,
                                "decision": decision,
                                "charge": charge,
                                "postcharge": decision > product.charged_decisions,
                                "floor_probability": distribution.floor_probability,
                                "cap_probability": distribution.cap_probability,
                                "expected_factor": law.expected_factor,
                                "tail_margin": tail_margin_at_stop,
                                "high_state_margin": -np.inf,
                                "status": "divergent_continuation_projection",
                                "boundary": None,
                                "stopping_side_gap_slope": None,
                            }
                        )
                    continue
                terminal_contract = ContractSpec(
                    discount_factor=beta,
                    termination_probability=0.03,
                    surrender_haircut=0.0,
                    death_guarantee=1.0,
                )
                tail, tail_slope, tail_margin = solve_stationary_with_tail(
                    grid=grid,
                    law=law,
                    contract=terminal_contract,
                    tolerance=1e-11,
                )
                schedule = solve_charge_schedule(
                    grid=grid,
                    transition_laws=[law] * len(product.decision_charges),
                    charges=product.decision_charges,
                    discount_factor=beta,
                    termination_probability=0.03,
                    death_guarantee=1.0,
                    terminal_value=tail.value,
                    terminal_high_state_slope=tail_slope,
                )
                for decision, item in enumerate(schedule.slices, start=1):
                    schedule_records.append(
                        {
                            "grid_points": grid_points,
                            "date": date,
                            "date_class": (
                                "regular_pilot" if date in REGULAR_DATES else "control"
                            ),
                            "product": product.code,
                            "decision": decision,
                            "charge": item.charge,
                            "postcharge": decision > product.charged_decisions,
                            "floor_probability": distribution.floor_probability,
                            "cap_probability": distribution.cap_probability,
                            "expected_factor": law.expected_factor,
                            "tail_margin": tail_margin,
                            "high_state_margin": item.high_state_margin,
                            "status": _status_for_slice(item),
                            "boundary": item.boundary,
                            "stopping_side_gap_slope": item.stopping_side_gap_slope,
                        }
                    )
                if grid_points == max(grid_sizes):
                    reference_objects[(date, product.code)] = (
                        grid,
                        distribution,
                        law,
                        tail,
                        tail_slope,
                        tail_margin,
                        schedule,
                    )

    for date in REGULAR_DATES:
        calibration = calibrations.loc[date]
        status = statuses.loc[date]
        beta = float(status["discount_factor"])
        discount_survival = beta * 0.97
        for product in PRODUCTS:
            (
                grid,
                distribution,
                law,
                tail,
                tail_slope,
                tail_margin,
                schedule,
            ) = reference_objects[(date, product.code)]
            start = product.charged_decisions
            window_slices = schedule.slices[start : start + 5]
            if len(window_slices) < 5 or any(item.boundary is None for item in window_slices):
                continue
            boundaries = tuple(float(item.boundary) for item in window_slices)
            slopes = tuple(float(item.stopping_side_gap_slope) for item in window_slices)
            gaps = tuple(item.gap for item in window_slices)
            local_initial_state = 0.96 * boundaries[0]
            local_audit = reference_forward_audit(
                log_grid=np.log(grid),
                initial_state=local_initial_state,
                initial_distribution=distribution,
                transition_distributions=[distribution] * 4,
                boundaries=boundaries,
            )
            issue_measure, preceding_survival = advance_issue_measure(
                log_grid=np.log(grid),
                initial_state=1.0,
                credit_distributions=[distribution] * (start + 1),
                preceding_boundaries=[item.boundary for item in schedule.slices[:start]],
            )
            issue_audit = reference_forward_audit_from_measure(
                initial_predecision_measure=issue_measure,
                transition_distributions=[distribution] * 4,
                boundaries=boundaries,
            )
            for decision, (pre, survivor) in enumerate(
                zip(issue_audit.predecision, issue_audit.survivors),
                start=start + 1,
            ):
                occupancy_records.append(
                    {
                        "date": date,
                        "product": product.code,
                        "decision": decision,
                        "boundary": boundaries[decision - start - 1],
                        "predecision_mass_from_issue": pre.mass,
                        "continuous_density_at_boundary": float(
                            np.interp(
                                np.log(boundaries[decision - start - 1]),
                                pre.log_grid,
                                pre.continuous_density,
                                left=0.0,
                                right=0.0,
                            )
                            / boundaries[decision - start - 1]
                        ),
                        "survivor_mass_from_issue": survivor.mass,
                        "minimum_reachable_state": 1.0,
                    }
                )
            for horizon in (3, 4, 5):
                for direction_name, relative_direction in DIRECTIONS.items():
                    relative = relative_direction[:horizon]
                    active_boundaries = boundaries[:horizon]
                    active_slopes = slopes[:horizon]
                    shifts = tuple(
                        relative[index] * active_boundaries[index]
                        for index in range(horizon)
                    )
                    for initialization, audit, start_power in (
                        ("local_rho_boundary", _slice_audit(local_audit, horizon), 1),
                        (
                            "issue_state",
                            _slice_audit(issue_audit, horizon),
                            start + 1,
                        ),
                    ):
                        coefficient = component_history_coefficient(
                            audit=audit,
                            boundaries=active_boundaries,
                            slopes=active_slopes,
                            shifts=shifts,
                            discount_survival=discount_survival,
                            floor_probabilities=[distribution.floor_probability]
                            * (horizon - 1),
                            alignment_groups=["postcharge_stationary"] * horizon,
                            discount_start_power=start_power,
                        )
                        coefficient_records.append(
                            {
                                "date": date,
                                "product": product.code,
                                "initialization": initialization,
                                "horizon": horizon,
                                "direction": direction_name,
                                "boundary": active_boundaries[0],
                                "ordinary_coefficient": coefficient.ordinary,
                                "ordered_prefix_coefficient": coefficient.resonant,
                                "full_coefficient": coefficient.full,
                                "copied_share": coefficient.copied_share,
                                "additive_pairwise": coefficient.additive_pairwise,
                                "coherent_labeled_pairwise": coefficient.coherent_pairwise,
                                "coherent_minus_full": (
                                    coefficient.coherent_pairwise - coefficient.full
                                ),
                                "additive_minus_full": (
                                    coefficient.additive_pairwise - coefficient.full
                                ),
                                "carrier_count": len(coefficient.carriers),
                                "resonant_carrier_count": sum(
                                    item.classification == "resonant"
                                    for item in coefficient.carriers
                                ),
                                "atom_boundary_collisions": len(
                                    audit.atom_boundary_collisions
                                ),
                            }
                        )
                        for epsilon in EPSILONS:
                            direct = direct_multidate_regret(
                                log_grid=np.log(grid),
                                state_grid=grid,
                                initial_state=local_initial_state,
                                initial_distribution=distribution,
                                transition_distributions=[distribution]
                                * (horizon - 1),
                                boundaries=active_boundaries,
                                gaps=gaps[:horizon],
                                shifts=shifts,
                                epsilon=epsilon,
                                discount_survival=discount_survival,
                                integration_nodes=args.integration_nodes,
                                discount_start_power=start_power,
                                initial_predecision_measure=(
                                    issue_measure
                                    if initialization == "issue_state"
                                    else None
                                ),
                            )
                            scaled = direct.total / epsilon**2
                            convergence_records.append(
                                {
                                    "date": date,
                                    "product": product.code,
                                    "initialization": initialization,
                                    "horizon": horizon,
                                    "direction": direction_name,
                                    "epsilon": epsilon,
                                    "actual_loss": direct.total,
                                    "actual_loss_per_100k": direct.total * 100_000,
                                    "actual_scaled_loss": scaled,
                                    "full_coefficient": coefficient.full,
                                    "absolute_full_error": abs(
                                        scaled - coefficient.full
                                    ),
                                    "relative_full_error": (
                                        abs(scaled - coefficient.full)
                                        / coefficient.full
                                        if coefficient.full > 0
                                        else 0.0 if scaled == 0 else np.inf
                                    ),
                                    "ordinary_error": abs(
                                        scaled - coefficient.ordinary
                                    ),
                                    "additive_pairwise_error": abs(
                                        scaled - coefficient.additive_pairwise
                                    ),
                                    "coherent_pairwise_error": abs(
                                        scaled - coefficient.coherent_pairwise
                                    ),
                                }
                            )

            lifetime = stationary_lifetime_sequence(
                log_grid=np.log(grid),
                initial_state=local_initial_state,
                distribution=distribution,
                boundary=boundaries[0],
                slope=slopes[0],
                shift=boundaries[0],
                discount_survival=discount_survival,
                maximum_dates=args.lifetime_dates,
            )
            diagonal_cumulative = 0.0
            for index in range(args.lifetime_dates):
                date_number = index + 1
                diagonal_increment = (
                    0.5
                    * slopes[0]
                    * boundaries[0] ** 2
                    * discount_survival**date_number
                    * lifetime.fresh_boundary_densities[index]
                )
                diagonal_cumulative += diagonal_increment
                lifetime_records.append(
                    {
                        "date": date,
                        "product": product.code,
                        "decision": date_number,
                        "fresh_boundary_density": lifetime.fresh_boundary_densities[
                            index
                        ],
                        "terminal_floor_run_density": (
                            lifetime.terminal_floor_run_densities[index]
                        ),
                        "coefficient_increment": lifetime.coefficient_increments[index],
                        "cumulative_full_coefficient": (
                            lifetime.cumulative_coefficients[index]
                        ),
                        "cumulative_fresh_diagonal": diagonal_cumulative,
                        "cumulative_copied_share": (
                            1.0
                            - diagonal_cumulative
                            / lifetime.cumulative_coefficients[index]
                            if lifetime.cumulative_coefficients[index] > 0
                            else 0.0
                        ),
                        "predecision_mass": lifetime.predecision_masses[index],
                        "survivor_mass": lifetime.survivor_masses[index],
                    }
                )

            varying_distributions = [
                _distribution(calibration, status, product, cap=cap)
                for cap in product.varying_caps
            ]
            varying_laws = [
                item.discretize(args.law_nodes) for item in varying_distributions
            ]
            cap_control = solve_charge_schedule(
                grid=grid,
                transition_laws=varying_laws,
                charges=[0.0] * 5,
                discount_factor=beta,
                termination_probability=0.03,
                death_guarantee=1.0,
                terminal_value=tail.value,
                terminal_high_state_slope=tail_slope,
            )
            previous_boundary = None
            for decision, (cap, item) in enumerate(
                zip(product.varying_caps, cap_control.slices), start=1
            ):
                varying_cap_records.append(
                    {
                        "date": date,
                        "product": product.code,
                        "decision": decision,
                        "next_credit_cap": cap,
                        "high_state_margin": item.high_state_margin,
                        "status": _status_for_slice(item),
                        "boundary": item.boundary,
                        "boundary_change": (
                            None
                            if item.boundary is None or previous_boundary is None
                            else item.boundary - previous_boundary
                        ),
                        "structurally_aligned_with_previous": False,
                    }
                )
                previous_boundary = item.boundary

    schedule_frame = pd.DataFrame(schedule_records)
    coefficient_frame = pd.DataFrame(coefficient_records)
    convergence_frame = pd.DataFrame(convergence_records)
    occupancy_frame = pd.DataFrame(occupancy_records)
    lifetime_frame = pd.DataFrame(lifetime_records)
    varying_frame = pd.DataFrame(varying_cap_records)
    schedule_frame.to_csv(output_dir / "schedule_grid_refinement.csv", index=False)
    coefficient_frame.to_csv(output_dir / "multidate_coefficients.csv", index=False)
    convergence_frame.to_csv(output_dir / "epsilon_convergence.csv", index=False)
    occupancy_frame.to_csv(output_dir / "issue_occupancy.csv", index=False)
    lifetime_frame.to_csv(output_dir / "lifetime_sequence.csv", index=False)
    varying_frame.to_csv(output_dir / "renewal_cap_control.csv", index=False)

    local_coefficients = coefficient_frame.loc[
        coefficient_frame["initialization"].eq("local_rho_boundary")
    ]
    issue_coefficients = coefficient_frame.loc[
        coefficient_frame["initialization"].eq("issue_state")
    ]
    smallest_epsilon = convergence_frame.loc[
        convergence_frame["epsilon"].eq(min(EPSILONS))
        & convergence_frame["initialization"].eq("local_rho_boundary")
    ]
    maximum_relative_error = float(smallest_epsilon["relative_full_error"].max())
    nonzero_local = int((local_coefficients["full_coefficient"] > 0).sum())
    nonzero_issue = int((issue_coefficients["full_coefficient"] > 0).sum())
    coherent_error = float(
        np.max(np.abs(coefficient_frame["coherent_minus_full"]))
    )
    additive_overcounts = int(
        (coefficient_frame["additive_minus_full"] > 1e-12).sum()
    )
    gate_classification = (
        "PROCEED TO COMPUTATIONAL ROBUSTNESS"
        if nonzero_issue > 0 and maximum_relative_error < 0.05
        else "APPLICATION REMAINS ILLUSTRATIVE"
    )
    summary = f"""# Phase II-A gate result

## Outcome

**{gate_classification}**

- Documented product/date projections with a regular five-date post-charge
  window: {coefficient_frame[['date', 'product']].drop_duplicates().shape[0]}.
- Nonzero local horizon/direction coefficients: {nonzero_local}.
- Nonzero issue-state horizon/direction coefficients: {nonzero_issue}.
- Maximum relative full-coefficient error at epsilon={min(EPSILONS):g} among
  nonzero local runs: {maximum_relative_error:.6g}.
- Maximum coherent-labeled-pairwise minus full coefficient:
  {coherent_error:.6g}.
- Additive-pairwise overcounts across reported rows: {additive_overcounts}.

## Gate interpretation

The fixed-cap, zero-MVA scalar projection has exact stationary post-charge
boundaries and therefore a theorem-valid multidate carrier under the local
`X0=0.96b` diagnostic.  It does not survive the economic-occupancy requirement:
the account factor is never below one, while every regular post-charge boundary
in the documented projections is below one.  A10 and AA7 have no upper stopping
tail during their charged decisions on the reported cases and stop all issue
paths when the first post-charge boundary appears; IP4 either stops all issue
paths at its first regular charged decision or has the same below-one problem.
Consequently the issue-state boundary trace and the finite perturbation loss
are zero.

The scalar carrier is exactly reconstructible from coherently labeled pairwise
minima.  Additive pairwise summation overcounts longer floor runs.  Varying
renewal caps move the boundary normal, while a rate-dependent MVA adds a second
state that the account floor does not copy.  These controls prevent the local
fixed-cap calculation from being interpreted as a complete product result.

Phase II-B is not authorized by the frozen gate.
"""
    (output_dir / "summary.md").write_text(summary, encoding="utf-8")
    parameters = {
        "regular_dates": list(REGULAR_DATES),
        "control_dates": list(CONTROL_DATES),
        "grid_points": list(grid_sizes),
        "law_nodes": args.law_nodes,
        "integration_nodes": args.integration_nodes,
        "epsilons": list(EPSILONS),
        "directions": {key: list(value) for key, value in DIRECTIONS.items()},
        "lifetime_dates": args.lifetime_dates,
        "annual_calibrations_sha256": _sha256(calibration_path),
        "date_status_sha256": _sha256(status_path),
        "gate_classification": gate_classification,
    }
    (output_dir / "parameters.json").write_text(
        json.dumps(parameters, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
