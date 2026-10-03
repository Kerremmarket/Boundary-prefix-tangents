"""Run the frozen five-snapshot cliquet feasibility pilot."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

from reproduction_support import validate_reproduction_file, validate_source_manifest

from cliquet import (
    Contract,
    CreditLaw,
    SlopeResult,
    TraceResult,
    build_law,
    forward_performance_difference,
    gate_integral,
    make_directions,
    one_sided_slopes,
    prefix_coefficient,
    quadrature_policy_regret,
    solve_boundary,
    solve_lattice,
    solve_quadrature,
    trace_densities,
)


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
RESULTS = ROOT / "results"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=REPO, text=True
    ).strip()


@dataclass
class PilotCase:
    date: str
    model: str
    contract_id: str
    market: dict[str, object]
    calibration: dict[str, object]
    law: CreditLaw
    contract: Contract
    boundary: float
    slopes: SlopeResult

    @property
    def key(self) -> tuple[str, str, str]:
        return self.date, self.contract_id, self.model


def load_inputs(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    observed_hashes: dict[str, str] = {}
    for relative, expected in config["input_hashes"].items():
        path = (ROOT / relative).resolve()
        observed = sha256(path)
        observed_hashes[relative] = observed
        validate_reproduction_file(path, expected, REPO)
    market = pd.read_csv(ROOT / "inputs/market_snapshots.csv")
    mixture = pd.read_csv(ROOT / "inputs/mixture_calibrations.csv")
    dates = market["date"].astype(str).tolist()
    if dates != config["snapshot_dates"]:
        raise RuntimeError("market snapshot order differs from frozen protocol")
    if set(mixture["date"].astype(str)) != set(dates):
        raise RuntimeError("mixture calibrations do not match the five snapshots")
    if not bool((mixture["status"] == "ok").all() and mixture["success"].all()):
        raise RuntimeError("a frozen mixture calibration is not usable")
    return market, mixture, observed_hashes


def build_cases(
    config: dict,
    market: pd.DataFrame,
    mixture: pd.DataFrame,
) -> dict[tuple[str, str, str], PilotCase]:
    mixture_by_date = {
        str(row["date"]): row.to_dict() for _, row in mixture.iterrows()
    }
    cases: dict[tuple[str, str, str], PilotCase] = {}
    for _, market_row in market.iterrows():
        date = str(market_row["date"])
        market_record = market_row.to_dict()
        calibration = mixture_by_date[date]
        for contract_spec in config["contracts"]:
            local_cap = float(contract_spec["local_cap"])
            global_cap = float(contract_spec["global_cap"])
            resets = int(contract_spec["resets"])
            beta = float(np.exp(-float(market_record["zero_rate"])))
            contract = Contract(local_cap, global_cap, resets, beta)
            for model in config["models"]:
                law = build_law(
                    model=model,
                    gross_forward=float(market_record["annual_gross_forward"]),
                    atm_volatility=float(market_record["atm_iv_1y"]),
                    local_cap=local_cap,
                    mixture_parameters=calibration,
                )
                result = solve_boundary(
                    law,
                    contract,
                    absolute_tolerance=float(
                        config["numerics"]["root_absolute_tolerance"]
                    ),
                )
                if result.status != "regular_interior_boundary" or result.boundary is None:
                    raise RuntimeError(
                        f"{date}/{contract_spec['id']}/{model}: {result.status}"
                    )
                boundary = result.boundary
                slopes = one_sided_slopes(law, contract, boundary)
                case = PilotCase(
                    date=date,
                    model=model,
                    contract_id=str(contract_spec["id"]),
                    market=market_record,
                    calibration=calibration,
                    law=law,
                    contract=contract,
                    boundary=boundary,
                    slopes=slopes,
                )
                cases[case.key] = case
    return cases


def policy_identifier(direction_name: str, eta: float) -> str:
    return f"{direction_name}__eta_{eta:.10f}"


def mechanism_policies(
    case: PilotCase,
    config: dict,
) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    directions = make_directions(case.contract.exercise_dates)
    records: list[dict[str, object]] = []
    for direction_name, direction in directions.items():
        for eta in config["etas"]:
            epsilon = float(eta) * case.contract.local_cap
            records.append(
                {
                    "policy_id": policy_identifier(direction_name, float(eta)),
                    "direction_name": direction_name,
                    "eta": float(eta),
                    "epsilon": epsilon,
                    "direction": direction,
                    "thresholds": case.boundary + epsilon * direction,
                }
            )
    return records, directions


def selected_independent_policy_ids(case: PilotCase, config: dict) -> set[str]:
    etas = [float(value) for value in config["etas"]]
    selected: set[str] = set()
    for name in ("all_up", "all_down", "alternating_up_down"):
        for eta in etas[-2:]:
            selected.add(policy_identifier(name, eta))
    middle = (case.contract.exercise_dates + 1) // 2
    for name in (f"single_up_t{middle}", f"single_down_t{middle}"):
        selected.add(policy_identifier(name, etas[-3]))
    return selected


def level_tasks(config: dict, cases: dict[tuple[str, str, str], PilotCase]):
    for level in config["numerics"]["levels"]:
        for case in cases.values():
            yield (
                str(level["name"]),
                int(level["intervals_per_local_cap"]),
                int(level["quadrature_nodes_per_component"]),
                case,
            )
    sentinel = config["numerics"]["sentinel"]
    for model in config["models"]:
        key = (
            str(sentinel["date"]),
            str(sentinel["contract_id"]),
            model,
        )
        yield (
            "sentinel",
            int(sentinel["intervals_per_local_cap"]),
            int(sentinel["quadrature_nodes_per_component"]),
            cases[key],
        )


def collision_metrics(case: PilotCase) -> tuple[float, float]:
    max_multiple = int(round(case.contract.global_cap / case.contract.local_cap))
    accumulator_distance = min(
        abs(case.boundary - count * case.contract.local_cap)
        for count in range(max_multiple + 1)
    )
    contact = case.contract.global_cap - case.boundary
    credit_contact_distance = min(
        abs(contact), abs(contact - case.contract.local_cap)
    )
    return accumulator_distance, credit_contact_distance


def eligible_trace(
    case: PilotCase,
    trace: TraceResult,
    density_tolerance: float,
    collision_tolerance: float,
) -> tuple[bool, int | None, int | None, float]:
    accumulator_distance, _ = collision_metrics(case)
    if accumulator_distance <= collision_tolerance:
        return False, None, None, 0.0
    # The source must leave at least two later exercise opportunities.  The
    # immediate next date is the first copied target used for this gate.
    for source in range(1, case.contract.exercise_dates - 1):
        target = source + 1
        intensity = (
            trace.fresh_density[source] * case.law.floor_probability
        )
        if (
            trace.fresh_density[source] > density_tolerance
            and intensity > density_tolerance
            and gate_integral(
                np.ones(case.contract.exercise_dates), source - 1, target - 1
            )
            > 0
        ):
            return True, source, target, float(intensity)
    return False, None, None, 0.0


def run() -> None:
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    config_path = ROOT / "config.json"
    config = json.loads(config_path.read_text())
    if not config.get("frozen_before_outcomes"):
        raise RuntimeError("refusing to run an unfrozen protocol")
    pre_outcome_code_commit = "54664b20f3227b56f08777e1602e2cfb5f358f17"
    validate_source_manifest(REPO)
    market, mixture, observed_hashes = load_inputs(config)
    cases = build_cases(config, market, mixture)
    if len(cases) != 30:
        raise RuntimeError(f"expected 30 frozen cases, found {len(cases)}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    boundary_rows: list[dict[str, object]] = []
    trace_rows: list[dict[str, object]] = []
    trace_component_rows: list[dict[str, object]] = []
    mechanism_rows: list[dict[str, object]] = []
    mechanism_date_rows: list[dict[str, object]] = []
    policy_occupancy_rows: list[dict[str, object]] = []
    coefficient_component_rows: list[dict[str, object]] = []
    independent_rows: list[dict[str, object]] = []
    finance_rows: list[dict[str, object]] = []
    timing_rows: list[dict[str, object]] = []

    base_contract_id = "c08_g24_n8"
    density_tolerance = float(
        config["numerics"]["positive_density_tolerance"]
    )
    collision_tolerance = float(
        config["numerics"]["atom_collision_tolerance"]
    )

    for level_name, intervals, quadrature_nodes, case in level_tasks(config, cases):
        case_started = time.perf_counter()
        other_model = next(
            model for model in config["models"] if model != case.model
        )
        other_case = cases[(case.date, case.contract_id, other_model)]
        native_thresholds = np.full(
            case.contract.exercise_dates, case.boundary
        )
        transfer_thresholds = np.full(
            case.contract.exercise_dates, other_case.boundary
        )
        policies: list[dict[str, object]] = [
            {
                "policy_id": "native_threshold",
                "kind": "finance_native",
                "thresholds": native_thresholds,
            },
            {
                "policy_id": "transferred_threshold",
                "kind": "finance_transfer",
                "thresholds": transfer_thresholds,
            },
        ]
        directions: dict[str, np.ndarray] = {}
        mechanism: list[dict[str, object]] = []
        if case.contract_id == base_contract_id:
            mechanism, directions = mechanism_policies(case, config)
            for record in mechanism:
                policies.append(
                    {
                        "policy_id": record["policy_id"],
                        "kind": "mechanism",
                        "thresholds": record["thresholds"],
                    }
                )

        threshold_matrix = np.vstack(
            [np.asarray(record["thresholds"], dtype=float) for record in policies]
        )
        lattice = solve_lattice(
            case.law,
            case.contract,
            intervals_per_cap=intervals,
            policy_thresholds=threshold_matrix,
        )
        if lattice.policy_issue_values is None:
            raise AssertionError("policy batch was not evaluated")
        forward = forward_performance_difference(
            lattice,
            case.contract,
            optimal_boundary=case.boundary,
            policy_thresholds=threshold_matrix,
        )

        independent_ids = selected_independent_policy_ids(case, config)
        quadrature_policy_indices = [0, 1]
        if mechanism:
            quadrature_policy_indices.extend(
                index
                for index, record in enumerate(policies)
                if str(record["policy_id"]) in independent_ids
            )
        quadrature_thresholds = threshold_matrix[quadrature_policy_indices]
        quadrature = solve_quadrature(
            case.law,
            case.contract,
            intervals_per_cap=intervals,
            nodes_per_component=quadrature_nodes,
            policy_thresholds=quadrature_thresholds,
        )
        if quadrature.policy_issue_values is None:
            raise AssertionError("quadrature policy batch was not evaluated")
        quadrature_direct_regrets = quadrature_policy_regret(
            quadrature,
            case.contract,
            policy_thresholds=quadrature_thresholds,
        )
        quadrature_by_policy = {
            str(policies[index]["policy_id"]): float(value)
            for index, value in zip(
                quadrature_policy_indices, quadrature.policy_issue_values
            )
        }
        quadrature_regret_by_policy = {
            str(policies[index]["policy_id"]): float(value)
            for index, value in zip(
                quadrature_policy_indices, quadrature_direct_regrets
            )
        }

        trace = trace_densities(
            case.law,
            case.contract,
            boundary=case.boundary,
            intervals_per_cap=intervals,
        )
        eligible, eligible_source, eligible_target, eligible_intensity = eligible_trace(
            case,
            trace,
            density_tolerance,
            collision_tolerance,
        )
        accumulator_distance, contact_distance = collision_metrics(case)
        credits, weights = case.law.quadrature(quadrature_nodes)
        expectation_quadrature = float(np.dot(credits, weights))

        for date in range(1, case.contract.exercise_dates + 1):
            remaining = case.contract.resets - date
            analytic_boundary = case.boundary
            boundary_rows.append(
                {
                    "date": case.date,
                    "contract_id": case.contract_id,
                    "model": case.model,
                    "level": level_name,
                    "intervals_per_local_cap": intervals,
                    "quadrature_nodes_per_component": quadrature_nodes,
                    "decision_date": date,
                    "remaining_resets": remaining,
                    "zero_rate": float(case.market["zero_rate"]),
                    "beta": case.contract.beta,
                    "boundary": analytic_boundary,
                    "lattice_boundary": lattice.grid_boundaries[date],
                    "quadrature_boundary": quadrature.grid_boundaries[date],
                    "lattice_boundary_error": lattice.grid_boundaries[date]
                    - analytic_boundary,
                    "quadrature_boundary_error": quadrature.grid_boundaries[date]
                    - analytic_boundary,
                    "local_cap": case.contract.local_cap,
                    "global_cap": case.contract.global_cap,
                    "floor_probability": case.law.floor_probability,
                    "cap_probability": case.law.cap_probability,
                    "expected_credit_analytic": case.law.expected_credit,
                    "expected_credit_quadrature": expectation_quadrature,
                    "expectation_error": expectation_quadrature
                    - case.law.expected_credit,
                    "cap_contact": case.slopes.cap_contact,
                    "right_gap_slope": case.slopes.right,
                    "left_gap_slope": case.slopes.left_by_remaining_resets[
                        remaining
                    ],
                    "accumulator_atom_distance": accumulator_distance,
                    "credit_contact_atom_distance": contact_distance,
                    "eligible_inception_trace": eligible,
                    "eligible_source_date": eligible_source,
                    "eligible_target_date": eligible_target,
                    "eligible_trace_intensity": eligible_intensity,
                }
            )
            trace_rows.append(
                {
                    "date": case.date,
                    "contract_id": case.contract_id,
                    "model": case.model,
                    "level": level_name,
                    "decision_date": date,
                    "total_density": trace.total_density[date],
                    "fresh_density": trace.fresh_density[date],
                    "combinatorial_density": trace.combinatorial_density[date],
                    "reconciliation_error": trace.component_reconciliation_error[
                        date
                    ],
                }
            )
            for source in range(1, date + 1):
                trace_component_rows.append(
                    {
                        "date": case.date,
                        "contract_id": case.contract_id,
                        "model": case.model,
                        "level": level_name,
                        "source_date": source,
                        "target_date": date,
                        "zero_suffix_length": date - source,
                        "history": "fresh"
                        if source == date
                        else "copied_zero_suffix",
                        "fresh_source_density": trace.fresh_density[source],
                        "floor_probability_power": case.law.floor_probability
                        ** (date - source),
                        "trace_intensity": trace.fresh_density[source]
                        * case.law.floor_probability ** (date - source),
                    }
                )

        finance_direction = (
            np.ones(case.contract.exercise_dates)
            if other_case.boundary > case.boundary
            else -np.ones(case.contract.exercise_dates)
        )
        finance_coefficient = prefix_coefficient(
            direction=finance_direction,
            contract=case.contract,
            law=case.law,
            trace=trace,
            slopes=case.slopes,
        )
        boundary_difference = other_case.boundary - case.boundary
        transfer_scale = abs(boundary_difference)
        transfer_policy_index = 1
        native_policy_index = 0
        transfer_forward_loss = float(
            forward.total_losses[transfer_policy_index]
        )
        native_forward_loss = float(forward.total_losses[native_policy_index])
        transfer_lattice_loss = float(
            lattice.optimal_issue_value
            - lattice.policy_issue_values[transfer_policy_index]
        )
        transfer_quadrature_loss = float(
            quadrature.optimal_issue_value
            - quadrature_by_policy["transferred_threshold"]
        )
        transfer_quadrature_direct = quadrature_regret_by_policy[
            "transferred_threshold"
        ]
        native_quadrature_loss = float(
            quadrature.optimal_issue_value
            - quadrature_by_policy["native_threshold"]
        )
        native_quadrature_direct = quadrature_regret_by_policy[
            "native_threshold"
        ]
        finance_rows.append(
            {
                "date": case.date,
                "contract_id": case.contract_id,
                "reference_model": case.model,
                "transferred_from_model": other_model,
                "level": level_name,
                "intervals_per_local_cap": intervals,
                "quadrature_nodes_per_component": quadrature_nodes,
                "reference_boundary": case.boundary,
                "transferred_boundary": other_case.boundary,
                "boundary_difference": boundary_difference,
                "boundary_difference_bps_accumulator": boundary_difference * 10_000,
                "optimal_premium_lattice": lattice.optimal_issue_value,
                "optimal_premium_quadrature": quadrature.optimal_issue_value,
                "european_value_lattice": lattice.european_issue_value,
                "european_value_quadrature": quadrature.european_issue_value,
                "early_exercise_value_lattice": lattice.optimal_issue_value
                - lattice.european_issue_value,
                "early_exercise_value_quadrature": quadrature.optimal_issue_value
                - quadrature.european_issue_value,
                "native_threshold_loss_forward": native_forward_loss,
                "native_threshold_loss_quadrature": native_quadrature_loss,
                "native_threshold_loss_quadrature_direct": native_quadrature_direct,
                "transfer_loss_forward": transfer_forward_loss,
                "transfer_loss_lattice_difference": transfer_lattice_loss,
                "transfer_loss_quadrature_difference": transfer_quadrature_loss,
                "transfer_loss_quadrature_direct": transfer_quadrature_direct,
                "transfer_internal_reconciliation": transfer_forward_loss
                - transfer_lattice_loss,
                "transfer_cross_evaluator_difference": transfer_forward_loss
                - transfer_quadrature_direct,
                "transfer_quadrature_cancellation_diagnostic": transfer_quadrature_loss
                - transfer_quadrature_direct,
                "fresh_diagonal_prediction": finance_coefficient.fresh_diagonal
                * transfer_scale**2,
                "additive_single_date_prediction": finance_coefficient.additive_single_date
                * transfer_scale**2,
                "full_prefix_prediction": finance_coefficient.full
                * transfer_scale**2,
                "prefix_change_prediction": (
                    finance_coefficient.full
                    - finance_coefficient.additive_single_date
                )
                * transfer_scale**2,
                "threshold_below_zero": bool(np.any(transfer_thresholds < 0)),
                "threshold_above_global_cap": bool(
                    np.any(transfer_thresholds > case.contract.global_cap)
                ),
                "calibration_status": case.calibration["status"]
                if case.model == "annual_lognormal_mixture"
                else "atm_summary_input",
                "calibration_standardized_rmse": float(
                    case.calibration["standardized_rmse"]
                )
                if case.model == "annual_lognormal_mixture"
                else np.nan,
                "calibration_forward_normalized_rmse": float(
                    case.calibration["forward_normalized_rmse"]
                )
                if case.model == "annual_lognormal_mixture"
                else np.nan,
                "calibration_within_spread_share": float(
                    case.calibration["within_spread_share"]
                )
                if case.model == "annual_lognormal_mixture"
                else np.nan,
            }
        )
        for policy_index, policy_role in (
            (native_policy_index, "reference_native"),
            (transfer_policy_index, "perturbed_transferred"),
        ):
            for date_index in range(case.contract.exercise_dates):
                policy_occupancy_rows.append(
                    {
                        "date": case.date,
                        "contract_id": case.contract_id,
                        "model": case.model,
                        "level": level_name,
                        "policy_role": policy_role,
                        "decision_date": date_index + 1,
                        "predecision_mass": forward.predecision_mass[
                            policy_index, date_index
                        ],
                        "survival_mass": forward.survival_mass[
                            policy_index, date_index
                        ],
                        "mismatch_mass": forward.mismatch_mass[
                            policy_index, date_index
                        ],
                        "discounted_date_loss": forward.date_losses[
                            policy_index, date_index
                        ],
                    }
                )

        if mechanism:
            coefficient_by_direction = {
                name: prefix_coefficient(
                    direction=direction,
                    contract=case.contract,
                    law=case.law,
                    trace=trace,
                    slopes=case.slopes,
                )
                for name, direction in directions.items()
            }
            policy_index = {
                str(record["policy_id"]): index
                for index, record in enumerate(policies)
            }
            for direction_name, coefficient in coefficient_by_direction.items():
                for component in coefficient.components:
                    coefficient_component_rows.append(
                        {
                            "date": case.date,
                            "contract_id": case.contract_id,
                            "model": case.model,
                            "level": level_name,
                            "direction_name": direction_name,
                            **component,
                        }
                    )
            for record in mechanism:
                identifier = str(record["policy_id"])
                index = policy_index[identifier]
                direction_name = str(record["direction_name"])
                eta = float(record["eta"])
                epsilon = float(record["epsilon"])
                coefficient = coefficient_by_direction[direction_name]
                exact_forward = float(forward.total_losses[index])
                exact_backward = float(
                    lattice.optimal_issue_value
                    - lattice.policy_issue_values[index]
                )
                mechanism_rows.append(
                    {
                        "date": case.date,
                        "contract_id": case.contract_id,
                        "model": case.model,
                        "level": level_name,
                        "policy_id": identifier,
                        "direction_name": direction_name,
                        "direction_class": "single_date"
                        if direction_name.startswith("single_")
                        else "multidate",
                        "eta": eta,
                        "epsilon": epsilon,
                        "exact_regret_forward": exact_forward,
                        "exact_regret_lattice_backward": exact_backward,
                        "internal_reconciliation_error": exact_forward
                        - exact_backward,
                        "fresh_diagonal_coefficient": coefficient.fresh_diagonal,
                        "additive_single_date_coefficient": coefficient.additive_single_date,
                        "full_prefix_coefficient": coefficient.full,
                        "copied_coefficient": coefficient.copied,
                        "fresh_diagonal_prediction": coefficient.fresh_diagonal
                        * epsilon**2,
                        "additive_single_date_prediction": coefficient.additive_single_date
                        * epsilon**2,
                        "full_prefix_prediction": coefficient.full * epsilon**2,
                        "threshold_below_zero": bool(
                            np.any(np.asarray(record["thresholds"]) < 0)
                        ),
                        "threshold_above_global_cap": bool(
                            np.any(
                                np.asarray(record["thresholds"])
                                > case.contract.global_cap
                            )
                        ),
                    }
                )
                for date_index in range(case.contract.exercise_dates):
                    mechanism_date_rows.append(
                        {
                            "date": case.date,
                            "contract_id": case.contract_id,
                            "model": case.model,
                            "level": level_name,
                            "policy_id": identifier,
                            "direction_name": direction_name,
                            "eta": eta,
                            "decision_date": date_index + 1,
                            "predecision_mass": forward.predecision_mass[
                                index, date_index
                            ],
                            "perturbed_survival_mass": forward.survival_mass[
                                index, date_index
                            ],
                            "mismatch_mass": forward.mismatch_mass[
                                index, date_index
                            ],
                            "discounted_date_loss": forward.date_losses[
                                index, date_index
                            ],
                        }
                    )
                if identifier in quadrature_by_policy:
                    quadrature_difference_loss = (
                        quadrature.optimal_issue_value
                        - quadrature_by_policy[identifier]
                    )
                    quadrature_direct_loss = quadrature_regret_by_policy[identifier]
                    independent_rows.append(
                        {
                            "date": case.date,
                            "contract_id": case.contract_id,
                            "model": case.model,
                            "level": level_name,
                            "policy_id": identifier,
                            "direction_name": direction_name,
                            "eta": eta,
                            "lattice_forward_regret": exact_forward,
                            "lattice_backward_regret": exact_backward,
                            "quadrature_backward_regret_direct": quadrature_direct_loss,
                            "quadrature_price_difference_regret": quadrature_difference_loss,
                            "lattice_identity_error": exact_forward
                            - exact_backward,
                            "cross_evaluator_error": exact_forward
                            - quadrature_direct_loss,
                            "quadrature_cancellation_diagnostic": quadrature_difference_loss
                            - quadrature_direct_loss,
                        }
                    )

        timing_rows.append(
            {
                "date": case.date,
                "contract_id": case.contract_id,
                "model": case.model,
                "level": level_name,
                "seconds": time.perf_counter() - case_started,
            }
        )
        elapsed = time.perf_counter() - started
        if elapsed > 60 * float(config["numerics"]["wall_clock_minutes"]):
            raise TimeoutError("frozen wall-clock compute limit exceeded")

    frames = {
        "boundaries.csv": pd.DataFrame(boundary_rows),
        "trace_audit.csv": pd.DataFrame(trace_rows),
        "trace_components.csv": pd.DataFrame(trace_component_rows),
        "mechanism_results.csv": pd.DataFrame(mechanism_rows),
        "mechanism_by_date.csv": pd.DataFrame(mechanism_date_rows),
        "policy_occupancy.csv": pd.DataFrame(policy_occupancy_rows),
        "coefficient_components.csv": pd.DataFrame(coefficient_component_rows),
        "independent_checks.csv": pd.DataFrame(independent_rows),
        "finance_valuation.csv": pd.DataFrame(finance_rows),
        "timings.csv": pd.DataFrame(timing_rows),
    }
    for filename, frame in frames.items():
        frame.to_csv(RESULTS / filename, index=False, float_format="%.17g")

    metadata = {
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.perf_counter() - started,
        "git_head_at_run": git("rev-parse", "HEAD"),
        "git_branch": git("branch", "--show-current"),
        "base_commit": config["base_commit"],
        "config_sha256": sha256(config_path),
        "observed_input_hashes": observed_hashes,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "case_count": len(cases),
        "outcome_row_counts": {
            filename: int(len(frame)) for filename, frame in frames.items()
        },
        "monte_carlo_used": False,
    }
    (RESULTS / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    run()
