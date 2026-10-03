"""Run the bounded prefix-versus-additive evidence revision.

This script writes raw deterministic calculations only.  It never modifies the
frozen cliquet packages from which cases and numerical primitives are imported.
"""

from __future__ import annotations

import hashlib
import json
import platform
import resource
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import scipy


PACKAGE = Path(__file__).resolve().parents[1]
REPO = PACKAGE.parents[1]
SOURCE = REPO / "research" / "cliquet_feasibility"
INTEGRATION = REPO / "research" / "cliquet_integration"
RESULTS = PACKAGE / "results"

sys.path.insert(0, str(SOURCE / "src"))
sys.path.insert(0, str(INTEGRATION / "src"))

from cliquet import (  # noqa: E402
    Contract,
    CreditLaw,
    LognormalComponent,
    build_law,
    make_directions,
    one_sided_slopes,
    prefix_coefficient,
    solve_boundary,
    trace_densities,
)
from comparator import (  # noqa: E402
    assert_boundary_action_equivalence,
    finite_lattice_decomposition,
    finite_lattice_decomposition_batch,
    quadrature_subtraction_decomposition,
    residual_components,
    singleton_thresholds,
)
from run_pilot import PilotCase, build_cases, load_inputs  # noqa: E402
from uncapped_cliquet import (  # noqa: E402
    local_domain,
    quadrature_policy_regret_uncapped,
    reachable_threshold_dates,
    solve_uncapped_lattice_reference,
    solve_uncapped_quadrature_reference,
)


EXPECTED_HASHES = {
    "research/cliquet_feasibility/config.json":
        "2fae9dd2d796c385af63da5dd8342724a74e48cbfc54b7f7e86317b854897d9e",
    "research/cliquet_feasibility/inputs/market_snapshots.csv":
        "6883f3d44877a78a83e63736007ca3a63121c0c6538ead67d3279f47c9c85540",
    "research/cliquet_feasibility/inputs/mixture_calibrations.csv":
        "a724dce29676a4c889c2297caad751babe854cb8cf3cdfa48e0994399cd6cce9",
    "research/cliquet_integration/tables/finance_all_30_audited.csv":
        "b5ab09f2ca5b040cbdf18f48e55a0800774e3099d961c27fc2d02153ff047fbd",
    "research/cliquet_integration/tables/mechanism_all_rows_audited.csv":
        "b563e7fb51352ca2cc0c5b69706d889ce2e6770ed8cfa917570541d84cc65387",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def verify_baseline(config: dict) -> dict[str, str]:
    from reproduction_support import validate_reproduction_file, validate_source_manifest
    if not config.get("frozen_before_new_outcomes"):
        raise RuntimeError("refusing to run an unfrozen experiment specification")
    validate_source_manifest(REPO)
    return {relative: validate_reproduction_file(REPO / relative, expected, REPO)
            for relative, expected in EXPECTED_HASHES.items()}


def pipeline_source_paths() -> list[Path]:
    """Return the frozen source files that determine the generated evidence."""

    return [
        PACKAGE / "src" / "comparator.py",
        PACKAGE / "src" / "run_experiments.py",
        PACKAGE / "src" / "analyze_results.py",
    ]


def atom_distance(boundary: float, contract: Contract) -> float:
    return float(
        min(
            abs(boundary - count * contract.local_cap)
            for count in range(contract.resets + 1)
        )
    )


def boundary_for(law: CreditLaw, contract: Contract, root_tolerance: float) -> float:
    result = solve_boundary(law, contract, absolute_tolerance=root_tolerance)
    if result.status != "regular_interior_boundary" or result.boundary is None:
        raise RuntimeError(f"nonregular comparative-static case: {result.status}")
    return float(result.boundary)


def coefficient_record(coefficient) -> dict[str, float]:
    return {
        "full_coefficient": float(coefficient.full),
        "additive_coefficient": float(coefficient.additive_single_date),
        "interaction_coefficient": float(
            coefficient.full - coefficient.additive_single_date
        ),
        "fresh_diagonal_coefficient": float(coefficient.fresh_diagonal),
        # This is the active nonfresh term for the supplied direction.  It is
        # not structural copied mass and can vanish for a common-down ray even
        # when copied mass changes the additive comparison.
        "active_nonfresh_coefficient": float(coefficient.copied),
    }


def decomposition_record(
    *,
    loss: float,
    singleton_sum: float,
    interaction: float,
    additive_prediction: float,
    full_prediction: float,
) -> dict[str, float]:
    one_resid, int_resid, full_resid, resid_error = residual_components(
        loss=loss,
        singleton_sum=singleton_sum,
        interaction=interaction,
        additive_prediction=additive_prediction,
        full_prediction=full_prediction,
    )
    additive_error = abs(loss - additive_prediction)
    prefix_error = abs(loss - full_prediction)
    return {
        "finite_loss": float(loss),
        "finite_singleton_sum": float(singleton_sum),
        "finite_interaction": float(interaction),
        "additive_prediction": float(additive_prediction),
        "full_prefix_prediction": float(full_prediction),
        "predicted_interaction": float(full_prediction - additive_prediction),
        "one_date_residual": one_resid,
        "interaction_residual": int_resid,
        "full_residual": full_resid,
        "residual_identity_error": resid_error,
        "additive_absolute_error": float(additive_error),
        "prefix_absolute_error": float(prefix_error),
        "gain": float(additive_error - prefix_error),
    }


def transfer_designs(
    config: dict,
    cases: dict[tuple[str, str, str], PilotCase],
) -> Iterable[tuple[PilotCase, PilotCase, float, np.ndarray]]:
    for case in cases.values():
        other_model = next(model for model in config["models"] if model != case.model)
        other = cases[(case.date, case.contract_id, other_model)]
        delta = float(other.boundary - case.boundary)
        physical_direction = np.full(case.contract.exercise_dates, delta)
        yield case, other, delta, physical_direction


def run_transfer_lattice(
    config: dict,
    cases: dict[tuple[str, str, str], PilotCase],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    dates: list[dict[str, object]] = []
    crosschecks: list[dict[str, object]] = []
    level_specs = [
        ("primary", int(config["numerics"]["primary_intervals_per_local_cap"])),
        ("fine", int(config["numerics"]["fine_intervals_per_local_cap"])),
    ]
    warning_key = (
        "2006-03-31",
        "c10_g30_n10",
        "annual_lognormal_mixture",
    )

    for case, other, delta, physical_direction in transfer_designs(config, cases):
        case_levels = list(level_specs)
        if case.key == warning_key:
            case_levels.append(
                ("warning_r1", int(config["numerics"]["warning_intervals_per_local_cap"]))
            )
        for level, intervals in case_levels:
            reference = solve_uncapped_lattice_reference(
                case.law, case.contract, intervals_per_local_cap=intervals
            )
            trace = trace_densities(
                case.law,
                case.contract,
                boundary=case.boundary,
                intervals_per_cap=intervals,
            )
            coefficient = prefix_coefficient(
                direction=physical_direction,
                contract=case.contract,
                law=case.law,
                trace=trace,
                slopes=case.slopes,
            )
            etas = np.asarray(config["transfer_homotopy_etas"], dtype=float)
            thresholds = np.vstack(
                [case.boundary + eta * physical_direction for eta in etas]
            )
            batch = finite_lattice_decomposition_batch(
                reference,
                case.contract,
                boundary=case.boundary,
                joint_thresholds=thresholds,
            )
            common = {
                "date": case.date,
                "contract_id": case.contract_id,
                "reference_model": case.model,
                "transferred_from_model": other.model,
                "level": level,
                "intervals_per_local_cap": intervals,
                "local_cap": case.contract.local_cap,
                "global_cap": case.contract.global_cap,
                "resets": case.contract.resets,
                "exercise_dates": case.contract.exercise_dates,
                "beta": case.contract.beta,
                "reference_boundary": case.boundary,
                "transferred_boundary": other.boundary,
                "boundary_difference": delta,
                "boundary_difference_bps_accumulator": delta * 10_000,
                "direction_sign": "up" if delta > 0 else "down",
                "floor_probability": case.law.floor_probability,
                "cap_probability": case.law.cap_probability,
                "boundary_atom_distance": atom_distance(case.boundary, case.contract),
                "native_loss": batch.native_loss,
                **coefficient_record(coefficient),
            }
            for index, eta in enumerate(etas):
                additive_prediction = eta**2 * coefficient.additive_single_date
                full_prediction = eta**2 * coefficient.full
                local, minimum, maximum = local_domain(
                    case.boundary,
                    case.contract.global_cap,
                    eta,
                    physical_direction,
                )
                record = {
                    **common,
                    "eta": float(eta),
                    "actual_transfer": bool(np.isclose(eta, 1.0)),
                    "threshold_min": minimum,
                    "threshold_max": maximum,
                    "strict_local_domain": bool(
                        local and minimum > 0 and maximum < case.contract.global_cap
                    ),
                    **decomposition_record(
                        loss=float(batch.loss[index]),
                        singleton_sum=float(batch.singleton_sum[index]),
                        interaction=float(batch.interaction[index]),
                        additive_prediction=float(additive_prediction),
                        full_prediction=float(full_prediction),
                    ),
                }
                rows.append(record)
                for date_index in range(case.contract.exercise_dates):
                    dates.append(
                        {
                            "date": case.date,
                            "contract_id": case.contract_id,
                            "reference_model": case.model,
                            "transferred_from_model": other.model,
                            "level": level,
                            "eta": float(eta),
                            "decision_date": date_index + 1,
                            "date_loss": float(batch.date_loss[index, date_index]),
                            "date_singleton": float(
                                batch.date_singleton[index, date_index]
                            ),
                            "date_interaction": float(
                                batch.date_interaction[index, date_index]
                            ),
                        }
                    )

            explicit = finite_lattice_decomposition(
                reference,
                case.contract,
                boundary=case.boundary,
                joint_thresholds=thresholds[0],
            )
            crosschecks.append(
                {
                    "date": case.date,
                    "contract_id": case.contract_id,
                    "reference_model": case.model,
                    "transferred_from_model": other.model,
                    "level": level,
                    "batch_minus_explicit_loss": float(batch.loss[0] - explicit.loss),
                    "batch_minus_explicit_singleton_sum": float(
                        batch.singleton_sum[0] - explicit.singleton_sum
                    ),
                    "batch_minus_explicit_interaction": float(
                        batch.interaction[0] - explicit.interaction
                    ),
                    "explicit_loss_identity_error": explicit.loss_identity_error,
                    "explicit_singleton_identity_error":
                        explicit.singleton_identity_error,
                    "explicit_interaction_identity_error":
                        explicit.interaction_identity_error,
                    "native_loss": explicit.native_loss,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(dates), pd.DataFrame(crosschecks)


def run_transfer_quadrature(
    config: dict,
    cases: dict[tuple[str, str, str], PilotCase],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    warning_key = (
        "2006-03-31",
        "c10_g30_n10",
        "annual_lognormal_mixture",
    )
    for case, other, delta, physical_direction in transfer_designs(config, cases):
        level_specs = [
            (
                "fine_quadrature",
                int(config["numerics"]["fine_intervals_per_local_cap"]),
                int(config["numerics"]["quadrature_nodes_per_component"]),
            )
        ]
        if case.key == warning_key:
            level_specs.append(
                (
                    "warning_r1_quadrature",
                    int(config["numerics"]["warning_intervals_per_local_cap"]),
                    int(
                        config["numerics"][
                            "warning_quadrature_nodes_per_component"
                        ]
                    ),
                )
            )
        for level, intervals, nodes in level_specs:
            reference = solve_uncapped_quadrature_reference(
                case.law,
                case.contract,
                intervals_per_local_cap=intervals,
                nodes_per_component=nodes,
            )
            assert_boundary_action_equivalence(
                reference,
                boundary=case.boundary,
                dates=case.contract.exercise_dates,
            )
            joint = case.boundary + physical_direction
            policies = np.vstack((joint, singleton_thresholds(case.boundary, joint)))
            regrets = quadrature_policy_regret_uncapped(
                reference, case.contract, policy_thresholds=policies
            )
            loss, singleton_sum, interaction = quadrature_subtraction_decomposition(
                regrets
            )
            rows.append(
                {
                    "date": case.date,
                    "contract_id": case.contract_id,
                    "reference_model": case.model,
                    "transferred_from_model": other.model,
                    "level": level,
                    "intervals_per_local_cap": intervals,
                    "nodes_per_component": nodes,
                    "finite_loss": loss,
                    "finite_singleton_sum": singleton_sum,
                    "finite_interaction": interaction,
                    "minimum_singleton_loss": float(np.min(regrets[1:])),
                    "maximum_singleton_loss": float(np.max(regrets[1:])),
                }
            )
    return pd.DataFrame(rows)


def mechanism_metadata(case: PilotCase, config: dict) -> tuple[list[dict], np.ndarray]:
    directions = make_directions(case.contract.exercise_dates)
    records: list[dict[str, object]] = []
    thresholds: list[np.ndarray] = []
    for name, direction in directions.items():
        for eta_value in config["mechanism_etas"]:
            eta = float(eta_value)
            epsilon = eta * case.contract.local_cap
            threshold = case.boundary + epsilon * direction
            records.append(
                {
                    "policy_id": f"{name}__eta_{eta:.10f}",
                    "direction_name": name,
                    "direction_class": (
                        "single_date" if name.startswith("single_") else "multidate"
                    ),
                    "eta": eta,
                    "epsilon": epsilon,
                    "direction": direction,
                }
            )
            thresholds.append(threshold)
    return records, np.vstack(thresholds)


def run_mechanism(
    config: dict,
    cases: dict[tuple[str, str, str], PilotCase],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    date_rows: list[dict[str, object]] = []
    explicit_rows: list[dict[str, object]] = []
    gate = pd.read_csv(INTEGRATION / "tables" / "mechanism_gate_40_audited.csv")
    gate_keys = set(zip(gate["date"], gate["model"], gate["policy_id"]))
    level_specs = [
        ("primary", int(config["numerics"]["primary_intervals_per_local_cap"])),
        ("fine", int(config["numerics"]["fine_intervals_per_local_cap"])),
    ]
    base_id = "c08_g24_n8"
    for case in cases.values():
        if case.contract_id != base_id:
            continue
        records, thresholds = mechanism_metadata(case, config)
        directions = make_directions(case.contract.exercise_dates)
        for level, intervals in level_specs:
            reference = solve_uncapped_lattice_reference(
                case.law, case.contract, intervals_per_local_cap=intervals
            )
            trace = trace_densities(
                case.law,
                case.contract,
                boundary=case.boundary,
                intervals_per_cap=intervals,
            )
            coefficients = {
                name: prefix_coefficient(
                    direction=direction,
                    contract=case.contract,
                    law=case.law,
                    trace=trace,
                    slopes=case.slopes,
                )
                for name, direction in directions.items()
            }
            for first in range(0, len(records), 12):
                last = min(first + 12, len(records))
                batch = finite_lattice_decomposition_batch(
                    reference,
                    case.contract,
                    boundary=case.boundary,
                    joint_thresholds=thresholds[first:last],
                )
                for local_index, record in enumerate(records[first:last]):
                    absolute_index = first + local_index
                    coefficient = coefficients[str(record["direction_name"])]
                    epsilon = float(record["epsilon"])
                    additive_prediction = epsilon**2 * coefficient.additive_single_date
                    full_prediction = epsilon**2 * coefficient.full
                    threshold = thresholds[absolute_index]
                    local, minimum, maximum = local_domain(
                        case.boundary,
                        case.contract.global_cap,
                        epsilon,
                        np.asarray(record["direction"]),
                    )
                    reachable = reachable_threshold_dates(threshold, case.contract)
                    rows.append(
                        {
                            "date": case.date,
                            "contract_id": case.contract_id,
                            "model": case.model,
                            "level": level,
                            "intervals_per_local_cap": intervals,
                            "policy_id": record["policy_id"],
                            "direction_name": record["direction_name"],
                            "direction_class": record["direction_class"],
                            "eta": record["eta"],
                            "epsilon": epsilon,
                            "boundary": case.boundary,
                            "threshold_min": minimum,
                            "threshold_max": maximum,
                            "strict_local_domain": bool(
                                local
                                and minimum > 0
                                and maximum < case.contract.global_cap
                            ),
                            "threshold_above_global_cap": bool(
                                np.any(threshold > case.contract.global_cap)
                            ),
                            "reachable_threshold_dates": ";".join(map(str, reachable)),
                            "floor_probability": case.law.floor_probability,
                            "boundary_atom_distance": atom_distance(
                                case.boundary, case.contract
                            ),
                            "native_loss": batch.native_loss,
                            **coefficient_record(coefficient),
                            **decomposition_record(
                                loss=float(batch.loss[local_index]),
                                singleton_sum=float(batch.singleton_sum[local_index]),
                                interaction=float(batch.interaction[local_index]),
                                additive_prediction=float(additive_prediction),
                                full_prediction=float(full_prediction),
                            ),
                        }
                    )
                    if level == "fine":
                        for date_index in range(case.contract.exercise_dates):
                            date_rows.append(
                                {
                                    "date": case.date,
                                    "contract_id": case.contract_id,
                                    "model": case.model,
                                    "policy_id": record["policy_id"],
                                    "direction_name": record["direction_name"],
                                    "eta": record["eta"],
                                    "decision_date": date_index + 1,
                                    "date_loss": float(
                                        batch.date_loss[local_index, date_index]
                                    ),
                                    "date_singleton": float(
                                        batch.date_singleton[local_index, date_index]
                                    ),
                                    "date_interaction": float(
                                        batch.date_interaction[local_index, date_index]
                                    ),
                                }
                            )
            if level == "fine":
                for index, record in enumerate(records):
                    key = (case.date, case.model, str(record["policy_id"]))
                    if key not in gate_keys:
                        continue
                    explicit = finite_lattice_decomposition(
                        reference,
                        case.contract,
                        boundary=case.boundary,
                        joint_thresholds=thresholds[index],
                    )
                    explicit_rows.append(
                        {
                            "date": case.date,
                            "contract_id": case.contract_id,
                            "model": case.model,
                            "policy_id": record["policy_id"],
                            "direction_name": record["direction_name"],
                            "eta": record["eta"],
                            "loss_identity_error": explicit.loss_identity_error,
                            "singleton_identity_error":
                                explicit.singleton_identity_error,
                            "interaction_identity_error":
                                explicit.interaction_identity_error,
                            "native_loss": explicit.native_loss,
                        }
                    )
    return pd.DataFrame(rows), pd.DataFrame(date_rows), pd.DataFrame(explicit_rows)


@dataclass(frozen=True)
class Scenario:
    experiment: str
    scenario_id: str
    date: str
    model: str
    law: CreditLaw
    contract: Contract
    directions: tuple[tuple[str, str, float], ...]
    changed_parameter: str
    changed_value: float
    baseline_value: float
    market: dict[str, object]
    calibration: dict[str, object]


def custom_case(
    *,
    date: str,
    model: str,
    market: dict[str, object],
    calibration: dict[str, object],
    local_cap: float,
    global_cap: float,
    resets: int,
    volatility_multiplier: float = 1.0,
) -> tuple[CreditLaw, Contract]:
    if model == "black_scholes_atm" and not np.isclose(volatility_multiplier, 1.0):
        law = CreditLaw(
            components=(
                LognormalComponent(
                    1.0,
                    float(market["annual_gross_forward"]),
                    float(market["atm_iv_1y"]) * volatility_multiplier,
                ),
            ),
            local_cap=local_cap,
        )
        law.validate()
    else:
        law = build_law(
            model=model,
            gross_forward=float(market["annual_gross_forward"]),
            atm_volatility=float(market["atm_iv_1y"]) * volatility_multiplier,
            local_cap=local_cap,
            mixture_parameters=calibration,
        )
    contract = Contract(
        local_cap,
        global_cap,
        resets,
        float(np.exp(-float(market["zero_rate"]))),
    )
    return law, contract


def comparative_scenarios(
    config: dict,
    market: pd.DataFrame,
    mixture: pd.DataFrame,
) -> list[Scenario]:
    settings = config["comparative_statics"]
    mix_by_date = {
        str(row["date"]): row.to_dict() for _, row in mixture.iterrows()
    }
    scenarios: list[Scenario] = []
    for market_row in market.itertuples(index=False):
        market_record = market_row._asdict()
        date = str(market_record["date"])
        calibration = mix_by_date[date]
        for model in config["models"]:
            for resets in settings["horizon_resets"]:
                law, contract = custom_case(
                    date=date,
                    model=model,
                    market=market_record,
                    calibration=calibration,
                    local_cap=0.08,
                    global_cap=0.24,
                    resets=int(resets),
                )
                fixed = float(settings["horizon_fixed_displacement"])
                budget = fixed * np.sqrt(
                    float(settings["horizon_reference_exercise_dates"])
                    / contract.exercise_dates
                )
                directions = tuple(
                    (normalization, "up" if sign > 0 else "down", sign * delta)
                    for normalization, delta in (
                        ("fixed_per_date", fixed),
                        ("fixed_squared_budget", float(budget)),
                    )
                    for sign in settings["signs"]
                )
                scenarios.append(
                    Scenario(
                        "horizon",
                        f"horizon__{date}__{model}__n{resets}",
                        date,
                        model,
                        law,
                        contract,
                        directions,
                        "resets",
                        float(resets),
                        8.0,
                        market_record,
                        calibration,
                    )
                )
            for local_cap in settings["local_caps_at_fixed_global_cap"]:
                law, contract = custom_case(
                    date=date,
                    model=model,
                    market=market_record,
                    calibration=calibration,
                    local_cap=float(local_cap),
                    global_cap=0.24,
                    resets=8,
                )
                delta = float(local_cap) * float(settings["relative_displacement"])
                directions = tuple(
                    ("relative_to_local_cap", "up" if sign > 0 else "down", sign * delta)
                    for sign in settings["signs"]
                )
                scenarios.append(
                    Scenario(
                        "local_cap",
                        f"local_cap__{date}__{model}__c{local_cap}",
                        date,
                        model,
                        law,
                        contract,
                        directions,
                        "local_cap",
                        float(local_cap),
                        0.08,
                        market_record,
                        calibration,
                    )
                )
            for global_cap in settings["global_caps_at_fixed_local_cap"]:
                law, contract = custom_case(
                    date=date,
                    model=model,
                    market=market_record,
                    calibration=calibration,
                    local_cap=0.08,
                    global_cap=float(global_cap),
                    resets=8,
                )
                delta = 0.08 * float(settings["relative_displacement"])
                directions = tuple(
                    ("relative_to_local_cap", "up" if sign > 0 else "down", sign * delta)
                    for sign in settings["signs"]
                )
                scenarios.append(
                    Scenario(
                        "global_cap",
                        f"global_cap__{date}__{model}__g{global_cap}",
                        date,
                        model,
                        law,
                        contract,
                        directions,
                        "global_cap",
                        float(global_cap),
                        0.24,
                        market_record,
                        calibration,
                    )
                )
        for multiplier in settings["volatility_multipliers"]:
            law, contract = custom_case(
                date=date,
                model="black_scholes_atm",
                market=market_record,
                calibration=calibration,
                local_cap=0.08,
                global_cap=0.24,
                resets=8,
                volatility_multiplier=float(multiplier),
            )
            delta = 0.08 * float(settings["relative_displacement"])
            directions = tuple(
                ("relative_to_local_cap", "up" if sign > 0 else "down", sign * delta)
                for sign in settings["signs"]
            )
            scenarios.append(
                Scenario(
                    "zero_credit_stress",
                    f"zero_credit__{date}__v{multiplier}",
                    date,
                    "black_scholes_atm",
                    law,
                    contract,
                    directions,
                    "volatility_multiplier",
                    float(multiplier),
                    1.0,
                    market_record,
                    calibration,
                )
            )
    return scenarios


def run_comparative_statics(
    config: dict,
    market: pd.DataFrame,
    mixture: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    root_tolerance = float(config["numerics"]["root_absolute_tolerance"])
    level_specs = [
        ("primary", int(config["numerics"]["primary_intervals_per_local_cap"])),
        ("fine", int(config["numerics"]["fine_intervals_per_local_cap"])),
    ]
    for scenario in comparative_scenarios(config, market, mixture):
        boundary = boundary_for(scenario.law, scenario.contract, root_tolerance)
        slopes = one_sided_slopes(scenario.law, scenario.contract, boundary)
        for level, intervals in level_specs:
            reference = solve_uncapped_lattice_reference(
                scenario.law,
                scenario.contract,
                intervals_per_local_cap=intervals,
            )
            trace = trace_densities(
                scenario.law,
                scenario.contract,
                boundary=boundary,
                intervals_per_cap=intervals,
            )
            threshold_rows: list[np.ndarray] = []
            coefficient_rows = []
            for _, _, delta in scenario.directions:
                direction = np.full(scenario.contract.exercise_dates, delta)
                threshold_rows.append(boundary + direction)
                coefficient_rows.append(
                    prefix_coefficient(
                        direction=direction,
                        contract=scenario.contract,
                        law=scenario.law,
                        trace=trace,
                        slopes=slopes,
                    )
                )
            batch = finite_lattice_decomposition_batch(
                reference,
                scenario.contract,
                boundary=boundary,
                joint_thresholds=np.vstack(threshold_rows),
            )
            for index, (normalization, sign_label, delta) in enumerate(
                scenario.directions
            ):
                coefficient = coefficient_rows[index]
                threshold = threshold_rows[index]
                local = bool(
                    np.all((threshold > 0) & (threshold < scenario.contract.global_cap))
                )
                rows.append(
                    {
                        "experiment": scenario.experiment,
                        "scenario_id": scenario.scenario_id,
                        "date": scenario.date,
                        "model": scenario.model,
                        "level": level,
                        "intervals_per_local_cap": intervals,
                        "changed_parameter": scenario.changed_parameter,
                        "changed_value": scenario.changed_value,
                        "baseline_value": scenario.baseline_value,
                        "normalization": normalization,
                        "direction_sign": sign_label,
                        "physical_displacement": float(delta),
                        "local_cap": scenario.contract.local_cap,
                        "global_cap": scenario.contract.global_cap,
                        "resets": scenario.contract.resets,
                        "exercise_dates": scenario.contract.exercise_dates,
                        "beta": scenario.contract.beta,
                        "boundary": boundary,
                        "boundary_atom_distance": atom_distance(
                            boundary, scenario.contract
                        ),
                        "strict_local_domain": local,
                        "floor_probability": scenario.law.floor_probability,
                        "cap_probability": scenario.law.cap_probability,
                        "expected_credit": scenario.law.expected_credit,
                        "right_gap_slope": slopes.right,
                        "minimum_left_gap_slope": min(
                            slopes.left_by_remaining_resets.values()
                        ),
                        "target_density_last_date": trace.total_density[
                            scenario.contract.exercise_dates
                        ],
                        "fresh_density_last_date": trace.fresh_density[
                            scenario.contract.exercise_dates
                        ],
                        "trace_reconciliation_error": max(
                            trace.component_reconciliation_error.values()
                        ),
                        "native_loss": batch.native_loss,
                        **coefficient_record(coefficient),
                        **decomposition_record(
                            loss=float(batch.loss[index]),
                            singleton_sum=float(batch.singleton_sum[index]),
                            interaction=float(batch.interaction[index]),
                            additive_prediction=float(coefficient.additive_single_date),
                            full_prediction=float(coefficient.full),
                        ),
                    }
                )
    return pd.DataFrame(rows)


def write(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(RESULTS / name, index=False, float_format="%.17g")


def run() -> None:
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    config_path = PACKAGE / "config.json"
    config = json.loads(config_path.read_text())
    source_paths = pipeline_source_paths()
    source_hashes_at_start = {
        str(path.relative_to(REPO)): sha256(path) for path in source_paths
    }
    config_hash_at_start = sha256(config_path)
    observed_hashes = verify_baseline(config)
    source_config = json.loads((SOURCE / "config.json").read_text())
    market, mixture, _ = load_inputs(source_config)
    cases = build_cases(source_config, market, mixture)
    if len(cases) != 30:
        raise RuntimeError(f"expected 30 source cases, found {len(cases)}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    transfer, transfer_dates, transfer_cross = run_transfer_lattice(config, cases)
    write(transfer, "transfer_lattice_levels.csv")
    write(transfer_dates, "transfer_date_decomposition.csv")
    write(transfer_cross, "transfer_singleton_crosschecks.csv")

    transfer_quadrature = run_transfer_quadrature(config, cases)
    write(transfer_quadrature, "transfer_quadrature_actual.csv")

    mechanism, mechanism_dates, mechanism_cross = run_mechanism(config, cases)
    write(mechanism, "mechanism_lattice_levels.csv")
    write(mechanism_dates, "mechanism_date_decomposition.csv")
    write(mechanism_cross, "mechanism_singleton_crosschecks.csv")

    comparative = run_comparative_statics(config, market, mixture)
    write(comparative, "comparative_statics_levels.csv")

    wall_seconds = time.perf_counter() - started
    wall_limit = 60 * float(config["numerics"]["wall_clock_minutes"])
    peak_rss_raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    peak_rss_gib = (
        peak_rss_raw / 1024**3
        if sys.platform == "darwin"
        else peak_rss_raw / 1024**2
    )
    memory_limit_gib = float(config["numerics"]["working_memory_gib"])
    source_hashes_at_finish = {
        str(path.relative_to(REPO)): sha256(path) for path in source_paths
    }
    config_hash_at_finish = sha256(config_path)
    observed_hashes_at_finish = {
        relative: sha256(REPO / relative) for relative in EXPECTED_HASHES
    }
    if source_hashes_at_finish != source_hashes_at_start:
        raise RuntimeError("pipeline source changed during the evidence run")
    if config_hash_at_finish != config_hash_at_start:
        raise RuntimeError("experiment config changed during the evidence run")
    if observed_hashes_at_finish != observed_hashes:
        raise RuntimeError("frozen baseline input changed during the evidence run")
    raw_output_paths = [
        RESULTS / "transfer_lattice_levels.csv",
        RESULTS / "transfer_date_decomposition.csv",
        RESULTS / "transfer_singleton_crosschecks.csv",
        RESULTS / "transfer_quadrature_actual.csv",
        RESULTS / "mechanism_lattice_levels.csv",
        RESULTS / "mechanism_date_decomposition.csv",
        RESULTS / "mechanism_singleton_crosschecks.csv",
        RESULTS / "comparative_statics_levels.csv",
    ]
    metadata = {
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": wall_seconds,
        "wall_limit_seconds": wall_limit,
        "within_predeclared_wall_limit": wall_seconds <= wall_limit,
        "peak_rss_raw": peak_rss_raw,
        "peak_rss_platform_units": "bytes" if sys.platform == "darwin" else "KiB",
        "peak_rss_gib": peak_rss_gib,
        "memory_limit_gib": memory_limit_gib,
        "within_predeclared_memory_limit": peak_rss_gib <= memory_limit_gib,
        "git_head_at_run": git("rev-parse", "HEAD"),
        "git_branch": git("branch", "--show-current"),
        "base_commit": config["base_commit"],
        "config_sha256_at_start": config_hash_at_start,
        "config_sha256_at_finish": config_hash_at_finish,
        "source_sha256_at_start": source_hashes_at_start,
        "source_sha256_at_finish": source_hashes_at_finish,
        "raw_output_sha256": {
            str(path.relative_to(PACKAGE)): sha256(path) for path in raw_output_paths
        },
        "observed_baseline_hashes": observed_hashes,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "monte_carlo_used": False,
        "row_counts": {
            "transfer_lattice_levels.csv": len(transfer),
            "transfer_date_decomposition.csv": len(transfer_dates),
            "transfer_singleton_crosschecks.csv": len(transfer_cross),
            "transfer_quadrature_actual.csv": len(transfer_quadrature),
            "mechanism_lattice_levels.csv": len(mechanism),
            "mechanism_date_decomposition.csv": len(mechanism_dates),
            "mechanism_singleton_crosschecks.csv": len(mechanism_cross),
            "comparative_statics_levels.csv": len(comparative),
        },
    }
    (RESULTS / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    if wall_seconds > wall_limit:
        raise RuntimeError("predeclared wall-clock limit exceeded")
    if peak_rss_gib > memory_limit_gib:
        raise RuntimeError("predeclared working-memory limit exceeded")


if __name__ == "__main__":
    run()
