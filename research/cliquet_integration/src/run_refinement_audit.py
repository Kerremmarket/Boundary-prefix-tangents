"""Run the predeclared independent gate and warning-row refinements."""

from __future__ import annotations

import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

INTEGRATION = Path(__file__).resolve().parents[1]
RESEARCH = INTEGRATION.parent
SOURCE = RESEARCH / "cliquet_feasibility"
RESULTS = INTEGRATION / "results"
sys.path.insert(0, str(SOURCE / "src"))
sys.path.insert(0, str(INTEGRATION / "src"))

from cliquet import (  # noqa: E402
    forward_performance_difference,
    make_directions,
    quadrature_policy_regret,
    solve_lattice,
    solve_quadrature,
)
from run_pilot import build_cases, load_inputs  # noqa: E402
from uncapped_cliquet import (  # noqa: E402
    forward_performance_difference_uncapped,
    solve_uncapped_lattice_reference,
    solve_uncapped_quadrature_reference,
    quadrature_policy_regret_uncapped,
)


def evaluate_capped(
    case,
    thresholds: np.ndarray,
    *,
    intervals: int,
    nodes: int,
) -> dict[str, float]:
    matrix = np.asarray(thresholds, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    lattice = solve_lattice(
        case.law,
        case.contract,
        intervals_per_cap=intervals,
        policy_thresholds=matrix,
    )
    forward = forward_performance_difference(
        lattice,
        case.contract,
        optimal_boundary=case.boundary,
        policy_thresholds=matrix,
    )
    quadrature = solve_quadrature(
        case.law,
        case.contract,
        intervals_per_cap=intervals,
        nodes_per_component=nodes,
        policy_thresholds=matrix,
    )
    direct = quadrature_policy_regret(
        quadrature,
        case.contract,
        policy_thresholds=matrix,
    )
    return {
        "lattice_forward": float(forward.total_losses[0]),
        "lattice_backward": float(
            lattice.optimal_issue_value - lattice.policy_issue_values[0]
        ),
        "quadrature_direct": float(direct[0]),
        "quadrature_price_difference": float(
            quadrature.optimal_issue_value - quadrature.policy_issue_values[0]
        ),
        "optimal_premium_lattice": float(lattice.optimal_issue_value),
        "optimal_premium_quadrature": float(quadrature.optimal_issue_value),
        "cross_evaluator_difference": float(
            forward.total_losses[0] - direct[0]
        ),
    }


def warning_refinement(cases: dict, audit_config: dict) -> pd.DataFrame:
    warning = audit_config["warning_case"]
    key = (
        warning["date"],
        warning["contract_id"],
        warning["reference_model"],
    )
    case = cases[key]
    other = cases[
        (
            warning["date"],
            warning["contract_id"],
            warning["transferred_from_model"],
        )
    ]
    thresholds = np.full(case.contract.exercise_dates, other.boundary)
    source_finance = pd.read_csv(SOURCE / "tables/finance_certified.csv")
    source_row = source_finance[
        (source_finance["date"] == warning["date"])
        & (source_finance["contract_id"] == warning["contract_id"])
        & (source_finance["reference_model"] == warning["reference_model"])
        & (
            source_finance["transferred_from_model"]
            == warning["transferred_from_model"]
        )
    ].iloc[0]
    rows: list[dict[str, object]] = [
        {
            "level": "source_fine",
            "intervals_per_local_cap": int(source_row["intervals_per_local_cap"]),
            "quadrature_nodes_per_component": int(
                source_row["quadrature_nodes_per_component"]
            ),
            "lattice_forward": float(source_row["transfer_loss_forward"]),
            "lattice_backward": float(
                source_row["transfer_loss_lattice_difference"]
            ),
            "quadrature_direct": float(
                source_row["transfer_loss_quadrature_direct"]
            ),
            "quadrature_price_difference": float(
                source_row["transfer_loss_quadrature_difference"]
            ),
            "optimal_premium_lattice": float(
                source_row["optimal_premium_lattice"]
            ),
            "optimal_premium_quadrature": float(
                source_row["optimal_premium_quadrature"]
            ),
            "cross_evaluator_difference": float(
                source_row["transfer_cross_evaluator_difference"]
            ),
            "uncapped_lattice_forward": np.nan,
            "capped_uncapped_lattice_difference": np.nan,
        }
    ]
    tolerance = float(
        audit_config["unchanged_tolerances"][
            "value_reconciliation_absolute"
        ]
    )
    for level in audit_config["refinement_levels"]:
        if level["name"] == "R2" and abs(
            float(rows[-1]["cross_evaluator_difference"])
        ) <= tolerance:
            break
        print(
            f"warning refinement {level['name']}: "
            f"{level['intervals_per_local_cap']}/{level['quadrature_nodes_per_component']}",
            flush=True,
        )
        computed = evaluate_capped(
            case,
            thresholds,
            intervals=int(level["intervals_per_local_cap"]),
            nodes=int(level["quadrature_nodes_per_component"]),
        )
        uncapped_reference = solve_uncapped_lattice_reference(
            case.law,
            case.contract,
            intervals_per_local_cap=int(level["intervals_per_local_cap"]),
        )
        uncapped_forward = forward_performance_difference_uncapped(
            uncapped_reference,
            case.contract,
            policy_thresholds=thresholds,
        )
        rows.append(
            {
                "level": level["name"],
                "intervals_per_local_cap": int(
                    level["intervals_per_local_cap"]
                ),
                "quadrature_nodes_per_component": int(
                    level["quadrature_nodes_per_component"]
                ),
                **computed,
                "uncapped_lattice_forward": float(
                    uncapped_forward.total_losses[0]
                ),
                "capped_uncapped_lattice_difference": float(
                    computed["lattice_forward"]
                    - uncapped_forward.total_losses[0]
                ),
            }
        )
    result = pd.DataFrame(rows)
    result.insert(0, "transferred_from_model", warning["transferred_from_model"])
    result.insert(0, "reference_model", warning["reference_model"])
    result.insert(0, "contract_id", warning["contract_id"])
    result.insert(0, "date", warning["date"])
    return result


def sentinel_refinement(cases: dict, audit_config: dict) -> pd.DataFrame:
    level = audit_config["refinement_levels"][0]
    source_raw = pd.read_csv(SOURCE / "results/finance_valuation.csv")
    rows: list[dict[str, object]] = []
    for model in ("black_scholes_atm", "annual_lognormal_mixture"):
        case = cases[("2016-10-31", "c08_g24_n8", model)]
        other_model = (
            "annual_lognormal_mixture"
            if model == "black_scholes_atm"
            else "black_scholes_atm"
        )
        other = cases[("2016-10-31", "c08_g24_n8", other_model)]
        thresholds = np.full(case.contract.exercise_dates, other.boundary)
        print(f"sentinel rerun: {model}", flush=True)
        computed = evaluate_capped(
            case,
            thresholds,
            intervals=int(level["intervals_per_local_cap"]),
            nodes=int(level["quadrature_nodes_per_component"]),
        )
        frozen = source_raw[
            (source_raw["date"] == "2016-10-31")
            & (source_raw["contract_id"] == "c08_g24_n8")
            & (source_raw["reference_model"] == model)
            & (source_raw["level"] == "sentinel")
        ].iloc[0]
        rows.append(
            {
                "date": case.date,
                "contract_id": case.contract_id,
                "reference_model": model,
                "transferred_from_model": other_model,
                "level": level["name"],
                "intervals_per_local_cap": int(
                    level["intervals_per_local_cap"]
                ),
                "quadrature_nodes_per_component": int(
                    level["quadrature_nodes_per_component"]
                ),
                **computed,
                "frozen_sentinel_lattice_forward": float(
                    frozen["transfer_loss_forward"]
                ),
                "rerun_minus_frozen": float(
                    computed["lattice_forward"]
                    - float(frozen["transfer_loss_forward"])
                ),
            }
        )
    return pd.DataFrame(rows)


def gate_quadrature(cases: dict, audit_config: dict) -> pd.DataFrame:
    gate = pd.read_csv(SOURCE / "tables/mechanism_gate_rows.csv")
    lattice_audit = pd.read_csv(RESULTS / "uncapped_mechanism_audit.csv")
    intervals = int(
        audit_config["uncapped_state_audit"]["intervals_per_local_cap"]
    )
    nodes = int(
        audit_config["uncapped_state_audit"][
            "quadrature_nodes_per_component"
        ]
    )
    rows: list[dict[str, object]] = []
    grouped = gate.groupby(["date", "contract_id", "model"], sort=False)
    for number, (key, frame) in enumerate(grouped, start=1):
        case = cases[tuple(str(item) for item in key)]
        directions = make_directions(case.contract.exercise_dates)
        frame = frame.sort_values(["direction_name", "eta"], kind="stable")
        thresholds = np.vstack(
            [
                case.boundary
                + float(record.epsilon)
                * directions[str(record.direction_name)]
                for record in frame.itertuples(index=False)
            ]
        )
        if np.any(thresholds <= 0) or np.any(
            thresholds >= case.contract.global_cap
        ):
            raise RuntimeError(f"gate row left strict local domain: {key}")
        print(
            f"uncapped quadrature gate case {number}/{len(grouped)}: "
            f"{case.date} {case.model}",
            flush=True,
        )
        reference = solve_uncapped_quadrature_reference(
            case.law,
            case.contract,
            intervals_per_local_cap=intervals,
            nodes_per_component=nodes,
        )
        regrets = quadrature_policy_regret_uncapped(
            reference,
            case.contract,
            policy_thresholds=thresholds,
        )
        for record, regret in zip(frame.itertuples(index=False), regrets):
            lattice = lattice_audit[
                (lattice_audit["date"] == case.date)
                & (lattice_audit["contract_id"] == case.contract_id)
                & (lattice_audit["model"] == case.model)
                & (lattice_audit["policy_id"] == record.policy_id)
            ].iloc[0]
            rows.append(
                {
                    "date": case.date,
                    "contract_id": case.contract_id,
                    "model": case.model,
                    "policy_id": record.policy_id,
                    "direction_name": record.direction_name,
                    "eta": float(record.eta),
                    "epsilon": float(record.epsilon),
                    "uncapped_lattice_forward": float(
                        lattice["uncapped_forward_loss"]
                    ),
                    "uncapped_quadrature_direct": float(regret),
                    "cross_evaluator_difference": float(
                        lattice["uncapped_forward_loss"] - regret
                    ),
                    "full_prefix_prediction": float(
                        record.full_prefix_prediction
                    ),
                    "frozen_certification_floor": float(
                        record.certification_floor
                    ),
                }
            )
    return pd.DataFrame(rows)


def run() -> None:
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    source_config = json.loads((SOURCE / "config.json").read_text())
    audit_config = json.loads((INTEGRATION / "config.json").read_text())
    market, mixture, _ = load_inputs(source_config)
    cases = build_cases(source_config, market, mixture)
    RESULTS.mkdir(parents=True, exist_ok=True)

    warning = warning_refinement(cases, audit_config)
    warning.to_csv(
        RESULTS / "warning_refinement.csv", index=False, float_format="%.17g"
    )
    sentinel = sentinel_refinement(cases, audit_config)
    sentinel.to_csv(
        RESULTS / "sentinel_reproduction.csv", index=False, float_format="%.17g"
    )
    gate = gate_quadrature(cases, audit_config)
    gate.to_csv(
        RESULTS / "gate_quadrature_audit.csv", index=False, float_format="%.17g"
    )

    tolerance = float(
        audit_config["unchanged_tolerances"][
            "value_reconciliation_absolute"
        ]
    )
    finest_warning = warning.iloc[-1]
    metadata = {
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.perf_counter() - started,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "warning_finest_level": str(finest_warning["level"]),
        "warning_finest_cross_evaluator_difference": float(
            finest_warning["cross_evaluator_difference"]
        ),
        "warning_original_tolerance_pass": bool(
            abs(float(finest_warning["cross_evaluator_difference"]))
            <= tolerance
        ),
        "warning_r2_required": bool((warning["level"] == "R2").any()),
        "sentinel_maximum_rerun_difference": float(
            sentinel["rerun_minus_frozen"].abs().max()
        ),
        "gate_rows": int(len(gate)),
        "gate_maximum_cross_evaluator_difference": float(
            gate["cross_evaluator_difference"].abs().max()
        ),
    }
    (RESULTS / "refinement_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    run()

