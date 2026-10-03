"""Run the frozen all-row literal-accumulator state/policy audit."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

INTEGRATION = Path(__file__).resolve().parents[1]
RESEARCH = INTEGRATION.parent
REPO = RESEARCH.parent
SOURCE = RESEARCH / "cliquet_feasibility"
RESULTS = INTEGRATION / "results"
sys.path.insert(0, str(SOURCE / "src"))
sys.path.insert(0, str(INTEGRATION / "src"))

from cliquet import make_directions  # noqa: E402
from run_pilot import build_cases, load_inputs  # noqa: E402
from uncapped_cliquet import (  # noqa: E402
    forward_performance_difference_uncapped,
    lattice_policy_values,
    local_domain,
    reachable_threshold_dates,
    solve_uncapped_lattice_reference,
)


EXPECTED_HASHES = {
    "config.json": "2fae9dd2d796c385af63da5dd8342724a74e48cbfc54b7f7e86317b854897d9e",
    "src/cliquet.py": "118d3666a5cc0ab71179c7a266aabe8fd7250dacefcee27248f9675143277836",
    "src/run_pilot.py": "6ce58b4a546058c83182ff37301bb92224c6de87b3089352314190e94d9fc940",
    "results/mechanism_results.csv": "fe40db2702e27e66ac119023af9b767f9a7f83d7f34cf30850b18124f9228e42",
    "tables/mechanism_gate_rows.csv": "1b2df05afbb2c4a6ff3329f214dece5ccfdf5cea069e520a83c9b2bf63db85a6",
    "tables/finance_certified.csv": "d057cd4e78e215fb08ba1f325349f6b110509b9fbb0489b4fc3469246080c7d2",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sources() -> dict[str, str]:
    # Public export: authenticate source against this release, and generated
    # artifacts against the local run record. Original hashes remain above.
    from reproduction_support import validate_reproduction_file, validate_source_manifest
    validate_source_manifest(REPO)
    return {relative: validate_reproduction_file(SOURCE / relative, expected, REPO)
            for relative, expected in EXPECTED_HASHES.items()}


def run() -> None:
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    observed_hashes = verify_sources()
    source_config = json.loads((SOURCE / "config.json").read_text())
    audit_config = json.loads((INTEGRATION / "config.json").read_text())
    if not audit_config.get("frozen_before_corrected_calculations"):
        raise RuntimeError("audit protocol is not frozen")
    market, mixture, input_hashes = load_inputs(source_config)
    cases = build_cases(source_config, market, mixture)
    certified = pd.read_csv(SOURCE / "tables/mechanism_certified.csv")
    certified = certified[certified["level"] == "fine"].copy()
    if len(certified) != 1140:
        raise RuntimeError(f"expected 1,140 fine mechanism rows, found {len(certified)}")

    intervals = int(
        audit_config["uncapped_state_audit"]["intervals_per_local_cap"]
    )
    batch_size = 12
    rows: list[dict[str, object]] = []
    date_rows: list[dict[str, object]] = []
    reference_rows: list[dict[str, object]] = []

    grouped = certified.groupby(["date", "contract_id", "model"], sort=False)
    for case_number, (key, frame) in enumerate(grouped, start=1):
        key_tuple = tuple(str(item) for item in key)
        case = cases[key_tuple]
        if case.contract_id != audit_config["base_contract_id"]:
            raise RuntimeError(f"unexpected mechanism contract {case.contract_id}")
        directions = make_directions(case.contract.exercise_dates)
        frame = frame.sort_values(["direction_name", "eta"], kind="stable").reset_index(drop=True)
        threshold_list: list[np.ndarray] = []
        direction_list: list[np.ndarray] = []
        for record in frame.itertuples(index=False):
            direction = directions[str(record.direction_name)]
            threshold_list.append(case.boundary + float(record.epsilon) * direction)
            direction_list.append(direction)
        thresholds = np.vstack(threshold_list)

        reference = solve_uncapped_lattice_reference(
            case.law,
            case.contract,
            intervals_per_local_cap=intervals,
        )
        policy_values = np.empty(len(frame))
        forward_losses = np.empty(len(frame))
        forward_date_losses = np.empty((len(frame), case.contract.exercise_dates))
        predecision_mass = np.empty_like(forward_date_losses)
        survival_mass = np.empty_like(forward_date_losses)
        mismatch_mass = np.empty_like(forward_date_losses)
        for first in range(0, len(frame), batch_size):
            last = min(first + batch_size, len(frame))
            batch = thresholds[first:last]
            policy_values[first:last] = lattice_policy_values(
                reference, case.contract, batch
            )
            forward = forward_performance_difference_uncapped(
                reference, case.contract, policy_thresholds=batch
            )
            forward_losses[first:last] = forward.total_losses
            forward_date_losses[first:last] = forward.date_losses
            predecision_mass[first:last] = forward.predecision_mass
            survival_mass[first:last] = forward.survival_mass
            mismatch_mass[first:last] = forward.mismatch_mass

        backward_losses = reference.optimal_issue_value - policy_values
        reference_rows.append(
            {
                "date": case.date,
                "contract_id": case.contract_id,
                "model": case.model,
                "intervals_per_local_cap": intervals,
                "uncapped_optimal_issue_value": reference.optimal_issue_value,
                "analytic_boundary": case.boundary,
                "maximum_grid_boundary_error": max(
                    abs(value - case.boundary)
                    for value in reference.grid_boundaries.values()
                ),
                "state_grid_maximum": float(reference.grid[-1]),
            }
        )

        for index, record in enumerate(frame.itertuples(index=False)):
            direction = direction_list[index]
            threshold = thresholds[index]
            is_local, threshold_min, threshold_max = local_domain(
                case.boundary,
                case.contract.global_cap,
                float(record.epsilon),
                direction,
            )
            reachable = reachable_threshold_dates(threshold, case.contract)
            above_dates = tuple(
                date
                for date, value in enumerate(threshold, start=1)
                if value > case.contract.global_cap
            )
            reachable_above_dates = tuple(
                date for date in above_dates if date in reachable
            )
            corrected = float(forward_losses[index])
            source_loss = float(record.exact_regret_forward)
            rows.append(
                {
                    "date": case.date,
                    "contract_id": case.contract_id,
                    "model": case.model,
                    "policy_id": record.policy_id,
                    "direction_name": record.direction_name,
                    "direction_class": record.direction_class,
                    "eta": float(record.eta),
                    "epsilon": float(record.epsilon),
                    "boundary": case.boundary,
                    "global_cap": case.contract.global_cap,
                    "threshold_min": threshold_min,
                    "threshold_max": threshold_max,
                    "strict_local_domain": bool(
                        threshold_min > 0.0
                        and threshold_max < case.contract.global_cap
                    ),
                    "closed_local_domain": is_local,
                    "threshold_above_global_cap": bool(above_dates),
                    "above_cap_dates": ";".join(map(str, above_dates)),
                    "reachable_above_cap_dates": ";".join(
                        map(str, reachable_above_dates)
                    ),
                    "above_cap_threshold_reachable": bool(reachable_above_dates),
                    "frozen_capped_loss": source_loss,
                    "uncapped_forward_loss": corrected,
                    "uncapped_backward_loss": float(backward_losses[index]),
                    "uncapped_internal_difference": corrected
                    - float(backward_losses[index]),
                    "uncapped_minus_frozen_capped": corrected - source_loss,
                    "full_prefix_prediction": float(record.full_prefix_prediction),
                    "additive_single_date_prediction": float(
                        record.additive_single_date_prediction
                    ),
                    "fresh_diagonal_prediction": float(
                        record.fresh_diagonal_prediction
                    ),
                    "frozen_certification_floor": float(
                        record.certification_floor
                    ),
                    "frozen_resolved": bool(record.resolved),
                }
            )
            if above_dates:
                for date_index in range(case.contract.exercise_dates):
                    date_rows.append(
                        {
                            "date": case.date,
                            "contract_id": case.contract_id,
                            "model": case.model,
                            "policy_id": record.policy_id,
                            "direction_name": record.direction_name,
                            "eta": float(record.eta),
                            "decision_date": date_index + 1,
                            "threshold": float(threshold[date_index]),
                            "global_cap": case.contract.global_cap,
                            "support_maximum": (date_index + 1)
                            * case.contract.local_cap,
                            "threshold_reachable": (date_index + 1)
                            in reachable,
                            "predecision_mass": float(
                                predecision_mass[index, date_index]
                            ),
                            "survival_mass": float(
                                survival_mass[index, date_index]
                            ),
                            "mismatch_mass": float(
                                mismatch_mass[index, date_index]
                            ),
                            "discounted_date_loss": float(
                                forward_date_losses[index, date_index]
                            ),
                        }
                    )
        print(
            f"uncapped lattice case {case_number}/{len(grouped)}: "
            f"{case.date} {case.model}",
            flush=True,
        )

    result = pd.DataFrame(rows)
    nonlocal_by_date = pd.DataFrame(date_rows)
    references = pd.DataFrame(reference_rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    result.to_csv(
        RESULTS / "uncapped_mechanism_audit.csv", index=False, float_format="%.17g"
    )
    nonlocal_by_date.to_csv(
        RESULTS / "uncapped_nonlocal_by_date.csv", index=False, float_format="%.17g"
    )
    references.to_csv(
        RESULTS / "uncapped_reference_audit.csv", index=False, float_format="%.17g"
    )

    local = result[result["strict_local_domain"]]
    nonlocal_rows = result[~result["strict_local_domain"]]
    above = result[result["threshold_above_global_cap"]]
    metadata = {
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.perf_counter() - started,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "source_hashes": observed_hashes,
        "input_hashes": input_hashes,
        "rows": int(len(result)),
        "strict_local_rows": int(len(local)),
        "nonlocal_rows": int(len(nonlocal_rows)),
        "above_cap_rows": int(len(above)),
        "above_cap_reachable_rows": int(above["above_cap_threshold_reachable"].sum()),
        "maximum_local_capped_uncapped_difference": float(
            local["uncapped_minus_frozen_capped"].abs().max()
        ),
        "maximum_uncapped_forward_backward_difference": float(
            result["uncapped_internal_difference"].abs().max()
        ),
        "maximum_nonlocal_correction": float(
            nonlocal_rows["uncapped_minus_frozen_capped"].abs().max()
        ),
        "all_above_cap_thresholds_reachable_at_some_decision": bool(
            above["above_cap_threshold_reachable"].all()
        ),
    }
    (RESULTS / "uncapped_audit_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    run()

