"""Fail-fast consistency checks for the comparator evidence package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


PACKAGE = Path(__file__).resolve().parents[1]
REPO = PACKAGE.parents[1]
RESULTS = PACKAGE / "results"
TABLES = PACKAGE / "tables"

TRANSFER_KEYS = [
    "date",
    "contract_id",
    "reference_model",
    "transferred_from_model",
    "eta",
]
MECHANISM_KEYS = ["date", "contract_id", "model", "policy_id"]
COMPARATIVE_KEYS = [
    "experiment",
    "scenario_id",
    "date",
    "model",
    "normalization",
    "direction_sign",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def require_close(values, target, tolerance: float, message: str) -> None:
    error = np.nanmax(np.abs(np.asarray(values, dtype=float) - target))
    require(error <= tolerance, f"{message}: maximum error {error:.6g}")


def require_levels(
    frame: pd.DataFrame,
    *,
    keys: list[str],
    expected: dict[str, int],
    name: str,
) -> None:
    observed = frame.groupby("level", dropna=False).size().to_dict()
    require(observed == expected, f"{name}: level counts {observed} != {expected}")
    require(
        not frame.duplicated(keys + ["level"]).any(),
        f"{name}: duplicate key/level rows",
    )


def check_metadata() -> dict:
    metadata = json.loads((RESULTS / "run_metadata.json").read_text())
    require(
        metadata["source_sha256_at_start"]
        == metadata["source_sha256_at_finish"],
        "pipeline source hashes changed during run",
    )
    for relative, expected in metadata["source_sha256_at_finish"].items():
        require(sha256(REPO / relative) == expected, f"current source hash: {relative}")
    require(
        metadata["config_sha256_at_start"]
        == metadata["config_sha256_at_finish"]
        == sha256(PACKAGE / "config.json"),
        "config hash mismatch",
    )
    for relative, expected in metadata["observed_baseline_hashes"].items():
        require(sha256(REPO / relative) == expected, f"baseline hash: {relative}")
    for relative, expected in metadata["raw_output_sha256"].items():
        require(sha256(PACKAGE / relative) == expected, f"raw output hash: {relative}")
    require(metadata["within_predeclared_wall_limit"], "wall-clock limit exceeded")
    require(metadata["within_predeclared_memory_limit"], "memory limit exceeded")
    require(not metadata["monte_carlo_used"], "unexpected Monte Carlo evidence")
    return metadata


def check_raw(metadata: dict) -> None:
    config = json.loads((PACKAGE / "config.json").read_text())
    identity_tol = float(config["numerics"]["identity_tolerance"])
    sign_tol = float(config["numerics"]["sign_tolerance"])

    transfer = pd.read_csv(RESULTS / "transfer_lattice_levels.csv")
    transfer_dates = pd.read_csv(RESULTS / "transfer_date_decomposition.csv")
    transfer_cross = pd.read_csv(RESULTS / "transfer_singleton_crosschecks.csv")
    quadrature = pd.read_csv(RESULTS / "transfer_quadrature_actual.csv")
    mechanism = pd.read_csv(RESULTS / "mechanism_lattice_levels.csv")
    mechanism_dates = pd.read_csv(RESULTS / "mechanism_date_decomposition.csv")
    mechanism_cross = pd.read_csv(RESULTS / "mechanism_singleton_crosschecks.csv")
    comparative = pd.read_csv(RESULTS / "comparative_statics_levels.csv")
    require_levels(
        transfer,
        keys=TRANSFER_KEYS,
        expected={"fine": 150, "primary": 150, "warning_r1": 5},
        name="transfer lattice",
    )
    require_levels(
        quadrature,
        keys=TRANSFER_KEYS[:-1],
        expected={"fine_quadrature": 30, "warning_r1_quadrature": 1},
        name="transfer quadrature",
    )
    require_levels(
        mechanism,
        keys=MECHANISM_KEYS,
        expected={"fine": 1140, "primary": 1140},
        name="mechanism",
    )
    require_levels(
        comparative,
        keys=COMPARATIVE_KEYS,
        expected={"fine": 270, "primary": 270},
        name="comparative statics",
    )

    expected_rows = {
        "transfer_lattice_levels.csv": 305,
        "transfer_date_decomposition.csv": 2145,
        "transfer_singleton_crosschecks.csv": 61,
        "transfer_quadrature_actual.csv": 31,
        "mechanism_lattice_levels.csv": 2280,
        "mechanism_date_decomposition.csv": 7980,
        "mechanism_singleton_crosschecks.csv": 40,
        "comparative_statics_levels.csv": 540,
    }
    require(metadata["row_counts"] == expected_rows, "metadata row-count map")

    require_close(
        quadrature["finite_loss"]
        - quadrature["finite_singleton_sum"]
        - quadrature["finite_interaction"],
        0.0,
        identity_tol,
        "quadrature L=S+J",
    )
    require(float(quadrature["minimum_singleton_loss"].min()) >= -sign_tol, "quadrature singleton losses")
    qfine = quadrature[quadrature["level"] == "fine_quadrature"]
    qwarning = quadrature[quadrature["level"] == "warning_r1_quadrature"]
    require(
        bool(
            (qfine["intervals_per_local_cap"] == 8192).all()
            and (qfine["nodes_per_component"] == 256).all()
            and (qwarning["intervals_per_local_cap"] == 16384).all()
            and (qwarning["nodes_per_component"] == 512).all()
        ),
        "quadrature refinement ladder",
    )

    transfer_date_sums = (
        transfer_dates.groupby(TRANSFER_KEYS + ["level"], as_index=False)[
            ["date_loss", "date_singleton", "date_interaction"]
        ]
        .sum()
        .rename(
            columns={
                "date_loss": "date_loss_sum",
                "date_singleton": "date_singleton_sum",
                "date_interaction": "date_interaction_sum",
            }
        )
    )
    transfer_join = transfer.merge(
        transfer_date_sums,
        on=TRANSFER_KEYS + ["level"],
        validate="one_to_one",
    )
    for raw_name, sum_name in (
        ("finite_loss", "date_loss_sum"),
        ("finite_singleton_sum", "date_singleton_sum"),
        ("finite_interaction", "date_interaction_sum"),
    ):
        require_close(
            transfer_join[raw_name],
            transfer_join[sum_name],
            identity_tol,
            f"transfer date sum {raw_name}",
        )

    mechanism_date_sums = (
        mechanism_dates.groupby(MECHANISM_KEYS, as_index=False)[
            ["date_loss", "date_singleton", "date_interaction"]
        ]
        .sum()
        .rename(
            columns={
                "date_loss": "date_loss_sum",
                "date_singleton": "date_singleton_sum",
                "date_interaction": "date_interaction_sum",
            }
        )
    )
    mechanism_fine = mechanism[mechanism["level"] == "fine"]
    mechanism_join = mechanism_fine.merge(
        mechanism_date_sums, on=MECHANISM_KEYS, validate="one_to_one"
    )
    for raw_name, sum_name in (
        ("finite_loss", "date_loss_sum"),
        ("finite_singleton_sum", "date_singleton_sum"),
        ("finite_interaction", "date_interaction_sum"),
    ):
        require_close(
            mechanism_join[raw_name],
            mechanism_join[sum_name],
            identity_tol,
            f"mechanism date sum {raw_name}",
        )

    for name, cross in (("transfer", transfer_cross), ("mechanism", mechanism_cross)):
        for column in cross.columns:
            if column.endswith("_error") or column.startswith("batch_minus_") or column == "native_loss":
                require(
                    float(cross[column].abs().max()) <= identity_tol,
                    f"{name} cross-check {column}",
                )

    for name, frame in (
        ("transfer", transfer),
        ("mechanism", mechanism),
        ("comparative", comparative),
    ):
        require_close(
            frame["finite_loss"]
            - frame["finite_singleton_sum"]
            - frame["finite_interaction"],
            0.0,
            identity_tol,
            f"{name} L=S+J",
        )
        require_close(
            frame["full_coefficient"]
            - frame["additive_coefficient"]
            - frame["interaction_coefficient"],
            0.0,
            identity_tol,
            f"{name} R=A+I",
        )
        require_close(
            frame["full_residual"]
            - frame["one_date_residual"]
            - frame["interaction_residual"],
            0.0,
            identity_tol,
            f"{name} residual decomposition",
        )
        require_close(
            frame["full_prefix_prediction"]
            - frame["additive_prediction"]
            - frame["predicted_interaction"],
            0.0,
            identity_tol,
            f"{name} predicted decomposition",
        )
        require(float(frame["native_loss"].abs().max()) <= identity_tol, f"{name} native loss")

    require_close(
        transfer["full_prefix_prediction"],
        transfer["eta"] ** 2 * transfer["full_coefficient"],
        identity_tol,
        "transfer physical-direction scaling",
    )
    require_close(
        mechanism["full_prefix_prediction"],
        mechanism["epsilon"] ** 2 * mechanism["full_coefficient"],
        identity_tol,
        "mechanism direction scaling",
    )
    require_close(
        comparative["full_prefix_prediction"],
        comparative["full_coefficient"],
        identity_tol,
        "comparative physical-direction convention",
    )

    up = transfer[transfer["direction_sign"] == "up"]
    down = transfer[transfer["direction_sign"] == "down"]
    require(float(up["finite_interaction"].min()) >= -sign_tol, "transfer J>=0")
    require(float(down["finite_interaction"].max()) <= sign_tol, "transfer J<=0")
    require(float(up["interaction_coefficient"].min()) >= -sign_tol, "transfer I>=0")
    require(float(down["interaction_coefficient"].max()) <= sign_tol, "transfer I<=0")

    singletons = mechanism[mechanism["direction_class"] == "single_date"]
    require_close(singletons["finite_interaction"], 0.0, identity_tol, "singleton J")
    require_close(singletons["interaction_coefficient"], 0.0, identity_tol, "singleton I")
    for names, inequality in (
        (["all_up", "ramp_up"], "up"),
        (["all_down", "ramp_down"], "down"),
    ):
        group = mechanism[mechanism["direction_name"].isin(names)]
        if inequality == "up":
            require(float(group["finite_interaction"].min()) >= -sign_tol, "mechanism J>=0")
            require(float(group["interaction_coefficient"].min()) >= -sign_tol, "mechanism I>=0")
        else:
            require(float(group["finite_interaction"].max()) <= sign_tol, "mechanism J<=0")
            require(float(group["interaction_coefficient"].max()) <= sign_tol, "mechanism I<=0")


def check_analysis() -> None:
    config = json.loads((PACKAGE / "config.json").read_text())
    multiplier = float(config["numerics"]["resolution_multiplier"])
    absolute = float(config["numerics"]["absolute_floor_minimum"])
    tables = [
        pd.read_csv(TABLES / "transfer_homotopy_all.csv"),
        pd.read_csv(TABLES / "mechanism_comparator_all.csv"),
        pd.read_csv(TABLES / "comparative_statics_all.csv"),
    ]
    for frame in tables:
        for stem, value in (
            ("loss", "finite_loss"),
            ("interaction", "finite_interaction"),
            ("gain", "gain"),
            ("additive_error", "additive_absolute_error"),
            ("prefix_error", "prefix_absolute_error"),
        ):
            floor = np.maximum(absolute, multiplier * frame[f"{stem}_discrepancy"])
            resolved = np.abs(frame[value]) > floor
            require(
                np.array_equal(resolved, frame[f"{stem}_resolved"].astype(bool)),
                f"{stem} resolution rule",
            )
        status = np.select(
            [frame["gain_resolved"] & (frame["gain"] > 0), frame["gain_resolved"]],
            ["prefix_closer_resolved", "additive_closer_resolved"],
            default="unresolved",
        )
        require(np.array_equal(status, frame["comparison_status"]), "comparison status")
        allowed_add_rel = frame["loss_resolved"] & frame["additive_error_resolved"]
        allowed_prefix_rel = frame["loss_resolved"] & frame["prefix_error_resolved"]
        allowed_improvement = frame["gain_resolved"] & frame["additive_error_resolved"]
        require(
            frame.loc[~allowed_add_rel, "additive_relative_error"].isna().all(),
            "unresolved additive relative errors must be blank",
        )
        require(
            frame.loc[~allowed_prefix_rel, "prefix_relative_error"].isna().all(),
            "unresolved prefix relative errors must be blank",
        )
        require(
            frame.loc[~allowed_improvement, "resolved_error_reduction_fraction"].isna().all(),
            "unresolved improvement ratios must be blank",
        )

    summary = json.loads((RESULTS / "analysis_summary.json").read_text())
    actual = tables[0][np.isclose(tables[0]["eta"], 1.0)]
    require(len(actual) == 30, "actual transfer count")
    require(int(actual["loss_resolved"].sum()) == 26, "reported resolved losses")
    require(
        int(actual["interaction_resolved"].sum()) == 24,
        "reported resolved interactions",
    )
    require(
        int(actual["additive_error_resolved"].sum()) == 14
        and int(actual["prefix_error_resolved"].sum()) == 0,
        "reported resolved prediction-error magnitudes",
    )
    require(
        int(actual["resolved_error_reduction_fraction"].notna().sum()) == 12,
        "reported denominator-resolved improvement ratios",
    )
    require(
        summary["transfer"]["actual_prefix_closer_resolved"]
        == int((actual["comparison_status"] == "prefix_closer_resolved").sum()),
        "summary prefix count",
    )
    require(
        summary["transfer"]["actual_additive_closer_resolved"]
        == int((actual["comparison_status"] == "additive_closer_resolved").sum()),
        "summary additive count",
    )
    require(
        (
            summary["transfer"]["actual_prefix_closer_resolved"],
            summary["transfer"]["actual_additive_closer_resolved"],
            summary["transfer"]["actual_comparison_unresolved"],
        )
        == (20, 0, 10),
        "reported actual-transfer 20-0-10 disposition",
    )
    require(
        summary["transfer"]["actual_prefix_nominally_closer_selected_lattice"]
        == 30
        and summary["transfer"]["actual_prefix_nominally_closer_fine_quadrature"]
        == 29,
        "reported nominal evaluator comparison",
    )
    require(
        summary["transfer"]["actual_opposite_residual_signs_selected_lattice"]
        == 18
        and summary["transfer"]["actual_opposite_residual_signs_all_evaluators"]
        == 10
        and summary["transfer"]["actual_both_component_residual_signs_resolved"]
        == 0,
        "reported residual-sign audit",
    )
    require(
        float(actual["finite_loss_bps"].max()) < 1.0,
        "all actual transfer losses below one basis point",
    )
    base = actual[actual["contract_id"] == "c08_g24_n8"]
    materiality = config["frozen_materiality_gate"]
    recomputed_pass = (
        base.loc[base["loss_resolved"], "finite_loss_bps"].median()
        >= float(materiality["base_median_transfer_bps_min"])
        and int(
            (
                base["loss_resolved"]
                & (
                    base["predicted_interaction_bps"].abs()
                    >= float(materiality["absolute_prefix_change_bps_min"])
                )
            ).sum()
        )
        >= int(materiality["material_base_cases_min"])
    )
    require(
        bool(summary["transfer"]["frozen_materiality_gate"]["pass"])
        == bool(recomputed_pass),
        "materiality gate",
    )
    require(not recomputed_pass, "reported frozen materiality failure")

    mechanism = tables[1]
    local_multidate = mechanism[
        mechanism["strict_local_domain"]
        & (mechanism["direction_class"] == "multidate")
    ]
    require(len(local_multidate) == 270, "local multidate row count")
    require(
        tuple(
            int((local_multidate["comparison_status"] == status).sum())
            for status in (
                "prefix_closer_resolved",
                "additive_closer_resolved",
                "unresolved",
            )
        )
        == (231, 0, 39),
        "reported local mechanism disposition",
    )
    nonlocal_rows = mechanism[~mechanism["strict_local_domain"]]
    require(
        tuple(
            int((nonlocal_rows["comparison_status"] == status).sum())
            for status in (
                "prefix_closer_resolved",
                "additive_closer_resolved",
                "unresolved",
            )
        )
        == (20, 10, 70),
        "reported nonlocal mechanism disposition",
    )

    comparative = tables[2]
    require(
        tuple(
            int((comparative["comparison_status"] == status).sum())
            for status in (
                "prefix_closer_resolved",
                "additive_closer_resolved",
                "unresolved",
            )
        )
        == (241, 0, 29),
        "reported controlled-comparison disposition",
    )
    physical_columns = [
        "date",
        "model",
        "local_cap",
        "global_cap",
        "resets",
        "direction_sign",
        "physical_displacement",
        "beta",
        "boundary",
        "floor_probability",
        "cap_probability",
        "expected_credit",
    ]
    require(
        len(comparative.drop_duplicates(physical_columns)) == 200,
        "controlled design has 200 distinct physical scenarios",
    )

    def monotone_count(
        frame: pd.DataFrame,
        *,
        group: list[str],
        value: str,
        increasing: bool,
    ) -> int:
        count = 0
        for _, rows in frame.groupby(group, sort=False):
            values = rows.sort_values("changed_value")[value].to_numpy(dtype=float)
            differences = np.diff(values)
            count += int(np.all(differences > 0 if increasing else differences < 0))
        return count

    comparative = comparative.assign(
        absolute_interaction_share=np.abs(
            comparative["interaction_coefficient"] / comparative["full_coefficient"]
        )
    )
    horizon = comparative[comparative["experiment"] == "horizon"]
    horizon_group = ["date", "model", "normalization", "direction_sign"]
    require(
        monotone_count(
            horizon,
            group=horizon_group,
            value="absolute_interaction_share",
            increasing=True,
        )
        == 40,
        "horizon interaction-share pattern",
    )
    require(
        monotone_count(
            horizon, group=horizon_group, value="finite_loss", increasing=True
        )
        == 35,
        "horizon loss pattern",
    )
    local_cap = comparative[comparative["experiment"] == "local_cap"]
    cap_group = ["date", "model", "direction_sign"]
    require(
        monotone_count(
            local_cap, group=cap_group, value="finite_loss", increasing=True
        )
        == 20
        and monotone_count(
            local_cap,
            group=cap_group,
            value="absolute_interaction_share",
            increasing=True,
        )
        == 20,
        "local-cap patterns",
    )
    global_cap = comparative[comparative["experiment"] == "global_cap"]
    require(
        monotone_count(
            global_cap, group=cap_group, value="finite_loss", increasing=False
        )
        == 18
        and monotone_count(
            global_cap,
            group=cap_group,
            value="absolute_interaction_share",
            increasing=False,
        )
        == 20,
        "global-cap patterns",
    )
    stress = comparative[comparative["experiment"] == "zero_credit_stress"]
    stress_p0 = stress.drop_duplicates(["date", "changed_value"])
    require(
        monotone_count(
            stress_p0, group=["date"], value="floor_probability", increasing=True
        )
        == 5,
        "volatility-stress floor-probability pattern",
    )
    stress_group = ["date", "direction_sign"]
    require(
        monotone_count(
            stress,
            group=stress_group,
            value="absolute_interaction_share",
            increasing=True,
        )
        == 10
        and monotone_count(
            stress, group=stress_group, value="finite_loss", increasing=False
        )
        == 9,
        "volatility-stress patterns",
    )
    for name in ("transfer_error_comparison", "transfer_residual_decomposition"):
        require((PACKAGE / "figures" / f"{name}.pdf").is_file(), f"figure {name}")
        require((REPO / "paper" / "figures" / f"{name}.pdf").is_file(), f"paper figure {name}")


def main() -> None:
    metadata = check_metadata()
    check_raw(metadata)
    check_analysis()
    analysis = json.loads((RESULTS / "analysis_summary.json").read_text())
    validation = {
        "status": "PASS",
        "source_hashes_stable_during_run": True,
        "config_hash_stable_during_run": True,
        "raw_output_hashes_match": True,
        "raw_row_counts": metadata["row_counts"],
        "actual_transfer_disposition": {
            "prefix_closer_resolved": analysis["transfer"][
                "actual_prefix_closer_resolved"
            ],
            "additive_closer_resolved": analysis["transfer"][
                "actual_additive_closer_resolved"
            ],
            "unresolved": analysis["transfer"]["actual_comparison_unresolved"],
        },
        "controlled_distinct_physical_scenarios": 200,
        "materiality_gate_pass": analysis["transfer"][
            "frozen_materiality_gate"
        ]["pass"],
    }
    (RESULTS / "validation_summary.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n"
    )
    print("prefix comparator validation: PASS")


if __name__ == "__main__":
    main()
