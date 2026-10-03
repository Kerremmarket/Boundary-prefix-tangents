"""Apply predeclared resolution rules and generate publication evidence."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PACKAGE = Path(__file__).resolve().parents[1]
REPO = PACKAGE.parents[1]
RESULTS = PACKAGE / "results"
TABLES = PACKAGE / "tables"
FIGURES = PACKAGE / "figures"
PAPER_FIGURES = REPO / "paper" / "figures"
INTEGRATION = REPO / "research" / "cliquet_integration"


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
METRICS = [
    "finite_loss",
    "finite_singleton_sum",
    "finite_interaction",
    "additive_prediction",
    "full_prefix_prediction",
    "predicted_interaction",
    "one_date_residual",
    "interaction_residual",
    "full_residual",
    "additive_absolute_error",
    "prefix_absolute_error",
    "gain",
    "full_coefficient",
    "additive_coefficient",
    "interaction_coefficient",
    "fresh_diagonal_coefficient",
]


def require_level_coverage(
    frame: pd.DataFrame,
    *,
    name: str,
    keys: list[str],
    expected: dict[str, int],
) -> None:
    """Fail before analysis if a raw table is incomplete or has duplicate keys."""

    observed = frame.groupby("level", dropna=False).size().to_dict()
    if observed != expected:
        raise RuntimeError(
            f"{name}: expected level counts {expected}, observed {observed}"
        )
    duplicate = frame.duplicated(keys + ["level"], keep=False)
    if duplicate.any():
        example = frame.loc[duplicate, keys + ["level"]].head().to_dict("records")
        raise RuntimeError(f"{name}: duplicate level keys, for example {example}")


def direct_change(best: pd.Series, other: pd.Series) -> np.ndarray:
    return np.abs(best.to_numpy(dtype=float) - other.to_numpy(dtype=float))


def apply_resolution(
    data: pd.DataFrame,
    *,
    absolute_floor: float,
    multiplier: float,
    loss_discrepancy: np.ndarray,
    interaction_discrepancy: np.ndarray,
    gain_discrepancy: np.ndarray,
    additive_error_discrepancy: np.ndarray,
    prefix_error_discrepancy: np.ndarray,
) -> pd.DataFrame:
    frame = data.copy()
    frame["loss_discrepancy"] = loss_discrepancy
    frame["interaction_discrepancy"] = interaction_discrepancy
    frame["gain_discrepancy"] = gain_discrepancy
    frame["additive_error_discrepancy"] = additive_error_discrepancy
    frame["prefix_error_discrepancy"] = prefix_error_discrepancy
    frame["loss_resolution_floor"] = np.maximum(
        absolute_floor, multiplier * loss_discrepancy
    )
    frame["interaction_resolution_floor"] = np.maximum(
        absolute_floor, multiplier * interaction_discrepancy
    )
    frame["gain_resolution_floor"] = np.maximum(
        absolute_floor, multiplier * gain_discrepancy
    )
    frame["additive_error_resolution_floor"] = np.maximum(
        absolute_floor, multiplier * additive_error_discrepancy
    )
    frame["prefix_error_resolution_floor"] = np.maximum(
        absolute_floor, multiplier * prefix_error_discrepancy
    )
    frame["loss_resolved"] = (
        frame["finite_loss"] > frame["loss_resolution_floor"]
    )
    frame["interaction_resolved"] = (
        frame["finite_interaction"].abs()
        > frame["interaction_resolution_floor"]
    )
    frame["gain_resolved"] = frame["gain"].abs() > frame["gain_resolution_floor"]
    frame["additive_error_resolved"] = (
        frame["additive_absolute_error"] > frame["additive_error_resolution_floor"]
    )
    frame["prefix_error_resolved"] = (
        frame["prefix_absolute_error"] > frame["prefix_error_resolution_floor"]
    )
    frame["comparison_status"] = np.select(
        [
            frame["gain_resolved"] & (frame["gain"] > 0),
            frame["gain_resolved"] & (frame["gain"] < 0),
        ],
        ["prefix_closer_resolved", "additive_closer_resolved"],
        default="unresolved",
    )
    frame["additive_relative_error"] = np.where(
        frame["loss_resolved"] & frame["additive_error_resolved"],
        frame["additive_absolute_error"] / frame["finite_loss"],
        np.nan,
    )
    frame["prefix_relative_error"] = np.where(
        frame["loss_resolved"] & frame["prefix_error_resolved"],
        frame["prefix_absolute_error"] / frame["finite_loss"],
        np.nan,
    )
    frame["resolved_error_reduction_fraction"] = np.where(
        frame["gain_resolved"] & frame["additive_error_resolved"],
        frame["gain"] / frame["additive_absolute_error"],
        np.nan,
    )
    frame["residuals_opposite_sign"] = (
        frame["one_date_residual"] * frame["interaction_residual"] < 0
    )
    residual_total = (
        frame["one_date_residual"].abs() + frame["interaction_residual"].abs()
    )
    frame["residual_cancellation_ratio"] = np.where(
        residual_total > absolute_floor,
        frame["full_residual"].abs() / residual_total,
        np.nan,
    )
    for column in [
        "finite_loss",
        "finite_singleton_sum",
        "finite_interaction",
        "additive_prediction",
        "full_prefix_prediction",
        "predicted_interaction",
        "one_date_residual",
        "interaction_residual",
        "full_residual",
        "additive_absolute_error",
        "prefix_absolute_error",
        "gain",
        "loss_discrepancy",
        "interaction_discrepancy",
        "gain_discrepancy",
        "additive_error_discrepancy",
        "prefix_error_discrepancy",
    ]:
        frame[f"{column}_bps"] = frame[column] * 10_000
    return frame


def transfer_table(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    raw = pd.read_csv(RESULTS / "transfer_lattice_levels.csv")
    require_level_coverage(
        raw,
        name="transfer_lattice_levels.csv",
        keys=TRANSFER_KEYS,
        expected={"fine": 150, "primary": 150, "warning_r1": 5},
    )
    primary = raw[raw["level"] == "primary"].copy()
    fine = raw[raw["level"] == "fine"].copy()
    warning = raw[raw["level"] == "warning_r1"].copy()
    warning_keys = set(map(tuple, warning[TRANSFER_KEYS].to_numpy()))
    best = fine[
        ~fine[TRANSFER_KEYS].apply(tuple, axis=1).isin(warning_keys)
    ].copy()
    best = pd.concat([best, warning], ignore_index=True)
    if len(best) != 150:
        raise RuntimeError(f"expected 150 best transfer rows, found {len(best)}")

    primary_keep = primary[TRANSFER_KEYS + METRICS].rename(
        columns={column: f"{column}_primary" for column in METRICS}
    )
    best = best.merge(primary_keep, on=TRANSFER_KEYS, validate="one_to_one")
    fine_keep = fine[TRANSFER_KEYS + METRICS].rename(
        columns={column: f"{column}_fine" for column in METRICS}
    )
    best = best.merge(fine_keep, on=TRANSFER_KEYS, validate="one_to_one")

    quadrature = pd.read_csv(RESULTS / "transfer_quadrature_actual.csv")
    qkeys = TRANSFER_KEYS[:-1]
    require_level_coverage(
        quadrature,
        name="transfer_quadrature_actual.csv",
        keys=qkeys,
        expected={"fine_quadrature": 30, "warning_r1_quadrature": 1},
    )
    qmetrics = ["finite_loss", "finite_singleton_sum", "finite_interaction"]
    quadrature_fine = quadrature[quadrature["level"] == "fine_quadrature"][
        qkeys + qmetrics
    ].rename(columns={column: f"{column}_quadrature_fine" for column in qmetrics})
    quadrature_warning = quadrature[
        quadrature["level"] == "warning_r1_quadrature"
    ][
        qkeys
        + qmetrics
    ].rename(
        columns={column: f"{column}_quadrature_warning" for column in qmetrics}
    )
    best = best.merge(
        quadrature_fine, on=qkeys, how="left", validate="many_to_one"
    )
    best = best.merge(
        quadrature_warning, on=qkeys, how="left", validate="many_to_one"
    )
    for metric in qmetrics:
        best[f"{metric}_quadrature"] = best[
            f"{metric}_quadrature_warning"
        ].fillna(best[f"{metric}_quadrature_fine"])
    not_actual = ~np.isclose(best["eta"], 1.0)
    quadrature_columns = [
        f"{metric}{suffix}"
        for metric in qmetrics
        for suffix in ("_quadrature", "_quadrature_fine", "_quadrature_warning")
    ]
    best.loc[not_actual, quadrature_columns] = np.nan

    audited = pd.read_csv(INTEGRATION / "tables" / "finance_all_30_audited.csv")
    audit_keep = audited[
        qkeys + ["audit_estimate", "audit_error_bound", "reporting_status"]
    ].rename(
        columns={
            "audit_estimate": "prior_audit_estimate",
            "audit_error_bound": "prior_audit_discrepancy",
            "reporting_status": "prior_audit_status",
        }
    )
    best = best.merge(audit_keep, on=qkeys, how="left", validate="many_to_one")

    loss_grid_primary = direct_change(
        best["finite_loss_fine"], best["finite_loss_primary"]
    )
    loss_grid_warning = direct_change(best["finite_loss"], best["finite_loss_fine"])
    loss_quad_best = np.abs(
        best["finite_loss"] - best["finite_loss_quadrature"]
    ).fillna(0.0).to_numpy()
    loss_quad_fine = np.abs(
        best["finite_loss_fine"] - best["finite_loss_quadrature_fine"]
    ).fillna(0.0).to_numpy()
    loss_quad_step = np.abs(
        best["finite_loss_quadrature"]
        - best["finite_loss_quadrature_fine"]
    ).fillna(0.0).to_numpy()
    loss_identity = best["residual_identity_error"].abs().to_numpy()
    prior_loss_discrepancy = np.where(
        np.isclose(best["eta"], 1.0), best["prior_audit_discrepancy"], 0.0
    )
    loss_discrepancy = np.maximum.reduce(
        [
            loss_grid_primary,
            loss_grid_warning,
            loss_quad_best,
            loss_quad_fine,
            loss_quad_step,
            loss_identity,
            prior_loss_discrepancy,
        ]
    )

    interaction_grid_primary = direct_change(
        best["finite_interaction_fine"], best["finite_interaction_primary"]
    )
    interaction_grid_warning = direct_change(
        best["finite_interaction"], best["finite_interaction_fine"]
    )
    interaction_quad_best = np.abs(
        best["finite_interaction"] - best["finite_interaction_quadrature"]
    ).fillna(0.0).to_numpy()
    interaction_quad_fine = np.abs(
        best["finite_interaction_fine"]
        - best["finite_interaction_quadrature_fine"]
    ).fillna(0.0).to_numpy()
    interaction_quad_step = np.abs(
        best["finite_interaction_quadrature"]
        - best["finite_interaction_quadrature_fine"]
    ).fillna(0.0).to_numpy()
    interaction_discrepancy = np.maximum.reduce(
        [
            interaction_grid_primary,
            interaction_grid_warning,
            interaction_quad_best,
            interaction_quad_fine,
            interaction_quad_step,
        ]
    )

    gain_quadrature = (
        (best["finite_loss_quadrature"] - best["additive_prediction"]).abs()
        - (best["finite_loss_quadrature"] - best["full_prefix_prediction"]).abs()
    )
    gain_quadrature_fine = (
        (
            best["finite_loss_quadrature_fine"]
            - best["additive_prediction_fine"]
        ).abs()
        - (
            best["finite_loss_quadrature_fine"]
            - best["full_prefix_prediction_fine"]
        ).abs()
    )
    gain_discrepancy = np.maximum.reduce(
        [
            direct_change(best["gain_fine"], best["gain_primary"]),
            direct_change(best["gain"], best["gain_fine"]),
            (best["gain"] - gain_quadrature).abs().fillna(0.0).to_numpy(),
            (best["gain_fine"] - gain_quadrature_fine)
            .abs()
            .fillna(0.0)
            .to_numpy(),
            (gain_quadrature - gain_quadrature_fine)
            .abs()
            .fillna(0.0)
            .to_numpy(),
        ]
    )

    additive_error_quadrature = (
        best["finite_loss_quadrature"] - best["additive_prediction"]
    ).abs()
    additive_error_quadrature_fine = (
        best["finite_loss_quadrature_fine"] - best["additive_prediction_fine"]
    ).abs()
    additive_error_discrepancy = np.maximum.reduce(
        [
            direct_change(
                best["additive_absolute_error_fine"],
                best["additive_absolute_error_primary"],
            ),
            direct_change(
                best["additive_absolute_error"],
                best["additive_absolute_error_fine"],
            ),
            (best["additive_absolute_error"] - additive_error_quadrature)
            .abs()
            .fillna(0.0)
            .to_numpy(),
            (best["additive_absolute_error_fine"] - additive_error_quadrature_fine)
            .abs()
            .fillna(0.0)
            .to_numpy(),
            (additive_error_quadrature - additive_error_quadrature_fine)
            .abs()
            .fillna(0.0)
            .to_numpy(),
        ]
    )
    prefix_error_quadrature = (
        best["finite_loss_quadrature"] - best["full_prefix_prediction"]
    ).abs()
    prefix_error_quadrature_fine = (
        best["finite_loss_quadrature_fine"]
        - best["full_prefix_prediction_fine"]
    ).abs()
    prefix_error_discrepancy = np.maximum.reduce(
        [
            direct_change(
                best["prefix_absolute_error_fine"],
                best["prefix_absolute_error_primary"],
            ),
            direct_change(
                best["prefix_absolute_error"], best["prefix_absolute_error_fine"]
            ),
            (best["prefix_absolute_error"] - prefix_error_quadrature)
            .abs()
            .fillna(0.0)
            .to_numpy(),
            (best["prefix_absolute_error_fine"] - prefix_error_quadrature_fine)
            .abs()
            .fillna(0.0)
            .to_numpy(),
            (prefix_error_quadrature - prefix_error_quadrature_fine)
            .abs()
            .fillna(0.0)
            .to_numpy(),
        ]
    )
    best["gain_quadrature"] = gain_quadrature
    best["gain_quadrature_fine"] = gain_quadrature_fine
    best = apply_resolution(
        best,
        absolute_floor=float(config["numerics"]["absolute_floor_minimum"]),
        multiplier=float(config["numerics"]["resolution_multiplier"]),
        loss_discrepancy=loss_discrepancy,
        interaction_discrepancy=interaction_discrepancy,
        gain_discrepancy=gain_discrepancy,
        additive_error_discrepancy=additive_error_discrepancy,
        prefix_error_discrepancy=prefix_error_discrepancy,
    )

    best["one_date_residual_quadrature"] = (
        best["finite_singleton_sum_quadrature"] - best["additive_prediction"]
    )
    best["interaction_residual_quadrature"] = (
        best["finite_interaction_quadrature"] - best["predicted_interaction"]
    )
    best["full_residual_quadrature"] = (
        best["finite_loss_quadrature"] - best["full_prefix_prediction"]
    )
    best["one_date_residual_quadrature_fine"] = (
        best["finite_singleton_sum_quadrature_fine"]
        - best["additive_prediction_fine"]
    )
    best["interaction_residual_quadrature_fine"] = (
        best["finite_interaction_quadrature_fine"]
        - best["predicted_interaction_fine"]
    )
    best["full_residual_quadrature_fine"] = (
        best["finite_loss_quadrature_fine"]
        - best["full_prefix_prediction_fine"]
    )
    for column in ("one_date_residual", "interaction_residual", "full_residual"):
        discrepancy = np.maximum.reduce(
            [
                direct_change(best[f"{column}_fine"], best[f"{column}_primary"]),
                direct_change(best[column], best[f"{column}_fine"]),
                (best[column] - best[f"{column}_quadrature"])
                .abs()
                .fillna(0.0)
                .to_numpy(),
                (best[f"{column}_fine"] - best[f"{column}_quadrature_fine"])
                .abs()
                .fillna(0.0)
                .to_numpy(),
                (
                    best[f"{column}_quadrature"]
                    - best[f"{column}_quadrature_fine"]
                )
                .abs()
                .fillna(0.0)
                .to_numpy(),
            ]
        )
        best[f"{column}_discrepancy"] = discrepancy
        best[f"{column}_sign_resolved"] = (
            best[column].abs()
            > np.maximum(
                float(config["numerics"]["absolute_floor_minimum"]),
                float(config["numerics"]["resolution_multiplier"]) * discrepancy,
            )
        )
    selected_opposite = best["one_date_residual"] * best["interaction_residual"] < 0
    primary_opposite = (
        best["one_date_residual_primary"] * best["interaction_residual_primary"] < 0
    )
    fine_opposite = (
        best["one_date_residual_fine"] * best["interaction_residual_fine"] < 0
    )
    quadrature_opposite = (
        best["one_date_residual_quadrature"]
        * best["interaction_residual_quadrature"]
        < 0
    )
    quadrature_fine_opposite = (
        best["one_date_residual_quadrature_fine"]
        * best["interaction_residual_quadrature_fine"]
        < 0
    )
    best["residuals_opposite_sign_all_available_evaluators"] = (
        selected_opposite
        & primary_opposite
        & fine_opposite
        & (best["finite_loss_quadrature_fine"].isna() | quadrature_fine_opposite)
        & (best["finite_loss_quadrature"].isna() | quadrature_opposite)
    )

    best["best_minus_prior_audit"] = np.where(
        np.isclose(best["eta"], 1.0),
        best["finite_loss"] - best["prior_audit_estimate"],
        np.nan,
    )
    actual = best[np.isclose(best["eta"], 1.0)].copy()
    base_actual = actual[actual["contract_id"] == "c08_g24_n8"]
    resolved_base = base_actual[base_actual["loss_resolved"]]
    materiality = config["frozen_materiality_gate"]
    qualifying_base = base_actual[
        base_actual["loss_resolved"]
        & (
            base_actual["predicted_interaction_bps"].abs()
            >= float(materiality["absolute_prefix_change_bps_min"])
        )
    ]
    base_median_resolved = float(resolved_base["finite_loss_bps"].median())
    materiality_pass = bool(
        base_median_resolved
        >= float(materiality["base_median_transfer_bps_min"])
        and len(qualifying_base) >= int(materiality["material_base_cases_min"])
    )

    metrics = {
        "homotopy_rows": int(len(best)),
        "actual_rows": int(len(actual)),
        "actual_loss_resolved": int(actual["loss_resolved"].sum()),
        "actual_interaction_resolved": int(actual["interaction_resolved"].sum()),
        "actual_prefix_nominally_closer_selected_lattice": int(
            (actual["gain"] > 0).sum()
        ),
        "actual_prefix_nominally_closer_fine_quadrature": int(
            (actual["gain_quadrature_fine"] > 0).sum()
        ),
        "actual_gain_sign_agreement_selected_lattice_and_fine_quadrature": int(
            (
                np.sign(actual["gain"])
                == np.sign(actual["gain_quadrature_fine"])
            ).sum()
        ),
        "actual_prefix_closer_resolved": int(
            (actual["comparison_status"] == "prefix_closer_resolved").sum()
        ),
        "actual_additive_closer_resolved": int(
            (actual["comparison_status"] == "additive_closer_resolved").sum()
        ),
        "actual_comparison_unresolved": int(
            (actual["comparison_status"] == "unresolved").sum()
        ),
        "actual_opposite_residual_signs_selected_lattice": int(
            actual["residuals_opposite_sign"].sum()
        ),
        "actual_opposite_residual_signs_all_evaluators": int(
            actual["residuals_opposite_sign_all_available_evaluators"].sum()
        ),
        "actual_both_component_residual_signs_resolved": int(
            (
                actual["one_date_residual_sign_resolved"]
                & actual["interaction_residual_sign_resolved"]
            ).sum()
        ),
        "actual_resolved_improvement_ratios": int(
            actual["resolved_error_reduction_fraction"].notna().sum()
        ),
        "actual_maximum_loss_bps_selected_lattice": float(
            actual["finite_loss_bps"].max()
        ),
        "actual_median_loss_bps_selected_lattice": float(
            actual["finite_loss_bps"].median()
        ),
        "actual_median_additive_error_bps_selected_lattice": float(
            actual["additive_absolute_error_bps"].median()
        ),
        "actual_median_prefix_error_bps_selected_lattice": float(
            actual["prefix_absolute_error_bps"].median()
        ),
        "actual_gain_bps_range_selected_lattice": [
            float(actual["gain_bps"].min()),
            float(actual["gain_bps"].max()),
        ],
        "descriptive_spearman_loss_vs_boundary_displacement_squared": float(
            actual["finite_loss"].corr(
                actual["boundary_difference"] ** 2, method="spearman"
            )
        ),
        "descriptive_spearman_loss_vs_unit_direction_full_coefficient": float(
            actual["finite_loss"].corr(
                actual["full_coefficient"]
                / (actual["boundary_difference"] ** 2),
                method="spearman",
            )
        ),
        "maximum_absolute_prior_audit_difference": float(
            actual["best_minus_prior_audit"].abs().max()
        ),
        "frozen_materiality_gate": {
            "pass": materiality_pass,
            "base_attempted": int(len(base_actual)),
            "base_loss_resolved": int(len(resolved_base)),
            "base_median_loss_bps_all_rows": float(
                base_actual["finite_loss_bps"].median()
            ),
            "base_median_loss_bps_resolved_rows": base_median_resolved,
            "required_base_median_loss_bps": float(
                materiality["base_median_transfer_bps_min"]
            ),
            "maximum_base_absolute_predicted_interaction_bps": float(
                base_actual["predicted_interaction_bps"].abs().max()
            ),
            "required_absolute_predicted_interaction_bps": float(
                materiality["absolute_prefix_change_bps_min"]
            ),
            "qualifying_base_cases": int(len(qualifying_base)),
            "required_qualifying_base_cases": int(
                materiality["material_base_cases_min"]
            ),
        },
    }
    return best, actual, metrics


def mechanism_table(config: dict) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(RESULTS / "mechanism_lattice_levels.csv")
    require_level_coverage(
        raw,
        name="mechanism_lattice_levels.csv",
        keys=MECHANISM_KEYS,
        expected={"fine": 1140, "primary": 1140},
    )
    fine = raw[raw["level"] == "fine"].copy()
    primary = raw[raw["level"] == "primary"].copy()
    primary_keep = primary[MECHANISM_KEYS + METRICS].rename(
        columns={column: f"{column}_primary" for column in METRICS}
    )
    fine = fine.merge(primary_keep, on=MECHANISM_KEYS, validate="one_to_one")
    prior = pd.read_csv(INTEGRATION / "tables" / "mechanism_all_rows_audited.csv")
    prior_keep = prior[
        MECHANISM_KEYS
        + ["uncapped_forward_loss", "audit_error_scale", "audit_resolved"]
    ].rename(
        columns={
            "uncapped_forward_loss": "prior_uncapped_loss",
            "audit_error_scale": "prior_loss_discrepancy",
            "audit_resolved": "prior_loss_resolved",
        }
    )
    fine = fine.merge(prior_keep, on=MECHANISM_KEYS, validate="one_to_one")
    quadrature = pd.read_csv(
        INTEGRATION / "results" / "gate_quadrature_audit.csv"
    )
    quadrature_keep = quadrature[
        MECHANISM_KEYS + ["uncapped_quadrature_direct"]
    ].rename(columns={"uncapped_quadrature_direct": "finite_loss_quadrature"})
    fine = fine.merge(
        quadrature_keep, on=MECHANISM_KEYS, how="left", validate="one_to_one"
    )
    loss_grid = direct_change(fine["finite_loss"], fine["finite_loss_primary"])
    loss_prior = np.abs(
        fine["finite_loss"] - fine["prior_uncapped_loss"]
    ).to_numpy()
    loss_discrepancy = np.maximum.reduce(
        [loss_grid, loss_prior, fine["prior_loss_discrepancy"].to_numpy()]
    )
    interaction_discrepancy = direct_change(
        fine["finite_interaction"], fine["finite_interaction_primary"]
    )
    gain_grid = direct_change(fine["gain"], fine["gain_primary"])
    gain_quadrature = (
        (
            (fine["finite_loss_quadrature"] - fine["additive_prediction"]).abs()
            - (fine["finite_loss_quadrature"] - fine["full_prefix_prediction"]).abs()
        )
        - fine["gain"]
    ).abs().fillna(0.0).to_numpy()
    gain_discrepancy = np.maximum(gain_grid, gain_quadrature)
    additive_error_grid = direct_change(
        fine["additive_absolute_error"], fine["additive_absolute_error_primary"]
    )
    additive_error_quadrature = (
        (fine["finite_loss_quadrature"] - fine["additive_prediction"]).abs()
        - fine["additive_absolute_error"]
    ).abs().fillna(0.0).to_numpy()
    additive_error_discrepancy = np.maximum(
        additive_error_grid, additive_error_quadrature
    )
    prefix_error_grid = direct_change(
        fine["prefix_absolute_error"], fine["prefix_absolute_error_primary"]
    )
    prefix_error_quadrature = (
        (fine["finite_loss_quadrature"] - fine["full_prefix_prediction"]).abs()
        - fine["prefix_absolute_error"]
    ).abs().fillna(0.0).to_numpy()
    prefix_error_discrepancy = np.maximum(
        prefix_error_grid, prefix_error_quadrature
    )
    fine = apply_resolution(
        fine,
        absolute_floor=float(config["numerics"]["absolute_floor_minimum"]),
        multiplier=float(config["numerics"]["resolution_multiplier"]),
        loss_discrepancy=loss_discrepancy,
        interaction_discrepancy=interaction_discrepancy,
        gain_discrepancy=gain_discrepancy,
        additive_error_discrepancy=additive_error_discrepancy,
        prefix_error_discrepancy=prefix_error_discrepancy,
    )
    summary: dict[str, object] = {
        "rows": int(len(fine)),
        "maximum_source_loss_difference": float(
            (fine["finite_loss"] - fine["prior_uncapped_loss"]).abs().max()
        ),
        "maximum_residual_identity_error": float(
            fine["residual_identity_error"].abs().max()
        ),
        "singleton_maximum_absolute_interaction": float(
            fine.loc[
                fine["direction_class"] == "single_date", "finite_interaction"
            ].abs().max()
        ),
        "strictly_local_rows": int(fine["strict_local_domain"].sum()),
        "nonlocal_rows": int((~fine["strict_local_domain"]).sum()),
        "by_domain": {},
    }
    for domain_name, domain in (
        ("strictly_local", fine[fine["strict_local_domain"]]),
        ("nonlocal", fine[~fine["strict_local_domain"]]),
    ):
        domain_summary: dict[str, object] = {
            "rows": int(len(domain)),
            "prefix_closer_resolved": int(
                (domain["comparison_status"] == "prefix_closer_resolved").sum()
            ),
            "additive_closer_resolved": int(
                (domain["comparison_status"] == "additive_closer_resolved").sum()
            ),
            "comparison_unresolved": int(
                (domain["comparison_status"] == "unresolved").sum()
            ),
            "by_direction": {},
        }
        for name, group in domain.groupby("direction_name", sort=True):
            domain_summary["by_direction"][name] = {
                "rows": int(len(group)),
                "loss_resolved": int(group["loss_resolved"].sum()),
                "gain_positive": int((group["gain"] > 0).sum()),
                "gain_resolved": int(group["gain_resolved"].sum()),
                "prefix_closer_resolved": int(
                    (group["comparison_status"] == "prefix_closer_resolved").sum()
                ),
                "additive_closer_resolved": int(
                    (group["comparison_status"] == "additive_closer_resolved").sum()
                ),
                "comparison_unresolved": int(
                    (group["comparison_status"] == "unresolved").sum()
                ),
            }
        summary["by_domain"][domain_name] = domain_summary
    return fine, summary


def comparative_table(config: dict) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(RESULTS / "comparative_statics_levels.csv")
    require_level_coverage(
        raw,
        name="comparative_statics_levels.csv",
        keys=COMPARATIVE_KEYS,
        expected={"fine": 270, "primary": 270},
    )
    fine = raw[raw["level"] == "fine"].copy()
    primary = raw[raw["level"] == "primary"].copy()
    primary_keep = primary[COMPARATIVE_KEYS + METRICS].rename(
        columns={column: f"{column}_primary" for column in METRICS}
    )
    fine = fine.merge(primary_keep, on=COMPARATIVE_KEYS, validate="one_to_one")
    loss_discrepancy = direct_change(fine["finite_loss"], fine["finite_loss_primary"])
    interaction_discrepancy = direct_change(
        fine["finite_interaction"], fine["finite_interaction_primary"]
    )
    gain_discrepancy = direct_change(fine["gain"], fine["gain_primary"])
    additive_error_discrepancy = direct_change(
        fine["additive_absolute_error"], fine["additive_absolute_error_primary"]
    )
    prefix_error_discrepancy = direct_change(
        fine["prefix_absolute_error"], fine["prefix_absolute_error_primary"]
    )
    fine = apply_resolution(
        fine,
        absolute_floor=float(config["numerics"]["absolute_floor_minimum"]),
        multiplier=float(config["numerics"]["resolution_multiplier"]),
        loss_discrepancy=loss_discrepancy,
        interaction_discrepancy=interaction_discrepancy,
        gain_discrepancy=gain_discrepancy,
        additive_error_discrepancy=additive_error_discrepancy,
        prefix_error_discrepancy=prefix_error_discrepancy,
    )
    horizon_boundaries = (
        fine[fine["experiment"] == "horizon"]
        .groupby(["date", "model", "scenario_id"])["boundary"]
        .first()
        .reset_index()
    )
    horizon_spans = horizon_boundaries.groupby(["date", "model"])["boundary"].agg(
        lambda values: float(values.max() - values.min())
    )
    summary: dict[str, object] = {
        "rows": int(len(fine)),
        "all_strictly_local": bool(fine["strict_local_domain"].all()),
        "maximum_trace_reconciliation_error": float(
            fine["trace_reconciliation_error"].max()
        ),
        "minimum_boundary_atom_distance": float(
            fine["boundary_atom_distance"].min()
        ),
        "maximum_horizon_boundary_span": float(horizon_spans.max()),
        "by_experiment": {},
    }
    for name, group in fine.groupby("experiment", sort=True):
        summary["by_experiment"][name] = {
            "rows": int(len(group)),
            "loss_resolved": int(group["loss_resolved"].sum()),
            "interaction_resolved": int(group["interaction_resolved"].sum()),
            "gain_resolved": int(group["gain_resolved"].sum()),
            "prefix_closer_resolved": int(
                (group["comparison_status"] == "prefix_closer_resolved").sum()
            ),
            "additive_closer_resolved": int(
                (group["comparison_status"] == "additive_closer_resolved").sum()
            ),
            "comparison_unresolved": int(
                (group["comparison_status"] == "unresolved").sum()
            ),
            "finite_loss_bps_range": [
                float(group["finite_loss_bps"].min()),
                float(group["finite_loss_bps"].max()),
            ],
            "interaction_share_range": [
                float((group["interaction_coefficient"] / group["full_coefficient"]).min()),
                float((group["interaction_coefficient"] / group["full_coefficient"]).max()),
            ],
        }
    return fine, summary


def error_figure(transfer: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    colors = {"additive": "#A44A3F", "prefix": "#285F8F"}
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8), sharey=True)
    for axis, sign in zip(axes, ("up", "down")):
        subset = transfer[transfer["direction_sign"] == sign]
        for _, path in subset.groupby(
            ["date", "contract_id", "reference_model"], sort=False
        ):
            path = path.sort_values("eta")
            axis.plot(
                path["eta"],
                np.maximum(path["additive_absolute_error_bps"], 1e-8),
                color=colors["additive"],
                alpha=0.15,
                linewidth=0.8,
            )
            axis.plot(
                path["eta"],
                np.maximum(path["prefix_absolute_error_bps"], 1e-8),
                color=colors["prefix"],
                alpha=0.15,
                linewidth=0.8,
            )
        for label, column in (
            ("additive one-date", "additive_absolute_error_bps"),
            ("ordered prefix", "prefix_absolute_error_bps"),
        ):
            median = subset.groupby("eta")[column].median().sort_index()
            axis.plot(
                median.index,
                np.maximum(median.values, 1e-8),
                color=colors["additive" if label.startswith("additive") else "prefix"],
                marker="o",
                markersize=3.5,
                linewidth=2,
                label=label,
            )
        actual = subset[np.isclose(subset["eta"], 1.0)]
        unresolved = actual[~actual["gain_resolved"]]
        axis.scatter(
            unresolved["eta"],
            np.maximum(unresolved["prefix_absolute_error_bps"], 1e-8),
            marker="x",
            color="black",
            s=25,
            zorder=5,
        )
        axis.axvline(1.0, color="black", linestyle="--", linewidth=0.9)
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.invert_xaxis()
        axis.grid(alpha=0.18)
        axis.set_title(f"common {sign} transfer")
        axis.set_xlabel(r"transfer homotopy $\eta$ ($\eta=1$ is actual)")
    axes[0].set_ylabel("absolute prediction error (bp of notional)")
    axes[1].legend(frameon=False, fontsize=8)
    fig.text(
        0.5,
        0.01,
        "Thin lines retain every case; crosses mark unresolved actual-transfer gains.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for suffix in ("pdf", "png"):
        fig.savefig(FIGURES / f"transfer_error_comparison.{suffix}", dpi=220)
        fig.savefig(PAPER_FIGURES / f"transfer_error_comparison.{suffix}", dpi=220)
    plt.close(fig)


def residual_figure(actual: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    base = actual[actual["contract_id"] == "c08_g24_n8"].copy()
    base["label"] = base["date"].str.slice(0, 4) + " " + np.where(
        base["reference_model"] == "black_scholes_atm", "BS", "Mix"
    )
    base = base.sort_values(["date", "reference_model"])
    y = np.arange(len(base))
    fig, axis = plt.subplots(figsize=(7.4, 4.6))
    axis.barh(
        y - 0.15,
        base["one_date_residual_bps"],
        height=0.28,
        color="#C88B3A",
        label="one-date residual",
    )
    axis.barh(
        y + 0.15,
        base["interaction_residual_bps"],
        height=0.28,
        color="#4B8F8C",
        label="interaction residual",
    )
    axis.scatter(
        base["full_residual_bps"],
        y,
        color="black",
        marker="|",
        s=70,
        label="sum (full residual)",
        zorder=5,
    )
    axis.axvline(0.0, color="black", linewidth=0.8)
    axis.set_yticks(y, base["label"])
    axis.set_xlabel("signed residual (bp of notional)")
    axis.set_title("Actual base-tuple transfer residual decomposition")
    axis.grid(axis="x", alpha=0.18)
    axis.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(FIGURES / f"transfer_residual_decomposition.{suffix}", dpi=220)
        fig.savefig(PAPER_FIGURES / f"transfer_residual_decomposition.{suffix}", dpi=220)
    plt.close(fig)


def write_compact_summaries(
    transfer: pd.DataFrame,
    actual: pd.DataFrame,
    mechanism: pd.DataFrame,
    comparative: pd.DataFrame,
) -> None:
    """Write manuscript-facing summaries from the complete evidence tables."""

    homotopy = (
        transfer.groupby(["eta", "direction_sign"], sort=True)
        .agg(
            attempted=("date", "size"),
            loss_resolved=("loss_resolved", "sum"),
            gain_resolved=("gain_resolved", "sum"),
            prefix_closer_resolved=(
                "comparison_status",
                lambda values: int((values == "prefix_closer_resolved").sum()),
            ),
            additive_closer_resolved=(
                "comparison_status",
                lambda values: int((values == "additive_closer_resolved").sum()),
            ),
            comparison_unresolved=(
                "comparison_status",
                lambda values: int((values == "unresolved").sum()),
            ),
            median_additive_error_bps=("additive_absolute_error_bps", "median"),
            median_prefix_error_bps=("prefix_absolute_error_bps", "median"),
            median_gain_bps=("gain_bps", "median"),
        )
        .reset_index()
    )
    homotopy.to_csv(
        TABLES / "transfer_homotopy_summary.csv", index=False, float_format="%.17g"
    )

    residual_subset = actual[actual["loss_resolved"] & actual["interaction_resolved"]]
    residual_summary = pd.DataFrame(
        [
            {
                "attempted_actual_transfers": len(actual),
                "loss_and_interaction_resolved": len(residual_subset),
                "opposite_residual_signs_selected_lattice_all_rows": int(
                    actual["residuals_opposite_sign"].sum()
                ),
                "opposite_residual_signs_all_evaluators_all_rows": int(
                    actual[
                        "residuals_opposite_sign_all_available_evaluators"
                    ].sum()
                ),
                "both_component_residual_signs_resolved": int(
                    (
                        actual["one_date_residual_sign_resolved"]
                        & actual["interaction_residual_sign_resolved"]
                    ).sum()
                ),
                "selected_lattice_strong_cancellation_resolved_subset": int(
                    (residual_subset["residual_cancellation_ratio"] < 0.5).sum()
                ),
                "selected_lattice_median_abs_one_date_residual_share": float(
                    np.median(
                        np.abs(
                            residual_subset["one_date_residual"]
                            / residual_subset["finite_singleton_sum"]
                        )
                    )
                ),
                "selected_lattice_median_abs_interaction_residual_share": float(
                    np.median(
                        np.abs(
                            residual_subset["interaction_residual"]
                            / residual_subset["finite_interaction"]
                        )
                    )
                ),
                "selected_lattice_median_abs_full_residual_share": float(
                    np.median(
                        np.abs(
                            residual_subset["full_residual"]
                            / residual_subset["finite_loss"]
                        )
                    )
                ),
            }
        ]
    )
    residual_summary.to_csv(
        TABLES / "transfer_residual_summary.csv", index=False, float_format="%.17g"
    )

    multidate = mechanism[mechanism["direction_class"] == "multidate"].copy()
    multidate["domain"] = np.where(
        multidate["strict_local_domain"], "strictly_local", "nonlocal"
    )
    direction_summary = (
        multidate.groupby(["domain", "direction_name"], sort=True)
        .agg(
            attempted=("date", "size"),
            loss_resolved=("loss_resolved", "sum"),
            interaction_resolved=("interaction_resolved", "sum"),
            prefix_closer_resolved=(
                "comparison_status",
                lambda values: int((values == "prefix_closer_resolved").sum()),
            ),
            additive_closer_resolved=(
                "comparison_status",
                lambda values: int((values == "additive_closer_resolved").sum()),
            ),
            comparison_unresolved=(
                "comparison_status",
                lambda values: int((values == "unresolved").sum()),
            ),
            positive_interactions=(
                "interaction_coefficient", lambda values: int((values > 0).sum())
            ),
            negative_interactions=(
                "interaction_coefficient", lambda values: int((values < 0).sum())
            ),
        )
        .reset_index()
    )
    direction_summary.to_csv(
        TABLES / "mechanism_direction_summary.csv", index=False, float_format="%.17g"
    )

    comparative_work = comparative.copy()
    comparative_work["interaction_share"] = (
        comparative_work["interaction_coefficient"]
        / comparative_work["full_coefficient"]
    )
    comparative_summary = (
        comparative_work.groupby(
            ["experiment", "changed_value", "normalization", "direction_sign"],
            sort=True,
        )
        .agg(
            attempted=("date", "size"),
            loss_resolved=("loss_resolved", "sum"),
            gain_resolved=("gain_resolved", "sum"),
            prefix_closer_resolved=(
                "comparison_status",
                lambda values: int((values == "prefix_closer_resolved").sum()),
            ),
            additive_closer_resolved=(
                "comparison_status",
                lambda values: int((values == "additive_closer_resolved").sum()),
            ),
            comparison_unresolved=(
                "comparison_status",
                lambda values: int((values == "unresolved").sum()),
            ),
            median_floor_probability=("floor_probability", "median"),
            median_boundary=("boundary", "median"),
            median_finite_loss_bps=("finite_loss_bps", "median"),
            median_additive_prediction_bps=("additive_prediction_bps", "median"),
            median_interaction_prediction_bps=("predicted_interaction_bps", "median"),
            median_interaction_share=("interaction_share", "median"),
            median_gain_bps=("gain_bps", "median"),
            median_target_density_last_date=("target_density_last_date", "median"),
            median_minimum_left_gap_slope=("minimum_left_gap_slope", "median"),
        )
        .reset_index()
    )
    comparative_summary.to_csv(
        TABLES / "comparative_statics_summary.csv", index=False, float_format="%.17g"
    )


def run() -> None:
    config = json.loads((PACKAGE / "config.json").read_text())
    TABLES.mkdir(parents=True, exist_ok=True)
    transfer, actual, transfer_summary = transfer_table(config)
    mechanism, mechanism_summary = mechanism_table(config)
    comparative, comparative_summary = comparative_table(config)
    transfer.to_csv(
        TABLES / "transfer_homotopy_all.csv", index=False, float_format="%.17g"
    )
    actual.to_csv(
        TABLES / "transfer_actual_all.csv", index=False, float_format="%.17g"
    )
    actual[actual["contract_id"] == "c08_g24_n8"].to_csv(
        TABLES / "transfer_actual_base.csv", index=False, float_format="%.17g"
    )
    mechanism.to_csv(
        TABLES / "mechanism_comparator_all.csv", index=False, float_format="%.17g"
    )
    comparative.to_csv(
        TABLES / "comparative_statics_all.csv", index=False, float_format="%.17g"
    )
    write_compact_summaries(transfer, actual, mechanism, comparative)
    error_figure(transfer)
    residual_figure(actual)
    summary = {
        "transfer": transfer_summary,
        "mechanism": mechanism_summary,
        "comparative_statics": comparative_summary,
    }
    (RESULTS / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    run()
