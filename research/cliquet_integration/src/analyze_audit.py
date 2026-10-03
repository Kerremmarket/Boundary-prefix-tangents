"""Create corrected audit tables, summaries, and the manuscript figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INTEGRATION = Path(__file__).resolve().parents[1]
RESEARCH = INTEGRATION.parent
SOURCE = RESEARCH / "cliquet_feasibility"
RESULTS = INTEGRATION / "results"
TABLES = INTEGRATION / "tables"
FIGURES = INTEGRATION / "figures"


KEYS = ["date", "contract_id", "model", "policy_id"]
FINANCE_KEYS = [
    "date",
    "contract_id",
    "reference_model",
    "transferred_from_model",
]
GLOBAL_CAP_BY_CONTRACT = {
    "c06_g18_n6": 0.18,
    "c08_g24_n8": 0.24,
    "c10_g30_n10": 0.30,
}


def mechanism_audit() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    uncapped = pd.read_csv(RESULTS / "uncapped_mechanism_audit.csv")
    source = pd.read_csv(SOURCE / "tables/mechanism_certified.csv")
    source_columns = [
        *KEYS,
        "successive_refinement_change",
        "independent_evaluator_difference",
        "internal_identity_difference",
        "certification_floor",
        "resolved",
        "full_prefix_coefficient",
        "additive_single_date_coefficient",
        "fresh_diagonal_coefficient",
    ]
    source = source[source["level"] == "fine"][source_columns]
    data = uncapped.merge(source, on=KEYS, how="left", validate="one_to_one")
    gate_cross = pd.read_csv(RESULTS / "gate_quadrature_audit.csv")[
        KEYS + ["cross_evaluator_difference"]
    ].rename(columns={"cross_evaluator_difference": "new_gate_cross_difference"})
    data = data.merge(gate_cross, on=KEYS, how="left", validate="one_to_one")
    data["audit_independent_difference"] = np.maximum(
        data["independent_evaluator_difference"].fillna(0.0).abs(),
        data["new_gate_cross_difference"].fillna(0.0).abs(),
    )
    numerical_scale = np.maximum.reduce(
        [
            data["successive_refinement_change"].to_numpy(),
            data["audit_independent_difference"].to_numpy(),
            data["uncapped_internal_difference"].abs().to_numpy(),
            np.full(len(data), 1000 * np.finfo(float).eps),
        ]
    )
    data["audit_error_scale"] = numerical_scale
    data["audit_certification_floor"] = np.maximum(1e-11, 10 * numerical_scale)
    data["audit_resolved"] = (
        data["uncapped_forward_loss"] > data["audit_certification_floor"]
    )
    data["full_absolute_error"] = (
        data["full_prefix_prediction"] - data["uncapped_forward_loss"]
    ).abs()
    data["additive_absolute_error"] = (
        data["additive_single_date_prediction"]
        - data["uncapped_forward_loss"]
    ).abs()
    relative_ok = data["strict_local_domain"] & data["audit_resolved"]
    data["full_relative_error"] = np.where(
        relative_ok,
        data["full_absolute_error"] / data["uncapped_forward_loss"],
        np.nan,
    )
    data["scaled_loss_to_full"] = np.where(
        relative_ok & (data["full_prefix_prediction"] > data["audit_certification_floor"]),
        data["uncapped_forward_loss"] / data["full_prefix_prediction"],
        np.nan,
    )
    data["full_closer_than_additive"] = (
        data["full_absolute_error"] < data["additive_absolute_error"]
    )
    data["prediction_difference_resolved"] = (
        data["full_prefix_prediction"]
        - data["additive_single_date_prediction"]
    ).abs() > data["audit_certification_floor"]
    data.to_csv(
        TABLES / "mechanism_all_rows_audited.csv",
        index=False,
        float_format="%.17g",
    )

    source_gate = pd.read_csv(SOURCE / "tables/mechanism_gate_rows.csv")
    gate_keys = source_gate[KEYS]
    gate = gate_keys.merge(data, on=KEYS, how="left", validate="one_to_one")
    gate.to_csv(
        TABLES / "mechanism_gate_40_audited.csv",
        index=False,
        float_format="%.17g",
    )
    nonlocal_rows = data[~data["strict_local_domain"]].copy()
    nonlocal_rows.to_csv(
        TABLES / "mechanism_nonlocal_corrections.csv",
        index=False,
        float_format="%.17g",
    )

    gate_resolved = gate[gate["audit_resolved"]]
    distinguishable = gate_resolved[
        gate_resolved["prediction_difference_resolved"]
    ]
    directions: dict[str, object] = {}
    for direction in ("all_up", "all_down"):
        subset = gate[gate["direction_name"] == direction]
        resolved = subset[subset["audit_resolved"]]
        distinct = resolved[resolved["prediction_difference_resolved"]]
        directions[direction] = {
            "rows": int(len(subset)),
            "resolved_rows": int(len(resolved)),
            "absolute_only_rows": int((~subset["audit_resolved"]).sum()),
            "median_full_relative_error_resolved": float(
                resolved["full_relative_error"].median()
            ),
            "maximum_full_relative_error_resolved": float(
                resolved["full_relative_error"].max()
            ),
            "full_closer_share_when_distinguishable": float(
                distinct["full_closer_than_additive"].mean()
            )
            if len(distinct)
            else None,
        }
    above = data[data["threshold_above_global_cap"]]
    changed = above[above["uncapped_minus_frozen_capped"].abs() > 1e-12]
    metrics = {
        "all_rows": int(len(data)),
        "strict_local_rows": int(data["strict_local_domain"].sum()),
        "above_cap_rows": int(len(above)),
        "above_cap_rows_with_reachable_threshold": int(
            above["above_cap_threshold_reachable"].sum()
        ),
        "above_cap_rows_with_material_state_correction": int(len(changed)),
        "maximum_absolute_nonlocal_state_correction": float(
            above["uncapped_minus_frozen_capped"].abs().max()
        ),
        "maximum_local_state_difference": float(
            data.loc[
                data["strict_local_domain"], "uncapped_minus_frozen_capped"
            ].abs().max()
        ),
        "gate_rows": int(len(gate)),
        "gate_resolved_rows": int(len(gate_resolved)),
        "gate_absolute_only_rows": int((~gate["audit_resolved"]).sum()),
        "gate_median_full_relative_error_resolved": float(
            gate_resolved["full_relative_error"].median()
        ),
        "gate_maximum_full_relative_error_resolved": float(
            gate_resolved["full_relative_error"].max()
        ),
        "gate_full_closer_share_when_distinguishable": float(
            distinguishable["full_closer_than_additive"].mean()
        )
        if len(distinguishable)
        else None,
        "directions": directions,
    }
    return data, gate, metrics


def finance_audit() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    source = pd.read_csv(SOURCE / "tables/finance_certified.csv")
    warning = pd.read_csv(RESULTS / "warning_refinement.csv")
    refined = warning.iloc[-1]
    data = source.copy()
    data["audit_estimate"] = data["transfer_loss_forward"]
    data["audit_quadrature_direct"] = data["transfer_loss_quadrature_direct"]
    data["audit_cross_evaluator_difference"] = data[
        "transfer_cross_evaluator_difference"
    ]
    data["audit_successive_lattice_change"] = data[
        "successive_refinement_change"
    ]
    data["audit_successive_quadrature_change"] = 0.0
    data["audit_optimal_premium"] = data["optimal_premium_lattice"]
    data["audit_early_exercise_value"] = data["early_exercise_value_lattice"]
    data["warning_refined"] = False

    mask = np.ones(len(data), dtype=bool)
    for key in FINANCE_KEYS:
        mask &= data[key].astype(str) == str(refined[key])
    if int(mask.sum()) != 1:
        raise RuntimeError("warning refinement did not match exactly one finance row")
    source_warning = data.loc[mask].iloc[0]
    data.loc[mask, "audit_estimate"] = float(refined["lattice_forward"])
    data.loc[mask, "audit_quadrature_direct"] = float(
        refined["quadrature_direct"]
    )
    data.loc[mask, "audit_cross_evaluator_difference"] = float(
        refined["cross_evaluator_difference"]
    )
    data.loc[mask, "audit_successive_lattice_change"] = abs(
        float(refined["lattice_forward"])
        - float(source_warning["transfer_loss_forward"])
    )
    data.loc[mask, "audit_successive_quadrature_change"] = abs(
        float(refined["quadrature_direct"])
        - float(source_warning["transfer_loss_quadrature_direct"])
    )
    data.loc[mask, "audit_optimal_premium"] = float(
        refined["optimal_premium_lattice"]
    )
    data.loc[mask, "warning_refined"] = True

    global_caps = data["contract_id"].map(GLOBAL_CAP_BY_CONTRACT)
    if global_caps.isna().any():
        unknown = sorted(data.loc[global_caps.isna(), "contract_id"].unique())
        raise RuntimeError(f"unknown contract ids in finance table: {unknown}")
    data["state_policy_equivalent"] = (
        (data["reference_boundary"] > 0)
        & (data["reference_boundary"] < global_caps)
        & (data["transferred_boundary"] > 0)
        & (data["transferred_boundary"] < global_caps)
    )
    data["audit_internal_difference"] = data[
        "transfer_internal_reconciliation"
    ].abs()
    scale = np.maximum.reduce(
        [
            data["audit_successive_lattice_change"].abs().to_numpy(),
            data["audit_successive_quadrature_change"].abs().to_numpy(),
            data["audit_cross_evaluator_difference"].abs().to_numpy(),
            data["audit_internal_difference"].to_numpy(),
            np.full(len(data), 1000 * np.finfo(float).eps),
        ]
    )
    data["audit_error_bound"] = scale
    data["audit_certification_floor"] = np.maximum(1e-11, 10 * scale)
    data["audit_resolved"] = data["audit_estimate"] > data[
        "audit_certification_floor"
    ]
    data["estimate_bps_notional"] = data["audit_estimate"] * 10_000
    data["error_bound_bps_notional"] = data["audit_error_bound"] * 10_000
    data["estimate_usd_per_million"] = data["audit_estimate"] * 1_000_000
    data["error_bound_usd_per_million"] = data["audit_error_bound"] * 1_000_000
    data["absolute_interval_lower"] = np.maximum(
        0.0, data["audit_estimate"] - data["audit_error_bound"]
    )
    data["absolute_interval_upper"] = (
        data["audit_estimate"] + data["audit_error_bound"]
    )
    data["reporting_status"] = np.where(
        data["audit_resolved"], "relative_certified", "absolute_only"
    )
    data["prefix_change_signed_bps"] = data["prefix_change_prediction"] * 10_000
    data["prefix_change_absolute_bps"] = data[
        "prefix_change_prediction"
    ].abs() * 10_000
    data["audit_full_relative_error"] = np.where(
        data["audit_resolved"],
        (data["full_prefix_prediction"] - data["audit_estimate"]).abs()
        / data["audit_estimate"],
        np.nan,
    )
    data["audit_loss_fraction_of_premium"] = np.where(
        data["audit_resolved"]
        & (data["audit_optimal_premium"] > data["audit_certification_floor"]),
        data["audit_estimate"] / data["audit_optimal_premium"],
        np.nan,
    )
    data["audit_loss_fraction_of_early_exercise_value"] = np.where(
        data["audit_resolved"]
        & (
            data["audit_early_exercise_value"]
            > data["audit_certification_floor"]
        ),
        data["audit_estimate"] / data["audit_early_exercise_value"],
        np.nan,
    )
    data.to_csv(
        TABLES / "finance_all_30_audited.csv", index=False, float_format="%.17g"
    )

    base = data[data["contract_id"] == "c08_g24_n8"].copy()
    base.to_csv(
        TABLES / "finance_base_10_audited.csv", index=False, float_format="%.17g"
    )
    source_prices = pd.read_csv(SOURCE / "tables/model_price_differences.csv")
    source_prices.to_csv(
        TABLES / "model_price_differences_frozen.csv",
        index=False,
        float_format="%.17g",
    )

    by_contract: dict[str, object] = {}
    for contract, frame in data.groupby("contract_id", sort=True):
        resolved = frame[frame["audit_resolved"]]
        by_contract[contract] = {
            "rows": int(len(frame)),
            "resolved_rows": int(len(resolved)),
            "median_resolved_loss_bps": float(
                resolved["estimate_bps_notional"].median()
            ),
            "maximum_resolved_loss_bps": float(
                resolved["estimate_bps_notional"].max()
            ),
            "median_absolute_prefix_change_bps": float(
                frame["prefix_change_absolute_bps"].median()
            ),
        }
    resolved_base = base[base["audit_resolved"]]
    source_warning_estimate = float(source_warning["transfer_loss_forward"])
    metrics = {
        "rows": int(len(data)),
        "resolved_rows": int(data["audit_resolved"].sum()),
        "absolute_only_rows": int((~data["audit_resolved"]).sum()),
        "all_transfer_thresholds_strictly_inside_cap": bool(
            data["state_policy_equivalent"].all()
        ),
        "base_rows": int(len(base)),
        "base_resolved_rows": int(len(resolved_base)),
        "base_median_resolved_loss_bps": float(
            resolved_base["estimate_bps_notional"].median()
        ),
        "base_maximum_resolved_loss_bps": float(
            resolved_base["estimate_bps_notional"].max()
        ),
        "warning_source_loss": source_warning_estimate,
        "warning_refined_loss": float(refined["lattice_forward"]),
        "warning_change_bps": float(
            (float(refined["lattice_forward"]) - source_warning_estimate) * 10_000
        ),
        "warning_refined_cross_evaluator_difference": abs(
            float(refined["cross_evaluator_difference"])
        ),
        "warning_refined_error_bound": float(data.loc[mask, "audit_error_bound"].iloc[0]),
        "warning_refined_certification_floor": float(
            data.loc[mask, "audit_certification_floor"].iloc[0]
        ),
        "frozen_materiality_gate_pass": False,
        "by_contract": by_contract,
    }
    return data, base, metrics


def signed_convergence_figure(mechanism: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    colors = {
        "black_scholes_atm": "#36648B",
        "annual_lognormal_mixture": "#B24C36",
    }
    labels = {
        "black_scholes_atm": "Black--Scholes ATM",
        "annual_lognormal_mixture": "annual mixture",
    }
    fig, axes = plt.subplots(1, 2, figsize=(9.3, 3.7), sharey=True)
    for axis, direction, title in zip(
        axes,
        ("all_up", "all_down"),
        ("common upward shift", "common downward shift"),
    ):
        subset = mechanism[
            (mechanism["direction_name"] == direction)
            & mechanism["strict_local_domain"]
        ].copy()
        for model, frame in subset.groupby("model", sort=True):
            for _, path in frame.groupby("date", sort=False):
                path = path.sort_values("eta")
                good = path[path["audit_resolved"]]
                axis.plot(
                    good["eta"],
                    good["scaled_loss_to_full"],
                    color=colors[model],
                    alpha=0.28,
                    linewidth=0.9,
                )
                bad = path[~path["audit_resolved"]]
                if len(bad):
                    axis.scatter(
                        bad["eta"],
                        np.full(len(bad), 0.965),
                        marker="x",
                        color=colors[model],
                        alpha=0.55,
                        s=18,
                    )
            grouped = (
                frame[frame["audit_resolved"]]
                .groupby("eta")["scaled_loss_to_full"]
                .median()
                .sort_index()
            )
            axis.plot(
                grouped.index,
                grouped.values,
                color=colors[model],
                marker="o",
                markersize=3.5,
                linewidth=1.8,
                label=labels[model],
            )
        axis.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
        axis.set_xscale("log", base=2)
        axis.invert_xaxis()
        axis.set_title(title)
        axis.set_xlabel(r"relative perturbation $\eta$ (smaller $\rightarrow$)")
        axis.grid(alpha=0.18)
        axis.set_ylim(0.96, 1.06)
    axes[0].set_ylabel("exact loss / full-prefix prediction")
    axes[1].legend(frameon=False, loc="upper right")
    fig.text(
        0.5,
        0.015,
        "Crosses mark absolute-only rows; no relative ratio is plotted below the audit floor.",
        ha="center",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(FIGURES / "signed_convergence.png", dpi=220)
    plt.close(fig)


def run() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    mechanism, _, mechanism_metrics = mechanism_audit()
    _, _, finance_metrics = finance_audit()
    signed_convergence_figure(mechanism)
    warning_metadata = json.loads(
        (RESULTS / "refinement_metadata.json").read_text()
    )
    summary = {
        "mechanism": mechanism_metrics,
        "finance": finance_metrics,
        "refinement": warning_metadata,
    }
    (RESULTS / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    run()
