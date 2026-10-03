"""Apply the frozen certification floors and verdict gates."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TABLES = ROOT / "tables"
FIGURES = ROOT / "figures"


def _merge_level(
    base: pd.DataFrame,
    source: pd.DataFrame,
    *,
    keys: list[str],
    level: str,
    columns: list[str],
) -> pd.DataFrame:
    selected = source[source["level"] == level][keys + columns].copy()
    selected = selected.rename(columns={column: f"{column}_{level}" for column in columns})
    return base.merge(selected, on=keys, how="left", validate="one_to_one")


def certify_mechanism(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    raw = pd.read_csv(RESULTS / "mechanism_results.csv")
    checks = pd.read_csv(RESULTS / "independent_checks.csv")
    keys = [
        "date",
        "contract_id",
        "model",
        "policy_id",
        "direction_name",
        "eta",
    ]
    fine = raw[raw["level"] == "fine"].copy()
    for level in ("primary", "coarse"):
        fine = _merge_level(
            fine,
            raw,
            keys=keys,
            level=level,
            columns=["exact_regret_forward"],
        )
    sentinel = raw[raw["level"] == "sentinel"][
        keys + ["exact_regret_forward"]
    ].rename(columns={"exact_regret_forward": "exact_regret_forward_sentinel"})
    fine = fine.merge(sentinel, on=keys, how="left", validate="one_to_one")
    independent = checks[checks["level"] == "fine"][
        keys
        + [
            "cross_evaluator_error",
            "quadrature_cancellation_diagnostic",
        ]
    ]
    fine = fine.merge(independent, on=keys, how="left", validate="one_to_one")

    differences = np.column_stack(
        [
            np.abs(
                fine["exact_regret_forward"]
                - fine["exact_regret_forward_primary"]
            ),
            np.abs(
                fine["exact_regret_forward_primary"]
                - fine["exact_regret_forward_coarse"]
            ),
            np.abs(
                fine["exact_regret_forward"]
                - fine["exact_regret_forward_sentinel"]
            ).fillna(0.0),
        ]
    )
    fine["successive_refinement_change"] = np.max(differences, axis=1)
    fine["independent_evaluator_difference"] = fine[
        "cross_evaluator_error"
    ].abs().fillna(0.0)
    fine["internal_identity_difference"] = fine[
        "internal_reconciliation_error"
    ].abs()
    numerical_scale = np.maximum.reduce(
        [
            fine["successive_refinement_change"].to_numpy(),
            fine["independent_evaluator_difference"].to_numpy(),
            fine["internal_identity_difference"].to_numpy(),
            np.full(len(fine), 1000 * np.finfo(float).eps),
        ]
    )
    fine["certification_floor"] = np.maximum(
        float(config["numerics"]["absolute_floor_minimum"]),
        float(config["numerics"]["floor_multiplier"]) * numerical_scale,
    )
    fine["resolved"] = (
        fine["exact_regret_forward"] > fine["certification_floor"]
    )
    fine["full_absolute_error"] = np.abs(
        fine["full_prefix_prediction"] - fine["exact_regret_forward"]
    )
    fine["additive_absolute_error"] = np.abs(
        fine["additive_single_date_prediction"]
        - fine["exact_regret_forward"]
    )
    fine["fresh_absolute_error"] = np.abs(
        fine["fresh_diagonal_prediction"] - fine["exact_regret_forward"]
    )
    for label, column in (
        ("full", "full_prefix_prediction"),
        ("additive", "additive_single_date_prediction"),
        ("fresh", "fresh_diagonal_prediction"),
    ):
        fine[f"{label}_relative_error"] = np.where(
            fine["resolved"],
            np.abs(fine[column] - fine["exact_regret_forward"])
            / fine["exact_regret_forward"],
            np.nan,
        )
        fine[f"exact_to_{label}_ratio"] = np.where(
            fine["resolved"] & (fine[column] > fine["certification_floor"]),
            fine["exact_regret_forward"] / fine[column],
            np.nan,
        )
    fine["prediction_difference_resolved"] = (
        np.abs(
            fine["full_prefix_prediction"]
            - fine["additive_single_date_prediction"]
        )
        > fine["certification_floor"]
    )
    fine["full_closer_than_additive"] = (
        fine["full_absolute_error"] < fine["additive_absolute_error"]
    )

    candidates = fine[
        fine["direction_name"].isin(["all_up", "all_down"])
        & fine["resolved"]
    ]
    selected = (
        candidates.sort_values("eta")
        .groupby(["date", "model", "direction_name"], as_index=False, group_keys=False)
        .head(2)
        .copy()
    )
    expected_gate_rows = 5 * 2 * 2 * 2
    direction_metrics = {}
    mechanism_pass = len(selected) == expected_gate_rows
    for direction in ("all_up", "all_down"):
        subset = selected[selected["direction_name"] == direction]
        distinguishable = subset[subset["prediction_difference_resolved"]]
        metric = {
            "rows": int(len(subset)),
            "median_full_relative_error": float(
                subset["full_relative_error"].median()
            ),
            "maximum_full_relative_error": float(
                subset["full_relative_error"].max()
            ),
            "full_closer_share_when_distinguishable": float(
                distinguishable["full_closer_than_additive"].mean()
            )
            if len(distinguishable)
            else None,
            "distinguishable_rows": int(len(distinguishable)),
        }
        direction_metrics[direction] = metric
        mechanism_pass = mechanism_pass and (
            metric["median_full_relative_error"]
            <= float(config["gates"]["mechanism_median_relative_error_max"])
            and metric["distinguishable_rows"] > 0
            and metric["full_closer_share_when_distinguishable"]
            >= float(config["gates"]["mechanism_prefix_closer_share_min"])
        )

    gate = {
        "pass": bool(mechanism_pass),
        "selected_rows": int(len(selected)),
        "expected_rows": expected_gate_rows,
        "directions": direction_metrics,
        "all_selected_median_full_relative_error": float(
            selected["full_relative_error"].median()
        ),
        "all_selected_maximum_full_relative_error": float(
            selected["full_relative_error"].max()
        ),
    }
    return fine, selected, gate


def certify_occupancy(config: dict, mechanism: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    boundaries = pd.read_csv(RESULTS / "boundaries.csv")
    base = boundaries[
        (boundaries["level"] == "fine")
        & (boundaries["contract_id"] == "c08_g24_n8")
        & (boundaries["decision_date"] == 1)
    ].copy()
    coefficients = mechanism[
        (mechanism["direction_name"] == "all_up")
        & np.isclose(mechanism["eta"], max(config["etas"]))
    ][
        ["date", "contract_id", "model", "copied_coefficient"]
    ]
    base = base.merge(
        coefficients,
        on=["date", "contract_id", "model"],
        how="left",
        validate="one_to_one",
    )
    base["clean_alignment"] = (
        (base["floor_probability"] > 0)
        & (base["right_gap_slope"] > 0)
        & (base["left_gap_slope"] > 0)
        & (
            base["accumulator_atom_distance"]
            > float(config["numerics"]["atom_collision_tolerance"])
        )
        & (base["copied_coefficient"] > 0)
    )
    counts = (
        base.groupby("model")["eligible_inception_trace"].sum().astype(int).to_dict()
    )
    minimum = int(config["gates"]["occupancy_min_snapshots_of_five"])
    occupancy_pass = all(count >= minimum for count in counts.values())
    alignment_pass = bool(base["clean_alignment"].all())
    gate = {
        "pass": bool(occupancy_pass and alignment_pass),
        "occupancy_pass": bool(occupancy_pass),
        "alignment_pass": alignment_pass,
        "eligible_snapshot_counts_by_model": counts,
        "required_count_by_model": minimum,
        "minimum_floor_probability": float(base["floor_probability"].min()),
        "minimum_atom_separation": float(base["accumulator_atom_distance"].min()),
        "minimum_one_sided_gap_slope": float(
            base[["right_gap_slope", "left_gap_slope"]].min().min()
        ),
    }
    return base, gate


def certify_finance(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    raw = pd.read_csv(RESULTS / "finance_valuation.csv")
    keys = [
        "date",
        "contract_id",
        "reference_model",
        "transferred_from_model",
    ]
    fine = raw[raw["level"] == "fine"].copy()
    columns = [
        "transfer_loss_forward",
        "optimal_premium_lattice",
        "early_exercise_value_lattice",
    ]
    for level in ("primary", "coarse"):
        fine = _merge_level(
            fine, raw, keys=keys, level=level, columns=columns
        )
    sentinel = raw[raw["level"] == "sentinel"][
        keys + ["transfer_loss_forward"]
    ].rename(columns={"transfer_loss_forward": "transfer_loss_forward_sentinel"})
    fine = fine.merge(sentinel, on=keys, how="left", validate="one_to_one")
    differences = np.column_stack(
        [
            np.abs(
                fine["transfer_loss_forward"]
                - fine["transfer_loss_forward_primary"]
            ),
            np.abs(
                fine["transfer_loss_forward_primary"]
                - fine["transfer_loss_forward_coarse"]
            ),
            np.abs(
                fine["transfer_loss_forward"]
                - fine["transfer_loss_forward_sentinel"]
            ).fillna(0.0),
        ]
    )
    fine["successive_refinement_change"] = np.max(differences, axis=1)
    fine["independent_evaluator_difference"] = fine[
        "transfer_cross_evaluator_difference"
    ].abs()
    numerical_scale = np.maximum.reduce(
        [
            fine["successive_refinement_change"].to_numpy(),
            fine["independent_evaluator_difference"].to_numpy(),
            fine["transfer_internal_reconciliation"].abs().to_numpy(),
            np.full(len(fine), 1000 * np.finfo(float).eps),
        ]
    )
    fine["certification_floor"] = np.maximum(
        float(config["numerics"]["absolute_floor_minimum"]),
        float(config["numerics"]["floor_multiplier"]) * numerical_scale,
    )
    fine["resolved"] = fine["transfer_loss_forward"] > fine["certification_floor"]
    fine["transfer_loss_bps_notional"] = fine["transfer_loss_forward"] * 10_000
    fine["transfer_loss_usd_per_million"] = (
        fine["transfer_loss_forward"]
        * float(config["illustrative_notional_usd"])
    )
    fine["loss_fraction_of_premium"] = np.where(
        fine["resolved"]
        & (fine["optimal_premium_lattice"] > fine["certification_floor"]),
        fine["transfer_loss_forward"] / fine["optimal_premium_lattice"],
        np.nan,
    )
    fine["loss_fraction_of_early_exercise_value"] = np.where(
        fine["resolved"]
        & (
            fine["early_exercise_value_lattice"]
            > fine["certification_floor"]
        ),
        fine["transfer_loss_forward"]
        / fine["early_exercise_value_lattice"],
        np.nan,
    )
    fine["prefix_change_bps"] = (
        np.abs(fine["prefix_change_prediction"]) * 10_000
    )
    fine["full_approximation_relative_error"] = np.where(
        fine["resolved"],
        np.abs(
            fine["full_prefix_prediction"] - fine["transfer_loss_forward"]
        )
        / fine["transfer_loss_forward"],
        np.nan,
    )
    fine["additive_approximation_relative_error"] = np.where(
        fine["resolved"],
        np.abs(
            fine["additive_single_date_prediction"]
            - fine["transfer_loss_forward"]
        )
        / fine["transfer_loss_forward"],
        np.nan,
    )

    price = fine.pivot(
        index=["date", "contract_id"],
        columns="reference_model",
        values=[
            "optimal_premium_lattice",
            "optimal_premium_quadrature",
            "optimal_premium_lattice_primary",
            "optimal_premium_lattice_coarse",
        ],
    )
    price.columns = ["__".join(column) for column in price.columns]
    price = price.reset_index()
    mix = "annual_lognormal_mixture"
    black = "black_scholes_atm"
    price["mixture_minus_black_price"] = (
        price[f"optimal_premium_lattice__{mix}"]
        - price[f"optimal_premium_lattice__{black}"]
    )
    price["mixture_minus_black_bps"] = (
        price["mixture_minus_black_price"] * 10_000
    )
    cross_error = np.maximum(
        np.abs(
            price[f"optimal_premium_lattice__{mix}"]
            - price[f"optimal_premium_quadrature__{mix}"]
        ),
        np.abs(
            price[f"optimal_premium_lattice__{black}"]
            - price[f"optimal_premium_quadrature__{black}"]
        ),
    )
    refinement = np.maximum.reduce(
        [
            np.abs(
                price[f"optimal_premium_lattice__{mix}"]
                - price[f"optimal_premium_lattice_primary__{mix}"]
            ),
            np.abs(
                price[f"optimal_premium_lattice_primary__{mix}"]
                - price[f"optimal_premium_lattice_coarse__{mix}"]
            ),
            np.abs(
                price[f"optimal_premium_lattice__{black}"]
                - price[f"optimal_premium_lattice_primary__{black}"]
            ),
            np.abs(
                price[f"optimal_premium_lattice_primary__{black}"]
                - price[f"optimal_premium_lattice_coarse__{black}"]
            ),
        ]
    )
    price["certification_floor"] = np.maximum(
        float(config["numerics"]["absolute_floor_minimum"]),
        float(config["numerics"]["floor_multiplier"])
        * np.maximum(cross_error, refinement),
    )
    price["resolved"] = (
        np.abs(price["mixture_minus_black_price"])
        > price["certification_floor"]
    )

    base = fine[fine["contract_id"] == "c08_g24_n8"].copy()
    resolved_base = base[base["resolved"]]
    median_bps = float(resolved_base["transfer_loss_bps_notional"].median())
    material = base[
        base["resolved"]
        & (
            base["prefix_change_bps"]
            >= float(config["gates"]["adopt_prefix_change_bps_min"])
        )
        & (
            base["full_approximation_relative_error"]
            <= float(config["gates"]["adopt_full_approx_relative_error_max"])
        )
    ]
    finance_adopt = (
        median_bps >= float(config["gates"]["adopt_median_transfer_bps_min"])
        and len(material)
        >= int(config["gates"]["adopt_material_base_cases_min"])
    )
    resolved_above_reject = resolved_base[
        resolved_base["transfer_loss_bps_notional"]
        > float(config["gates"]["reject_all_transfer_bps_below"])
    ]
    gate = {
        "adopt_pass": bool(finance_adopt),
        "base_rows": int(len(base)),
        "base_resolved_rows": int(len(resolved_base)),
        "median_resolved_base_transfer_bps": median_bps,
        "maximum_resolved_base_transfer_bps": float(
            resolved_base["transfer_loss_bps_notional"].max()
        ),
        "material_prefix_cases": int(len(material)),
        "resolved_base_cases_above_reject_floor": int(
            len(resolved_above_reject)
        ),
        "all_base_transfer_losses_below_reject_floor": bool(
            len(resolved_above_reject) == 0
        ),
        "median_base_prefix_change_bps": float(
            base["prefix_change_bps"].median()
        ),
        "maximum_base_prefix_change_bps": float(
            base["prefix_change_bps"].max()
        ),
    }
    return fine, price, gate


def create_figures(mechanism: pd.DataFrame, finance: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    subset = mechanism[
        mechanism["direction_name"].isin(["all_up", "all_down"])
        & ~mechanism["threshold_above_global_cap"]
        & ~mechanism["threshold_below_zero"]
    ]
    grouped = (
        subset.groupby(["direction_name", "eta"])[
            ["full_relative_error", "additive_relative_error"]
        ]
        .median()
        .reset_index()
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharey=True)
    for axis, direction in zip(axes, ("all_up", "all_down")):
        data = grouped[grouped["direction_name"] == direction]
        axis.loglog(
            data["eta"],
            data["full_relative_error"],
            marker="o",
            label="full prefix",
        )
        axis.loglog(
            data["eta"],
            data["additive_relative_error"],
            marker="s",
            label="additive date-wise",
        )
        axis.set_title(direction.replace("_", " "))
        axis.set_xlabel(r"relative boundary scale $\eta$")
        axis.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel("median relative error (resolved rows)")
    axes[0].legend(frameon=False)
    fig.text(
        0.5,
        0.005,
        "Local-domain rows only; all out-of-domain finite-scale failures remain in the audit table.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIGURES / "mechanism_convergence.png", dpi=180)
    plt.close(fig)

    base = finance[finance["contract_id"] == "c08_g24_n8"].copy()
    dates = list(dict.fromkeys(base["date"].tolist()))
    models = ["black_scholes_atm", "annual_lognormal_mixture"]
    labels = ["Black--Scholes", "mixture"]
    x = np.arange(len(dates))
    width = 0.36
    fig, axis = plt.subplots(figsize=(9.2, 4.2))
    for offset, model, label in zip((-0.5, 0.5), models, labels):
        values = (
            base[base["reference_model"] == model]
            .set_index("date")
            .loc[dates, "transfer_loss_bps_notional"]
        )
        axis.bar(x + offset * width, values, width=width, label=label)
    axis.axhline(1.0, color="black", linestyle="--", linewidth=1, label="1 bp adopt gate")
    axis.set_xticks(x, dates, rotation=30, ha="right")
    axis.set_ylabel("policy-transfer loss (bp of notional)")
    axis.set_title("Base proposed cliquet tuple")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "finance_transfer_loss.png", dpi=180)
    plt.close(fig)


def run() -> None:
    config = json.loads((ROOT / "config.json").read_text())
    TABLES.mkdir(parents=True, exist_ok=True)
    mechanism, mechanism_gate_rows, mechanism_gate = certify_mechanism(config)
    occupancy, occupancy_gate = certify_occupancy(config, mechanism)
    finance, model_prices, finance_gate = certify_finance(config)

    if not occupancy_gate["pass"] or not mechanism_gate["pass"]:
        verdict = "REJECT"
    elif finance_gate["all_base_transfer_losses_below_reject_floor"]:
        verdict = "REJECT"
    elif finance_gate["adopt_pass"]:
        verdict = "ADOPT FOR EXPANSION"
    else:
        verdict = "LIMITED VALUE"

    mechanism.to_csv(
        TABLES / "mechanism_certified.csv", index=False, float_format="%.17g"
    )
    mechanism_gate_rows.to_csv(
        TABLES / "mechanism_gate_rows.csv", index=False, float_format="%.17g"
    )
    occupancy.to_csv(
        TABLES / "occupancy_alignment.csv", index=False, float_format="%.17g"
    )
    finance.to_csv(
        TABLES / "finance_certified.csv", index=False, float_format="%.17g"
    )
    model_prices.to_csv(
        TABLES / "model_price_differences.csv", index=False, float_format="%.17g"
    )

    mechanism_summary = (
        mechanism.assign(
            direction_group=np.where(
                mechanism["direction_name"].str.startswith("single_up"),
                "single_up",
                np.where(
                    mechanism["direction_name"].str.startswith("single_down"),
                    "single_down",
                    mechanism["direction_name"],
                ),
            )
        )
        .groupby("direction_group")
        .agg(
            rows=("date", "size"),
            resolved_rows=("resolved", "sum"),
            median_full_relative_error=("full_relative_error", "median"),
            maximum_full_relative_error=("full_relative_error", "max"),
            median_additive_relative_error=("additive_relative_error", "median"),
            median_copied_coefficient=("copied_coefficient", "median"),
        )
        .reset_index()
    )
    mechanism_summary.to_csv(
        TABLES / "mechanism_summary.csv", index=False, float_format="%.17g"
    )
    finance_summary = (
        finance.groupby("contract_id")
        .agg(
            directed_rows=("date", "size"),
            resolved_rows=("resolved", "sum"),
            median_transfer_bps=("transfer_loss_bps_notional", "median"),
            maximum_transfer_bps=("transfer_loss_bps_notional", "max"),
            median_transfer_usd_per_million=(
                "transfer_loss_usd_per_million",
                "median",
            ),
            median_prefix_change_bps=("prefix_change_bps", "median"),
            maximum_prefix_change_bps=("prefix_change_bps", "max"),
        )
        .reset_index()
    )
    finance_summary.to_csv(
        TABLES / "finance_summary.csv", index=False, float_format="%.17g"
    )
    create_figures(mechanism, finance)

    summary = {
        "verdict": verdict,
        "occupancy_alignment_gate": occupancy_gate,
        "mechanism_gate": mechanism_gate,
        "finance_gate": finance_gate,
        "counts": {
            "certified_mechanism_rows": int(len(mechanism)),
            "resolved_mechanism_rows": int(mechanism["resolved"].sum()),
            "certified_finance_rows": int(len(finance)),
            "resolved_finance_rows": int(finance["resolved"].sum()),
            "resolved_model_price_rows": int(model_prices["resolved"].sum()),
        },
    }
    (RESULTS / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    run()
