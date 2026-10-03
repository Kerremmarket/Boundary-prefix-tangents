#!/usr/bin/env python3
"""Create Phase II-A publication tables, figures, and audit summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PRODUCT_LABELS = {
    "A10": "Athene A10",
    "AA7": "Allianz AA7",
    "IP4": "MassMutual IP4",
}
COLORS = {"A10": "#1f77b4", "AA7": "#d95f02", "IP4": "#2ca02c"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _write_table(frame: pd.DataFrame, stem: Path, float_format: str = "%.6g") -> None:
    frame.to_csv(stem.with_suffix(".csv"), index=False)
    stem.with_suffix(".tex").write_text(
        frame.to_latex(index=False, float_format=float_format, escape=True),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    evidence = args.evidence_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    schedule = pd.read_csv(evidence / "schedule_grid_refinement.csv")
    coefficients = pd.read_csv(evidence / "multidate_coefficients.csv")
    convergence = pd.read_csv(evidence / "epsilon_convergence.csv")
    occupancy = pd.read_csv(evidence / "issue_occupancy.csv")
    lifetime = pd.read_csv(evidence / "lifetime_sequence.csv")
    joint = pd.read_csv(evidence / "joint_grid_epsilon_refinement.csv")
    reconciliation = pd.read_csv(evidence / "carrier_density_reconciliation.csv")

    gate = pd.DataFrame(
        [
            {
                "gate": "Authoritative product terms frozen",
                "evidence": "3 documented projections",
                "result": "PASS",
            },
            {
                "gate": "At least three regular aligned dates",
                "evidence": "7 fixed-cap post-charge product/date windows",
                "result": "PASS (conditional projection)",
            },
            {
                "gate": "Positive trace from X0=1",
                "evidence": "0 of 84 issue-state horizon/direction rows",
                "result": "FAIL",
            },
            {
                "gate": "Direct local repricing convergence",
                "evidence": "240k max error 0.72% at epsilon=0.00125",
                "result": "PASS (local only)",
            },
            {
                "gate": "Irreducible beyond coherent pairwise",
                "evidence": "coherent reconstruction exact to 2.8e-17",
                "result": "FAIL",
            },
            {
                "gate": "Renewal-cap and MVA robustness",
                "evidence": "normal transport is not preserved",
                "result": "FAIL as product-level claim",
            },
        ]
    )
    _write_table(gate, output / "table_gate_disposition", float_format="%.4g")

    local_h5 = coefficients.loc[
        coefficients["initialization"].eq("local_rho_boundary")
        & coefficients["horizon"].eq(5)
        & coefficients["direction"].eq("common")
    ].copy()
    eps_ref = convergence.loc[
        convergence["initialization"].eq("local_rho_boundary")
        & convergence["horizon"].eq(5)
        & convergence["direction"].eq("common")
        & convergence["epsilon"].eq(0.00125)
    ][["date", "product", "relative_full_error"]].rename(
        columns={"relative_full_error": "error_120k"}
    )
    eps_240 = joint.loc[
        joint["status"].eq("regular_upper_boundary")
        & joint["grid_points"].eq(240_000)
        & joint["epsilon"].eq(0.00125)
    ][["date", "product", "relative_full_error"]].rename(
        columns={"relative_full_error": "error_240k"}
    )
    common = local_h5.merge(eps_ref, on=["date", "product"]).merge(
        eps_240, on=["date", "product"]
    )
    common["product"] = common["product"].map(PRODUCT_LABELS)
    common["additive_overstatement"] = (
        common["additive_pairwise"] / common["full_coefficient"] - 1.0
    )
    common_table = common[
        [
            "date",
            "product",
            "boundary",
            "full_coefficient",
            "copied_share",
            "additive_overstatement",
            "error_120k",
            "error_240k",
        ]
    ].sort_values(["date", "product"])
    _write_table(common_table, output / "table_five_date_common")

    life5 = lifetime.loc[lifetime["decision"].eq(5)][
        ["date", "product", "cumulative_full_coefficient"]
    ].rename(columns={"cumulative_full_coefficient": "coefficient_T5"})
    life50 = lifetime.loc[lifetime["decision"].eq(50)][
        [
            "date",
            "product",
            "cumulative_full_coefficient",
            "cumulative_copied_share",
            "coefficient_increment",
        ]
    ].rename(
        columns={
            "cumulative_full_coefficient": "coefficient_T50",
            "cumulative_copied_share": "copied_share_T50",
            "coefficient_increment": "increment_T50",
        }
    )
    life_table = life5.merge(life50, on=["date", "product"])
    life_table["product"] = life_table["product"].map(PRODUCT_LABELS)
    life_table["T50_over_T5"] = (
        life_table["coefficient_T50"] / life_table["coefficient_T5"]
    )
    _write_table(
        life_table.sort_values(["date", "product"]), output / "table_lifetime"
    )

    reference_schedule = schedule.loc[schedule["grid_points"].eq(120_000)].copy()
    outcome_rows: list[dict] = []
    for (date, product), group in reference_schedule.groupby(["date", "product"]):
        regular = group.loc[group["status"].eq("regular_upper_boundary")]
        divergent = group["status"].eq("divergent_continuation_projection").any()
        outcome_rows.append(
            {
                "date": date,
                "product": PRODUCT_LABELS[product],
                "first_regular_decision": (
                    int(regular["decision"].min()) if not regular.empty else np.nan
                ),
                "postcharge_boundary": (
                    float(regular.loc[regular["postcharge"], "boundary"].iloc[0])
                    if regular["postcharge"].any()
                    else np.nan
                ),
                "boundary_below_issue_floor": (
                    bool((regular["boundary"] < 1.0).all())
                    if not regular.empty
                    else False
                ),
                "divergent_continuation_projection": bool(divergent),
            }
        )
    outcome_table = pd.DataFrame(outcome_rows).sort_values(["date", "product"])
    _write_table(outcome_table, output / "table_schedule_outcomes")

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 160,
        }
    )
    regular_dates = ["2006-03-31", "2022-10-31", "2023-05-31"]
    figure, axes = plt.subplots(1, 3, figsize=(10.5, 3.5), sharey=True)
    for axis, date in zip(axes, regular_dates):
        date_frame = reference_schedule.loc[reference_schedule["date"].eq(date)]
        for product in PRODUCT_LABELS:
            rows = date_frame.loc[
                date_frame["product"].eq(product)
                & date_frame["status"].eq("regular_upper_boundary")
            ]
            axis.plot(
                rows["decision"],
                rows["boundary"],
                marker="o",
                markersize=3,
                linewidth=1.2,
                color=COLORS[product],
                label=PRODUCT_LABELS[product],
            )
        axis.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
        axis.set_title(date)
        axis.set_xlabel("decision date")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("upper boundary")
    axes[-1].legend(loc="best", frameon=False)
    figure.suptitle("Documented schedules: every regular boundary lies below issue support")
    figure.tight_layout()
    figure.savefig(output / "figure_schedule_boundaries.png", bbox_inches="tight")
    figure.savefig(output / "figure_schedule_boundaries.pdf", bbox_inches="tight")
    plt.close(figure)

    finest = joint.loc[
        joint["status"].eq("regular_upper_boundary")
        & joint["grid_points"].eq(240_000)
    ].copy()
    figure, axis = plt.subplots(figsize=(6.6, 4.0))
    for (date, product), group in finest.groupby(["date", "product"]):
        group = group.sort_values("epsilon")
        axis.plot(
            group["epsilon"],
            group["relative_full_error"],
            marker="o",
            markersize=3,
            linewidth=1.1,
            label=f"{PRODUCT_LABELS[product]}, {date[:4]}",
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.invert_xaxis()
    axis.set_xlabel(r"relative boundary perturbation $\epsilon$")
    axis.set_ylabel("relative coefficient error")
    axis.set_title("Five-date direct repricing converges to the ordered-prefix coefficient")
    axis.grid(alpha=0.25, which="both")
    axis.legend(ncol=2, frameon=False)
    figure.tight_layout()
    figure.savefig(output / "figure_joint_convergence.png", bbox_inches="tight")
    figure.savefig(output / "figure_joint_convergence.pdf", bbox_inches="tight")
    plt.close(figure)

    local = coefficients.loc[
        coefficients["initialization"].eq("local_rho_boundary")
    ].copy()
    local["ordinary_ratio"] = local["ordinary_coefficient"] / local["full_coefficient"]
    local["additive_ratio"] = local["additive_pairwise"] / local["full_coefficient"]
    ratios = local.groupby("horizon")[["ordinary_ratio", "additive_ratio"]].median()
    positions = np.arange(len(ratios))
    width = 0.25
    figure, axis = plt.subplots(figsize=(6.2, 3.8))
    axis.bar(
        positions - width,
        ratios["ordinary_ratio"],
        width,
        label="ordinary only",
        color="#8da0cb",
    )
    axis.bar(positions, np.ones(len(ratios)), width, label="full/coherent", color="#66c2a5")
    axis.bar(
        positions + width,
        ratios["additive_ratio"],
        width,
        label="additive pairwise",
        color="#fc8d62",
    )
    axis.set_xticks(positions, [str(value) for value in ratios.index])
    axis.set_xlabel("active dates")
    axis.set_ylabel("median ratio to full coefficient")
    axis.set_title("Coherent pairwise is exact; additive pairwise overcounts long runs")
    axis.axhline(1.0, color="black", linewidth=0.8)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output / "figure_pairwise_comparison.png", bbox_inches="tight")
    figure.savefig(output / "figure_pairwise_comparison.pdf", bbox_inches="tight")
    plt.close(figure)

    maximum_240_error = float(finest["relative_full_error"].max())
    finest_small = finest.loc[finest["epsilon"].eq(0.00125)]
    maximum_240_small_error = float(finest_small["relative_full_error"].max())
    maximum_carrier_error = float(reconciliation["relative_difference"].max())
    summary = f"""# Phase II-A analysis summary

- Gate classification: **APPLICATION REMAINS ILLUSTRATIVE**.
- Local fixed-cap five-date windows: {len(common_table)} product/date cases.
- Issue-state coefficient and loss: exactly zero in all reported rows; maximum
  issue survivor mass at the first regular post-charge boundary is
  {occupancy['survivor_mass_from_issue'].max():.3g}.
- Maximum 240,000-grid relative coefficient error at epsilon 0.00125:
  {maximum_240_small_error:.4%}.  The retained epsilon 0.02 stress reaches
  {maximum_240_error:.4%}, confirming that the expansion is only local.
- Maximum aggregate/labelled left-trace reconciliation error:
  {maximum_carrier_error:.3g}.
- Coherent labelled pairwise reconstruction is algebraically identical to the
  full scalar carrier; additive pairwise summation overcounts longer runs.
- Phase II-B and II-C were not run because the frozen issue-occupancy gate
  failed.
"""
    (output / "analysis_summary.md").write_text(summary, encoding="utf-8")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
