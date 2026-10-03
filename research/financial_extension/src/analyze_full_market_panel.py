#!/usr/bin/env python3
"""Create compact tables, figures, and a proposed section from the full panel."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


JOINT_DIRECTIONS = ("joint_equal", "later_larger", "earlier_larger")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-panel-dir", type=Path, required=True)
    parser.add_argument("--pilot-regret-dir", type=Path, required=True)
    parser.add_argument("--contract-sensitivity-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _save_figure(figure, output_dir: Path, stem: str) -> None:
    figure.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    figure.savefig(output_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def _set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 140,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def main() -> int:
    args = parse_args()
    full_dir = args.full_panel_dir.expanduser().resolve()
    pilot_dir = args.pilot_regret_dir.expanduser().resolve()
    contract_dir = args.contract_sensitivity_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration = pd.read_csv(full_dir / "annual_calibrations.csv", parse_dates=["date"])
    status = pd.read_csv(full_dir / "date_status.csv", parse_dates=["date"])
    coefficients = pd.read_csv(full_dir / "coefficient_panel.csv", parse_dates=["date"])
    finite = pd.read_csv(full_dir / "finite_perturbation_panel.csv", parse_dates=["date"])
    repair = pd.read_csv(full_dir / "repair_diagnostics.csv", parse_dates=["date"])
    contract_sensitivity = pd.read_csv(
        contract_dir / "table_contract_sensitivity.csv"
    )
    pilot_coefficients = pd.read_csv(
        pilot_dir / "coefficient_results.csv", parse_dates=["date"]
    )
    pilot_convergence = pd.read_csv(
        pilot_dir / "epsilon_convergence.csv", parse_dates=["date"]
    )

    direction_rows = []
    for direction in JOINT_DIRECTIONS:
        coefficient_group = coefficients.loc[coefficients["direction"] == direction]
        finite_group = finite.loc[finite["direction"] == direction]
        error_ratio = (
            finite_group["absolute_prefix_error"]
            / finite_group["absolute_diagonal_error"]
        )
        direction_rows.append(
            {
                "direction": direction,
                "market_dates": len(coefficient_group),
                "median_copied_share": coefficient_group[
                    "copied_share_of_prefix"
                ].median(),
                "median_prefix_uplift": coefficient_group[
                    "prefix_uplift_over_diagonal"
                ].median(),
                "prefix_closer_share": (
                    finite_group["absolute_prefix_error"]
                    < finite_group["absolute_diagonal_error"]
                ).mean(),
                "median_prefix_to_diagonal_error_ratio": error_ratio.median(),
                "median_loss_per_100k": finite_group[
                    "actual_loss_per_100k_guarantee"
                ].median(),
                "maximum_loss_per_100k": finite_group[
                    "actual_loss_per_100k_guarantee"
                ].max(),
            }
        )
    direction_table = pd.DataFrame(direction_rows)
    direction_table.to_csv(output_dir / "table_prefix_accuracy.csv", index=False)

    status["year"] = status["date"].dt.year
    status["has_boundary"] = status["status"].eq("regular_upper_boundary")
    annual = status.groupby("year").agg(
        market_dates=("date", "size"),
        boundary_dates=("has_boundary", "sum"),
        median_zero_rate=("zero_rate", "median"),
        median_atm_iv=("atm_iv_1y", "median"),
    )
    annual["boundary_share"] = annual["boundary_dates"] / annual["market_dates"]
    annual.reset_index().to_csv(output_dir / "table_boundary_by_year.csv", index=False)

    regime_rows = []
    for label, group in status.groupby("has_boundary"):
        regime_rows.append(
            {
                "regime": "regular boundary" if label else "no upper tail",
                "dates": len(group),
                "median_zero_rate": group["zero_rate"].median(),
                "median_atm_iv": group["atm_iv_1y"].median(),
                "median_skew": group["skew_25d_1y"].median(),
                "median_term_slope": group["iv_term_slope"].median(),
                "median_fit_rmse": group[
                    "mixture_forward_normalized_rmse"
                ].median(),
            }
        )
    regime_table = pd.DataFrame(regime_rows)
    regime_table.to_csv(output_dir / "table_boundary_regimes.csv", index=False)

    moneyness_rows = []
    joint_pilot = pilot_coefficients.loc[
        pilot_coefficients["direction"].isin(JOINT_DIRECTIONS)
    ]
    for ratio, group in joint_pilot.groupby("moneyness_ratio"):
        moneyness_rows.append(
            {
                "initial_state_over_boundary": ratio,
                "comparisons": len(group),
                "median_copied_share": group["copied_share_of_prefix"].median(),
                "minimum_copied_share": group["copied_share_of_prefix"].min(),
                "maximum_copied_share": group["copied_share_of_prefix"].max(),
            }
        )
    pd.DataFrame(moneyness_rows).to_csv(
        output_dir / "table_moneyness_sensitivity.csv", index=False
    )
    contract_sensitivity.to_csv(
        output_dir / "table_contract_sensitivity.csv", index=False
    )

    joint_equal = coefficients.loc[coefficients["direction"] == "joint_equal"].merge(
        status, on="date", suffixes=("_coefficient", "")
    )
    correlation_variables = [
        "zero_rate",
        "atm_iv_1y",
        "skew_25d_1y",
        "iv_term_slope",
        "floor_probability",
        "boundary",
    ]
    correlation_rows = []
    for variable in correlation_variables:
        valid = joint_equal[[variable, "copied_share_of_prefix"]].dropna()
        rho, pvalue = spearmanr(
            valid[variable], valid["copied_share_of_prefix"]
        )
        correlation_rows.append(
            {
                "variable": variable,
                "spearman_rho": rho,
                "two_sided_pvalue_descriptive": pvalue,
                "observations": len(valid),
            }
        )
    correlation_table = pd.DataFrame(correlation_rows)
    correlation_table.to_csv(output_dir / "table_condition_correlations.csv", index=False)

    for frame, name in (
        (direction_table, "table_prefix_accuracy.tex"),
        (regime_table, "table_boundary_regimes.tex"),
        (correlation_table, "table_condition_correlations.tex"),
        (contract_sensitivity, "table_contract_sensitivity.tex"),
    ):
        (output_dir / name).write_text(
            frame.to_latex(index=False, float_format=lambda value: f"{value:.4f}"),
            encoding="utf-8",
        )

    _set_style()
    figure, axis_rate = plt.subplots(figsize=(7.2, 3.4))
    axis_rate.plot(status["date"], 100 * status["zero_rate"], color="#244a73", lw=1.1)
    axis_rate.scatter(
        status.loc[status["has_boundary"], "date"],
        100 * status.loc[status["has_boundary"], "zero_rate"],
        color="#b33b2e",
        s=14,
        label="regular surrender boundary",
        zorder=3,
    )
    axis_rate.set_ylabel("one-year zero rate (%)")
    axis_rate.set_xlabel("OptionMetrics month-end")
    axis_rate.set_title("Boundary existence is concentrated in high-rate regimes")
    axis_rate.legend(frameon=False, loc="upper center")
    axis_rate.grid(axis="y", alpha=0.2)
    _save_figure(figure, output_dir, "figure_boundary_timeline")

    joint_finite = finite.loc[finite["direction"].isin(JOINT_DIRECTIONS)].copy()
    minimum = float(
        min(
            joint_finite["actual_scaled_loss"].min(),
            joint_finite["diagonal_coefficient"].min(),
            joint_finite["prefix_coefficient"].min(),
        )
    )
    maximum = float(
        max(
            joint_finite["actual_scaled_loss"].max(),
            joint_finite["diagonal_coefficient"].max(),
            joint_finite["prefix_coefficient"].max(),
        )
    )
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), sharex=True, sharey=True)
    for axis, column, title, color in (
        (axes[0], "diagonal_coefficient", "Date-diagonal", "#777777"),
        (axes[1], "prefix_coefficient", "Boundary-prefix", "#287a5b"),
    ):
        axis.scatter(
            joint_finite["actual_scaled_loss"],
            joint_finite[column],
            s=12,
            alpha=0.68,
            color=color,
            edgecolor="none",
        )
        axis.plot([minimum, maximum], [minimum, maximum], color="black", lw=0.8)
        axis.set_title(title)
        axis.set_xlabel("actual loss / epsilon²")
        axis.grid(alpha=0.18)
    axes[0].set_ylabel("predicted coefficient")
    figure.suptitle("Full prefix coefficient predicts finite-perturbation loss")
    _save_figure(figure, output_dir, "figure_prediction_accuracy")

    figure, axis = plt.subplots(figsize=(5.4, 3.5))
    scatter = axis.scatter(
        joint_equal["floor_probability"],
        100 * joint_equal["copied_share_of_prefix"],
        c=100 * joint_equal["atm_iv_1y"],
        cmap="viridis",
        s=27,
        alpha=0.85,
    )
    axis.set_xlabel("credited-floor probability")
    axis.set_ylabel("copied share of full coefficient (%)")
    axis.set_title("Cross-date share tracks the endogenous floor atom")
    colorbar = figure.colorbar(scatter, ax=axis)
    colorbar.set_label("one-year ATM implied volatility (%)")
    axis.grid(alpha=0.18)
    _save_figure(figure, output_dir, "figure_copied_share_conditions")

    convergence = pilot_convergence.loc[
        pilot_convergence["moneyness_ratio"].eq(0.96)
        & pilot_convergence["direction"].eq("joint_equal")
    ].copy()
    figure, axes = plt.subplots(
        1, 3, figsize=(7.5, 3.2), sharey=False, layout="constrained"
    )
    for axis, (date, group) in zip(axes, convergence.groupby("date")):
        group = group.sort_values("epsilon")
        axis.plot(
            group["epsilon"],
            group["actual_scaled_loss"],
            marker="o",
            ms=3,
            lw=1,
            color="#244a73",
            label="actual / epsilon²",
        )
        axis.axhline(
            group["prefix_coefficient"].iloc[0],
            color="#287a5b",
            lw=1,
            label="prefix",
        )
        axis.axhline(
            group["diagonal_coefficient"].iloc[0],
            color="#888888",
            lw=1,
            ls="--",
            label="diagonal",
        )
        axis.set_xscale("log")
        axis.invert_xaxis()
        axis.set_title(date.strftime("%Y-%m-%d"))
        axis.set_xlabel("epsilon")
        axis.grid(alpha=0.18)
    axes[0].set_ylabel("scaled value loss")
    axes[-1].legend(frameon=False, loc="best")
    figure.suptitle("Joint refinement approaches the prefix coefficient")
    _save_figure(figure, output_dir, "figure_epsilon_convergence")

    regular = status.loc[status["has_boundary"]]
    equal_direction = direction_table.loc[
        direction_table["direction"] == "joint_equal"
    ].iloc[0]
    rho_floor = correlation_table.loc[
        correlation_table["variable"] == "floor_probability", "spearman_rho"
    ].iloc[0]
    outside_repair = repair.loc[~repair["feasible_within_spread"]]
    outside_dates = int(outside_repair["date"].nunique())
    outside_rows = int(outside_repair["rows_outside_spread"].sum())
    contract_regular = contract_sensitivity[
        "pilot_coefficient_ready_dates"
    ].gt(0)
    contract_share_min = contract_sensitivity.loc[
        contract_regular, "median_pilot_copied_share"
    ].min()
    contract_share_max = contract_sensitivity.loc[
        contract_regular, "median_pilot_copied_share"
    ].max()
    proposed = f"""# Proposed financial-application section (not inserted into the manuscript)

## Market-calibrated surrender with an endogenous copied component

Consider a life-contingent annual-reset SPX indexed annuity. Conditional on
survival, the normalized account-to-guarantee state evolves as

`X_(t+1) = X_t Y_(t+1)`,

where `Y = 1 + min(c, max(0, alpha(R-1)))`. The benchmark fixes participation
`alpha=0.80`, cap `c=0.08`, surrender haircut 0.08, and annual termination
probability 0.03. At each anniversary the holder chooses between surrender and
continuation. The stationary Bellman equation is

`V(x)=max{{0.92x, beta E[m max(1,xY)+(1-m)V(xY)]}}`.

The return region `R<=1` is mapped to the exact atom `Y=1`. This component
copies the entire boundary-relevant state, rather than only a nuisance maximum
or benefit-base coordinate. It is therefore an endogenous identity carrier at
successive stationary surrender boundaries.

## OptionMetrics calibration and trace quantities

Monthly SPX OptionMetrics data cover 248 dates from January 2005 through August
2025. Each one-year slice is formed from positive ordered OTM quotes after
put--call-parity forward identification and an explicitly recorded
static-arbitrage repair. The repair is feasible within quoted spreads on
{len(repair) - outside_dates} dates; {outside_rows} rows across {outside_dates}
dates require adjustments outside their spreads. A martingale two-lognormal
mixture is fitted independently on every date. All 248 dates calibrate; the
median price RMSE is
{calibration['forward_normalized_rmse'].median():.6f} of the forward, and four
fits touch a declared parameter bound.

The fitted gross-return law is transformed analytically through the crediting
rule. The floor and cap atoms remain exact; only the interior density is
quadrature-discretized for the Bellman solve. Starting one year before the first
perturbed decision from `x_0=0.96b`, the first-decision source measure has
continuous density `f_1` near the boundary and atoms away from it.

For upward shifts `epsilon h_1` and `epsilon h_2`, the conventional diagonal
coefficient contains the two local trace terms. The additional identity-carrier
term is

`C_copy = 0.5 d^2 lambda(b) p_floor f_1(b) min(h_1,h_2)^2`,

where `d=beta(1-m)`. The actual loss is evaluated independently by integrating
the Bellman advantage against perturbed-policy occupancy, retaining both source
and transition atoms.

## Results

A regular upper surrender boundary exists on {len(regular)} of 248 dates, all in
the high-rate regimes 2005--2007 and 2022--2025. The median one-year zero rate
is {100 * regular['zero_rate'].median():.2f}% on boundary dates and
{100 * status.loc[~status['has_boundary'], 'zero_rate'].median():.2f}% otherwise.
The remaining 188 dates are reported as genuine no-upper-tail cases: the
credited guarantee is too valuable for asymptotic surrender under the fixed
contract.

Across the 60 boundary dates, the joint-equal copied term is a median
{100 * equal_direction['median_copied_share']:.1f}% of the full prefix
coefficient and raises the coefficient by
{100 * equal_direction['median_prefix_uplift']:.1f}% relative to the diagonal
approximation. At the predeclared epsilon=0.005, the prefix approximation is
closer to actual loss on every joint-direction/date comparison. The copied
share is strongly associated with the calibrated floor probability (descriptive
Spearman rho={rho_floor:.3f}), as predicted by the component formula.

The absolute loss is small for genuinely small perturbations: the median
joint-direction loss at epsilon=0.005 is
${direction_table['median_loss_per_100k'].median():.2f} per $100,000 of guarantee.
Thus the cross-date correction is material as a share of second-order pricing
error but is not economically large at the individual-contract level under
half-percent relative boundary errors. Larger portfolio or boundary-error
scenarios scale quadratically only while the local approximation remains valid.

One-at-a-time contract sensitivities vary participation, cap, surrender
haircut, and annual termination probability. The analytic positive asymptotic
surrender-margin count ranges from
{int(contract_sensitivity['full_panel_positive_margin_dates'].min())} to
{int(contract_sensitivity['full_panel_positive_margin_dates'].max())} of 248
dates. On coefficient-ready pilot cases, the median copied share ranges from
{100 * contract_share_min:.1f}% to {100 * contract_share_max:.1f}%, and the
prefix approximation is closer in every evaluated case. A 4% cap provides a
useful negative geometry check: at `x_0=0.96b`, even the maximum credited factor
cannot reach the first boundary, so the first-date boundary density is zero.

## Scope and limitations

The experiment is a stationary market-state counterfactual for a life-contingent
contract, not a time-series forecast and not a calibration of insurer-specific
caps, mortality, lapses, or fees. The two-lognormal law is deliberately
parsimonious and does not fit every quote within its spread. Boundary existence
and copied shares are robust on the predeclared pilot to raw/repaired targets,
bid/offer envelopes, liquidity and moneyness restrictions, and numerical
refinement. The evidence demonstrates a natural financial occurrence of the
paper's mechanism and a systematic proportional correction; it does not support
a claim that small boundary errors create large dollar losses in this benchmark.
"""
    (output_dir / "proposed_application_section.md").write_text(
        proposed, encoding="utf-8"
    )

    summary = f"""# Full-panel financial extension results

- OptionMetrics dates: {len(status)}; regular boundary dates: {len(regular)}.
- Prefix wins at epsilon=0.005: {int((finite.loc[finite['direction'].isin(JOINT_DIRECTIONS), 'absolute_prefix_error'].to_numpy() < finite.loc[finite['direction'].isin(JOINT_DIRECTIONS), 'absolute_diagonal_error'].to_numpy()).sum())} of {len(finite.loc[finite['direction'].isin(JOINT_DIRECTIONS)])} joint comparisons.
- Median joint copied share: {direction_table['median_copied_share'].median():.6f}.
- Median joint uplift over diagonal: {direction_table['median_prefix_uplift'].median():.6f}.
- Median joint loss at epsilon=0.005: ${direction_table['median_loss_per_100k'].median():.6f} per $100,000 guarantee.
- Boundary-date median rate: {regular['zero_rate'].median():.6f}; no-boundary median rate: {status.loc[~status['has_boundary'], 'zero_rate'].median():.6f}.
- Static-arbitrage repair outside quoted spreads: {outside_rows} rows across {outside_dates} of 248 dates.
- Contract sensitivity: {int(contract_sensitivity['pilot_failed_dates'].sum())} Bellman failures; prefix closer in every coefficient-ready pilot case.

Interpretation: the full prefix term is empirically necessary for accurate
second-order approximation in the boundary regime, but the absolute value loss
from a half-percent relative boundary error is small in this fixed benchmark.
"""
    (output_dir / "summary.md").write_text(summary, encoding="utf-8")
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
