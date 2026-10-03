"""Generate compact tables, figures, and summaries for the diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATE_ORDER = [
    "2016-10-31",
    "2008-11-28",
    "2020-02-28",
    "2009-04-30",
    "2006-03-31",
]


def _write_latex(frame: pd.DataFrame, path: Path, *, column_format: str) -> None:
    latex = frame.to_latex(
        index=False,
        escape=False,
        column_format=column_format,
    )
    path.write_text(latex, encoding="utf-8")


def analyze(evidence_dir: Path, audit_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    market = pd.read_csv(evidence_dir / "market_inputs.csv")
    boundaries = pd.read_csv(evidence_dir / "boundaries.csv")
    refinement = pd.read_csv(evidence_dir / "reference_refinement.csv")
    eta = pd.read_csv(evidence_dir / "eta_path_results.csv")
    gates = pd.read_csv(evidence_dir / "gate_results.csv")
    native = pd.read_csv(evidence_dir / "native_policy_evaluation.csv")
    validation = json.loads((audit_dir / "validation_metrics.json").read_text())

    reference_boundaries = boundaries[
        boundaries["method"] == "reference_fft_65537"
    ][
        [
            "date",
            "exercise_index",
            "exercise_time",
            "log_boundary",
            "spot_boundary",
        ]
    ].rename(
        columns={
            "log_boundary": "reference_log_boundary",
            "spot_boundary": "reference_spot_boundary",
        }
    )
    coarse_boundaries = boundaries[boundaries["method"] == "coarse_fft_257"]
    coarse_displacement = coarse_boundaries.merge(
        reference_boundaries, on=["date", "exercise_index"]
    )
    coarse_displacement["abs_shift"] = np.abs(
        coarse_displacement["log_boundary"]
        - coarse_displacement["reference_log_boundary"]
    )
    coarse_max_shift = coarse_displacement.groupby("date")["abs_shift"].max()

    primary_refinement = refinement[
        (refinement["method"] == "fft_convolution")
        & (refinement["grid_nodes"] == 65_537)
    ].set_index("date")
    coarse_gates = gates[gates["method"] == "coarse_fft_257"].set_index("date")
    ref_first = reference_boundaries[
        reference_boundaries["exercise_index"] == 1
    ].set_index("date")
    ref_last = reference_boundaries[
        reference_boundaries["exercise_index"] == 7
    ].set_index("date")
    market_indexed = market.set_index("date")

    reference_rows = []
    for date in DATE_ORDER:
        reference_rows.append(
            {
                "Date": date,
                "$r$": market_indexed.loc[date, "zero_rate"],
                "$q$": market_indexed.loc[
                    date, "black_scholes_dividend_yield"
                ],
                "$\\sigma$": market_indexed.loc[date, "atm_iv_1y"],
                "$B_1$": ref_first.loc[date, "reference_spot_boundary"],
                "$B_7$": ref_last.loc[date, "reference_spot_boundary"],
                "$V_0$": primary_refinement.loc[date, "value"],
                "max $|\\widehat b-b|$": coarse_max_shift.loc[date],
                "slope": coarse_gates.loc[
                    date, "log_log_slope_eta_le_quarter"
                ],
                "ratio $1/16$": coarse_gates.loc[
                    date, "ratio_eta_one_sixteenth"
                ],
                "gate": "pass" if coarse_gates.loc[date, "gate_pass"] else "fail",
            }
        )
    reference_table = pd.DataFrame(reference_rows)
    reference_table.to_csv(output_dir / "table_reference_coarse.csv", index=False)
    reference_display = reference_table.copy()
    for col in ["$r$", "$q$", "$\\sigma$", "$B_1$", "$B_7$", "$V_0$"]:
        reference_display[col] = reference_display[col].map(lambda x: f"{x:.4f}")
    reference_display["max $|\\widehat b-b|$"] = reference_display[
        "max $|\\widehat b-b|$"
    ].map(lambda x: f"{x:.4f}")
    reference_display["slope"] = reference_display["slope"].map(
        lambda x: f"{x:.3f}"
    )
    reference_display["ratio $1/16$"] = reference_display["ratio $1/16$"].map(
        lambda x: f"{x:.3f}"
    )
    _write_latex(
        reference_display,
        output_dir / "table_reference_coarse.tex",
        column_format="lrrrrrrrrrl",
    )

    lsmc_boundaries = boundaries[boundaries["method"] == "lsmc_linear"].merge(
        reference_boundaries, on=["date", "exercise_index"]
    )
    lsmc_boundaries["abs_shift"] = np.abs(
        lsmc_boundaries["log_boundary"]
        - lsmc_boundaries["reference_log_boundary"]
    )
    seed_shift = lsmc_boundaries.groupby(["date", "seed"])["abs_shift"].max()
    lsmc_gates = gates[gates["method"] == "lsmc_linear"]
    lsmc_rows = []
    for date in DATE_ORDER:
        date_gates = lsmc_gates[lsmc_gates["date"] == date]
        date_native = native[native["date"] == date]
        shifts = seed_shift.loc[date]
        lsmc_rows.append(
            {
                "Date": date,
                "regular": f"{int(date_gates['boundary_regular'].sum())}/10",
                "gate passes": f"{int(date_gates['gate_pass'].sum())}/10",
                "median max $|h|$": float(shifts.median()),
                "median slope": float(
                    date_gates["log_log_slope_eta_le_quarter"].median()
                ),
                "median ratio $1/16$": float(
                    date_gates["ratio_eta_one_sixteenth"].median()
                ),
                "ratio range $1/16$": (
                    f"{date_gates['ratio_eta_one_sixteenth'].min():.3f}--"
                    f"{date_gates['ratio_eta_one_sixteenth'].max():.3f}"
                ),
                "median native loss": float(
                    date_native["deterministic_native_loss"].median()
                ),
                "MC consistent": f"{int(date_native['heldout_consistent'].sum())}/10",
            }
        )
    lsmc_table = pd.DataFrame(lsmc_rows)
    lsmc_table.to_csv(output_dir / "table_lsmc_audit.csv", index=False)
    lsmc_display = lsmc_table.copy()
    lsmc_display["median max $|h|$"] = lsmc_display[
        "median max $|h|$"
    ].map(lambda x: f"{x:.3f}")
    lsmc_display["median slope"] = lsmc_display["median slope"].map(
        lambda x: f"{x:.3f}"
    )
    lsmc_display["median ratio $1/16$"] = lsmc_display[
        "median ratio $1/16$"
    ].map(lambda x: f"{x:.3f}")
    lsmc_display["median native loss"] = lsmc_display[
        "median native loss"
    ].map(lambda x: f"{x:.2e}")
    _write_latex(
        lsmc_display,
        output_dir / "table_lsmc_audit.tex",
        column_format="lccrrrrrc",
    )

    eta_rows = []
    for method in ["coarse_fft_257", "lsmc_linear"]:
        subset = eta[eta["method"] == method]
        for date in DATE_ORDER:
            date_subset = subset[subset["date"] == date]
            grouped = date_subset.groupby("eta")["actual_to_diagonal"]
            gate_subset = gates[(gates["method"] == method) & (gates["date"] == date)]
            eta_rows.append(
                {
                    "Method": "coarse DP" if method == "coarse_fft_257" else "LSMC median",
                    "Date": date,
                    "$\\eta=1$": grouped.median().loc[1.0],
                    "$\\eta=1/4$": grouped.median().loc[0.25],
                    "$\\eta=1/16$": grouped.median().loc[0.0625],
                    "slope": gate_subset[
                        "log_log_slope_eta_le_quarter"
                    ].median(),
                    "passes": int(gate_subset["gate_pass"].sum()),
                    "total": len(gate_subset),
                }
            )
    eta_table = pd.DataFrame(eta_rows)
    eta_table.to_csv(output_dir / "table_eta_summary.csv", index=False)
    eta_display = eta_table.copy()
    for col in ["$\\eta=1$", "$\\eta=1/4$", "$\\eta=1/16$", "slope"]:
        eta_display[col] = eta_display[col].map(lambda x: f"{x:.3f}")
    eta_display["gate"] = eta_display["passes"].astype(str) + "/" + eta_display[
        "total"
    ].astype(str)
    eta_display = eta_display.drop(columns=["passes", "total"])
    _write_latex(
        eta_display,
        output_dir / "table_eta_summary.tex",
        column_format="llrrrrc",
    )

    taxonomy = pd.DataFrame(
        [
            {
                "Case": "smooth Bermudan put",
                "Transport geometry": "smooth transition density",
                "Eligible occupancy": "positive at each regular boundary",
                "Second-order result": "date diagonal only",
                "Disposition": "coarse control passes; LSMC often nonlocal",
            },
            {
                "Case": "full-recall search",
                "Transport geometry": "copied state aligned with stopping normal",
                "Eligible occupancy": "positive",
                "Second-order result": "occupied prefix correction",
                "Disposition": "natural illustration retained in Paper I",
            },
            {
                "Case": "stationary annuity local",
                "Transport geometry": "floor identity copies scalar normal",
                "Eligible occupancy": "zero from issue",
                "Second-order result": "local prefix, coherently pairwise reconstructible",
                "Disposition": "falsification diagnostic only",
            },
            {
                "Case": "annuity renewal/MVA controls",
                "Transport geometry": "boundary moves or rate normal is not copied",
                "Eligible occupancy": "not sufficient",
                "Second-order result": "exact resonance fails",
                "Disposition": "no product-level prefix claim",
            },
        ]
    )
    taxonomy.to_csv(output_dir / "table_taxonomy.csv", index=False)
    _write_latex(
        taxonomy,
        output_dir / "table_taxonomy.tex",
        column_format=(
            ">{\\raggedright\\arraybackslash}p{0.17\\linewidth}"
            ">{\\raggedright\\arraybackslash}p{0.23\\linewidth}"
            ">{\\raggedright\\arraybackslash}p{0.17\\linewidth}"
            ">{\\raggedright\\arraybackslash}p{0.20\\linewidth}"
            ">{\\raggedright\\arraybackslash}p{0.18\\linewidth}"
        ),
    )

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 160,
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2), sharex=True)
    for ax, date in zip(axes.flat, DATE_ORDER, strict=False):
        ref = reference_boundaries[reference_boundaries["date"] == date]
        coarse = coarse_boundaries[coarse_boundaries["date"] == date]
        lsmc = boundaries[
            (boundaries["method"] == "lsmc_linear") & (boundaries["date"] == date)
        ]
        for _, seed_frame in lsmc.groupby("seed"):
            ax.plot(
                seed_frame["exercise_time"],
                seed_frame["spot_boundary"],
                color="#6baed6",
                alpha=0.22,
                linewidth=0.7,
            )
        median = lsmc.groupby("exercise_time", as_index=False)["spot_boundary"].median()
        ax.plot(
            median["exercise_time"],
            median["spot_boundary"],
            color="#2171b5",
            linewidth=1.8,
            label="LSMC median",
        )
        ax.plot(
            coarse["exercise_time"],
            coarse["spot_boundary"],
            "--",
            color="#d95f0e",
            linewidth=1.5,
            label="coarse DP",
        )
        ax.plot(
            ref["exercise_time"],
            ref["reference_spot_boundary"],
            color="black",
            linewidth=1.8,
            label="reference",
        )
        ax.set_title(date)
        ax.grid(alpha=0.2)
        ax.set_ylim(0.25, 1.02)
    axes.flat[-1].axis("off")
    for ax in axes[-1, :2]:
        ax.set_xlabel("exercise time")
    for ax in axes[:, 0]:
        ax.set_ylabel("spot boundary / strike")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.94, 0.13))
    fig.suptitle("Native Bermudan boundaries: reference, coarse DP, and ten LSMC seeds")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_dir / "figure_boundaries.png", bbox_inches="tight")
    fig.savefig(output_dir / "figure_boundaries.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), sharex=True)
    for date in DATE_ORDER:
        coarse = eta[
            (eta["method"] == "coarse_fft_257") & (eta["date"] == date)
        ].sort_values("eta")
        axes[0].plot(
            coarse["eta"], coarse["actual_to_diagonal"], marker="o", label=date
        )
        lsmc = eta[(eta["method"] == "lsmc_linear") & (eta["date"] == date)]
        grouped = lsmc.groupby("eta")["actual_to_diagonal"]
        med = grouped.median().sort_index()
        low = grouped.quantile(0.1).reindex(med.index)
        high = grouped.quantile(0.9).reindex(med.index)
        line = axes[1].plot(med.index, med.values, marker="o", label=date)[0]
        axes[1].fill_between(
            med.index, low.values, high.values, color=line.get_color(), alpha=0.15
        )
    for ax, title in zip(
        axes,
        ["Coarse DP: all five dates", "Linear LSMC: median and 10--90% seed band"],
        strict=True,
    ):
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
        ax.set_xscale("log", base=2)
        ax.invert_xaxis()
        ax.set_xlabel(r"boundary scale $\eta$")
        ax.set_ylabel(r"actual loss / $(\eta^2 C_{\rm diag})$")
        ax.set_title(title)
        ax.grid(alpha=0.2)
    axes[1].set_yscale("log")
    axes[0].legend(ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "figure_eta_diagnostic.png", bbox_inches="tight")
    fig.savefig(output_dir / "figure_eta_diagnostic.pdf", bbox_inches="tight")
    plt.close(fig)

    lsmc_gate_count = int(
        gates[gates["method"] == "lsmc_linear"]["gate_pass"].sum()
    )
    summary = f"""# Smooth-control and annuity falsification diagnostic

## Result

The smooth Bermudan control gives a clean positive control for the
date-diagonal theory, but the regression exercise is too weak for main-text
integration. All 55 approximate policies have one correctly oriented native
boundary at each of seven nonterminal dates. Every 257-node coarse-DP policy
passes the frozen local gate: at `eta=1/16`, actual-to-diagonal ratios range
from {coarse_gates['ratio_eta_one_sixteenth'].min():.4f} to
{coarse_gates['ratio_eta_one_sixteenth'].max():.4f}, and fitted log--log slopes
range from {coarse_gates['log_log_slope_eta_le_quarter'].min():.4f} to
{coarse_gates['log_log_slope_eta_le_quarter'].max():.4f}.

The linear LSMC policies remain boundary-regular, and held-out repricing is
consistent in {int(native['heldout_consistent'].sum())}/50 cases. But only
{lsmc_gate_count}/50 pass the local scaling/diagonal gate. On three dates the
true early stopping boundary is deep in a rarely occupied tail; LSMC places it
much closer to the strike. Even after multiplying the displacement by 1/16,
the path is not local enough: median actual-to-diagonal ratios are
{lsmc_table.set_index('Date').loc['2016-10-31', 'median ratio $1/16$']:.3f},
{lsmc_table.set_index('Date').loc['2020-02-28', 'median ratio $1/16$']:.3f}, and
{lsmc_table.set_index('Date').loc['2009-04-30', 'median ratio $1/16$']:.3f} on
those dates. Seed replication makes the failure more credible, not less.

This is not evidence of a missing prefix term. The Black--Scholes transition
has a smooth density, and the maximum joint-minus-one-date interaction share at
`eta=1/16` is
{validation['bermudan']['max_lsmc_interaction_share_eta_one_sixteenth']:.2%}.
The failure is finite-scale curvature caused by a nonlocal supplied boundary.

## Frozen annuity stress test

The annuity evidence was read, not rerun. Its stationary local identity remains
valid: 84/84 local coefficients are positive and direct losses converge. But
coherent labelled pairwise reconstruction is exact to
{validation['annuity_read_only_audit']['max_coherent_pairwise_error']:.2e}, and
issue-state occupancy is positive in 0/84 rows. Renewal-cap changes range from
{validation['annuity_read_only_audit']['renewal_boundary_change_min']:.4f} to
{validation['annuity_read_only_audit']['renewal_boundary_change_max']:.4f}; the
MVA/rate-state control also loses complete transport of the stopping normal.

The combined necessary condition is sharper than “there is an atom”: a
singular component must transport the complete target normal exactly and carry
positive occupied eligible trace.

## Integration recommendation

The coarse Bermudan result is a useful control and the four-case taxonomy is
conceptually clean. The LSMC path, however, does not meet the predeclared local
gate on three of five dates. Inserting this as a successful financial or
statistical validation would overstate the evidence. Retain the package as a
standalone appendix/internal audit; do not modify Paper I.

**USEFUL BUT TOO WEAK FOR MAIN TEXT**
"""
    (output_dir / "analysis_summary.md").write_text(summary, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=Path("research/financial_extension/diagnostic/evidence/bermudan"),
    )
    parser.add_argument(
        "--audit-dir",
        type=Path,
        default=Path("research/financial_extension/diagnostic/audit"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("research/financial_extension/diagnostic/publication"),
    )
    args = parser.parse_args()
    analyze(args.evidence_dir, args.audit_dir, args.output_dir)


if __name__ == "__main__":
    main()
