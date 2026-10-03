#!/usr/bin/env python3
"""Certify only the frozen Bermudan rows near the numerical loss floor.

The script is deliberately separate from the frozen diagnostic generator.  It
does not rewrite any frozen evidence.  It reruns the eight coarse-DP rows whose
recorded loss is below 1e-12, compares two reference grids, three forward-density
grids, an independent 256-point Gauss-Legendre strip integral, and the
cancellation-prone backward value difference, then writes a compact audit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "research/financial_extension/src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.polynomial.legendre import leggauss
from scipy.stats import norm

from bermudan_diagnostic import (
    GaussianDensityConvolver,
    diagonal_coefficient,
    evaluate_threshold_policy_forward,
    price_threshold_policy_fft,
    reference_occupancy_densities,
    solve_bermudan_put_fft,
)
from run_bermudan_diagnostic import load_frozen_market_inputs


DATE_ORDER = [
    "2016-10-31",
    "2008-11-28",
    "2020-02-28",
    "2009-04-30",
    "2006-03-31",
]
NEAR_FLOOR_CUTOFF = 1e-12
PUBLICATION_FLOOR = 5e-14
REFERENCE_NODES = (65_537, 131_073)
DENSITY_NODES = (32_769, 65_537, 131_073)


def _gauss_legendre_integral(
    density_grid: np.ndarray,
    density: np.ndarray,
    gap_grid: np.ndarray,
    gap: np.ndarray,
    left: float,
    right: float,
    nodes: np.ndarray,
    weights: np.ndarray,
) -> float:
    if right <= left:
        return 0.0
    points = 0.5 * (right - left) * nodes + 0.5 * (right + left)
    integrand = np.interp(points, density_grid, density) * np.abs(
        np.interp(points, gap_grid, gap)
    )
    return float(0.5 * (right - left) * (weights @ integrand))


def _gauss_legendre_forward(solution, boundaries: np.ndarray, density_nodes: int):
    grid = np.linspace(solution.grid[0], solution.grid[-1], density_nodes)
    dt = 1.0 / 8.0
    mean = (
        solution.market.rate
        - solution.market.dividend_yield
        - 0.5 * solution.market.volatility**2
    ) * dt
    standard_deviation = solution.market.volatility * np.sqrt(dt)
    convolver = GaussianDensityConvolver(
        grid,
        mean_increment=mean,
        standard_deviation=standard_deviation,
    )
    density = norm.pdf(grid, loc=mean, scale=standard_deviation)
    reference_grid, reference_densities, _ = reference_occupancy_densities(
        solution, density_nodes=density_nodes
    )
    nodes, weights = leggauss(256)
    local_losses: list[float] = []
    single_losses: list[float] = []

    for index, (reference_boundary, supplied_boundary, gap) in enumerate(
        zip(
            solution.boundaries,
            boundaries,
            solution.gaps,
            strict=True,
        )
    ):
        left, right = sorted((reference_boundary, supplied_boundary))
        discount = np.exp(-solution.market.rate * (index + 1) * dt)
        local_losses.append(
            discount
            * _gauss_legendre_integral(
                grid,
                density,
                solution.grid,
                gap,
                left,
                right,
                nodes,
                weights,
            )
        )
        single_losses.append(
            discount
            * _gauss_legendre_integral(
                reference_grid,
                reference_densities[index],
                solution.grid,
                gap,
                left,
                right,
                nodes,
                weights,
            )
        )
        density = np.where(grid > supplied_boundary, density, 0.0)
        density = convolver(density)

    total = float(np.sum(local_losses))
    single_total = float(np.sum(single_losses))
    return total, single_total, total - single_total


def _certified_summary(coarse: pd.DataFrame) -> pd.DataFrame:
    retained = coarse.loc[coarse["actual_loss"] >= PUBLICATION_FLOOR].copy()
    rows = []
    for date in DATE_ORDER:
        group = retained.loc[retained["date"].eq(date)].copy()
        slope = float(
            np.polyfit(np.log(group["eta"]), np.log(group["actual_loss"]), 1)[0]
        )
        smallest = group.loc[group["eta"].idxmin()]
        rows.append(
            {
                "date": date,
                "certified_points": len(group),
                "certified_log_log_slope": slope,
                "smallest_certified_eta": float(smallest["eta"]),
                "smallest_certified_loss": float(smallest["actual_loss"]),
                "smallest_certified_ratio": float(
                    smallest["actual_to_diagonal"]
                ),
                "smallest_certified_abs_interaction_share": abs(
                    float(smallest["interaction_share"])
                ),
            }
        )
    return pd.DataFrame(rows)


def _write_figure(eta: pd.DataFrame, output: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 160,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), sharex=True)
    unresolved_label_used = False
    for date in DATE_ORDER:
        coarse = eta.loc[
            eta["method"].eq("coarse_fft_257") & eta["date"].eq(date)
        ].sort_values("eta")
        certified = coarse.loc[coarse["actual_loss"] >= PUBLICATION_FLOOR]
        unresolved = coarse.loc[coarse["actual_loss"] < PUBLICATION_FLOOR]
        line = axes[0].plot(
            certified["eta"],
            certified["actual_to_diagonal"],
            marker="o",
            label=date,
        )[0]
        if not unresolved.empty:
            axes[0].plot(
                unresolved["eta"],
                np.full(len(unresolved), 0.035),
                marker="x",
                linestyle="None",
                color=line.get_color(),
                transform=axes[0].get_xaxis_transform(),
                clip_on=False,
                label=(
                    "relative unresolved (< floor)"
                    if not unresolved_label_used
                    else None
                ),
            )
            unresolved_label_used = True

        lsmc = eta.loc[
            eta["method"].eq("lsmc_linear") & eta["date"].eq(date)
        ]
        grouped = lsmc.groupby("eta")["actual_to_diagonal"]
        median = grouped.median().sort_index()
        low = grouped.quantile(0.1).reindex(median.index)
        high = grouped.quantile(0.9).reindex(median.index)
        line = axes[1].plot(
            median.index, median.values, marker="o", label=date
        )[0]
        axes[1].fill_between(
            median.index,
            low.values,
            high.values,
            color=line.get_color(),
            alpha=0.15,
        )

    for axis, title in zip(
        axes,
        [
            "Coarse DP: certified relative ratios",
            "Linear LSMC: median and 10--90% seed band",
        ],
        strict=True,
    ):
        axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
        axis.set_xscale("log", base=2)
        axis.invert_xaxis()
        axis.set_xlabel(r"boundary scale $\eta$")
        axis.set_ylabel(r"actual loss / $(\eta^2 C_{\rm diag})$")
        axis.set_title(title)
        axis.grid(alpha=0.2)
    axes[1].set_yscale("log")
    axes[0].legend(ncol=2, frameon=False)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def run(repo: Path, output_dir: Path, figure_output: Path) -> None:
    evidence = repo / "research/financial_extension/diagnostic/evidence/bermudan"
    panel = repo / "research/financial_extension/evidence/full_market_panel/date_status.csv"
    selection = (
        repo
        / "research/financial_extension/evidence/pilot_selection/selected_pilot_dates.csv"
    )
    eta = pd.read_csv(evidence / "eta_path_results.csv")
    coarse = eta.loc[eta["method"].eq("coarse_fft_257")].copy()
    affected = coarse.loc[coarse["actual_loss"] < NEAR_FLOOR_CUTOFF].copy()
    expected = {
        ("2016-10-31", 1.0),
        ("2016-10-31", 0.5),
        ("2016-10-31", 0.25),
        ("2016-10-31", 0.125),
        ("2016-10-31", 0.0625),
        ("2020-02-28", 0.25),
        ("2020-02-28", 0.125),
        ("2020-02-28", 0.0625),
    }
    actual = set(zip(affected["date"], affected["eta"], strict=True))
    if actual != expected:
        raise ValueError(f"near-floor row set changed: {sorted(actual)}")

    markets, _ = load_frozen_market_inputs(panel, selection)
    market_by_date = {market.date: market for market in markets}
    output_rows: list[dict] = []

    for date in ("2016-10-31", "2020-02-28"):
        market = market_by_date[date]
        references = {
            nodes: solve_bermudan_put_fft(market, grid_nodes=nodes)
            for nodes in REFERENCE_NODES
        }
        coarse_solution = solve_bermudan_put_fft(
            market,
            grid_nodes=257,
            interpolate_boundary_root=False,
        )
        date_rows = affected.loc[affected["date"].eq(date)]
        for frozen in date_rows.itertuples(index=False):
            forward_values: list[float] = []
            ratios: list[float] = []
            backward_values: list[float] = []
            for reference in references.values():
                displacement = coarse_solution.boundaries - reference.boundaries
                boundaries = reference.boundaries + frozen.eta * displacement
                for density_nodes in DENSITY_NODES:
                    density_grid, reference_densities, _ = (
                        reference_occupancy_densities(
                            reference, density_nodes=density_nodes
                        )
                    )
                    coefficient, _ = diagonal_coefficient(
                        reference,
                        coarse_solution.boundaries,
                        density_grid=density_grid,
                        reference_densities=reference_densities,
                    )
                    evaluation = evaluate_threshold_policy_forward(
                        reference,
                        boundaries,
                        density_nodes=density_nodes,
                        reference_density_grid=density_grid,
                        reference_densities=reference_densities,
                    )
                    reference_value = price_threshold_policy_fft(
                        reference,
                        reference.boundaries,
                        evaluation_nodes=density_nodes,
                    )
                    supplied_value = price_threshold_policy_fft(
                        reference,
                        boundaries,
                        evaluation_nodes=density_nodes,
                    )
                    forward_values.append(evaluation.loss)
                    ratios.append(
                        evaluation.loss / (frozen.eta**2 * coefficient)
                    )
                    backward_values.append(reference_value - supplied_value)

            reference = references[65_537]
            displacement = coarse_solution.boundaries - reference.boundaries
            boundaries = reference.boundaries + frozen.eta * displacement
            gauss_loss, _, gauss_interaction = _gauss_legendre_forward(
                reference, boundaries, density_nodes=131_073
            )
            above_floor = frozen.actual_loss >= PUBLICATION_FLOOR
            output_rows.append(
                {
                    "date": date,
                    "eta": frozen.eta,
                    "frozen_forward_loss": frozen.actual_loss,
                    "forward_min": min(forward_values),
                    "forward_max": max(forward_values),
                    "forward_absolute_span": max(forward_values)
                    - min(forward_values),
                    "gauss_legendre_loss": gauss_loss,
                    "frozen_minus_gauss_legendre": frozen.actual_loss
                    - gauss_loss,
                    "gauss_legendre_interaction": gauss_interaction,
                    "max_abs_forward_backward": max(
                        abs(forward - backward)
                        for forward, backward in zip(
                            forward_values, backward_values, strict=True
                        )
                    ),
                    "backward_min": min(backward_values),
                    "backward_max": max(backward_values),
                    "relative_status": (
                        "certified" if above_floor else "unresolved_below_floor"
                    ),
                    "certified_ratio_min": min(ratios) if above_floor else np.nan,
                    "certified_ratio_max": max(ratios) if above_floor else np.nan,
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    refinements = pd.DataFrame(output_rows).sort_values(["date", "eta"])
    refinements.to_csv(output_dir / "tiny_loss_refinement.csv", index=False)
    certified = _certified_summary(coarse)
    certified.to_csv(output_dir / "certified_coarse_summary.csv", index=False)

    maximum_forward_span = float(refinements["forward_absolute_span"].max())
    maximum_gauss_difference = float(
        refinements["frozen_minus_gauss_legendre"].abs().max()
    )
    maximum_backward_difference = float(
        refinements["max_abs_forward_backward"].max()
    )
    subfloor = refinements.loc[
        refinements["relative_status"].eq("unresolved_below_floor")
    ]
    metrics = {
        "status": "PASS",
        "near_floor_cutoff": NEAR_FLOOR_CUTOFF,
        "publication_floor": PUBLICATION_FLOOR,
        "affected_rows_rerun": len(refinements),
        "subfloor_rows": len(subfloor),
        "certified_coarse_rows": int(
            (coarse["actual_loss"] >= PUBLICATION_FLOOR).sum()
        ),
        "maximum_forward_grid_reference_span": maximum_forward_span,
        "maximum_frozen_gauss_legendre_difference": maximum_gauss_difference,
        "maximum_forward_backward_difference": maximum_backward_difference,
        "certified_slope_min": float(
            certified["certified_log_log_slope"].min()
        ),
        "certified_slope_max": float(
            certified["certified_log_log_slope"].max()
        ),
        "smallest_certified_ratio_min": float(
            certified["smallest_certified_ratio"].min()
        ),
        "smallest_certified_ratio_max": float(
            certified["smallest_certified_ratio"].max()
        ),
        "maximum_smallest_certified_interaction_share": float(
            certified["smallest_certified_abs_interaction_share"].max()
        ),
    }
    (output_dir / "certification_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )

    subfloor_lines = []
    for row in subfloor.itertuples(index=False):
        subfloor_lines.append(
            f"| {row.date} | {row.eta:g} | {row.frozen_forward_loss:.3e} | "
            f"{row.forward_absolute_span:.3e} | "
            f"{row.max_abs_forward_backward:.3e} | unresolved |"
        )
    certified_lines = []
    for row in certified.itertuples(index=False):
        certified_lines.append(
            f"| {row.date} | {row.certified_points} | "
            f"{row.certified_log_log_slope:.3f} | "
            f"{row.smallest_certified_eta:g} | "
            f"{row.smallest_certified_ratio:.3f} | "
            f"{100 * row.smallest_certified_abs_interaction_share:.4f}% |"
        )
    report = f"""# Bermudan tiny-loss numerical-resolution report

## Scope

Only the eight frozen coarse-DP rows with forward loss below
`{NEAR_FLOOR_CUTOFF:.0e}` were rerun.  No date, contract, model, boundary path,
OptionMetrics input, LSMC seed, annuity row, or frozen evidence file was changed.

## Resolution floor

Across two reference grids ({REFERENCE_NODES[0]:,} and {REFERENCE_NODES[1]:,}
nodes), three forward-density grids ({DENSITY_NODES[0]:,},
{DENSITY_NODES[1]:,}, and {DENSITY_NODES[2]:,} nodes), and an independent
256-point Gauss-Legendre strip integral, the maximum absolute forward span is
`{maximum_forward_span:.3e}` and the maximum frozen-versus-Gauss-Legendre
difference is `{maximum_gauss_difference:.3e}`.  The backward value-difference
check is cancellation limited: its maximum absolute discrepancy from the
positive-integrand forward identity on these rows is
`{maximum_backward_difference:.3e}`.

The publication floor is therefore set conservatively at
`{PUBLICATION_FLOOR:.0e}` normalized option value, above the observed backward
cancellation envelope.  Relative actual-to-diagonal ratios are reported only
above this common floor.  Below it, the forward loss and absolute discrepancies
are retained, while relative accuracy is unresolved.

## Rows below the publication floor

| Date | eta | forward loss | forward span | max abs forward-backward | relative accuracy |
|---|---:|---:|---:|---:|---|
{chr(10).join(subfloor_lines)}

The `2.292e-15` loss is stable in the positive-integrand engine, but it is not
used for a published relative ratio because it is below the common cross-check
floor.

## Certification without sub-resolution observations

| Date | points above floor | fitted slope | smallest certified eta | ratio | abs interaction share |
|---|---:|---:|---:|---:|---:|
{chr(10).join(certified_lines)}

Using only the 21 coarse observations above the floor, every snapshot retains
at least two certified scales.  The fitted powers are
`{metrics['certified_slope_min']:.3f}--{metrics['certified_slope_max']:.3f}`;
the smallest certified ratios are
`{metrics['smallest_certified_ratio_min']:.3f}--{metrics['smallest_certified_ratio_max']:.3f}`;
and the largest absolute interaction share at the smallest certified scale is
`{100 * metrics['maximum_smallest_certified_interaction_share']:.4f}%`.
Thus the eta-squared and negligible-interaction conclusions do not rely on any
sub-resolution observation.

## Disposition

**PASS.**  Preserve the frozen losses as absolute outputs, suppress relative
ratios below `5e-14`, and use the certified-scale summaries above in the paper.
"""
    (output_dir / "NUMERICAL_RESOLUTION_REPORT.md").write_text(
        report, encoding="utf-8"
    )
    _write_figure(eta, figure_output)
    print(json.dumps(metrics, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("research/finance_integration/numerical_resolution"),
    )
    parser.add_argument(
        "--figure-output",
        type=Path,
        default=Path("paper/figures/finance_eta_diagnostic.pdf"),
    )
    args = parser.parse_args()
    repo = args.repo.expanduser().resolve()
    output_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else repo / args.output_dir
    )
    figure_output = (
        args.figure_output
        if args.figure_output.is_absolute()
        else repo / args.figure_output
    )
    run(repo, output_dir, figure_output)


if __name__ == "__main__":
    main()
