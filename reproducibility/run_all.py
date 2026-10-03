#!/usr/bin/env python3
"""Regenerate every numerical result and diagnostic in one deterministic run."""

from __future__ import annotations

import csv
import json
import math
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.compute import (  # noqa: E402
    all_sign_direct,
    all_sign_formula,
    cubic_correction,
    diffuse_tube_probability,
    error_coupling_checks,
    full_recall_coefficients,
    full_recall_uniform_channels,
    full_recall_uniform_exact,
    gaussian_crossover,
    gaussian_diffuse_endpoint,
    log_slope,
    resurrection_coefficients,
    running_max_coefficients,
    singular_tube_probability,
)


OUTPUT = ROOT / "outputs"
CSV_DIR = OUTPUT / "csv"
FIG_DIR = OUTPUT / "figures"


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT.parent,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "UNCOMMITTED"


def write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def svg_lines(
    path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    series: list[tuple[str, str, list[tuple[float, float]]]],
    log_x: bool = False,
    log_y: bool = False,
) -> None:
    width, height = 760, 480
    left, right, top, bottom = 88, 26, 55, 70
    points = [p for _, _, values in series for p in values]
    tx = (lambda x: math.log10(x)) if log_x else (lambda x: x)
    ty = (lambda y: math.log10(y)) if log_y else (lambda y: y)
    xs, ys = [tx(x) for x, _ in points], [ty(y) for _, y in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if ymin == ymax:
        ymin, ymax = ymin - 1.0, ymax + 1.0
    xpad, ypad = 0.04 * (xmax - xmin), 0.08 * (ymax - ymin)
    xmin, xmax, ymin, ymax = xmin - xpad, xmax + xpad, ymin - ypad, ymax + ypad

    def px(x: float) -> float:
        return left + (tx(x) - xmin) * (width - left - right) / (xmax - xmin)

    def py(y: float) -> float:
        return height - bottom - (ty(y) - ymin) * (height - top - bottom) / (ymax - ymin)

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="18">{title}</text>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="black"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="black"/>',
        f'<text x="{width/2}" y="{height-18}" text-anchor="middle" font-family="sans-serif" font-size="14">{xlabel}</text>',
        f'<text x="20" y="{height/2}" text-anchor="middle" transform="rotate(-90 20 {height/2})" font-family="sans-serif" font-size="14">{ylabel}</text>',
    ]
    for label, color, values in series:
        coords = " ".join(f"{px(x):.2f},{py(y):.2f}" for x, y in values)
        out.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.5"/>')
    for idx, (label, color, _) in enumerate(series):
        y = 55 + 20 * idx
        out.append(f'<line x1="{width-190}" y1="{y}" x2="{width-165}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        out.append(f'<text x="{width-157}" y="{y+5}" font-family="sans-serif" font-size="12">{label}</text>')
    out.append("</svg>")
    path.write_text("\n".join(out), encoding="utf-8")


def main() -> None:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    grid = [-2.0, -1.0, -0.25, 0.25, 1.0, 2.0]
    slopes = [-2.0, -0.5, 0.5, 2.0]
    rows: list[list[object]] = []
    max_error = 0.0
    for a in slopes:
        for h1 in grid:
            for h2 in grid:
                formula = all_sign_formula(h1, h2, a, 1.3, 0.7, 0.9)
                direct = all_sign_direct(h1, h2, a, 1.3, 0.7, 0.9)
                error = abs(formula - direct)
                max_error = max(max_error, error)
                rows.append([a, h1, h2, formula, direct, error])
    write_csv(
        CSV_DIR / "all_sign_grid.csv",
        ["a", "h1", "h2", "formula", "direct_quadrature", "absolute_error"],
        rows,
    )

    kappas = [10 ** (-2.0 + 4.0 * i / 100.0) for i in range(101)]
    cross_rows = []
    cross_values = []
    for kappa in kappas:
        value = gaussian_crossover(1.0, 1.0, 1.0, kappa)
        cross_rows.append([kappa, value])
        cross_values.append((kappa, value))
    write_csv(CSV_DIR / "gaussian_crossover.csv", ["kappa", "coefficient"], cross_rows)

    epsilons = [10 ** (-1.5 - 0.08 * i) for i in range(50)]
    cubic = [(eps, cubic_correction(eps)) for eps in epsilons]
    cubic_fit = log_slope([x for x, _ in cubic[-30:]], [y for _, y in cubic[-30:]])
    write_csv(CSV_DIR / "cubic_correction.csv", ["epsilon", "correction"], cubic)

    deltas = [10 ** (-3.5 + 0.06 * i) for i in range(50)]
    singular = [(d, singular_tube_probability(d)) for d in deltas]
    diffuse = [(d, diffuse_tube_probability(d, 0.3)) for d in deltas]
    singular_fit = log_slope([x for x, _ in singular[:20]], [y for _, y in singular[:20]])
    diffuse_fit = log_slope([x for x, _ in diffuse[:20]], [y for _, y in diffuse[:20]])
    write_csv(
        CSV_DIR / "tube_probabilities.csv",
        ["delta", "singular_probability", "diffuse_probability"],
        [[d, ps, pd] for (d, ps), (_, pd) in zip(singular, diffuse)],
    )

    rho1, rho2, eta = running_max_coefficients()
    singular_share = eta / (rho1 + rho2 + eta)
    uplift = eta / (rho1 + rho2)
    coupling = error_coupling_checks()
    model_a, model_b = resurrection_coefficients(1.0, 1.0)
    search_date1, search_new, search_copied = full_recall_coefficients()
    search_channels_eps_01 = full_recall_uniform_channels(0.1)
    search_exact_eps_01 = full_recall_uniform_exact(0.1)

    results = {
        "all_sign_max_absolute_error": max_error,
        "gaussian_resonant_endpoint": all_sign_formula(1.0, 1.0, 1.0),
        "gaussian_diffuse_endpoint": gaussian_diffuse_endpoint(1.0, 1.0, 1.0),
        "cubic_fitted_exponent": cubic_fit,
        "singular_tube_fitted_exponent": singular_fit,
        "diffuse_tube_fitted_exponent": diffuse_fit,
        "running_max": {
            "rho1": rho1,
            "rho2_plus_reference_alive": rho2,
            "eta": eta,
            "singular_share": singular_share,
            "uplift_over_diagonal": uplift,
        },
        "full_recall_search": {
            "uniform_cost": 0.125,
            "boundary": 0.5,
            "stopping_side_slope": 0.5,
            "date_1": search_date1,
            "new_record": search_new,
            "copied_state": search_copied,
            "total": search_date1 + search_new + search_copied,
            "diagonal": search_date1 + search_new,
            "uplift_over_diagonal": search_copied / (search_date1 + search_new),
            "finite_epsilon_0.1_channels": {
                "date_1": search_channels_eps_01[0],
                "new_record": search_channels_eps_01[1],
                "copied_state": search_channels_eps_01[2],
            },
            "exact_regret_epsilon_0.1": search_exact_eps_01,
        },
        "resurrection_unit_shifts": {"model_A": model_a, "model_B": model_b},
        "error_couplings": coupling,
        "quadrature": {
            "rule": "Gauss-Legendre",
            "default_order": 256,
            "running_max_order": 1024,
            "running_max_lower_tail": "12 time-t1 lognormal standard deviations",
        },
        "tolerances": {
            "all_sign_absolute": 2e-12,
            "running_max_absolute": 5e-12,
            "slope_absolute": 5e-4,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "git_commit": git_commit(),
            "dependencies": "Python standard library only",
        },
    }
    (OUTPUT / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    svg_lines(
        FIG_DIR / "gaussian_crossover.svg",
        "Critical Gaussian transport",
        "noise/error ratio kappa",
        "normalized regret coefficient",
        [
            ("crossover", "#1f77b4", cross_values),
            ("resonant endpoint", "#d62728", [(k, results["gaussian_resonant_endpoint"]) for k in kappas]),
            ("diffuse endpoint", "#2ca02c", [(k, results["gaussian_diffuse_endpoint"]) for k in kappas]),
        ],
        log_x=True,
    )
    svg_lines(
        FIG_DIR / "tube_probabilities.svg",
        "Joint boundary-tube probability",
        "tube width delta",
        "probability",
        [("singular", "#d62728", singular), ("diffuse", "#1f77b4", diffuse)],
        log_x=True,
        log_y=True,
    )
    svg_lines(
        FIG_DIR / "cubic_correction.svg",
        "Diffuse cross-date correction",
        "boundary scale epsilon",
        "absolute correction",
        [("exact Gaussian correction", "#9467bd", cubic)],
        log_x=True,
        log_y=True,
    )

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
