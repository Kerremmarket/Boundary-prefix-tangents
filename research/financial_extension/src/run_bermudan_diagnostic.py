"""Run the frozen smooth Bermudan financial diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from bermudan_diagnostic import (
    MarketInput,
    diagonal_coefficient,
    evaluate_native_policy_forward,
    evaluate_policy_on_paths,
    evaluate_threshold_policy_forward,
    log_log_slope,
    lsmc_exercise_mask,
    price_threshold_policy_fft,
    reference_occupancy_densities,
    simulate_black_scholes_paths,
    solve_bermudan_put,
    solve_bermudan_put_fft,
    train_lsmc_policy,
)


SELECTED_DATES = [
    "2016-10-31",
    "2008-11-28",
    "2020-02-28",
    "2009-04-30",
    "2006-03-31",
]
TRAINING_SEEDS = list(range(2026082701, 2026082711))
EVALUATION_SEEDS = list(range(3026082701, 3026082706))
ETAS = [1.0, 0.5, 0.25, 0.125, 0.0625]


def load_frozen_market_inputs(
    panel_path: Path, selection_path: Path
) -> tuple[list[MarketInput], pd.DataFrame]:
    panel = pd.read_csv(panel_path)
    selection = pd.read_csv(selection_path)
    first_five = selection.sort_values("selection_order").head(5)["date"].tolist()
    if first_five != SELECTED_DATES:
        raise ValueError(
            f"predeclared selection changed: expected {SELECTED_DATES}, got {first_five}"
        )
    rows = panel[panel["date"].isin(SELECTED_DATES)].copy()
    if len(rows) != len(SELECTED_DATES) or rows["date"].nunique() != len(SELECTED_DATES):
        raise ValueError("frozen panel does not contain exactly one row per selected date")
    rows["selection_order"] = rows["date"].map(
        {date: i + 1 for i, date in enumerate(SELECTED_DATES)}
    )
    rows = rows.sort_values("selection_order").reset_index(drop=True)
    required = [
        "zero_rate",
        "annual_gross_forward",
        "atm_iv_1y",
        "spot",
        "one_year_forward",
    ]
    if rows[required].isna().any().any():
        raise ValueError("selected market input contains missing values")
    rows["black_scholes_dividend_yield"] = rows["zero_rate"] - np.log(
        rows["annual_gross_forward"]
    )
    markets = [
        MarketInput(
            date=row.date,
            rate=float(row.zero_rate),
            dividend_yield=float(row.black_scholes_dividend_yield),
            volatility=float(row.atm_iv_1y),
            annual_gross_forward=float(row.annual_gross_forward),
        )
        for row in rows.itertuples()
    ]
    for market in markets:
        market.validate()
    return markets, rows


def _boundary_rows(solution, *, method: str, seed: int | None = None) -> list[dict]:
    rows = []
    for i, diagnostic in enumerate(solution.boundary_diagnostics, start=1):
        rows.append(
            {
                "date": solution.market.date,
                "method": method,
                "seed": seed,
                "exercise_index": i,
                "exercise_time": i / 8.0,
                "crossing_count": diagnostic.crossing_count,
                "orientation_ok": diagnostic.orientation_ok,
                "log_boundary": diagnostic.boundary,
                "spot_boundary": (
                    np.exp(diagnostic.boundary)
                    if diagnostic.boundary is not None
                    else np.nan
                ),
            }
        )
    return rows


def _gate_row(
    *,
    date: str,
    method: str,
    seed: int | None,
    regular: bool,
    eta_rows: list[dict],
    mc_pass: bool,
) -> dict:
    if regular and eta_rows:
        eta_frame = pd.DataFrame(eta_rows).sort_values("eta", ascending=False)
        slope = log_log_slope(eta_frame["eta"], eta_frame["actual_loss"])
        row_18 = eta_frame[np.isclose(eta_frame["eta"], 0.125)].iloc[0]
        row_116 = eta_frame[np.isclose(eta_frame["eta"], 0.0625)].iloc[0]
        ratio_18 = float(row_18["actual_to_diagonal"])
        ratio_116 = float(row_116["actual_to_diagonal"])
        interaction_share = abs(float(row_116["interaction_share"]))
    else:
        slope = ratio_18 = ratio_116 = interaction_share = np.nan
    passed = bool(
        regular
        and 1.90 <= slope <= 2.10
        and abs(ratio_18 - 1.0) <= 0.10
        and abs(ratio_116 - 1.0) <= 0.10
        and interaction_share <= 0.10
        and mc_pass
    )
    return {
        "date": date,
        "method": method,
        "seed": seed,
        "boundary_regular": regular,
        "log_log_slope_eta_le_quarter": slope,
        "ratio_eta_one_eighth": ratio_18,
        "ratio_eta_one_sixteenth": ratio_116,
        "interaction_share_eta_one_sixteenth": interaction_share,
        "heldout_mc_consistent": mc_pass,
        "gate_pass": passed,
    }


def _evaluate_boundary_vector(
    *,
    reference,
    approximate_boundaries: np.ndarray,
    method: str,
    seed: int | None,
    density_grid: np.ndarray,
    reference_densities: list[np.ndarray],
    fixed_reference_value: float,
) -> tuple[list[dict], float, np.ndarray]:
    coefficient, contributions = diagonal_coefficient(
        reference,
        approximate_boundaries,
        density_grid=density_grid,
        reference_densities=reference_densities,
    )
    displacement = approximate_boundaries - reference.boundaries
    rows: list[dict] = []
    for eta in ETAS:
        boundaries = reference.boundaries + eta * displacement
        evaluation = evaluate_threshold_policy_forward(
            reference,
            boundaries,
            density_nodes=len(density_grid),
            reference_density_grid=density_grid,
            reference_densities=reference_densities,
        )
        diagonal_prediction = eta**2 * coefficient
        backward_value = price_threshold_policy_fft(
            reference, boundaries, evaluation_nodes=len(density_grid)
        )
        backward_loss = fixed_reference_value - backward_value
        rows.append(
            {
                "date": reference.market.date,
                "method": method,
                "seed": seed,
                "eta": eta,
                "actual_loss": evaluation.loss,
                "diagonal_prediction": diagonal_prediction,
                "actual_to_diagonal": (
                    evaluation.loss / diagonal_prediction
                    if diagonal_prediction > 0
                    else np.nan
                ),
                "single_date_sum": evaluation.single_date_sum,
                "interaction": evaluation.interaction,
                "interaction_share": (
                    evaluation.interaction / evaluation.loss
                    if evaluation.loss > 0
                    else 0.0
                ),
                "backward_loss_audit": backward_loss,
                "forward_minus_backward": evaluation.loss - backward_loss,
                "max_abs_log_boundary_shift": float(
                    np.max(np.abs(eta * displacement))
                ),
            }
        )
    return rows, coefficient, contributions


def run(
    *,
    panel_path: Path,
    selection_path: Path,
    output_dir: Path,
    training_paths: int = 60_000,
    evaluation_paths: int = 200_000,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    markets, market_rows = load_frozen_market_inputs(panel_path, selection_path)
    market_columns = [
        "selection_order",
        "date",
        "spot",
        "one_year_forward",
        "actual_maturity",
        "annual_gross_forward",
        "zero_rate",
        "black_scholes_dividend_yield",
        "atm_iv_1y",
        "skew_25d_1y",
        "iv_term_slope",
    ]
    market_rows[market_columns].to_csv(output_dir / "market_inputs.csv", index=False)

    boundary_rows: list[dict] = []
    refinement_rows: list[dict] = []
    eta_rows_all: list[dict] = []
    coefficient_rows: list[dict] = []
    lsmc_audit_rows: list[dict] = []
    native_eval_rows: list[dict] = []
    gate_rows: list[dict] = []
    occupancy_rows: list[dict] = []

    for market_index, market in enumerate(markets):
        print(f"[{market_index + 1}/{len(markets)}] reference {market.date}", flush=True)
        references = {
            nodes: solve_bermudan_put_fft(market, grid_nodes=nodes)
            for nodes in (32_769, 65_537, 131_073)
        }
        reference = references[65_537]
        gh_audit = solve_bermudan_put(
            market, grid_nodes=65_537, quadrature_nodes=256
        )
        boundary_rows.extend(_boundary_rows(reference, method="reference_fft_65537"))
        for nodes, solution in references.items():
            refinement_rows.append(
                {
                    "date": market.date,
                    "method": "fft_convolution",
                    "grid_nodes": nodes,
                    "quadrature_nodes": 0,
                    "value": solution.value_at_zero,
                    "max_boundary_difference_vs_65537": float(
                        np.max(np.abs(solution.boundaries - reference.boundaries))
                    ),
                    "value_difference_vs_65537": (
                        solution.value_at_zero - reference.value_at_zero
                    ),
                    "all_boundaries_regular": all(
                        item.crossing_count == 1 and item.orientation_ok
                        for item in solution.boundary_diagnostics
                    ),
                }
            )
        refinement_rows.append(
            {
                "date": market.date,
                "method": "gauss_hermite_audit",
                "grid_nodes": 65_537,
                "quadrature_nodes": 256,
                "value": gh_audit.value_at_zero,
                "max_boundary_difference_vs_65537": float(
                    np.max(np.abs(gh_audit.boundaries - reference.boundaries))
                ),
                "value_difference_vs_65537": (
                    gh_audit.value_at_zero - reference.value_at_zero
                ),
                "all_boundaries_regular": all(
                    item.crossing_count == 1 and item.orientation_ok
                    for item in gh_audit.boundary_diagnostics
                ),
            }
        )

        density_grid, reference_densities, occupancy_masses = (
            reference_occupancy_densities(reference, density_nodes=32_769)
        )
        for i, (boundary, slope, density, mass) in enumerate(
            zip(
                reference.boundaries,
                reference.gap_slopes,
                reference_densities,
                occupancy_masses,
                strict=True,
            ),
            start=1,
        ):
            occupancy_rows.append(
                {
                    "date": market.date,
                    "exercise_index": i,
                    "exercise_time": i / 8.0,
                    "log_boundary": boundary,
                    "spot_boundary": np.exp(boundary),
                    "gap_slope_log_coordinate": slope,
                    "surviving_mass_before_decision": mass,
                    "occupancy_density_at_boundary": np.interp(
                        boundary, density_grid, density
                    ),
                }
            )
        fixed_reference_value = price_threshold_policy_fft(
            reference, reference.boundaries, evaluation_nodes=32_769
        )
        annual_mean = market.rate - market.dividend_yield - 0.5 * market.volatility**2
        one_year_tail = norm.cdf((-4.0 - annual_mean) / market.volatility) + norm.sf(
            (2.0 - annual_mean) / market.volatility
        )
        refinement_rows[-1]["one_year_grid_tail_probability"] = one_year_tail

        print(f"[{market_index + 1}/{len(markets)}] coarse policies", flush=True)
        coarse_solutions = {
            nodes: solve_bermudan_put_fft(
                market,
                grid_nodes=nodes,
                interpolate_boundary_root=False,
            )
            for nodes in (129, 257, 513)
        }
        for nodes, solution in coarse_solutions.items():
            boundary_rows.extend(_boundary_rows(solution, method=f"coarse_fft_{nodes}"))
        coarse = coarse_solutions[257]
        coarse_regular = bool(np.all(np.isfinite(coarse.boundaries)))
        coarse_eta_rows: list[dict] = []
        if coarse_regular:
            coarse_eta_rows, coefficient, contributions = _evaluate_boundary_vector(
                reference=reference,
                approximate_boundaries=coarse.boundaries,
                method="coarse_fft_257",
                seed=None,
                density_grid=density_grid,
                reference_densities=reference_densities,
                fixed_reference_value=fixed_reference_value,
            )
            eta_rows_all.extend(coarse_eta_rows)
            for i, (shift, contribution) in enumerate(
                zip(
                    coarse.boundaries - reference.boundaries,
                    contributions,
                    strict=True,
                ),
                start=1,
            ):
                coefficient_rows.append(
                    {
                        "date": market.date,
                        "method": "coarse_fft_257",
                        "seed": None,
                        "exercise_index": i,
                        "log_boundary_displacement": shift,
                        "diagonal_contribution": contribution,
                        "diagonal_total": coefficient,
                    }
                )
        gate_rows.append(
            _gate_row(
                date=market.date,
                method="coarse_fft_257",
                seed=None,
                regular=coarse_regular,
                eta_rows=coarse_eta_rows,
                mc_pass=True,
            )
        )

        print(f"[{market_index + 1}/{len(markets)}] LSMC seeds", flush=True)
        evaluation_spots = simulate_black_scholes_paths(
            market,
            paths=evaluation_paths,
            seed=EVALUATION_SEEDS[market_index],
            exercise_dates=8,
        )
        reference_path_payoff = evaluate_policy_on_paths(
            market,
            evaluation_spots,
            lambda i, x: x <= reference.boundaries[i],
        )
        for seed in TRAINING_SEEDS:
            policy = train_lsmc_policy(
                market, seed=seed, paths=training_paths, exercise_dates=8
            )
            policy_like_solution = type(
                "BoundaryContainer",
                (),
                {
                    "market": market,
                    "boundary_diagnostics": policy.boundary_diagnostics,
                },
            )()
            boundary_rows.extend(
                _boundary_rows(policy_like_solution, method="lsmc_linear", seed=seed)
            )
            for i, (coefficient, count, rank, condition) in enumerate(
                zip(
                    policy.coefficients,
                    policy.sample_counts,
                    policy.ranks,
                    policy.condition_numbers,
                    strict=True,
                ),
                start=1,
            ):
                lsmc_audit_rows.append(
                    {
                        "date": market.date,
                        "seed": seed,
                        "exercise_index": i,
                        "sample_count": count,
                        "rank": rank,
                        "condition_number": condition,
                        "coefficient_intercept": coefficient[0],
                        "coefficient_linear": coefficient[1],
                        "training_value": policy.training_value,
                    }
                )
            native = evaluate_native_policy_forward(
                reference,
                lambda i, x, p=policy: lsmc_exercise_mask(x, p.coefficients[i]),
                density_nodes=32_769,
            )
            approximate_path_payoff = evaluate_policy_on_paths(
                market,
                evaluation_spots,
                lambda i, x, p=policy: lsmc_exercise_mask(x, p.coefficients[i]),
            )
            differences = reference_path_payoff - approximate_path_payoff
            mc_loss = float(np.mean(differences))
            mc_se = float(np.std(differences, ddof=1) / np.sqrt(evaluation_paths))
            mc_pass = bool(
                native.loss >= mc_loss - 1.96 * mc_se - 2e-5
                and native.loss <= mc_loss + 1.96 * mc_se + 2e-5
            )
            regular = bool(np.all(np.isfinite(policy.boundaries)))
            native_eval_rows.append(
                {
                    "date": market.date,
                    "method": "lsmc_linear",
                    "seed": seed,
                    "boundary_regular": regular,
                    "training_value": policy.training_value,
                    "deterministic_native_loss": native.loss,
                    "deterministic_native_value": reference.value_at_zero - native.loss,
                    "heldout_mc_loss": mc_loss,
                    "heldout_mc_standard_error": mc_se,
                    "heldout_reference_value": float(np.mean(reference_path_payoff)),
                    "heldout_approximate_value": float(
                        np.mean(approximate_path_payoff)
                    ),
                    "heldout_consistent": mc_pass,
                }
            )
            policy_eta_rows: list[dict] = []
            if regular:
                policy_eta_rows, coefficient, contributions = _evaluate_boundary_vector(
                    reference=reference,
                    approximate_boundaries=policy.boundaries,
                    method="lsmc_linear",
                    seed=seed,
                    density_grid=density_grid,
                    reference_densities=reference_densities,
                    fixed_reference_value=fixed_reference_value,
                )
                eta_rows_all.extend(policy_eta_rows)
                for i, (shift, contribution) in enumerate(
                    zip(
                        policy.boundaries - reference.boundaries,
                        contributions,
                        strict=True,
                    ),
                    start=1,
                ):
                    coefficient_rows.append(
                        {
                            "date": market.date,
                            "method": "lsmc_linear",
                            "seed": seed,
                            "exercise_index": i,
                            "log_boundary_displacement": shift,
                            "diagonal_contribution": contribution,
                            "diagonal_total": coefficient,
                        }
                    )
                native_eval_rows[-1]["threshold_eta_one_loss"] = policy_eta_rows[0][
                    "actual_loss"
                ]
                native_eval_rows[-1]["native_minus_threshold_loss"] = (
                    native.loss - policy_eta_rows[0]["actual_loss"]
                )
            gate_rows.append(
                _gate_row(
                    date=market.date,
                    method="lsmc_linear",
                    seed=seed,
                    regular=regular,
                    eta_rows=policy_eta_rows,
                    mc_pass=mc_pass,
                )
            )

    outputs = {
        "boundaries.csv": boundary_rows,
        "reference_refinement.csv": refinement_rows,
        "occupancy_and_slopes.csv": occupancy_rows,
        "diagonal_coefficients.csv": coefficient_rows,
        "eta_path_results.csv": eta_rows_all,
        "lsmc_regression_audit.csv": lsmc_audit_rows,
        "native_policy_evaluation.csv": native_eval_rows,
        "gate_results.csv": gate_rows,
    }
    for name, rows in outputs.items():
        pd.DataFrame(rows).to_csv(output_dir / name, index=False)

    gates = pd.DataFrame(gate_rows)
    refinements = pd.DataFrame(refinement_rows)
    fft_refinements = refinements[refinements["method"] == "fft_convolution"]
    summary = {
        "selected_dates": SELECTED_DATES,
        "training_seeds": TRAINING_SEEDS,
        "evaluation_seeds": EVALUATION_SEEDS,
        "training_paths_per_seed": training_paths,
        "evaluation_paths_per_date": evaluation_paths,
        "approximate_policies": int(len(gates)),
        "regular_policies": int(gates["boundary_regular"].sum()),
        "gate_passes": int(gates["gate_pass"].sum()),
        "all_policy_gates_pass": bool(gates["gate_pass"].all()),
        "max_fft_boundary_refinement_error": float(
            fft_refinements["max_boundary_difference_vs_65537"].max()
        ),
        "max_fft_value_refinement_error": float(
            fft_refinements["value_difference_vs_65537"].abs().max()
        ),
        "paper1_modified": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--panel-path",
        type=Path,
        default=Path(
            "research/financial_extension/evidence/full_market_panel/date_status.csv"
        ),
    )
    parser.add_argument(
        "--selection-path",
        type=Path,
        default=Path(
            "research/financial_extension/evidence/pilot_selection/selected_pilot_dates.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("research/financial_extension/diagnostic/evidence/bermudan"),
    )
    parser.add_argument("--training-paths", type=int, default=60_000)
    parser.add_argument("--evaluation-paths", type=int, default=200_000)
    args = parser.parse_args()
    run(
        panel_path=args.panel_path,
        selection_path=args.selection_path,
        output_dir=args.output_dir,
        training_paths=args.training_paths,
        evaluation_paths=args.evaluation_paths,
    )


if __name__ == "__main__":
    main()
