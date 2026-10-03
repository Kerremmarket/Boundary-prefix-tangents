#!/usr/bin/env python3
"""Calibrate and evaluate the fixed annual contract on all 248 market dates."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from calibration import fit_lognormal_mixture
from clean_quotes import prepare_calibration_quotes
from indexed_annuity import (
    ContractSpec,
    CreditingSpec,
    MixtureCreditedDistribution,
    log_state_grid,
    solve_stationary_bellman,
    upper_regular_boundary,
)
from market_models import LognormalMixtureParams
from prefix_regret import (
    market_two_date_actual_regret,
    market_two_date_coefficients,
)
from run_market_pilot import repair_target_slices, select_target_maturity_slices


DIRECTIONS = {
    "joint_equal": (1.0, 1.0),
    "later_larger": (0.8, 1.1),
    "earlier_larger": (1.1, 0.8),
    "earlier_only": (1.0, 0.0),
    "later_only": (0.0, 1.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--market-features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-strikes", type=int, default=12)
    parser.add_argument("--maximum-maturity-distance", type=int, default=120)
    parser.add_argument("--grid-points", type=int, default=30_000)
    parser.add_argument("--law-nodes", type=int, default=64)
    parser.add_argument("--integration-nodes", type=int, default=48)
    parser.add_argument("--epsilon", type=float, default=0.005)
    return parser.parse_args()


def _single(frame: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(frame[column], errors="coerce").dropna().unique()
    if len(values) != 1:
        raise ValueError(f"{column} is not constant within the selected slice")
    return float(values[0])


def _parameter_bound_hit(parameters: dict[str, float], tolerance: float = 1e-5) -> bool:
    bounds = {
        "low_weight": (0.05, 0.90),
        "low_forward_multiplier": (0.60, 0.999),
        "low_volatility": (0.02, 1.20),
        "high_volatility": (0.02, 1.20),
    }
    direct_hit = any(
        abs(parameters[name] - lower) <= tolerance
        or abs(parameters[name] - upper) <= tolerance
        for name, (lower, upper) in bounds.items()
    )
    implied_hit = parameters["high_forward_multiplier"] >= 3.0 - tolerance
    return direct_hit or implied_hit


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    options = pd.read_parquet(
        data_dir / "spx_option_prices_monthly_2005_2025.parquet"
    )
    forwards = pd.read_parquet(
        data_dir / "spx_forward_prices_monthly_2005_2025.parquet"
    )
    zero = pd.read_parquet(
        data_dir / "spx_zero_curve_monthly_2005_2025.parquet"
    )
    spot = pd.read_parquet(data_dir / "spx_spot_monthly_2005_2025.parquet")
    for frame in (options, forwards, zero, spot):
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    cleaned = prepare_calibration_quotes(options, forwards, zero, spot)
    (output_dir / "cleaning_attrition.json").write_text(
        json.dumps(
            {"attrition": cleaned.attrition, "diagnostics": cleaned.diagnostics},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    features = pd.read_csv(
        args.market_features.expanduser().resolve(), parse_dates=["date"]
    )
    features["date"] = features["date"].dt.normalize()
    feature_lookup = features.set_index("date").to_dict("index")
    dates = sorted(cleaned.quotes["date"].unique())
    crediting = CreditingSpec(0.80, 0.08)
    calibration_records: list[dict] = []
    repair_records: list[dict] = []
    status_records: list[dict] = []
    coefficient_records: list[dict] = []
    perturbation_records: list[dict] = []

    for date_value in dates:
        date = pd.Timestamp(date_value).normalize()
        date_text = date.strftime("%Y-%m-%d")
        date_quotes = cleaned.quotes.loc[cleaned.quotes["date"].eq(date)]
        calibration_completed = False
        try:
            slices = select_target_maturity_slices(
                date_quotes,
                target_dtes=(365,),
                min_strikes=args.min_strikes,
            )
            selected = slices[365]
            maturity_distance = float(selected["target_dte_distance"].iloc[0])
            if maturity_distance > args.maximum_maturity_distance:
                raise ValueError(
                    f"nearest eligible expiry is {maturity_distance:.0f} days from one year"
                )
            repaired, repair_diagnostics = repair_target_slices(slices)
            for record in repair_diagnostics:
                repair_records.append({"date": date_text, **record})
            fit = fit_lognormal_mixture(repaired)
            if not fit.success:
                raise RuntimeError(
                    "annual-mixture optimizer did not converge: " + fit.message
                )
            payload = asdict(fit)
            payload.pop("fitted_prices")
            parameters = payload.pop("parameters")
            bound_hit = _parameter_bound_hit(parameters)
            calibration_record = {
                "date": date_text,
                "status": "ok",
                "maturity_distance": maturity_distance,
                "parameter_bound_hit": bound_hit,
                **payload,
                **parameters,
            }
            calibration_records.append(calibration_record)
            calibration_completed = True

            spot_value = _single(repaired, "spot")
            forward_value = _single(repaired, "matched_forward_price")
            maturity = _single(repaired, "year_fraction")
            zero_rate = _single(repaired, "zero_rate")
            annual_gross_forward = float(
                np.exp(np.log(forward_value / spot_value) / maturity)
            )
            mixture = LognormalMixtureParams(
                parameters["low_weight"],
                parameters["low_forward_multiplier"],
                parameters["low_volatility"],
                parameters["high_volatility"],
            )
            distribution = MixtureCreditedDistribution(
                gross_forward=annual_gross_forward,
                mixture=mixture,
                crediting=crediting,
            )
            law = distribution.discretize(args.law_nodes)
            contract = ContractSpec(
                discount_factor=float(np.exp(-zero_rate)),
                termination_probability=0.03,
                surrender_haircut=0.08,
                death_guarantee=1.0,
            )
            liquidation_slope = 1.0 - contract.surrender_haircut
            high_state_margin = liquidation_slope - (
                contract.discount_factor
                * law.expected_factor
                * (
                    contract.termination_probability
                    + (1.0 - contract.termination_probability)
                    * liquidation_slope
                )
            )
            market_features = feature_lookup.get(date, {})
            status_record = {
                "date": date_text,
                "status": "no_upper_stopping_tail",
                "spot": spot_value,
                "one_year_forward": forward_value,
                "actual_maturity": maturity,
                "maturity_distance": maturity_distance,
                "annual_gross_forward": annual_gross_forward,
                "zero_rate": zero_rate,
                "discount_factor": contract.discount_factor,
                "floor_probability": distribution.floor_probability,
                "cap_probability": distribution.cap_probability,
                "expected_credited_factor": law.expected_factor,
                "high_state_surrender_margin": high_state_margin,
                "mixture_standardized_rmse": fit.standardized_rmse,
                "mixture_forward_normalized_rmse": fit.forward_normalized_rmse,
                "mixture_within_spread_share": fit.within_spread_share,
                "parameter_bound_hit": bound_hit,
                **market_features,
            }
            if high_state_margin <= 0:
                status_records.append(status_record)
                continue

            solution = solve_stationary_bellman(
                grid=log_state_grid(0.03, 20.0, args.grid_points),
                law=law,
                contract=contract,
                tolerance=1e-11,
            )
            boundary, slope = upper_regular_boundary(solution)
            initial_state = 0.96 * boundary
            status_record.update(
                {
                    "status": "regular_upper_boundary",
                    "boundary": boundary,
                    "stopping_side_gap_slope": slope,
                    "bellman_iterations": solution.iterations,
                    "bellman_sup_error": solution.sup_error,
                }
            )
            status_records.append(status_record)

            for direction, (relative_h1, relative_h2) in DIRECTIONS.items():
                h1 = relative_h1 * boundary
                h2 = relative_h2 * boundary
                coefficients = market_two_date_coefficients(
                    boundary=boundary,
                    stopping_side_gap_slope=slope,
                    h1=h1,
                    h2=h2,
                    initial_state=initial_state,
                    distribution=distribution,
                    contract=contract,
                    integration_nodes=args.integration_nodes,
                )
                coefficient_records.append(
                    {
                        "date": date_text,
                        "direction": direction,
                        "relative_h1": relative_h1,
                        "relative_h2": relative_h2,
                        "boundary": boundary,
                        "initial_state": initial_state,
                        "date1_diagonal": coefficients.date1_diagonal,
                        "date2_diagonal": coefficients.date2_diagonal,
                        "diagonal": coefficients.diagonal,
                        "copied_prefix": coefficients.copied_prefix,
                        "prefix": coefficients.prefix,
                        "copied_share_of_prefix": (
                            coefficients.copied_prefix / coefficients.prefix
                            if coefficients.prefix > 0
                            else 0.0
                        ),
                        "prefix_uplift_over_diagonal": (
                            coefficients.copied_prefix / coefficients.diagonal
                            if coefficients.diagonal > 0
                            else 0.0
                        ),
                    }
                )
                regret = market_two_date_actual_regret(
                    epsilon=args.epsilon,
                    boundary=boundary,
                    h1=h1,
                    h2=h2,
                    initial_state=initial_state,
                    distribution=distribution,
                    solution=solution,
                    contract=contract,
                    integration_nodes=args.integration_nodes,
                )
                scaled = regret.total / args.epsilon**2
                perturbation_records.append(
                    {
                        "date": date_text,
                        "direction": direction,
                        "epsilon": args.epsilon,
                        "actual_loss": regret.total,
                        "actual_loss_per_100k_guarantee": regret.total * 100_000,
                        "actual_scaled_loss": scaled,
                        "diagonal_coefficient": coefficients.diagonal,
                        "prefix_coefficient": coefficients.prefix,
                        "absolute_diagonal_error": abs(
                            scaled - coefficients.diagonal
                        ),
                        "absolute_prefix_error": abs(
                            scaled - coefficients.prefix
                        ),
                    }
                )
        except Exception as error:
            if not calibration_completed:
                calibration_records.append(
                    {
                        "date": date_text,
                        "status": "failed",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
            status_records.append(
                {
                    "date": date_text,
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )

    calibration_frame = pd.DataFrame(calibration_records)
    repair_frame = pd.DataFrame(repair_records)
    status_frame = pd.DataFrame(status_records)
    coefficient_frame = pd.DataFrame(coefficient_records)
    perturbation_frame = pd.DataFrame(perturbation_records)
    calibration_frame.to_csv(output_dir / "annual_calibrations.csv", index=False)
    repair_frame.to_csv(output_dir / "repair_diagnostics.csv", index=False)
    status_frame.to_csv(output_dir / "date_status.csv", index=False)
    coefficient_frame.to_csv(output_dir / "coefficient_panel.csv", index=False)
    perturbation_frame.to_csv(
        output_dir / "finite_perturbation_panel.csv", index=False
    )

    successful = calibration_frame.loc[calibration_frame["status"] == "ok"]
    regular = status_frame.loc[
        status_frame["status"] == "regular_upper_boundary"
    ]
    joint = coefficient_frame.loc[
        coefficient_frame["direction"].isin(
            ["joint_equal", "later_larger", "earlier_larger"]
        )
    ]
    finite_joint = perturbation_frame.loc[
        perturbation_frame["direction"].isin(
            ["joint_equal", "later_larger", "earlier_larger"]
        )
    ]
    prefix_wins = int(
        (
            finite_joint["absolute_prefix_error"]
            < finite_joint["absolute_diagonal_error"]
        ).sum()
    )
    outside_repair = repair_frame.loc[~repair_frame["feasible_within_spread"]]
    summary = f"""# Full 248-date annual OptionMetrics panel

- Market dates in cleaned data: {len(dates)}.
- Successful annual-mixture calibrations: {len(successful)}.
- Optimizer convergence successes: {int(successful['success'].sum())} of {len(successful)}.
- Calibration/date failures: {int((calibration_frame['status'] == 'failed').sum())}.
- Parameter-bound hits: {int(successful['parameter_bound_hit'].sum()) if len(successful) else 0}.
- Median forward-normalized price RMSE: {successful['forward_normalized_rmse'].median() if len(successful) else float('nan'):.6f}.
- Maximum forward-normalized price RMSE: {successful['forward_normalized_rmse'].max() if len(successful) else float('nan'):.6f}.
- Repair outside quoted spreads: {int(outside_repair['rows_outside_spread'].sum())} rows across {outside_repair['date'].nunique()} dates.
- Regular upper surrender boundaries: {len(regular)}.
- No upper stopping tail: {int((status_frame['status'] == 'no_upper_stopping_tail').sum())}.
- Positive copied-prefix joint coefficients: {int((joint['copied_prefix'] > 0).sum())} of {len(joint)}.
- Prefix closer than diagonal at epsilon={args.epsilon}: {prefix_wins} of {len(finite_joint)} joint comparisons.
- Median copied share of full joint prefix: {joint['copied_share_of_prefix'].median() if len(joint) else float('nan'):.6f}.
- Median prefix uplift over diagonal: {joint['prefix_uplift_over_diagonal'].median() if len(joint) else float('nan'):.6f}.
- Median actual loss per $100,000 guarantee at epsilon={args.epsilon}: {finite_joint['actual_loss_per_100k_guarantee'].median() if len(finite_joint) else float('nan'):.6f} dollars.

Every date is retained with an explicit calibration, no-boundary, or failure
status. The annual mixture is a market-state counterfactual fitted to the nearest
eligible one-year slice; it is not a forecast of a time-homogeneous physical
return law.
"""
    (output_dir / "summary.md").write_text(summary, encoding="utf-8")
    print(
        f"dates={len(dates)} calibrations={len(successful)} "
        f"boundaries={len(regular)} failures={(calibration_frame.status == 'failed').sum()}"
    )
    print(output_dir)
    return 1 if (calibration_frame["status"] == "failed").any() else 0


if __name__ == "__main__":
    raise SystemExit(main())
