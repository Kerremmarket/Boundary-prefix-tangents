#!/usr/bin/env python3
"""Stress the annual-mixture pilot across quote and weighting choices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from calibration import fit_lognormal_mixture
from indexed_annuity import (
    ContractSpec,
    CreditingSpec,
    MixtureCreditedDistribution,
    log_state_grid,
    solve_stationary_bellman,
    upper_regular_boundary,
)
from market_models import LognormalMixtureParams
from prefix_regret import market_two_date_coefficients


SCENARIOS = (
    "repaired_mid",
    "raw_mid",
    "bid_envelope",
    "offer_envelope",
    "wider_price_floor",
    "strict_liquidity",
    "central_moneyness",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market-pilot-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _single(frame: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(frame[column], errors="coerce").dropna().unique()
    if len(values) != 1:
        raise ValueError(f"{column} is not constant within the target slice")
    return float(values[0])


def _scenario_quotes(
    quotes: pd.DataFrame, scenario: str
) -> tuple[pd.DataFrame, str | None, float]:
    if scenario == "repaired_mid":
        return quotes, "repaired_call_price", 0.05
    if scenario == "raw_mid":
        return quotes, "call_equivalent_mid", 0.05
    if scenario == "bid_envelope":
        return quotes, "call_equivalent_bid", 0.05
    if scenario == "offer_envelope":
        return quotes, "call_equivalent_offer", 0.05
    if scenario == "wider_price_floor":
        return quotes, "repaired_call_price", 0.25
    if scenario == "strict_liquidity":
        return (
            quotes.loc[quotes["strict_liquidity_sensitivity"]].copy(),
            "repaired_call_price",
            0.05,
        )
    if scenario == "central_moneyness":
        return (
            quotes.loc[quotes["log_forward_moneyness"].abs() <= 0.25].copy(),
            "repaired_call_price",
            0.05,
        )
    raise ValueError(f"unknown scenario {scenario}")


def main() -> int:
    args = parse_args()
    pilot_dir = args.market_pilot_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    quotes = pd.read_parquet(
        pilot_dir / "pilot_repaired_quotes_and_fits.parquet"
    )
    one_year = quotes.loc[quotes["target_dte"].eq(365)].copy()
    dates = sorted(pd.to_datetime(one_year["date"]).dt.normalize().unique())
    crediting = CreditingSpec(participation=0.80, cap=0.08)
    records: list[dict] = []

    for date_value in dates:
        date = pd.Timestamp(date_value).normalize()
        date_quotes = one_year.loc[
            pd.to_datetime(one_year["date"]).dt.normalize().eq(date)
        ].copy()
        spot = _single(date_quotes, "spot")
        forward = _single(date_quotes, "matched_forward_price")
        maturity = _single(date_quotes, "year_fraction")
        zero_rate = _single(date_quotes, "zero_rate")
        annual_gross_forward = float(np.exp(np.log(forward / spot) / maturity))
        contract = ContractSpec(
            discount_factor=float(np.exp(-zero_rate)),
            termination_probability=0.03,
            surrender_haircut=0.08,
            death_guarantee=1.0,
        )
        for scenario_index, scenario in enumerate(SCENARIOS):
            scenario_quotes, target_column, price_floor = _scenario_quotes(
                date_quotes, scenario
            )
            if len(scenario_quotes) < 12:
                records.append(
                    {
                        "date": date.strftime("%Y-%m-%d"),
                        "scenario": scenario,
                        "status": "insufficient_quotes",
                        "quotes": len(scenario_quotes),
                    }
                )
                continue
            result = fit_lognormal_mixture(
                scenario_quotes,
                price_floor=price_floor,
                seed=74291 + scenario_index,
                target_column=target_column,
            )
            params = LognormalMixtureParams(
                low_weight=result.parameters["low_weight"],
                low_forward_multiplier=result.parameters[
                    "low_forward_multiplier"
                ],
                low_volatility=result.parameters["low_volatility"],
                high_volatility=result.parameters["high_volatility"],
            )
            distribution = MixtureCreditedDistribution(
                gross_forward=annual_gross_forward,
                mixture=params,
                crediting=crediting,
            )
            law = distribution.discretize(64)
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
            record = {
                "date": date.strftime("%Y-%m-%d"),
                "scenario": scenario,
                "status": "no_upper_stopping_tail",
                "quotes": len(scenario_quotes),
                "optimizer_success": result.success,
                "standardized_rmse": result.standardized_rmse,
                "forward_normalized_rmse": result.forward_normalized_rmse,
                "within_spread_share": result.within_spread_share,
                **result.parameters,
                "floor_probability": distribution.floor_probability,
                "cap_probability": distribution.cap_probability,
                "expected_credited_factor": law.expected_factor,
                "high_state_surrender_margin": high_state_margin,
            }
            if high_state_margin > 0:
                solution = solve_stationary_bellman(
                    grid=log_state_grid(0.03, 20.0, 30_000),
                    law=law,
                    contract=contract,
                    tolerance=1e-11,
                )
                boundary, slope = upper_regular_boundary(solution)
                initial_state = 0.96 * boundary
                coefficients = market_two_date_coefficients(
                    boundary=boundary,
                    stopping_side_gap_slope=slope,
                    h1=boundary,
                    h2=boundary,
                    initial_state=initial_state,
                    distribution=distribution,
                    contract=contract,
                    integration_nodes=64,
                )
                record.update(
                    {
                        "status": "regular_upper_boundary",
                        "boundary": boundary,
                        "stopping_side_gap_slope": slope,
                        "diagonal_coefficient": coefficients.diagonal,
                        "copied_prefix": coefficients.copied_prefix,
                        "prefix_coefficient": coefficients.prefix,
                        "copied_share_of_prefix": (
                            coefficients.copied_prefix / coefficients.prefix
                        ),
                    }
                )
            records.append(record)

    frame = pd.DataFrame(records)
    frame.to_csv(output_dir / "calibration_sensitivity.csv", index=False)
    central_dates = frame.loc[
        (frame["scenario"] == "repaired_mid")
        & (frame["status"] == "regular_upper_boundary"),
        "date",
    ]
    boundary_sensitivity = frame.loc[frame["date"].isin(central_dates)].copy()
    summary = {
        "dates": int(frame["date"].nunique()),
        "scenarios": list(SCENARIOS),
        "fits": int(len(frame)),
        "optimizer_failures": int((frame["optimizer_success"] == False).sum()),  # noqa: E712
        "central_boundary_dates": central_dates.tolist(),
        "central_boundary_date_scenario_count": int(len(boundary_sensitivity)),
        "central_boundary_status_preserved": int(
            (boundary_sensitivity["status"] == "regular_upper_boundary").sum()
        ),
        "central_boundary_min": float(boundary_sensitivity["boundary"].min()),
        "central_boundary_max": float(boundary_sensitivity["boundary"].max()),
        "central_floor_probability_min": float(
            boundary_sensitivity["floor_probability"].min()
        ),
        "central_floor_probability_max": float(
            boundary_sensitivity["floor_probability"].max()
        ),
        "central_copied_share_min": float(
            boundary_sensitivity["copied_share_of_prefix"].min()
        ),
        "central_copied_share_max": float(
            boundary_sensitivity["copied_share_of_prefix"].max()
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(
        f"""# Annual-mixture calibration sensitivity

- Dates: {summary['dates']}; deterministic scenarios: {len(SCENARIOS)}.
- Optimizer failures: {summary['optimizer_failures']}.
- Central boundary dates: {', '.join(summary['central_boundary_dates'])}.
- Boundary status preserved across {summary['central_boundary_status_preserved']} of {summary['central_boundary_date_scenario_count']} scenarios on those dates.
- Boundary range on those scenarios: {summary['central_boundary_min']:.6f} to {summary['central_boundary_max']:.6f}.
- Floor-probability range: {summary['central_floor_probability_min']:.6f} to {summary['central_floor_probability_max']:.6f}.
- Copied share of the joint-equal prefix coefficient: {summary['central_copied_share_min']:.6f} to {summary['central_copied_share_max']:.6f}.

The scenarios use repaired mids, raw mids, all-bid and all-offer quote envelopes,
a wider price-error floor, a strict-liquidity subset, and a central-moneyness
subset. They diagnose target/weight sensitivity; they are not confidence
intervals and do not turn quote envelopes into coherent probability statements.
""",
        encoding="utf-8",
    )
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
