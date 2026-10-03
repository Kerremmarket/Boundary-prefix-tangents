"""Deterministic bid--ask-scaled calibration for the pilot market models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from market_models import (
    KouParams,
    LognormalMixtureParams,
    black_option_price,
    kou_call_price,
    lognormal_mixture_option_price,
)


@dataclass(frozen=True)
class CalibrationResult:
    model: str
    parameters: dict[str, float]
    success: bool
    message: str
    evaluations: int
    standardized_rmse: float
    price_rmse: float
    forward_normalized_rmse: float
    within_spread_share: float
    fitted_prices: np.ndarray


def _calibration_arrays(
    quotes: pd.DataFrame,
    price_floor: float,
    target_column: str | None = None,
):
    required = {
        "strike",
        "matched_forward_price",
        "discount_factor",
        "year_fraction",
        "call_equivalent_bid",
        "call_equivalent_offer",
        "call_equivalent_mid",
    }
    missing = sorted(required - set(quotes.columns))
    if missing:
        raise ValueError(f"calibration quotes are missing columns: {missing}")
    if len(quotes) < 5:
        raise ValueError("at least five quotes are required for calibration")
    if target_column is None:
        target_column = (
            "repaired_call_price"
            if "repaired_call_price" in quotes.columns
            else "call_equivalent_mid"
        )
    if target_column not in quotes:
        raise ValueError(f"target column {target_column!r} is unavailable")
    target = quotes[target_column].to_numpy(dtype=float)
    bid = quotes["call_equivalent_bid"].to_numpy(dtype=float)
    offer = quotes["call_equivalent_offer"].to_numpy(dtype=float)
    half_spread = (offer - bid) / 2.0
    scale = np.maximum(half_spread, price_floor)
    arrays = {
        "forward": quotes["matched_forward_price"].to_numpy(dtype=float),
        "strike": quotes["strike"].to_numpy(dtype=float),
        "discount": quotes["discount_factor"].to_numpy(dtype=float),
        "maturity": quotes["year_fraction"].to_numpy(dtype=float),
        "target": target,
        "bid": bid,
        "offer": offer,
        "scale": scale,
    }
    if not all(np.all(np.isfinite(value)) for value in arrays.values()):
        raise ValueError("calibration arrays contain nonfinite values")
    if (
        np.any(arrays["forward"] <= 0)
        or np.any(arrays["strike"] <= 0)
        or np.any(arrays["discount"] <= 0)
        or np.any(arrays["maturity"] <= 0)
        or np.any(arrays["scale"] <= 0)
    ):
        raise ValueError("calibration arrays contain invalid positive quantities")
    return arrays


def _result(
    *,
    model: str,
    parameters: dict[str, float],
    optimizer,
    fitted: np.ndarray,
    arrays,
) -> CalibrationResult:
    error = fitted - arrays["target"]
    forward_scale = float(np.median(arrays["forward"]))
    within = (fitted >= arrays["bid"] - 1e-10) & (
        fitted <= arrays["offer"] + 1e-10
    )
    return CalibrationResult(
        model=model,
        parameters=parameters,
        success=bool(optimizer.success),
        message=str(optimizer.message),
        evaluations=int(optimizer.nfev),
        standardized_rmse=float(np.sqrt(np.mean((error / arrays["scale"]) ** 2))),
        price_rmse=float(np.sqrt(np.mean(error**2))),
        forward_normalized_rmse=float(np.sqrt(np.mean(error**2)) / forward_scale),
        within_spread_share=float(np.mean(within)),
        fitted_prices=np.asarray(fitted),
    )


def fit_black(
    quotes: pd.DataFrame,
    *,
    price_floor: float = 0.05,
    target_column: str | None = None,
) -> CalibrationResult:
    arrays = _calibration_arrays(quotes, price_floor, target_column)

    def prices(volatility: float) -> np.ndarray:
        return black_option_price(
            forward=arrays["forward"],
            strike=arrays["strike"],
            discount=arrays["discount"],
            maturity=arrays["maturity"],
            volatility=volatility,
            cp_flag=np.full(len(arrays["strike"]), "C"),
        )

    optimizer = least_squares(
        lambda value: (prices(float(value[0])) - arrays["target"]) / arrays["scale"],
        x0=np.array([0.20]),
        bounds=(np.array([0.01]), np.array([1.50])),
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
        max_nfev=500,
    )
    volatility = float(optimizer.x[0])
    fitted = prices(volatility)
    return _result(
        model="black_scholes",
        parameters={"volatility": volatility},
        optimizer=optimizer,
        fitted=fitted,
        arrays=arrays,
    )


def _best_multistart(
    residual: Callable[[np.ndarray], np.ndarray],
    starts: list[np.ndarray],
    lower: np.ndarray,
    upper: np.ndarray,
    max_nfev: int,
):
    best = None
    best_cost = np.inf
    for start in starts:
        clipped = np.minimum(np.maximum(start, lower + 1e-9), upper - 1e-9)
        candidate = least_squares(
            residual,
            x0=clipped,
            bounds=(lower, upper),
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
            max_nfev=max_nfev,
        )
        cost = float(np.dot(candidate.fun, candidate.fun))
        if cost < best_cost:
            best, best_cost = candidate, cost
    assert best is not None
    return best


def fit_lognormal_mixture(
    quotes: pd.DataFrame,
    *,
    price_floor: float = 0.05,
    seed: int = 74291,
    target_column: str | None = None,
    max_nfev: int = 10_000,
) -> CalibrationResult:
    if max_nfev < 100:
        raise ValueError("max_nfev must be at least 100")
    arrays = _calibration_arrays(quotes, price_floor, target_column)

    def make_params(value: np.ndarray) -> LognormalMixtureParams:
        return LognormalMixtureParams(
            low_weight=float(value[0]),
            low_forward_multiplier=float(value[1]),
            low_volatility=float(value[2]),
            high_volatility=float(value[3]),
        )

    def prices(value: np.ndarray) -> np.ndarray:
        params = make_params(value)
        return lognormal_mixture_option_price(
            forward=arrays["forward"],
            strike=arrays["strike"],
            discount=arrays["discount"],
            maturity=arrays["maturity"],
            cp_flag=np.full(len(arrays["strike"]), "C"),
            params=params,
        )

    def residual(value: np.ndarray) -> np.ndarray:
        params = make_params(value)
        price_residual = (prices(value) - arrays["target"]) / arrays["scale"]
        # Keep the data residual in the objective even outside the preferred
        # implied-high-forward range.  Replacing it by a vanishing constant at
        # the boundary would create a spurious near-zero-cost solution.
        penalty = 1_000.0 * max(params.high_forward_multiplier - 3.0, 0.0)
        return np.concatenate((price_residual, np.array([penalty])))

    lower = np.array([0.05, 0.60, 0.02, 0.02])
    upper = np.array([0.90, 0.999, 1.20, 1.20])
    rng = np.random.default_rng(seed)
    starts = [
        np.array([0.70, 0.90, 0.28, 0.15]),
        np.array([0.50, 0.85, 0.35, 0.12]),
        np.array([0.80, 0.95, 0.22, 0.18]),
    ]
    starts.extend(rng.uniform(lower, upper) for _ in range(5))
    optimizer = _best_multistart(
        residual, starts, lower, upper, max_nfev=max_nfev
    )
    params = make_params(optimizer.x)
    fitted = prices(optimizer.x)
    return _result(
        model="annual_lognormal_mixture",
        parameters={
            "low_weight": params.low_weight,
            "low_forward_multiplier": params.low_forward_multiplier,
            "high_forward_multiplier": params.high_forward_multiplier,
            "low_volatility": params.low_volatility,
            "high_volatility": params.high_volatility,
        },
        optimizer=optimizer,
        fitted=fitted,
        arrays=arrays,
    )


def fit_kou(
    quotes: pd.DataFrame,
    *,
    price_floor: float = 0.05,
    seed: int = 19487,
    integration_nodes: int = 192,
    target_column: str | None = None,
) -> CalibrationResult:
    arrays = _calibration_arrays(quotes, price_floor, target_column)

    def make_params(value: np.ndarray) -> KouParams:
        return KouParams(
            diffusion_volatility=float(value[0]),
            jump_intensity=float(value[1]),
            up_probability=float(value[2]),
            up_rate=float(value[3]),
            down_rate=float(value[4]),
        )

    # Raw expiry slices have constant maturities; rounded grouping also supports
    # the declared multi-tenor pilot without pretending every row is identical.
    groups = {}
    for maturity in np.unique(np.round(arrays["maturity"], 10)):
        groups[float(maturity)] = np.flatnonzero(
            np.isclose(arrays["maturity"], maturity, atol=1e-10, rtol=0)
        )

    def prices(value: np.ndarray) -> np.ndarray:
        params = make_params(value)
        fitted = np.empty(len(arrays["target"]))
        for maturity, indices in groups.items():
            fitted[indices] = kou_call_price(
                forward=arrays["forward"][indices],
                strike=arrays["strike"][indices],
                discount=arrays["discount"][indices],
                maturity=maturity,
                params=params,
                integration_nodes=integration_nodes,
                integration_limit=140.0,
            )
        return fitted

    def residual(value: np.ndarray) -> np.ndarray:
        return (prices(value) - arrays["target"]) / arrays["scale"]

    lower = np.array([0.02, 0.0, 0.03, 1.80, 0.50])
    upper = np.array([0.80, 5.0, 0.97, 20.0, 30.0])
    rng = np.random.default_rng(seed)
    starts = [
        np.array([0.16, 0.50, 0.30, 4.0, 6.0]),
        np.array([0.20, 0.20, 0.40, 6.0, 8.0]),
        np.array([0.12, 1.00, 0.25, 3.0, 5.0]),
    ]
    starts.extend(rng.uniform(lower, upper) for _ in range(3))
    optimizer = _best_multistart(
        residual, starts, lower, upper, max_nfev=1_200
    )
    params = make_params(optimizer.x)
    fitted = prices(optimizer.x)
    return _result(
        model="kou_double_exponential_jump_diffusion",
        parameters={
            "diffusion_volatility": params.diffusion_volatility,
            "jump_intensity": params.jump_intensity,
            "up_probability": params.up_probability,
            "up_rate": params.up_rate,
            "down_rate": params.down_rate,
        },
        optimizer=optimizer,
        fitted=fitted,
        arrays=arrays,
    )
