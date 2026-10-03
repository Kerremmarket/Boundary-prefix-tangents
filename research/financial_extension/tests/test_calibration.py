from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE))

from calibration import fit_black, fit_lognormal_mixture  # noqa: E402
from market_models import (  # noqa: E402
    LognormalMixtureParams,
    black_option_price,
    lognormal_mixture_option_price,
)


def quote_frame(prices: np.ndarray, half_spread: float = 0.10) -> pd.DataFrame:
    strikes = np.linspace(70.0, 130.0, len(prices))
    return pd.DataFrame(
        {
            "strike": strikes,
            "matched_forward_price": 100.0,
            "discount_factor": 0.96,
            "year_fraction": 1.0,
            "call_equivalent_bid": prices - half_spread,
            "call_equivalent_offer": prices + half_spread,
            "call_equivalent_mid": prices,
        }
    )


def test_black_calibration_recovers_synthetic_volatility() -> None:
    strikes = np.linspace(70.0, 130.0, 25)
    prices = black_option_price(
        forward=100.0,
        strike=strikes,
        discount=0.96,
        maturity=1.0,
        volatility=0.23,
        cp_flag=["C"] * len(strikes),
    )
    result = fit_black(quote_frame(prices), price_floor=0.01)
    assert result.success
    assert abs(result.parameters["volatility"] - 0.23) < 1e-8
    assert result.standardized_rmse < 1e-8
    assert result.within_spread_share == 1.0


def test_mixture_calibration_improves_over_black_on_mixture_smile() -> None:
    strikes = np.linspace(70.0, 130.0, 31)
    true = LognormalMixtureParams(
        low_weight=0.70,
        low_forward_multiplier=0.90,
        low_volatility=0.28,
        high_volatility=0.15,
    )
    prices = lognormal_mixture_option_price(
        forward=100.0,
        strike=strikes,
        discount=0.96,
        maturity=1.0,
        cp_flag=["C"] * len(strikes),
        params=true,
    )
    quotes = quote_frame(prices, half_spread=0.05)
    black = fit_black(quotes, price_floor=0.01)
    mixture = fit_lognormal_mixture(quotes, price_floor=0.01, seed=7)
    assert mixture.success
    assert mixture.standardized_rmse < 0.05 * black.standardized_rmse
    assert mixture.forward_normalized_rmse < 1e-5
    martingale_mean = (
        mixture.parameters["low_weight"]
        * mixture.parameters["low_forward_multiplier"]
        + (1.0 - mixture.parameters["low_weight"])
        * mixture.parameters["high_forward_multiplier"]
    )
    assert np.isclose(martingale_mean, 1.0)


def test_explicit_target_column_is_used_for_quote_envelope_sensitivity() -> None:
    strikes = np.linspace(70.0, 130.0, 25)
    mids = black_option_price(
        forward=100.0,
        strike=strikes,
        discount=0.96,
        maturity=1.0,
        volatility=0.20,
        cp_flag=["C"] * len(strikes),
    )
    quotes = quote_frame(mids, half_spread=0.10)
    bid_fit = fit_black(quotes, target_column="call_equivalent_bid")
    offer_fit = fit_black(quotes, target_column="call_equivalent_offer")
    assert bid_fit.parameters["volatility"] < offer_fit.parameters["volatility"]
