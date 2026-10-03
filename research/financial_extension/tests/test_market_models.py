from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE))

from market_models import (  # noqa: E402
    KouParams,
    LognormalMixtureParams,
    black_option_price,
    kou_normalized_characteristic,
    kou_option_price,
    lognormal_mixture_option_price,
)


def test_black_put_call_parity() -> None:
    forward = 100.0
    strike = np.array([80.0, 100.0, 120.0])
    discount = np.exp(-0.04)
    calls = black_option_price(
        forward=forward,
        strike=strike,
        discount=discount,
        maturity=1.0,
        volatility=0.20,
        cp_flag=["C", "C", "C"],
    )
    puts = black_option_price(
        forward=forward,
        strike=strike,
        discount=discount,
        maturity=1.0,
        volatility=0.20,
        cp_flag=["P", "P", "P"],
    )
    assert np.allclose(calls - puts, discount * (forward - strike))


def test_lognormal_mixture_respects_martingale_parity() -> None:
    params = LognormalMixtureParams(
        low_weight=0.70,
        low_forward_multiplier=0.90,
        low_volatility=0.28,
        high_volatility=0.15,
    )
    assert np.isclose(
        params.low_weight * params.low_forward_multiplier
        + (1.0 - params.low_weight) * params.high_forward_multiplier,
        1.0,
    )
    strikes = np.array([80.0, 100.0, 120.0])
    calls = lognormal_mixture_option_price(
        forward=100.0,
        strike=strikes,
        discount=0.96,
        maturity=1.0,
        cp_flag=["C"] * 3,
        params=params,
    )
    puts = lognormal_mixture_option_price(
        forward=100.0,
        strike=strikes,
        discount=0.96,
        maturity=1.0,
        cp_flag=["P"] * 3,
        params=params,
    )
    assert np.allclose(calls - puts, 0.96 * (100.0 - strikes))


def test_kou_martingale_moment_is_one() -> None:
    params = KouParams(
        diffusion_volatility=0.18,
        jump_intensity=0.7,
        up_probability=0.35,
        up_rate=4.0,
        down_rate=5.0,
    )
    moment = kou_normalized_characteristic(-1j, maturity=1.7, params=params)
    assert np.isclose(moment, 1.0 + 0.0j, atol=1e-12)


def test_kou_zero_intensity_matches_black() -> None:
    strikes = np.array([70.0, 85.0, 100.0, 115.0, 130.0])
    discount = np.exp(-0.05)
    params = KouParams(
        diffusion_volatility=0.20,
        jump_intensity=0.0,
        up_probability=0.5,
        up_rate=5.0,
        down_rate=5.0,
    )
    kou = kou_option_price(
        forward=100.0,
        strike=strikes,
        discount=discount,
        maturity=1.0,
        cp_flag=["C"] * len(strikes),
        params=params,
        integration_limit=180.0,
        integration_nodes=384,
    )
    black = black_option_price(
        forward=100.0,
        strike=strikes,
        discount=discount,
        maturity=1.0,
        volatility=0.20,
        cp_flag=["C"] * len(strikes),
    )
    assert np.allclose(kou, black, atol=2e-8, rtol=0)


def test_kou_put_call_parity() -> None:
    strikes = np.array([85.0, 100.0, 115.0])
    params = KouParams(0.16, 0.5, 0.3, 4.5, 6.0)
    calls = kou_option_price(
        forward=100.0,
        strike=strikes,
        discount=0.96,
        maturity=1.0,
        cp_flag=["C"] * 3,
        params=params,
        integration_nodes=256,
    )
    puts = kou_option_price(
        forward=100.0,
        strike=strikes,
        discount=0.96,
        maturity=1.0,
        cp_flag=["P"] * 3,
        params=params,
        integration_nodes=256,
    )
    assert np.all(calls >= 0)
    assert np.all(puts >= 0)
    assert np.allclose(calls - puts, 0.96 * (100.0 - strikes), atol=1e-10)
