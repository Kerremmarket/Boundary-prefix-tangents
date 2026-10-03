"""Forward-based European option prices for the pilot market models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.stats import norm


def _broadcast_inputs(*values):
    return np.broadcast_arrays(*[np.asarray(value, dtype=float) for value in values])


def black_option_price(
    *,
    forward,
    strike,
    discount,
    maturity,
    volatility,
    cp_flag,
) -> np.ndarray:
    forward, strike, discount, maturity, volatility = _broadcast_inputs(
        forward, strike, discount, maturity, volatility
    )
    flags = np.broadcast_to(np.asarray(cp_flag), forward.shape)
    if (
        np.any(forward <= 0)
        or np.any(strike <= 0)
        or np.any(discount <= 0)
        or np.any(maturity <= 0)
        or np.any(volatility <= 0)
    ):
        raise ValueError("Black inputs must be positive")
    total_volatility = volatility * np.sqrt(maturity)
    d1 = np.log(forward / strike) / total_volatility + 0.5 * total_volatility
    d2 = d1 - total_volatility
    call = discount * (forward * norm.cdf(d1) - strike * norm.cdf(d2))
    put = discount * (strike * norm.cdf(-d2) - forward * norm.cdf(-d1))
    is_call = np.char.upper(flags.astype(str)) == "C"
    if not np.all(is_call | (np.char.upper(flags.astype(str)) == "P")):
        raise ValueError("cp_flag must contain only C or P")
    return np.where(is_call, call, put)


@dataclass(frozen=True)
class LognormalMixtureParams:
    low_weight: float
    low_forward_multiplier: float
    low_volatility: float
    high_volatility: float

    @property
    def high_forward_multiplier(self) -> float:
        return (1.0 - self.low_weight * self.low_forward_multiplier) / (
            1.0 - self.low_weight
        )

    def validate(self) -> None:
        if not 0 < self.low_weight < 1:
            raise ValueError("mixture weight must lie in (0,1)")
        if not 0 < self.low_forward_multiplier < 1:
            raise ValueError("low component forward multiplier must lie in (0,1)")
        if self.low_volatility <= 0 or self.high_volatility <= 0:
            raise ValueError("component volatilities must be positive")
        if self.high_forward_multiplier <= 0:
            raise ValueError("martingale-implied high component forward is invalid")


def lognormal_mixture_option_price(
    *,
    forward,
    strike,
    discount,
    maturity,
    cp_flag,
    params: LognormalMixtureParams,
) -> np.ndarray:
    params.validate()
    low = black_option_price(
        forward=np.asarray(forward) * params.low_forward_multiplier,
        strike=strike,
        discount=discount,
        maturity=maturity,
        volatility=params.low_volatility,
        cp_flag=cp_flag,
    )
    high = black_option_price(
        forward=np.asarray(forward) * params.high_forward_multiplier,
        strike=strike,
        discount=discount,
        maturity=maturity,
        volatility=params.high_volatility,
        cp_flag=cp_flag,
    )
    return params.low_weight * low + (1.0 - params.low_weight) * high


@dataclass(frozen=True)
class KouParams:
    diffusion_volatility: float
    jump_intensity: float
    up_probability: float
    up_rate: float
    down_rate: float

    def validate(self, damping: float | None = None) -> None:
        if self.diffusion_volatility <= 0:
            raise ValueError("diffusion volatility must be positive")
        if self.jump_intensity < 0:
            raise ValueError("jump intensity cannot be negative")
        if not 0 < self.up_probability < 1:
            raise ValueError("up_probability must lie in (0,1)")
        if self.up_rate <= 1:
            raise ValueError("up_rate must exceed one for the martingale moment")
        if self.down_rate <= 0:
            raise ValueError("down_rate must be positive")
        if damping is not None and self.up_rate <= damping + 1:
            raise ValueError("up_rate is too small for the requested Carr--Madan damping")

    @property
    def jump_compensator(self) -> float:
        self.validate()
        mean_gross_jump = (
            self.up_probability * self.up_rate / (self.up_rate - 1.0)
            + (1.0 - self.up_probability)
            * self.down_rate
            / (self.down_rate + 1.0)
        )
        return mean_gross_jump - 1.0


def kou_normalized_characteristic(
    argument: np.ndarray | complex,
    *,
    maturity: float,
    params: KouParams,
) -> np.ndarray:
    """Characteristic function of log(S_T/F_T) under the martingale law."""

    params.validate()
    if maturity <= 0:
        raise ValueError("maturity must be positive")
    argument = np.asarray(argument, dtype=complex)
    sigma = params.diffusion_volatility
    intensity = params.jump_intensity
    probability = params.up_probability
    up_rate = params.up_rate
    down_rate = params.down_rate
    jump_cf = (
        probability * up_rate / (up_rate - 1j * argument)
        + (1.0 - probability) * down_rate / (down_rate + 1j * argument)
    )
    drift = -0.5 * sigma**2 - intensity * params.jump_compensator
    exponent = maturity * (
        1j * argument * drift
        - 0.5 * sigma**2 * argument**2
        + intensity * (jump_cf - 1.0)
    )
    return np.exp(exponent)


def kou_call_price(
    *,
    forward,
    strike,
    discount,
    maturity: float,
    params: KouParams,
    damping: float = 0.75,
    integration_limit: float = 160.0,
    integration_nodes: int = 384,
) -> np.ndarray:
    """Price calls by Carr--Madan inversion in forward-normalized units."""

    params.validate(damping=damping)
    if maturity <= 0 or integration_limit <= 0 or integration_nodes < 64:
        raise ValueError("invalid maturity or integration controls")
    forward, strike, discount = _broadcast_inputs(forward, strike, discount)
    if np.any(forward <= 0) or np.any(strike <= 0) or np.any(discount <= 0):
        raise ValueError("forward, strike, and discount must be positive")
    log_moneyness = np.log(strike / forward).reshape(-1)

    nodes, weights = leggauss(integration_nodes)
    frequency = 0.5 * integration_limit * (nodes + 1.0)
    integration_weights = 0.5 * integration_limit * weights
    shifted = frequency - 1j * (damping + 1.0)
    characteristic = kou_normalized_characteristic(
        shifted, maturity=maturity, params=params
    )
    denominator = (
        damping**2
        + damping
        - frequency**2
        + 1j * (2.0 * damping + 1.0) * frequency
    )
    transform = characteristic / denominator
    oscillation = np.exp(-1j * np.outer(log_moneyness, frequency))
    integral = np.real(oscillation * transform) @ integration_weights
    normalized_call = np.exp(-damping * log_moneyness) * integral / np.pi
    prices = (
        discount.reshape(-1)
        * forward.reshape(-1)
        * np.maximum(normalized_call, 0.0)
    )
    return prices.reshape(forward.shape)


def kou_option_price(
    *,
    forward,
    strike,
    discount,
    maturity: float,
    cp_flag,
    params: KouParams,
    **integration_controls,
) -> np.ndarray:
    forward, strike, discount = _broadcast_inputs(forward, strike, discount)
    calls = kou_call_price(
        forward=forward,
        strike=strike,
        discount=discount,
        maturity=maturity,
        params=params,
        **integration_controls,
    )
    puts = calls - discount * (forward - strike)
    flags = np.broadcast_to(np.asarray(cp_flag), forward.shape)
    upper = np.char.upper(flags.astype(str))
    if not np.all((upper == "C") | (upper == "P")):
        raise ValueError("cp_flag must contain only C or P")
    return np.where(upper == "C", calls, puts)
