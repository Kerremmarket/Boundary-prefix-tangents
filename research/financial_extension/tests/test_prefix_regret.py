from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE))

from indexed_annuity import (  # noqa: E402
    ContractSpec,
    CreditingSpec,
    MixtureCreditedDistribution,
    black_scholes_credited_law,
    log_state_grid,
    solve_stationary_bellman,
    upper_regular_boundary,
)
from market_models import LognormalMixtureParams  # noqa: E402
from prefix_regret import (  # noqa: E402
    lognormal_source_density,
    market_two_date_actual_regret,
    market_two_date_coefficients,
    two_date_actual_regret,
    two_date_coefficients,
)


def test_actual_regret_converges_to_prefix_not_diagonal_coefficient() -> None:
    law = black_scholes_credited_law(
        annual_rate=0.05,
        dividend_yield=0.02,
        volatility=0.20,
        crediting=CreditingSpec(participation=0.80, cap=0.08),
        quadrature_nodes=32,
    )
    contract = ContractSpec(
        discount_factor=np.exp(-0.05),
        termination_probability=0.10,
        surrender_haircut=0.08,
        death_guarantee=1.0,
    )
    solution = solve_stationary_bellman(
        grid=log_state_grid(0.05, 8.0, 30_000),
        law=law,
        contract=contract,
        tolerance=1e-12,
    )
    boundary, slope = upper_regular_boundary(solution)
    source_density = lognormal_source_density(median=0.82, log_sigma=0.16)
    coefficients = two_date_coefficients(
        boundary=boundary,
        stopping_side_gap_slope=slope,
        h1=0.8,
        h2=1.1,
        source_density=source_density,
        law=law,
        contract=contract,
    )
    assert coefficients.copied_prefix > 0

    scaled = []
    for epsilon in (0.01, 0.005, 0.0025, 0.00125):
        regret = two_date_actual_regret(
            epsilon=epsilon,
            boundary=boundary,
            h1=0.8,
            h2=1.1,
            source_density=source_density,
            solution=solution,
            law=law,
            contract=contract,
            integration_nodes=64,
        )
        scaled.append(regret.total / epsilon**2)

    prefix_errors = [abs(value - coefficients.prefix) for value in scaled]
    assert prefix_errors[-1] < prefix_errors[0]
    assert prefix_errors[-1] < 0.01
    assert abs(scaled[-1] - coefficients.prefix) < abs(
        scaled[-1] - coefficients.diagonal
    )
    assert (scaled[-1] - coefficients.diagonal) > 0.5 * coefficients.copied_prefix


def test_market_mixed_law_regret_converges_to_prefix_coefficient() -> None:
    distribution = MixtureCreditedDistribution(
        gross_forward=1.02,
        mixture=LognormalMixtureParams(
            low_weight=0.35,
            low_forward_multiplier=0.80,
            low_volatility=0.30,
            high_volatility=0.14,
        ),
        crediting=CreditingSpec(participation=0.80, cap=0.08),
    )
    law = distribution.discretize(quadrature_nodes=48)
    contract = ContractSpec(
        discount_factor=np.exp(-0.05),
        termination_probability=0.10,
        surrender_haircut=0.08,
        death_guarantee=1.0,
    )
    solution = solve_stationary_bellman(
        grid=log_state_grid(0.05, 8.0, 30_000),
        law=law,
        contract=contract,
        tolerance=1e-12,
    )
    boundary, slope = upper_regular_boundary(solution)
    initial_state = 0.96 * boundary
    h1, h2 = 0.8 * boundary, 1.1 * boundary
    coefficients = market_two_date_coefficients(
        boundary=boundary,
        stopping_side_gap_slope=slope,
        h1=h1,
        h2=h2,
        initial_state=initial_state,
        distribution=distribution,
        contract=contract,
        integration_nodes=96,
    )
    assert coefficients.copied_prefix > 0

    scaled = []
    # Stop before epsilon approaches the fixed Bellman grid spacing; the market
    # runner performs the required joint epsilon/grid refinement separately.
    for epsilon in (0.02, 0.01, 0.005):
        regret = market_two_date_actual_regret(
            epsilon=epsilon,
            boundary=boundary,
            h1=h1,
            h2=h2,
            initial_state=initial_state,
            distribution=distribution,
            solution=solution,
            contract=contract,
            integration_nodes=48,
        )
        scaled.append(regret.total / epsilon**2)

    prefix_errors = [abs(value - coefficients.prefix) for value in scaled]
    assert prefix_errors[-1] < prefix_errors[0]
    assert abs(scaled[-1] - coefficients.prefix) < abs(
        scaled[-1] - coefficients.diagonal
    )
