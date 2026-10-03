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


def fixture_law():
    return black_scholes_credited_law(
        annual_rate=0.05,
        dividend_yield=0.02,
        volatility=0.20,
        crediting=CreditingSpec(participation=0.80, cap=0.08),
        quadrature_nodes=96,
    )


def fixture_contract() -> ContractSpec:
    return ContractSpec(
        discount_factor=np.exp(-0.05),
        termination_probability=0.10,
        surrender_haircut=0.08,
        death_guarantee=1.0,
    )


def test_credited_law_has_exact_copy_and_cap_atoms() -> None:
    law = fixture_law()
    assert law.factors[0] == 1.0
    assert np.isclose(law.factors[-1], 1.08)
    assert np.isclose(law.weights.sum(), 1.0)
    assert 0.45 < law.floor_probability < 0.51
    assert 0.30 < law.cap_probability < 0.37
    assert 1.02 < law.expected_factor < 1.05


def test_stationary_solver_finds_regular_upper_boundary() -> None:
    solution = solve_stationary_bellman(
        grid=log_state_grid(0.05, 8.0, 3000),
        law=fixture_law(),
        contract=fixture_contract(),
        tolerance=1e-11,
    )
    boundary, slope = upper_regular_boundary(solution)
    assert solution.sup_error <= 1e-11
    assert 0.89 < boundary < 0.91
    assert slope > 0.10
    assert np.all(solution.stopping[-50:])


def test_boundary_is_stable_under_grid_refinement() -> None:
    coarse = solve_stationary_bellman(
        grid=log_state_grid(0.05, 8.0, 1800),
        law=fixture_law(),
        contract=fixture_contract(),
    )
    fine = solve_stationary_bellman(
        grid=log_state_grid(0.05, 8.0, 3600),
        law=fixture_law(),
        contract=fixture_contract(),
    )
    coarse_boundary, coarse_slope = upper_regular_boundary(coarse)
    fine_boundary, fine_slope = upper_regular_boundary(fine)
    assert abs(coarse_boundary - fine_boundary) < 5e-4
    assert abs(coarse_slope - fine_slope) < 0.02


def test_minimum_surrender_value_can_generate_two_regular_boundaries() -> None:
    contract = ContractSpec(
        discount_factor=np.exp(-0.05),
        termination_probability=0.10,
        surrender_haircut=0.08,
        death_guarantee=1.0,
        minimum_surrender_value=0.75,
    )
    solution = solve_stationary_bellman(
        grid=log_state_grid(0.20, 8.0, 3000),
        law=fixture_law(),
        contract=contract,
    )
    assert len(solution.roots) == 2
    assert solution.root_slopes[0] < 0 < solution.root_slopes[1]


def test_mixture_credited_distribution_preserves_atoms_and_probability() -> None:
    distribution = MixtureCreditedDistribution(
        gross_forward=1.03,
        mixture=LognormalMixtureParams(
            low_weight=0.35,
            low_forward_multiplier=0.82,
            low_volatility=0.30,
            high_volatility=0.14,
        ),
        crediting=CreditingSpec(participation=0.80, cap=0.08),
    )
    law = distribution.discretize(quadrature_nodes=64)
    assert law.factors[0] == 1.0
    assert law.factors[-1] == 1.08
    assert np.isclose(law.weights.sum(), 1.0)
    assert np.isclose(law.floor_probability, distribution.floor_probability, atol=1e-12)
    assert np.isclose(law.cap_probability, distribution.cap_probability, atol=1e-12)
    assert 0 < distribution.floor_probability < 1
    assert 0 < distribution.cap_probability < 1

    grid = np.linspace(1.0, 1.08, 20_001)
    interior_mass = np.trapz(distribution.factor_density(grid), grid)
    assert np.isclose(
        distribution.floor_probability
        + interior_mass
        + distribution.cap_probability,
        1.0,
        atol=2e-5,
    )
