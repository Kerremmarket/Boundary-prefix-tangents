from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE))

from bermudan_diagnostic import (  # noqa: E402
    GaussianDensityConvolver,
    MarketInput,
    diagonal_coefficient,
    evaluate_threshold_policy_forward,
    log_log_slope,
    reference_occupancy_densities,
    solve_bermudan_put_fft,
    train_lsmc_policy,
)


def _market() -> MarketInput:
    return MarketInput(
        date="test",
        rate=0.03,
        dividend_yield=0.015,
        volatility=0.20,
        annual_gross_forward=float(np.exp(0.015)),
    )


def test_fft_reference_has_regular_put_boundaries_and_converges() -> None:
    coarse = solve_bermudan_put_fft(_market(), grid_nodes=8_193)
    fine = solve_bermudan_put_fft(_market(), grid_nodes=16_385)
    assert all(item.crossing_count == 1 for item in fine.boundary_diagnostics)
    assert all(item.orientation_ok for item in fine.boundary_diagnostics)
    assert np.all(np.diff(fine.boundaries) > 0.0)
    assert np.max(np.abs(coarse.boundaries - fine.boundaries)) < 2e-5
    assert abs(coarse.value_at_zero - fine.value_at_zero) < 1e-6


def test_gaussian_density_convolution_preserves_interior_mass() -> None:
    grid = np.linspace(-4.0, 2.0, 16_385)
    density = np.exp(-0.5 * ((grid + 0.1) / 0.2) ** 2) / (
        0.2 * np.sqrt(2.0 * np.pi)
    )
    convolver = GaussianDensityConvolver(
        grid, mean_increment=0.01, standard_deviation=0.08
    )
    transitioned = convolver(density)
    assert abs(np.trapz(density, grid) - 1.0) < 1e-10
    assert abs(np.trapz(transitioned, grid) - 1.0) < 1e-9
    assert np.all(transitioned >= 0.0)


def test_forward_loss_is_zero_at_reference_and_matches_quadratic_limit() -> None:
    reference = solve_bermudan_put_fft(_market(), grid_nodes=32_769)
    density_grid, densities, _ = reference_occupancy_densities(
        reference, density_nodes=16_385
    )
    zero = evaluate_threshold_policy_forward(
        reference,
        reference.boundaries,
        density_nodes=16_385,
        reference_density_grid=density_grid,
        reference_densities=densities,
    )
    assert zero.loss == 0.0

    displacement = np.array([-0.008, 0.002, 0.008, -0.011, -0.011, 0.005, 0.009])
    coefficient, _ = diagonal_coefficient(
        reference,
        reference.boundaries + displacement,
        density_grid=density_grid,
        reference_densities=densities,
    )
    etas = np.array([0.25, 0.125, 0.0625])
    losses = []
    for eta in etas:
        evaluation = evaluate_threshold_policy_forward(
            reference,
            reference.boundaries + eta * displacement,
            density_nodes=16_385,
            reference_density_grid=density_grid,
            reference_densities=densities,
        )
        losses.append(evaluation.loss)
    losses = np.asarray(losses)
    assert 1.95 < log_log_slope(etas, losses) < 2.05
    assert abs(losses[-1] / (etas[-1] ** 2 * coefficient) - 1.0) < 0.01


def test_linear_lsmc_retains_native_single_crossing_on_sanity_case() -> None:
    policy = train_lsmc_policy(_market(), seed=2026082701, paths=20_000)
    assert np.all(np.isfinite(policy.boundaries))
    assert all(item.crossing_count == 1 for item in policy.boundary_diagnostics)
    assert all(item.orientation_ok for item in policy.boundary_diagnostics)
    assert np.all(policy.ranks == 2)
    assert np.all(np.isfinite(policy.condition_numbers))
