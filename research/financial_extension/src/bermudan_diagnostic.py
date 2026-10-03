"""Smooth Black--Scholes Bermudan control for the financial diagnostic.

The module deliberately separates policy construction from policy evaluation.
It contains no statistical tangent or learned-policy theorem: LSMC merely
supplies fixed boundary graphs that are evaluated against a deterministic
reference problem.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Sequence

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.fft import irfft, next_fast_len, rfft
from scipy.optimize import brentq
from scipy.stats import norm


@dataclass(frozen=True)
class MarketInput:
    date: str
    rate: float
    dividend_yield: float
    volatility: float
    annual_gross_forward: float

    def validate(self) -> None:
        if not self.date:
            raise ValueError("date is required")
        if not np.isfinite(self.rate) or not np.isfinite(self.dividend_yield):
            raise ValueError("rate and dividend yield must be finite")
        if self.volatility <= 0 or not np.isfinite(self.volatility):
            raise ValueError("volatility must be positive")
        if self.annual_gross_forward <= 0:
            raise ValueError("annual gross forward must be positive")


@dataclass(frozen=True)
class BoundaryDiagnostic:
    crossing_count: int
    orientation_ok: bool
    boundary: float | None


@dataclass
class BermudanSolution:
    market: MarketInput
    grid: np.ndarray
    gaps: list[np.ndarray]
    exercise_masks: list[np.ndarray]
    boundaries: np.ndarray
    boundary_diagnostics: list[BoundaryDiagnostic]
    gap_slopes: np.ndarray
    value_at_zero: float
    grid_nodes: int
    quadrature_nodes: int
    solver: str


@dataclass
class LSMCPolicy:
    market: MarketInput
    seed: int
    coefficients: list[np.ndarray]
    sample_counts: np.ndarray
    ranks: np.ndarray
    condition_numbers: np.ndarray
    boundaries: np.ndarray
    boundary_diagnostics: list[BoundaryDiagnostic]
    training_value: float


@dataclass
class ForwardEvaluation:
    loss: float
    local_losses: np.ndarray
    single_date_sum: float
    interaction: float
    predecision_masses: np.ndarray


def put_payoff(log_spot: np.ndarray | float) -> np.ndarray:
    log_spot = np.asarray(log_spot, dtype=float)
    return np.maximum(1.0 - np.exp(log_spot), 0.0)


@lru_cache(maxsize=None)
def _normal_quadrature(nodes: int) -> tuple[np.ndarray, np.ndarray]:
    if nodes < 8:
        raise ValueError("at least eight quadrature nodes are required")
    x, w = hermgauss(nodes)
    return np.sqrt(2.0) * x, w / np.sqrt(np.pi)


def _interpolate_option_value(
    query: np.ndarray, grid: np.ndarray, values: np.ndarray
) -> np.ndarray:
    result = np.interp(query, grid, values)
    below = query < grid[0]
    above = query > grid[-1]
    if np.any(below):
        result[below] = put_payoff(query[below])
    if np.any(above):
        result[above] = 0.0
    return result


def transition_expectation(
    values: np.ndarray,
    grid: np.ndarray,
    *,
    mean_increment: float,
    standard_deviation: float,
    quadrature_nodes: int,
) -> np.ndarray:
    """Conditional expectation of a next-date value on a fixed log grid."""

    z, weights = _normal_quadrature(quadrature_nodes)
    result = np.zeros_like(grid)
    for node, weight in zip(z, weights, strict=True):
        query = grid + mean_increment + standard_deviation * node
        result += weight * _interpolate_option_value(query, grid, values)
    return result


def point_transition_expectation(
    values: np.ndarray,
    grid: np.ndarray,
    *,
    point: float,
    mean_increment: float,
    standard_deviation: float,
    quadrature_nodes: int,
) -> float:
    z, weights = _normal_quadrature(quadrature_nodes)
    query = point + mean_increment + standard_deviation * z
    interpolated = _interpolate_option_value(query.copy(), grid, values)
    return float(weights @ interpolated)


def diagnose_exercise_mask(
    grid: np.ndarray,
    exercise_mask: np.ndarray,
    gap: np.ndarray,
    *,
    interpolate_root: bool,
) -> BoundaryDiagnostic:
    """Diagnose native put-side signs without imposing a threshold."""

    eligible = np.flatnonzero(grid < 0.0)
    if eligible.size < 3:
        raise ValueError("grid must contain at least three in-the-money nodes")
    last = eligible[-1]
    signs = np.asarray(exercise_mask[: last + 1], dtype=bool)
    changes = np.flatnonzero(signs[:-1] != signs[1:])
    orientation_ok = bool(signs[0] and not signs[-1])
    if changes.size != 1 or not orientation_ok:
        return BoundaryDiagnostic(int(changes.size), orientation_ok, None)
    j = int(changes[0])
    if not signs[j] or signs[j + 1]:
        return BoundaryDiagnostic(1, False, None)
    if interpolate_root:
        y0, y1 = float(gap[j]), float(gap[j + 1])
        if y0 == y1:
            boundary = 0.5 * (grid[j] + grid[j + 1])
        else:
            boundary = grid[j] - y0 * (grid[j + 1] - grid[j]) / (y1 - y0)
    else:
        boundary = 0.5 * (grid[j] + grid[j + 1])
    return BoundaryDiagnostic(1, True, float(boundary))


def solve_bermudan_put(
    market: MarketInput,
    *,
    grid_nodes: int = 65_537,
    quadrature_nodes: int = 64,
    exercise_dates: int = 8,
    log_grid_min: float = -4.0,
    log_grid_max: float = 2.0,
    interpolate_boundary_root: bool = True,
) -> BermudanSolution:
    market.validate()
    if grid_nodes < 65 or exercise_dates < 2:
        raise ValueError("grid and exercise-date counts are too small")
    grid = np.linspace(log_grid_min, log_grid_max, grid_nodes)
    dt = 1.0 / exercise_dates
    discount = np.exp(-market.rate * dt)
    mean = (
        market.rate
        - market.dividend_yield
        - 0.5 * market.volatility**2
    ) * dt
    sd = market.volatility * np.sqrt(dt)
    payoff = put_payoff(grid)
    value = payoff.copy()
    gaps_reversed: list[np.ndarray] = []
    masks_reversed: list[np.ndarray] = []
    diagnostics_reversed: list[BoundaryDiagnostic] = []

    for _ in range(exercise_dates - 1):
        continuation = discount * transition_expectation(
            value,
            grid,
            mean_increment=mean,
            standard_deviation=sd,
            quadrature_nodes=quadrature_nodes,
        )
        gap = payoff - continuation
        exercise = (payoff > 0.0) & (gap >= 0.0)
        diagnostic = diagnose_exercise_mask(
            grid,
            exercise,
            gap,
            interpolate_root=interpolate_boundary_root,
        )
        gaps_reversed.append(gap)
        masks_reversed.append(exercise)
        diagnostics_reversed.append(diagnostic)
        value = np.where(exercise, payoff, continuation)

    gaps = list(reversed(gaps_reversed))
    masks = list(reversed(masks_reversed))
    diagnostics = list(reversed(diagnostics_reversed))
    if any(item.boundary is None for item in diagnostics):
        boundaries = np.full(exercise_dates - 1, np.nan)
        slopes = np.full(exercise_dates - 1, np.nan)
    else:
        boundaries = np.array([item.boundary for item in diagnostics], dtype=float)
        slopes = np.array(
            [
                np.interp(boundary, grid, np.gradient(gap, grid))
                for boundary, gap in zip(boundaries, gaps, strict=True)
            ]
        )
    value0 = discount * point_transition_expectation(
        value,
        grid,
        point=0.0,
        mean_increment=mean,
        standard_deviation=sd,
        quadrature_nodes=quadrature_nodes,
    )
    return BermudanSolution(
        market=market,
        grid=grid,
        gaps=gaps,
        exercise_masks=masks,
        boundaries=boundaries,
        boundary_diagnostics=diagnostics,
        gap_slopes=slopes,
        value_at_zero=float(value0),
        grid_nodes=grid_nodes,
        quadrature_nodes=quadrature_nodes,
        solver="gauss_hermite",
    )


def solve_bermudan_put_fft(
    market: MarketInput,
    *,
    grid_nodes: int = 65_537,
    exercise_dates: int = 8,
    log_grid_min: float = -4.0,
    log_grid_max: float = 2.0,
    interpolate_boundary_root: bool = True,
) -> BermudanSolution:
    """Solve the Bermudan problem by full Gaussian convolution on a log grid."""

    market.validate()
    if grid_nodes < 65 or exercise_dates < 2:
        raise ValueError("grid and exercise-date counts are too small")
    grid = np.linspace(log_grid_min, log_grid_max, grid_nodes)
    dt = 1.0 / exercise_dates
    discount = np.exp(-market.rate * dt)
    mean = (
        market.rate
        - market.dividend_yield
        - 0.5 * market.volatility**2
    ) * dt
    sd = market.volatility * np.sqrt(dt)
    # Backward expectation uses the adjoint orientation of the forward-density
    # convolution: k(y-x; mean) = k(x-y; -mean).
    value_operator = GaussianDensityConvolver(
        grid, mean_increment=-mean, standard_deviation=sd
    )
    payoff = put_payoff(grid)
    value = payoff.copy()
    gaps_reversed: list[np.ndarray] = []
    masks_reversed: list[np.ndarray] = []
    diagnostics_reversed: list[BoundaryDiagnostic] = []

    for _ in range(exercise_dates - 1):
        continuation = discount * value_operator(value)
        gap = payoff - continuation
        exercise = (payoff > 0.0) & (gap >= 0.0)
        diagnostic = diagnose_exercise_mask(
            grid,
            exercise,
            gap,
            interpolate_root=interpolate_boundary_root,
        )
        gaps_reversed.append(gap)
        masks_reversed.append(exercise)
        diagnostics_reversed.append(diagnostic)
        value = np.where(exercise, payoff, continuation)

    gaps = list(reversed(gaps_reversed))
    masks = list(reversed(masks_reversed))
    diagnostics = list(reversed(diagnostics_reversed))
    if any(item.boundary is None for item in diagnostics):
        boundaries = np.full(exercise_dates - 1, np.nan)
        slopes = np.full(exercise_dates - 1, np.nan)
    else:
        boundaries = np.array([item.boundary for item in diagnostics], dtype=float)
        slopes = np.array(
            [
                np.interp(boundary, grid, np.gradient(gap, grid))
                for boundary, gap in zip(boundaries, gaps, strict=True)
            ]
        )
    value0_grid = discount * value_operator(value)
    value0 = float(np.interp(0.0, grid, value0_grid))
    return BermudanSolution(
        market=market,
        grid=grid,
        gaps=gaps,
        exercise_masks=masks,
        boundaries=boundaries,
        boundary_diagnostics=diagnostics,
        gap_slopes=slopes,
        value_at_zero=value0,
        grid_nodes=grid_nodes,
        quadrature_nodes=0,
        solver="fft_convolution",
    )


def shifted_legendre_basis(spot: np.ndarray | float) -> np.ndarray:
    spot = np.asarray(spot, dtype=float)
    z = 2.0 * spot - 1.0
    return np.column_stack(
        [
            np.ones_like(z),
            z,
            0.5 * (3.0 * z**2 - 1.0),
            0.5 * (5.0 * z**3 - 3.0 * z),
        ]
    )


def simulate_black_scholes_paths(
    market: MarketInput,
    *,
    paths: int,
    seed: int,
    exercise_dates: int = 8,
) -> np.ndarray:
    market.validate()
    if paths < 1:
        raise ValueError("paths must be positive")
    dt = 1.0 / exercise_dates
    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal((paths, exercise_dates))
    increments = (
        market.rate
        - market.dividend_yield
        - 0.5 * market.volatility**2
    ) * dt + market.volatility * np.sqrt(dt) * shocks
    return np.exp(np.cumsum(increments, axis=1))


def _lsmc_gap(log_spot: np.ndarray, coefficient: np.ndarray) -> np.ndarray:
    spot = np.exp(np.asarray(log_spot, dtype=float))
    design = shifted_legendre_basis(spot)[:, : len(coefficient)]
    continuation = np.maximum(design @ coefficient, 0.0)
    return put_payoff(log_spot) - continuation


def lsmc_exercise_mask(log_spot: np.ndarray, coefficient: np.ndarray) -> np.ndarray:
    intrinsic = put_payoff(log_spot)
    return (intrinsic > 0.0) & (_lsmc_gap(log_spot, coefficient) >= 0.0)


def _diagnose_lsmc_boundary(
    coefficient: np.ndarray,
    *,
    scan_nodes: int = 65_537,
) -> BoundaryDiagnostic:
    grid = np.linspace(-4.0, 0.0, scan_nodes, endpoint=False)
    gap = _lsmc_gap(grid, coefficient)
    mask = (put_payoff(grid) > 0.0) & (gap >= 0.0)
    diagnostic = diagnose_exercise_mask(
        grid, mask, gap, interpolate_root=False
    )
    if diagnostic.boundary is None:
        return diagnostic
    changes = np.flatnonzero(mask[:-1] != mask[1:])
    j = int(changes[0])
    left, right = float(grid[j]), float(grid[j + 1])
    try:
        root = brentq(
            lambda x: float(_lsmc_gap(np.array([x]), coefficient)[0]),
            left,
            right,
            xtol=1e-13,
            rtol=1e-13,
        )
    except ValueError:
        root = 0.5 * (left + right)
    return BoundaryDiagnostic(1, True, float(root))


def train_lsmc_policy(
    market: MarketInput,
    *,
    seed: int,
    paths: int = 60_000,
    exercise_dates: int = 8,
) -> LSMCPolicy:
    spots = simulate_black_scholes_paths(
        market, paths=paths, seed=seed, exercise_dates=exercise_dates
    )
    dt = 1.0 / exercise_dates
    payoff = np.maximum(1.0 - spots, 0.0)
    cashflow = payoff[:, -1].copy()
    exercise_index = np.full(paths, exercise_dates - 1, dtype=int)
    coefficients: list[np.ndarray] = [np.zeros(2) for _ in range(exercise_dates - 1)]
    counts = np.zeros(exercise_dates - 1, dtype=int)
    ranks = np.zeros(exercise_dates - 1, dtype=int)
    conditions = np.zeros(exercise_dates - 1, dtype=float)

    for i in range(exercise_dates - 2, -1, -1):
        intrinsic = payoff[:, i]
        itm = intrinsic > 0.0
        design = shifted_legendre_basis(spots[itm, i])[:, :2]
        target = cashflow[itm] * np.exp(
            -market.rate * (exercise_index[itm] - i) * dt
        )
        coefficient, _, rank, singular_values = np.linalg.lstsq(
            design, target, rcond=None
        )
        continuation = np.zeros(paths)
        continuation[itm] = np.maximum(design @ coefficient, 0.0)
        exercise = itm & (intrinsic >= continuation)
        cashflow[exercise] = intrinsic[exercise]
        exercise_index[exercise] = i
        coefficients[i] = coefficient
        counts[i] = int(itm.sum())
        ranks[i] = int(rank)
        conditions[i] = float(
            singular_values[0] / singular_values[-1]
            if singular_values[-1] > 0
            else np.inf
        )

    discounted = cashflow * np.exp(-market.rate * (exercise_index + 1) * dt)
    diagnostics = [_diagnose_lsmc_boundary(coef) for coef in coefficients]
    boundaries = np.array(
        [item.boundary if item.boundary is not None else np.nan for item in diagnostics]
    )
    return LSMCPolicy(
        market=market,
        seed=seed,
        coefficients=coefficients,
        sample_counts=counts,
        ranks=ranks,
        condition_numbers=conditions,
        boundaries=boundaries,
        boundary_diagnostics=diagnostics,
        training_value=float(np.mean(discounted)),
    )


class GaussianDensityConvolver:
    """Uniform-grid Gaussian transition for subprobability densities."""

    def __init__(
        self,
        grid: np.ndarray,
        *,
        mean_increment: float,
        standard_deviation: float,
    ) -> None:
        grid = np.asarray(grid, dtype=float)
        if grid.ndim != 1 or grid.size < 3:
            raise ValueError("a one-dimensional uniform grid is required")
        differences = np.diff(grid)
        if not np.allclose(differences, differences[0], rtol=1e-10, atol=1e-14):
            raise ValueError("density grid must be uniform")
        self.grid = grid
        self.dx = float(differences[0])
        self.size = grid.size
        offsets = np.arange(-(self.size - 1), self.size) * self.dx
        kernel = norm.pdf(
            offsets, loc=mean_increment, scale=standard_deviation
        )
        full_length = self.size + kernel.size - 1
        self.fft_length = next_fast_len(full_length)
        self.kernel_fft = rfft(kernel, self.fft_length)
        self.slice_start = self.size - 1

    def __call__(self, density: np.ndarray) -> np.ndarray:
        density = np.asarray(density, dtype=float)
        if density.shape != self.grid.shape:
            raise ValueError("density has the wrong shape")
        convolution = irfft(
            rfft(density, self.fft_length) * self.kernel_fft,
            self.fft_length,
        )
        result = convolution[
            self.slice_start : self.slice_start + self.size
        ] * self.dx
        result[result < 0.0] = 0.0
        return result


def _strip_integral(
    density_grid: np.ndarray,
    density: np.ndarray,
    gap_grid: np.ndarray,
    gap: np.ndarray,
    left: float,
    right: float,
) -> float:
    if right <= left:
        return 0.0
    width = right - left
    spacing = min(np.diff(density_grid).min(), np.diff(gap_grid).min())
    nodes = max(65, int(np.ceil(width / spacing)) * 2 + 1)
    nodes = min(nodes, 4097)
    x = np.linspace(left, right, nodes)
    integrand = np.interp(x, density_grid, density) * np.abs(
        np.interp(x, gap_grid, gap)
    )
    return float(np.trapz(integrand, x))


def reference_occupancy_densities(
    solution: BermudanSolution,
    *,
    density_nodes: int = 32_769,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    market = solution.market
    grid = np.linspace(solution.grid[0], solution.grid[-1], density_nodes)
    dt = 1.0 / 8.0
    mean = (
        market.rate
        - market.dividend_yield
        - 0.5 * market.volatility**2
    ) * dt
    sd = market.volatility * np.sqrt(dt)
    convolver = GaussianDensityConvolver(
        grid, mean_increment=mean, standard_deviation=sd
    )
    density = norm.pdf(grid, loc=mean, scale=sd)
    densities: list[np.ndarray] = []
    masses: list[float] = []
    for boundary in solution.boundaries:
        densities.append(density.copy())
        masses.append(float(np.trapz(density, grid)))
        density = np.where(grid > boundary, density, 0.0)
        density = convolver(density)
    return grid, densities, np.asarray(masses)


def diagonal_coefficient(
    solution: BermudanSolution,
    approximate_boundaries: np.ndarray,
    *,
    density_grid: np.ndarray,
    reference_densities: Sequence[np.ndarray],
) -> tuple[float, np.ndarray]:
    displacement = np.asarray(approximate_boundaries) - solution.boundaries
    times = np.arange(1, 8) / 8.0
    contributions = []
    for i, (boundary, slope, shift, density) in enumerate(
        zip(
            solution.boundaries,
            solution.gap_slopes,
            displacement,
            reference_densities,
            strict=True,
        )
    ):
        occupied_density = np.interp(boundary, density_grid, density)
        contributions.append(
            0.5
            * np.exp(-solution.market.rate * times[i])
            * occupied_density
            * abs(slope)
            * shift**2
        )
    values = np.asarray(contributions)
    return float(values.sum()), values


def evaluate_threshold_policy_forward(
    solution: BermudanSolution,
    approximate_boundaries: np.ndarray,
    *,
    density_nodes: int = 32_769,
    reference_density_grid: np.ndarray | None = None,
    reference_densities: Sequence[np.ndarray] | None = None,
) -> ForwardEvaluation:
    approximate_boundaries = np.asarray(approximate_boundaries, dtype=float)
    if approximate_boundaries.shape != solution.boundaries.shape:
        raise ValueError("boundary vector has the wrong shape")
    market = solution.market
    grid = np.linspace(solution.grid[0], solution.grid[-1], density_nodes)
    dt = 1.0 / 8.0
    mean = (
        market.rate
        - market.dividend_yield
        - 0.5 * market.volatility**2
    ) * dt
    sd = market.volatility * np.sqrt(dt)
    convolver = GaussianDensityConvolver(
        grid, mean_increment=mean, standard_deviation=sd
    )
    density = norm.pdf(grid, loc=mean, scale=sd)
    local_losses: list[float] = []
    masses: list[float] = []
    single_losses: list[float] = []
    if reference_density_grid is None or reference_densities is None:
        reference_density_grid, reference_densities, _ = (
            reference_occupancy_densities(solution, density_nodes=density_nodes)
        )

    for i, (reference_boundary, approximate_boundary, gap) in enumerate(
        zip(
            solution.boundaries,
            approximate_boundaries,
            solution.gaps,
            strict=True,
        )
    ):
        masses.append(float(np.trapz(density, grid)))
        left = min(reference_boundary, approximate_boundary)
        right = max(reference_boundary, approximate_boundary)
        discount = np.exp(-market.rate * (i + 1) * dt)
        local = discount * _strip_integral(
            grid, density, solution.grid, gap, left, right
        )
        single = discount * _strip_integral(
            reference_density_grid,
            np.asarray(reference_densities[i]),
            solution.grid,
            gap,
            left,
            right,
        )
        local_losses.append(local)
        single_losses.append(single)
        density = np.where(grid > approximate_boundary, density, 0.0)
        density = convolver(density)

    total = float(np.sum(local_losses))
    single_total = float(np.sum(single_losses))
    return ForwardEvaluation(
        loss=total,
        local_losses=np.asarray(local_losses),
        single_date_sum=single_total,
        interaction=total - single_total,
        predecision_masses=np.asarray(masses),
    )


def evaluate_native_policy_forward(
    solution: BermudanSolution,
    exercise_rule: Callable[[int, np.ndarray], np.ndarray],
    *,
    density_nodes: int = 32_769,
) -> ForwardEvaluation:
    market = solution.market
    grid = np.linspace(solution.grid[0], solution.grid[-1], density_nodes)
    dt = 1.0 / 8.0
    mean = (
        market.rate
        - market.dividend_yield
        - 0.5 * market.volatility**2
    ) * dt
    sd = market.volatility * np.sqrt(dt)
    convolver = GaussianDensityConvolver(
        grid, mean_increment=mean, standard_deviation=sd
    )
    density = norm.pdf(grid, loc=mean, scale=sd)
    local_losses: list[float] = []
    masses: list[float] = []
    for i, (reference_boundary, gap) in enumerate(
        zip(solution.boundaries, solution.gaps, strict=True)
    ):
        masses.append(float(np.trapz(density, grid)))
        approximate_exercise = np.asarray(exercise_rule(i, grid), dtype=bool)
        reference_exercise = grid <= reference_boundary
        mismatch = approximate_exercise != reference_exercise
        integrand = density * np.abs(np.interp(grid, solution.grid, gap)) * mismatch
        local = np.exp(-market.rate * (i + 1) * dt) * float(
            np.trapz(integrand, grid)
        )
        local_losses.append(local)
        density = np.where(~approximate_exercise, density, 0.0)
        density = convolver(density)
    total = float(np.sum(local_losses))
    return ForwardEvaluation(
        loss=total,
        local_losses=np.asarray(local_losses),
        single_date_sum=np.nan,
        interaction=np.nan,
        predecision_masses=np.asarray(masses),
    )


def price_threshold_policy_backward(
    solution: BermudanSolution,
    boundaries: np.ndarray,
    *,
    quadrature_nodes: int = 64,
) -> float:
    boundaries = np.asarray(boundaries, dtype=float)
    if boundaries.shape != solution.boundaries.shape:
        raise ValueError("boundary vector has the wrong shape")
    grid = solution.grid
    payoff = put_payoff(grid)
    market = solution.market
    dt = 1.0 / 8.0
    discount = np.exp(-market.rate * dt)
    mean = (
        market.rate
        - market.dividend_yield
        - 0.5 * market.volatility**2
    ) * dt
    sd = market.volatility * np.sqrt(dt)
    value = payoff.copy()
    for boundary in boundaries[::-1]:
        continuation = discount * transition_expectation(
            value,
            grid,
            mean_increment=mean,
            standard_deviation=sd,
            quadrature_nodes=quadrature_nodes,
        )
        value = np.where(grid <= boundary, payoff, continuation)
    return discount * point_transition_expectation(
        value,
        grid,
        point=0.0,
        mean_increment=mean,
        standard_deviation=sd,
        quadrature_nodes=quadrature_nodes,
    )


def price_threshold_policy_fft(
    solution: BermudanSolution,
    boundaries: np.ndarray,
    *,
    evaluation_nodes: int | None = None,
) -> float:
    """Independently reprice a fixed threshold policy by Gaussian convolution."""

    boundaries = np.asarray(boundaries, dtype=float)
    if boundaries.shape != solution.boundaries.shape:
        raise ValueError("boundary vector has the wrong shape")
    nodes = evaluation_nodes or solution.grid_nodes
    grid = np.linspace(solution.grid[0], solution.grid[-1], nodes)
    payoff = put_payoff(grid)
    market = solution.market
    dt = 1.0 / 8.0
    discount = np.exp(-market.rate * dt)
    mean = (
        market.rate
        - market.dividend_yield
        - 0.5 * market.volatility**2
    ) * dt
    sd = market.volatility * np.sqrt(dt)
    operator = GaussianDensityConvolver(
        grid, mean_increment=-mean, standard_deviation=sd
    )
    value = payoff.copy()
    for boundary in boundaries[::-1]:
        continuation = discount * operator(value)
        value = np.where(grid <= boundary, payoff, continuation)
    value0 = discount * operator(value)
    return float(np.interp(0.0, grid, value0))


def evaluate_policy_on_paths(
    market: MarketInput,
    spots: np.ndarray,
    exercise_rule: Callable[[int, np.ndarray], np.ndarray],
) -> np.ndarray:
    paths, exercise_dates = spots.shape
    alive = np.ones(paths, dtype=bool)
    discounted_payoff = np.zeros(paths)
    dt = 1.0 / exercise_dates
    for i in range(exercise_dates - 1):
        active = np.flatnonzero(alive)
        if active.size == 0:
            break
        log_spot = np.log(spots[active, i])
        exercise = np.asarray(exercise_rule(i, log_spot), dtype=bool)
        chosen = active[exercise]
        discounted_payoff[chosen] = np.exp(-market.rate * (i + 1) * dt) * np.maximum(
            1.0 - spots[chosen, i], 0.0
        )
        alive[chosen] = False
    active = np.flatnonzero(alive)
    discounted_payoff[active] = np.exp(-market.rate) * np.maximum(
        1.0 - spots[active, -1], 0.0
    )
    return discounted_payoff


def heldout_policy_loss(
    solution: BermudanSolution,
    policy: LSMCPolicy,
    *,
    paths: int,
    seed: int,
) -> tuple[float, float, float, float]:
    spots = simulate_black_scholes_paths(
        solution.market, paths=paths, seed=seed, exercise_dates=8
    )
    reference_payoff = evaluate_policy_on_paths(
        solution.market,
        spots,
        lambda i, x: x <= solution.boundaries[i],
    )
    approximate_payoff = evaluate_policy_on_paths(
        solution.market,
        spots,
        lambda i, x: lsmc_exercise_mask(x, policy.coefficients[i]),
    )
    difference = reference_payoff - approximate_payoff
    standard_error = float(np.std(difference, ddof=1) / np.sqrt(paths))
    return (
        float(np.mean(difference)),
        standard_error,
        float(np.mean(reference_payoff)),
        float(np.mean(approximate_payoff)),
    )


def log_log_slope(etas: Sequence[float], losses: Sequence[float]) -> float:
    etas = np.asarray(etas, dtype=float)
    losses = np.asarray(losses, dtype=float)
    mask = (etas <= 0.25) & (losses > 0.0)
    if mask.sum() < 2:
        return np.nan
    slope, _ = np.polyfit(np.log(etas[mask]), np.log(losses[mask]), 1)
    return float(slope)
