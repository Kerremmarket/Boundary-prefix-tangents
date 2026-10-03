"""Deterministic globally capped cliquet pilot primitives.

The module deliberately contains no data selection or outcome-dependent
contract tuning.  It implements the frozen one-dimensional contract, two
independent deterministic discretizations, and the ordered-prefix coefficient.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial, log
from typing import Iterable

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import brentq
from scipy.signal import fftconvolve
from scipy.stats import norm


@dataclass(frozen=True)
class LognormalComponent:
    weight: float
    gross_mean: float
    volatility: float

    def validate(self) -> None:
        if not np.isfinite(self.weight) or self.weight <= 0:
            raise ValueError("component weight must be positive")
        if not np.isfinite(self.gross_mean) or self.gross_mean <= 0:
            raise ValueError("component gross mean must be positive")
        if not np.isfinite(self.volatility) or self.volatility <= 0:
            raise ValueError("component volatility must be positive")

    @property
    def log_mean(self) -> float:
        return log(self.gross_mean) - 0.5 * self.volatility**2


@dataclass(frozen=True)
class CreditLaw:
    """Law of ``C=min(c,max(Z-1,0))`` for a lognormal mixture ``Z``."""

    components: tuple[LognormalComponent, ...]
    local_cap: float

    def validate(self) -> None:
        if not self.components:
            raise ValueError("at least one lognormal component is required")
        for component in self.components:
            component.validate()
        if not np.isclose(
            sum(component.weight for component in self.components),
            1.0,
            atol=1e-13,
            rtol=0,
        ):
            raise ValueError("component weights must sum to one")
        if not np.isfinite(self.local_cap) or self.local_cap <= 0:
            raise ValueError("local cap must be positive")

    def gross_cdf(self, value) -> np.ndarray:
        self.validate()
        values = np.asarray(value, dtype=float)
        result = np.zeros_like(values)
        positive = values > 0
        if np.any(positive):
            x = values[positive]
            for component in self.components:
                z = (np.log(x) - component.log_mean) / component.volatility
                result[positive] += component.weight * norm.cdf(z)
        return result

    def gross_density(self, value) -> np.ndarray:
        self.validate()
        values = np.asarray(value, dtype=float)
        result = np.zeros_like(values)
        positive = values > 0
        if np.any(positive):
            x = values[positive]
            for component in self.components:
                z = (np.log(x) - component.log_mean) / component.volatility
                result[positive] += (
                    component.weight
                    * norm.pdf(z)
                    / (x * component.volatility)
                )
        return result

    @property
    def floor_probability(self) -> float:
        return float(self.gross_cdf(np.array([1.0]))[0])

    @property
    def cap_probability(self) -> float:
        return float(
            1.0 - self.gross_cdf(np.array([1.0 + self.local_cap]))[0]
        )

    @property
    def interior_probability(self) -> float:
        return 1.0 - self.floor_probability - self.cap_probability

    def credit_density(self, value) -> np.ndarray:
        values = np.asarray(value, dtype=float)
        result = np.zeros_like(values)
        interior = (values > 0) & (values < self.local_cap)
        if np.any(interior):
            result[interior] = self.gross_density(1.0 + values[interior])
        return result

    def credit_density_trace(self, value) -> np.ndarray:
        """Interior density with its finite endpoint traces retained."""

        values = np.asarray(value, dtype=float)
        result = np.zeros_like(values)
        closed = (values >= 0) & (values <= self.local_cap)
        if np.any(closed):
            result[closed] = self.gross_density(1.0 + values[closed])
        return result

    def probability_credit_less(self, level: float) -> float:
        """Return ``P(C<level)`` with the cap atom treated exactly."""

        if level <= 0:
            return 0.0
        if level < self.local_cap:
            return float(self.gross_cdf(np.array([1.0 + level]))[0])
        if np.isclose(level, self.local_cap, atol=1e-14, rtol=0):
            return 1.0 - self.cap_probability
        return 1.0

    def probability_positive_leq(self, level: float) -> float:
        """Return ``P(0<C<=level)`` with both credit atoms explicit."""

        if level <= 0:
            return 0.0
        if level < self.local_cap:
            return float(
                self.gross_cdf(np.array([1.0 + level]))[0]
                - self.floor_probability
            )
        return 1.0 - self.floor_probability

    @staticmethod
    def _truncated_first_moment(
        component: LognormalComponent, upper: float
    ) -> float:
        if upper <= 0:
            return 0.0
        z = (
            np.log(upper)
            - component.log_mean
            - component.volatility**2
        ) / component.volatility
        return component.gross_mean * float(norm.cdf(z))

    def expected_min_credit(self, level: float) -> float:
        """Analytic ``E[min(C,level)]`` for ``level>=0``."""

        if level <= 0:
            return 0.0
        upper_credit = min(float(level), self.local_cap)
        upper_gross = 1.0 + upper_credit
        total = 0.0
        for component in self.components:
            p_one = float(
                norm.cdf(
                    (0.0 - component.log_mean) / component.volatility
                )
            )
            p_upper = float(
                norm.cdf(
                    (np.log(upper_gross) - component.log_mean)
                    / component.volatility
                )
            )
            first_one = self._truncated_first_moment(component, 1.0)
            first_upper = self._truncated_first_moment(
                component, upper_gross
            )
            interior = (first_upper - first_one) - (p_upper - p_one)
            tail = upper_credit * (1.0 - p_upper)
            total += component.weight * (interior + tail)
        return float(total)

    @property
    def expected_credit(self) -> float:
        return self.expected_min_credit(self.local_cap)

    def quadrature(self, nodes_per_component: int) -> tuple[np.ndarray, np.ndarray]:
        """Exact atoms plus mass-matched Gauss--Legendre interior nodes."""

        if nodes_per_component < 16:
            raise ValueError("at least 16 nodes per component are required")
        base_nodes, base_weights = leggauss(nodes_per_component)
        credits: list[np.ndarray] = [np.array([0.0])]
        weights: list[np.ndarray] = [np.array([self.floor_probability])]
        for component in self.components:
            lower_z = (0.0 - component.log_mean) / component.volatility
            upper_z = (
                np.log(1.0 + self.local_cap) - component.log_mean
            ) / component.volatility
            z = 0.5 * (upper_z - lower_z) * base_nodes + 0.5 * (
                upper_z + lower_z
            )
            raw_weights = (
                component.weight
                * 0.5
                * (upper_z - lower_z)
                * base_weights
                * norm.pdf(z)
            )
            exact_mass = component.weight * (
                norm.cdf(upper_z) - norm.cdf(lower_z)
            )
            raw_weights *= exact_mass / raw_weights.sum()
            gross = np.exp(component.log_mean + component.volatility * z)
            credits.append(gross - 1.0)
            weights.append(raw_weights)
        credits.append(np.array([self.local_cap]))
        weights.append(np.array([self.cap_probability]))
        credit_array = np.concatenate(credits)
        weight_array = np.concatenate(weights)
        order = np.argsort(credit_array, kind="stable")
        credit_array = credit_array[order]
        weight_array = weight_array[order]
        if not np.isclose(weight_array.sum(), 1.0, atol=2e-13, rtol=0):
            raise ArithmeticError("quadrature law does not preserve total mass")
        return credit_array, weight_array

    def lattice_mass(self, intervals_per_cap: int) -> tuple[np.ndarray, float]:
        """Mass-preserving lattice law with exact floor and cap atoms.

        The continuous interior is assigned to the nearest strictly interior
        lattice node using exact CDF bin masses.  The two edge half-cells are
        folded into the nearest interior node, so neither the floor nor cap
        atom is contaminated by continuous mass.  Thus the requested number
        of intervals is actually used by the independent lattice evaluator.
        """

        if intervals_per_cap < 128 or intervals_per_cap % 2:
            raise ValueError("intervals_per_cap must be an even integer >=128")
        step = self.local_cap / intervals_per_cap
        result = np.zeros(intervals_per_cap + 1)
        result[0] = self.floor_probability
        result[-1] = self.cap_probability
        for index in range(1, intervals_per_cap):
            lower = 0.0 if index == 1 else (index - 0.5) * step
            upper = (
                self.local_cap
                if index == intervals_per_cap - 1
                else (index + 0.5) * step
            )
            mass = float(
                self.gross_cdf(np.array([1.0 + upper]))[0]
                - self.gross_cdf(np.array([1.0 + lower]))[0]
            )
            result[index] = mass
        if not np.isclose(result.sum(), 1.0, atol=2e-13, rtol=0):
            raise ArithmeticError("lattice law does not preserve total mass")
        return result, step


@dataclass(frozen=True)
class Contract:
    local_cap: float
    global_cap: float
    resets: int
    beta: float

    def validate(self) -> None:
        if self.local_cap <= 0 or self.global_cap <= self.local_cap:
            raise ValueError("caps must satisfy 0<c<G")
        if self.resets < 3:
            raise ValueError("at least three resets are required")
        if self.beta <= 0 or not np.isfinite(self.beta):
            raise ValueError("beta must be finite and positive")
        ratio = self.global_cap / self.local_cap
        if not np.isclose(ratio, round(ratio), atol=1e-12, rtol=0):
            raise ValueError("the deterministic lattice requires integer G/c")

    @property
    def exercise_dates(self) -> int:
        return self.resets - 1


@dataclass(frozen=True)
class BoundaryResult:
    status: str
    boundary: float | None
    residual: float | None


def solve_boundary(
    law: CreditLaw,
    contract: Contract,
    *,
    absolute_tolerance: float = 1e-14,
) -> BoundaryResult:
    """Solve the proven common finite-horizon threshold equation."""

    law.validate()
    contract.validate()
    if law.expected_credit <= 1e-16:
        return BoundaryResult("degenerate_zero_credit_no_interior_boundary", None, None)
    if contract.beta >= 1.0:
        return BoundaryResult("beta_at_least_one_no_interior_boundary", None, None)

    def residual(boundary: float) -> float:
        return (
            (1.0 - contract.beta) * boundary
            - contract.beta
            * law.expected_min_credit(contract.global_cap - boundary)
        )

    boundary = float(
        brentq(
            residual,
            0.0,
            contract.global_cap,
            xtol=absolute_tolerance,
            rtol=4 * np.finfo(float).eps,
        )
    )
    return BoundaryResult("regular_interior_boundary", boundary, residual(boundary))


@dataclass(frozen=True)
class SlopeResult:
    right: float
    left_by_remaining_resets: dict[int, float]
    value_left_derivative: dict[int, float]
    cap_contact: float


def one_sided_slopes(
    law: CreditLaw,
    contract: Contract,
    boundary: float,
) -> SlopeResult:
    cap_contact = contract.global_cap - boundary
    p0 = law.floor_probability
    q = law.probability_positive_leq(cap_contact)
    right = 1.0 - contract.beta * law.probability_credit_less(cap_contact)
    derivatives = {0: 1.0}
    left: dict[int, float] = {}
    for remaining in range(1, contract.resets):
        derivatives[remaining] = contract.beta * (
            q + p0 * derivatives[remaining - 1]
        )
        left[remaining] = 1.0 - derivatives[remaining]
    if right <= 0 or any(value <= 0 for value in left.values()):
        raise ArithmeticError("the claimed regular one-sided slope is not positive")
    return SlopeResult(right, left, derivatives, cap_contact)


def _grid(contract: Contract, intervals_per_cap: int) -> tuple[np.ndarray, float]:
    contract.validate()
    total_intervals = int(
        round(contract.global_cap / contract.local_cap) * intervals_per_cap
    )
    grid = np.linspace(0.0, contract.global_cap, total_intervals + 1)
    return grid, contract.local_cap / intervals_per_cap


def _transition_lattice(values: np.ndarray, credit_mass: np.ndarray) -> np.ndarray:
    """Compute ``sum_k w_k V_(i+k)`` with a constant capped tail."""

    values = np.asarray(values, dtype=float)
    support = len(credit_mass) - 1
    tail_shape = values.shape[:-1] + (support,)
    tail = np.broadcast_to(values[..., -1:], tail_shape)
    extended = np.concatenate((values, tail), axis=-1)
    kernel_shape = (1,) * (values.ndim - 1) + (len(credit_mass),)
    convolution = fftconvolve(
        extended,
        credit_mass[::-1].reshape(kernel_shape),
        mode="full",
        axes=-1,
    )
    start = support
    return convolution[..., start : start + values.shape[-1]]


def _advance_mass_lattice(
    survivor_mass: np.ndarray,
    credit_mass: np.ndarray,
) -> np.ndarray:
    survivor_mass = np.asarray(survivor_mass, dtype=float)
    kernel_shape = (1,) * (survivor_mass.ndim - 1) + (len(credit_mass),)
    raw = fftconvolve(
        survivor_mass,
        credit_mass.reshape(kernel_shape),
        mode="full",
        axes=-1,
    )
    state_points = survivor_mass.shape[-1]
    result = raw[..., :state_points].copy()
    result[..., -1] = raw[..., state_points - 1 :].sum(axis=-1)
    tiny_negative = (result < 0) & (result > -2e-14)
    result[tiny_negative] = 0.0
    return result


@dataclass
class LatticeSolution:
    grid: np.ndarray
    credit_mass: np.ndarray
    optimal_issue_value: float
    optimal_values_date1: np.ndarray
    optimal_gaps: dict[int, np.ndarray]
    grid_boundaries: dict[int, float]
    policy_issue_values: np.ndarray | None
    european_issue_value: float


def _crossing(grid: np.ndarray, gap: np.ndarray) -> float:
    indices = np.flatnonzero((gap[:-1] < 0) & (gap[1:] >= 0))
    if len(indices) != 1:
        raise ArithmeticError(f"expected one negative-to-positive root, found {len(indices)}")
    index = int(indices[0])
    weight = -gap[index] / (gap[index + 1] - gap[index])
    return float(grid[index] + weight * (grid[index + 1] - grid[index]))


def solve_lattice(
    law: CreditLaw,
    contract: Contract,
    *,
    intervals_per_cap: int,
    policy_thresholds: np.ndarray | None = None,
) -> LatticeSolution:
    """Mass-lattice Bellman solution and optional fixed-policy batch."""

    if not np.isclose(law.local_cap, contract.local_cap, atol=1e-15, rtol=0):
        raise ValueError("law and contract local caps differ")
    grid, _ = _grid(contract, intervals_per_cap)
    credit_mass, _ = law.lattice_mass(intervals_per_cap)

    next_optimal = grid.copy()
    next_european = grid.copy()
    gaps: dict[int, np.ndarray] = {}
    boundaries: dict[int, float] = {}
    for date in range(contract.exercise_dates, 0, -1):
        continuation = contract.beta * _transition_lattice(
            next_optimal, credit_mass
        )
        gap = grid - continuation
        gaps[date] = gap
        boundaries[date] = _crossing(grid, gap)
        next_optimal = np.maximum(grid, continuation)
        next_european = contract.beta * _transition_lattice(
            next_european, credit_mass
        )

    optimal_issue = float(
        contract.beta * np.dot(credit_mass, next_optimal[: len(credit_mass)])
    )
    european_issue = float(
        contract.beta * np.dot(credit_mass, next_european[: len(credit_mass)])
    )

    policy_values: np.ndarray | None = None
    if policy_thresholds is not None:
        thresholds = np.asarray(policy_thresholds, dtype=float)
        if thresholds.ndim == 1:
            thresholds = thresholds[None, :]
        if thresholds.shape[1] != contract.exercise_dates:
            raise ValueError("one policy threshold is required per exercise date")
        next_policy = np.broadcast_to(
            grid, (thresholds.shape[0], len(grid))
        ).copy()
        for date in range(contract.exercise_dates, 0, -1):
            continuation = contract.beta * _transition_lattice(
                next_policy, credit_mass
            )
            stop = grid[None, :] >= thresholds[:, date - 1, None]
            next_policy = np.where(stop, grid[None, :], continuation)
        policy_values = contract.beta * (
            next_policy[:, : len(credit_mass)] @ credit_mass
        )

    return LatticeSolution(
        grid=grid,
        credit_mass=credit_mass,
        optimal_issue_value=optimal_issue,
        optimal_values_date1=next_optimal,
        optimal_gaps=gaps,
        grid_boundaries=boundaries,
        policy_issue_values=policy_values,
        european_issue_value=european_issue,
    )


@dataclass
class ForwardResult:
    total_losses: np.ndarray
    date_losses: np.ndarray
    predecision_mass: np.ndarray
    survival_mass: np.ndarray
    mismatch_mass: np.ndarray


def forward_performance_difference(
    solution: LatticeSolution,
    contract: Contract,
    *,
    optimal_boundary: float,
    policy_thresholds: np.ndarray,
) -> ForwardResult:
    """Forward occupancy evaluation of the exact performance-difference sum."""

    thresholds = np.asarray(policy_thresholds, dtype=float)
    if thresholds.ndim == 1:
        thresholds = thresholds[None, :]
    policy_count, dates = thresholds.shape
    if dates != contract.exercise_dates:
        raise ValueError("threshold matrix has the wrong number of dates")
    if not 0 < optimal_boundary < contract.global_cap:
        raise ValueError("a regular analytic boundary is required")
    state_points = len(solution.grid)
    survivor = np.zeros((policy_count, state_points))
    survivor[:, 0] = 1.0
    date_losses = np.zeros((policy_count, dates))
    predecision_mass = np.zeros((policy_count, dates))
    survival_mass = np.zeros((policy_count, dates))
    mismatch_mass = np.zeros((policy_count, dates))
    for date in range(1, dates + 1):
        predecision = _advance_mass_lattice(survivor, solution.credit_mass)
        policy_stop = (
            solution.grid[None, :] >= thresholds[:, date - 1, None]
        )
        # Use the action selected by this numerical Bellman solve for the exact
        # discrete performance-difference identity.  The separately reported
        # grid-boundary audit checks it against ``optimal_boundary``.
        optimal_stop = solution.optimal_gaps[date] >= 0
        mismatch = policy_stop != optimal_stop[None, :]
        weighted_gap = np.abs(solution.optimal_gaps[date])[None, :]
        date_losses[:, date - 1] = (
            contract.beta**date
            * np.sum(predecision * mismatch * weighted_gap, axis=1)
        )
        predecision_mass[:, date - 1] = predecision.sum(axis=1)
        mismatch_mass[:, date - 1] = np.sum(predecision * mismatch, axis=1)
        survivor = np.where(policy_stop, 0.0, predecision)
        survival_mass[:, date - 1] = survivor.sum(axis=1)

    return ForwardResult(
        total_losses=date_losses.sum(axis=1),
        date_losses=date_losses,
        predecision_mass=predecision_mass,
        survival_mass=survival_mass,
        mismatch_mass=mismatch_mass,
    )


def _transition_quadrature(
    values: np.ndarray,
    grid: np.ndarray,
    credits: np.ndarray,
    weights: np.ndarray,
    global_cap: float,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    result = np.zeros_like(values)
    for credit, weight in zip(credits, weights):
        query = np.minimum(grid + credit, global_cap)
        if values.ndim == 1:
            interpolated = np.interp(query, grid, values)
        else:
            step = grid[1] - grid[0]
            coordinate = query / step
            lower = np.floor(coordinate).astype(int)
            lower = np.minimum(lower, len(grid) - 1)
            upper = np.minimum(lower + 1, len(grid) - 1)
            fraction = coordinate - lower
            interpolated = (
                values[:, lower] * (1.0 - fraction)[None, :]
                + values[:, upper] * fraction[None, :]
            )
        result += weight * interpolated
    return result


@dataclass
class QuadratureSolution:
    grid: np.ndarray
    credits: np.ndarray
    weights: np.ndarray
    optimal_issue_value: float
    policy_issue_values: np.ndarray | None
    european_issue_value: float
    grid_boundaries: dict[int, float]
    optimal_gaps: dict[int, np.ndarray]


def solve_quadrature(
    law: CreditLaw,
    contract: Contract,
    *,
    intervals_per_cap: int,
    nodes_per_component: int,
    policy_thresholds: np.ndarray | None = None,
) -> QuadratureSolution:
    """Independent continuous-state interpolation/Gauss quadrature evaluator."""

    grid, _ = _grid(contract, intervals_per_cap)
    credits, weights = law.quadrature(nodes_per_component)
    next_optimal = grid.copy()
    next_european = grid.copy()
    boundaries: dict[int, float] = {}
    gaps: dict[int, np.ndarray] = {}
    for date in range(contract.exercise_dates, 0, -1):
        continuation = contract.beta * _transition_quadrature(
            next_optimal, grid, credits, weights, contract.global_cap
        )
        gap = grid - continuation
        gaps[date] = gap
        boundaries[date] = _crossing(grid, gap)
        next_optimal = np.maximum(grid, continuation)
        next_european = contract.beta * _transition_quadrature(
            next_european, grid, credits, weights, contract.global_cap
        )
    issue_query = np.minimum(credits, contract.global_cap)
    optimal_issue = float(
        contract.beta * np.dot(weights, np.interp(issue_query, grid, next_optimal))
    )
    european_issue = float(
        contract.beta * np.dot(weights, np.interp(issue_query, grid, next_european))
    )

    policy_values: np.ndarray | None = None
    if policy_thresholds is not None:
        thresholds = np.asarray(policy_thresholds, dtype=float)
        if thresholds.ndim == 1:
            thresholds = thresholds[None, :]
        next_policy = np.broadcast_to(
            grid, (thresholds.shape[0], len(grid))
        ).copy()
        for date in range(contract.exercise_dates, 0, -1):
            continuation = contract.beta * _transition_quadrature(
                next_policy, grid, credits, weights, contract.global_cap
            )
            stop = grid[None, :] >= thresholds[:, date - 1, None]
            next_policy = np.where(stop, grid[None, :], continuation)
        policy_values = np.empty(thresholds.shape[0])
        for index, values in enumerate(next_policy):
            policy_values[index] = contract.beta * np.dot(
                weights, np.interp(issue_query, grid, values)
            )

    return QuadratureSolution(
        grid=grid,
        credits=credits,
        weights=weights,
        optimal_issue_value=optimal_issue,
        policy_issue_values=policy_values,
        european_issue_value=european_issue,
        grid_boundaries=boundaries,
        optimal_gaps=gaps,
    )


def quadrature_policy_regret(
    solution: QuadratureSolution,
    contract: Contract,
    *,
    policy_thresholds: np.ndarray,
) -> np.ndarray:
    """Direct backward regret recursion under the quadrature discretization.

    This evaluates loss itself instead of subtracting two nearly equal issue
    prices.  It is therefore the preferred independent check near the
    numerical floor; the price-difference calculation remains an explicit
    cancellation diagnostic.
    """

    thresholds = np.asarray(policy_thresholds, dtype=float)
    if thresholds.ndim == 1:
        thresholds = thresholds[None, :]
    if thresholds.shape[1] != contract.exercise_dates:
        raise ValueError("threshold matrix has the wrong number of dates")
    loss_next = np.zeros((thresholds.shape[0], len(solution.grid)))
    for date in range(contract.exercise_dates, 0, -1):
        future_loss = contract.beta * _transition_quadrature(
            loss_next,
            solution.grid,
            solution.credits,
            solution.weights,
            contract.global_cap,
        )
        gap = solution.optimal_gaps[date][None, :]
        stop = solution.grid[None, :] >= thresholds[:, date - 1, None]
        stop_loss = np.maximum(-gap, 0.0)
        continue_loss = np.maximum(gap, 0.0) + future_loss
        loss_next = np.where(stop, stop_loss, continue_loss)

    issue_query = np.minimum(solution.credits, contract.global_cap)
    result = np.empty(thresholds.shape[0])
    for index, loss in enumerate(loss_next):
        result[index] = contract.beta * np.dot(
            solution.weights,
            np.interp(issue_query, solution.grid, loss),
        )
    return result


@dataclass
class TraceResult:
    grid: np.ndarray
    total_density: dict[int, float]
    fresh_density: dict[int, float]
    combinatorial_density: dict[int, float]
    component_reconciliation_error: dict[int, float]
    density_arrays: dict[int, np.ndarray]


def trace_densities(
    law: CreditLaw,
    contract: Contract,
    *,
    boundary: float,
    intervals_per_cap: int,
) -> TraceResult:
    """Continuous boundary traces by recurrence and component enumeration."""

    grid, step = _grid(contract, intervals_per_cap)
    cap_shift = intervals_per_cap
    density = law.credit_density_trace(grid)
    density[grid > law.local_cap] = 0.0
    density_weights = density.copy()
    density_weights[0] *= 0.5
    density_weights[cap_shift] *= 0.5
    p0 = law.floor_probability
    pcap = law.cap_probability

    previous = np.zeros_like(grid)
    atoms = np.array([1.0])
    arrays: dict[int, np.ndarray] = {}
    total: dict[int, float] = {}
    fresh: dict[int, float] = {}

    for date in range(1, contract.exercise_dates + 1):
        convolution = step * fftconvolve(previous, density_weights)[: len(grid)]
        current = p0 * previous + convolution
        if cap_shift < len(grid):
            current[cap_shift:] += pcap * previous[:-cap_shift]
        for cap_count, atom_mass in enumerate(atoms):
            shift = cap_count * cap_shift
            if shift >= len(grid):
                continue
            current[shift:] += atom_mass * density[: len(grid) - shift]
        arrays[date] = current
        total[date] = float(np.interp(boundary, grid, current))
        previous_at_boundary = float(np.interp(boundary, grid, previous))
        fresh[date] = total[date] - p0 * previous_at_boundary
        previous = current
        atoms = np.convolve(atoms, np.array([p0, pcap]))

    powers: dict[int, np.ndarray] = {1: density}
    for count in range(2, contract.exercise_dates + 1):
        powers[count] = step * fftconvolve(
            powers[count - 1], density_weights
        )[: len(grid)]

    combinatorial: dict[int, float] = {}
    for date in range(1, contract.exercise_dates + 1):
        value = 0.0
        for continuous_count in range(1, date + 1):
            remainder = date - continuous_count
            for cap_count in range(remainder + 1):
                zero_count = remainder - cap_count
                argument = boundary - cap_count * law.local_cap
                if argument < 0 or argument > contract.global_cap:
                    continue
                coefficient = (
                    factorial(date)
                    / (
                        factorial(continuous_count)
                        * factorial(cap_count)
                        * factorial(zero_count)
                    )
                    * pcap**cap_count
                    * p0**zero_count
                )
                value += coefficient * float(
                    np.interp(argument, grid, powers[continuous_count])
                )
        combinatorial[date] = value

    reconciliation: dict[int, float] = {}
    for target in range(1, contract.exercise_dates + 1):
        reconstructed = sum(
            fresh[source] * p0 ** (target - source)
            for source in range(1, target + 1)
        )
        reconciliation[target] = max(
            abs(reconstructed - total[target]),
            abs(combinatorial[target] - total[target]),
        )

    return TraceResult(
        grid=grid,
        total_density=total,
        fresh_density=fresh,
        combinatorial_density=combinatorial,
        component_reconciliation_error=reconciliation,
        density_arrays=arrays,
    )


def gate_integral(direction: np.ndarray, source: int, target: int) -> float:
    """Exact scalar-ray gate integral, with zero-based date indices."""

    h = np.asarray(direction, dtype=float)
    target_value = float(h[target])
    if target_value > 0:
        upper = max(0.0, float(np.min(h[source : target + 1])))
        return 0.5 * upper**2
    if target_value < 0:
        earlier_min = (
            float(np.min(h[source:target])) if source < target else np.inf
        )
        upper = min(0.0, earlier_min)
        if upper <= target_value:
            return 0.0
        return 0.5 * (target_value**2 - upper**2)
    return 0.0


@dataclass(frozen=True)
class CoefficientResult:
    full: float
    additive_single_date: float
    fresh_diagonal: float
    copied: float
    components: tuple[dict[str, float | int | str], ...]


def prefix_coefficient(
    *,
    direction: Iterable[float],
    contract: Contract,
    law: CreditLaw,
    trace: TraceResult,
    slopes: SlopeResult,
) -> CoefficientResult:
    h = np.asarray(tuple(direction), dtype=float)
    if h.shape != (contract.exercise_dates,):
        raise ValueError("direction has the wrong number of exercise dates")
    p0 = law.floor_probability
    full = 0.0
    fresh_diagonal = 0.0
    components: list[dict[str, float | int | str]] = []
    for target0 in range(contract.exercise_dates):
        target = target0 + 1
        remaining = contract.resets - target
        target_slope = (
            slopes.right
            if h[target0] > 0
            else slopes.left_by_remaining_resets[remaining]
        )
        for source0 in range(target0 + 1):
            source = source0 + 1
            intensity = trace.fresh_density[source] * p0 ** (target - source)
            integral = gate_integral(h, source0, target0)
            contribution = (
                contract.beta**target * target_slope * intensity * integral
            )
            full += contribution
            if source == target:
                fresh_diagonal += contribution
            components.append(
                {
                    "source_date": source,
                    "target_date": target,
                    "history": "fresh" if source == target else "copied_zero_suffix",
                    "zero_suffix_length": target - source,
                    "trace_intensity": intensity,
                    "gate_integral": integral,
                    "target_slope": target_slope,
                    "contribution": contribution,
                }
            )

    additive = 0.0
    for target0, target_value in enumerate(h):
        if target_value == 0:
            continue
        isolated = np.zeros_like(h)
        isolated[target0] = target_value
        for source0 in range(target0 + 1):
            target = target0 + 1
            source = source0 + 1
            remaining = contract.resets - target
            target_slope = (
                slopes.right
                if target_value > 0
                else slopes.left_by_remaining_resets[remaining]
            )
            intensity = trace.fresh_density[source] * p0 ** (target - source)
            additive += (
                contract.beta**target
                * target_slope
                * intensity
                * gate_integral(isolated, source0, target0)
            )

    return CoefficientResult(
        full=float(full),
        additive_single_date=float(additive),
        fresh_diagonal=float(fresh_diagonal),
        copied=float(full - fresh_diagonal),
        components=tuple(components),
    )


def make_directions(exercise_dates: int) -> dict[str, np.ndarray]:
    if exercise_dates < 2:
        raise ValueError("at least two exercise dates are required")
    directions: dict[str, np.ndarray] = {
        "all_up": np.ones(exercise_dates),
        "all_down": -np.ones(exercise_dates),
        "ramp_up": np.linspace(1.0 / exercise_dates, 1.0, exercise_dates),
        "ramp_down": -np.linspace(1.0 / exercise_dates, 1.0, exercise_dates),
        "alternating_up_down": np.where(
            np.arange(exercise_dates) % 2 == 0, 1.0, -1.0
        ),
    }
    for date in range(1, exercise_dates + 1):
        for sign, label in ((1.0, "up"), (-1.0, "down")):
            value = np.zeros(exercise_dates)
            value[date - 1] = sign
            directions[f"single_{label}_t{date}"] = value
    return directions


def build_law(
    *,
    model: str,
    gross_forward: float,
    atm_volatility: float,
    local_cap: float,
    mixture_parameters: dict[str, float] | None = None,
) -> CreditLaw:
    if model == "black_scholes_atm":
        components = (
            LognormalComponent(1.0, gross_forward, atm_volatility),
        )
    elif model == "annual_lognormal_mixture":
        if mixture_parameters is None:
            raise ValueError("mixture parameters are required")
        low_weight = float(mixture_parameters["low_weight"])
        components = (
            LognormalComponent(
                low_weight,
                gross_forward
                * float(mixture_parameters["low_forward_multiplier"]),
                float(mixture_parameters["low_volatility"]),
            ),
            LognormalComponent(
                1.0 - low_weight,
                gross_forward
                * float(mixture_parameters["high_forward_multiplier"]),
                float(mixture_parameters["high_volatility"]),
            ),
        )
    else:
        raise ValueError(f"unknown model {model!r}")
    law = CreditLaw(components=components, local_cap=local_cap)
    law.validate()
    return law
