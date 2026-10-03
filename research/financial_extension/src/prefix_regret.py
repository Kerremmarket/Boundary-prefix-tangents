"""Two-date actual, diagonal, and copied-prefix regret calculations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.stats import lognorm

from indexed_annuity import (
    BellmanSolution,
    ContractSpec,
    DiscreteCreditedLaw,
    MixtureCreditedDistribution,
)


Density = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class CoefficientBreakdown:
    date1_diagonal: float
    date2_diagonal: float
    copied_prefix: float

    @property
    def diagonal(self) -> float:
        return self.date1_diagonal + self.date2_diagonal

    @property
    def prefix(self) -> float:
        return self.diagonal + self.copied_prefix


@dataclass(frozen=True)
class RegretBreakdown:
    date1: float
    date2: float

    @property
    def total(self) -> float:
        return self.date1 + self.date2


def lognormal_source_density(*, median: float, log_sigma: float) -> Density:
    if median <= 0 or log_sigma <= 0:
        raise ValueError("median and log_sigma must be positive")

    def density(state: np.ndarray) -> np.ndarray:
        return lognorm.pdf(np.asarray(state, dtype=float), s=log_sigma, scale=median)

    return density


def _gauss_integral(
    function: Callable[[np.ndarray], np.ndarray],
    lower: float,
    upper: float,
    nodes: int,
) -> float:
    if upper <= lower:
        return 0.0
    points, weights = leggauss(nodes)
    states = 0.5 * (upper - lower) * points + 0.5 * (upper + lower)
    values = np.asarray(function(states), dtype=float)
    return float(0.5 * (upper - lower) * np.dot(weights, values))


def _gap_interpolator(solution: BellmanSolution) -> Callable[[np.ndarray], np.ndarray]:
    def gap(state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=float)
        return np.interp(state, solution.grid, solution.gap)

    return gap


def two_date_coefficients(
    *,
    boundary: float,
    stopping_side_gap_slope: float,
    h1: float,
    h2: float,
    source_density: Density,
    law: DiscreteCreditedLaw,
    contract: ContractSpec,
) -> CoefficientBreakdown:
    """Compute the independent diagonal and identity-branch coefficients.

    The formula is for nonnegative upward shifts of one regular upper boundary.
    The target date is discounted one year and requires survival of termination.
    """

    if boundary <= 0 or stopping_side_gap_slope <= 0:
        raise ValueError("a positive regular upper boundary and slope are required")
    if h1 < 0 or h2 < 0:
        raise ValueError("this helper is restricted to nonnegative upward shifts")
    law.validate()
    contract.validate()
    source_at_boundary = float(np.asarray(source_density(np.array([boundary])))[0])
    if source_at_boundary <= 0 or not np.isfinite(source_at_boundary):
        raise ValueError("source density must be finite and positive at the boundary")

    # Under the reference earlier gate x<b, the identity factor cannot deliver
    # an upward target mismatch.  Every factor strictly above one contributes
    # to the ordinary target density by change of variables.
    target_density = 0.0
    for factor, weight in zip(law.factors, law.weights):
        if factor > 1.0 + 1e-14:
            target_density += weight * float(
                np.asarray(source_density(np.array([boundary / factor])))[0]
            ) / factor

    target_discount = contract.discount_factor * (
        1.0 - contract.termination_probability
    )
    date1 = (
        0.5 * stopping_side_gap_slope * source_at_boundary * h1**2
    )
    date2 = (
        0.5
        * target_discount
        * stopping_side_gap_slope
        * target_density
        * h2**2
    )
    copied = (
        0.5
        * target_discount
        * stopping_side_gap_slope
        * law.floor_probability
        * source_at_boundary
        * min(h1, h2) ** 2
    )
    return CoefficientBreakdown(
        date1_diagonal=float(date1),
        date2_diagonal=float(date2),
        copied_prefix=float(copied),
    )


def two_date_actual_regret(
    *,
    epsilon: float,
    boundary: float,
    h1: float,
    h2: float,
    source_density: Density,
    solution: BellmanSolution,
    law: DiscreteCreditedLaw,
    contract: ContractSpec,
    integration_nodes: int = 64,
) -> RegretBreakdown:
    """Evaluate the exact two-date latent-path regret for upward shifts.

    Date two uses the same stationary Bellman gap and the exact stationary value
    as its tail.  The calculation does not re-optimize after perturbing a gate.
    """

    if epsilon <= 0 or h1 < 0 or h2 < 0:
        raise ValueError("epsilon must be positive and shifts nonnegative")
    if integration_nodes < 16:
        raise ValueError("integration_nodes must be at least 16")
    gap = _gap_interpolator(solution)

    date1 = _gauss_integral(
        lambda state: source_density(state) * np.abs(gap(state)),
        boundary,
        boundary + epsilon * h1,
        integration_nodes,
    )

    date2_undiscounted = 0.0
    for factor, weight in zip(law.factors, law.weights):
        lower = boundary / factor
        upper = min(
            (boundary + epsilon * h2) / factor,
            boundary + epsilon * h1,
        )
        date2_undiscounted += weight * _gauss_integral(
            lambda state, factor=factor: source_density(state)
            * np.abs(gap(state * factor)),
            lower,
            upper,
            integration_nodes,
        )
    date2 = (
        contract.discount_factor
        * (1.0 - contract.termination_probability)
        * date2_undiscounted
    )
    return RegretBreakdown(date1=float(date1), date2=float(date2))


def market_two_date_coefficients(
    *,
    boundary: float,
    stopping_side_gap_slope: float,
    h1: float,
    h2: float,
    initial_state: float,
    distribution: MixtureCreditedDistribution,
    contract: ContractSpec,
    integration_nodes: int = 96,
) -> CoefficientBreakdown:
    """Coefficients under the exact mixed source and transition law.

    The first exercise-date state is generated by one credited return from
    ``initial_state``.  Its floor and cap atoms are retained explicitly.  The
    ordinary target density includes the continuous images of those atoms and
    the cap branch, while the floor identity branch at the earlier boundary is
    reserved for the copied-prefix term.
    """

    if boundary <= 0 or stopping_side_gap_slope <= 0:
        raise ValueError("a positive regular upper boundary and slope are required")
    if h1 < 0 or h2 < 0:
        raise ValueError("this helper is restricted to nonnegative upward shifts")
    if integration_nodes < 16:
        raise ValueError("integration_nodes must be at least 16")
    distribution.validate()
    contract.validate()

    source_density = lambda state: distribution.state_continuous_density(
        state, initial_state
    )
    source_at_boundary = float(source_density(np.array([boundary]))[0])
    if source_at_boundary <= 0 or not np.isfinite(source_at_boundary):
        raise ValueError(
            "the first-date state must have positive continuous density at the boundary"
        )

    cap_factor = distribution.cap_factor
    source_lower = initial_state
    source_upper = initial_state * cap_factor
    lower = max(source_lower, boundary / cap_factor)
    upper = min(source_upper, boundary)
    continuous_target_density = _gauss_integral(
        lambda state: source_density(state)
        * distribution.factor_density(boundary / state)
        / state,
        lower,
        upper,
        integration_nodes,
    )

    atom_target_density = 0.0
    for atom_state, atom_mass in distribution.state_atoms(initial_state):
        if atom_state < boundary:
            atom_target_density += atom_mass * float(
                distribution.factor_density(np.array([boundary / atom_state]))[0]
            ) / atom_state

    cap_target_density = 0.0
    cap_source = boundary / cap_factor
    if cap_source < boundary:
        cap_target_density = (
            distribution.cap_probability
            * float(source_density(np.array([cap_source]))[0])
            / cap_factor
        )

    target_density = (
        continuous_target_density + atom_target_density + cap_target_density
    )
    initial_discount = contract.discount_factor * (
        1.0 - contract.termination_probability
    )
    target_discount = contract.discount_factor * (
        1.0 - contract.termination_probability
    )
    date1 = (
        initial_discount
        * 0.5
        * stopping_side_gap_slope
        * source_at_boundary
        * h1**2
    )
    date2 = (
        initial_discount
        * 0.5
        * target_discount
        * stopping_side_gap_slope
        * target_density
        * h2**2
    )
    copied = (
        initial_discount
        * 0.5
        * target_discount
        * stopping_side_gap_slope
        * distribution.floor_probability
        * source_at_boundary
        * min(h1, h2) ** 2
    )
    return CoefficientBreakdown(
        date1_diagonal=float(date1),
        date2_diagonal=float(date2),
        copied_prefix=float(copied),
    )


def market_two_date_actual_regret(
    *,
    epsilon: float,
    boundary: float,
    h1: float,
    h2: float,
    initial_state: float,
    distribution: MixtureCreditedDistribution,
    solution: BellmanSolution,
    contract: ContractSpec,
    integration_nodes: int = 64,
) -> RegretBreakdown:
    """Exact two-date regret under the fitted mixed credited-return law.

    The performance-difference decomposition uses occupancy under the perturbed
    upward gates.  Both source atoms and second-transition atoms are included;
    continuous--continuous transport is evaluated by nested Gauss quadrature.
    """

    if epsilon <= 0 or h1 < 0 or h2 < 0:
        raise ValueError("epsilon must be positive and shifts nonnegative")
    if integration_nodes < 16:
        raise ValueError("integration_nodes must be at least 16")
    distribution.validate()
    contract.validate()
    gap = _gap_interpolator(solution)
    source_density = lambda state: distribution.state_continuous_density(
        state, initial_state
    )
    source_atoms = distribution.state_atoms(initial_state)
    cap_factor = distribution.cap_factor
    source_lower = initial_state
    source_upper = initial_state * cap_factor
    gate1 = boundary + epsilon * h1
    target_upper = boundary + epsilon * h2

    date1 = _gauss_integral(
        lambda state: source_density(state) * np.abs(gap(state)),
        boundary,
        min(gate1, source_upper),
        integration_nodes,
    )
    for atom_state, atom_mass in source_atoms:
        if boundary <= atom_state <= gate1:
            date1 += atom_mass * float(np.abs(gap(np.array([atom_state])))[0])

    def continuous_target_density_at(target_state: np.ndarray) -> np.ndarray:
        targets = np.asarray(target_state, dtype=float)
        result = np.zeros_like(targets)
        for index, target in np.ndenumerate(targets):
            lower = max(source_lower, target / cap_factor)
            upper = min(source_upper, gate1, target)
            transported = _gauss_integral(
                lambda state, target=target: source_density(state)
                * distribution.factor_density(target / state)
                / state,
                lower,
                upper,
                integration_nodes,
            )
            for atom_state, atom_mass in source_atoms:
                if atom_state < gate1:
                    transported += atom_mass * float(
                        distribution.factor_density(
                            np.array([target / atom_state])
                        )[0]
                    ) / atom_state
            result[index] = transported
        return result

    continuous_branch = _gauss_integral(
        lambda target: np.abs(gap(target))
        * continuous_target_density_at(target),
        boundary,
        target_upper,
        integration_nodes,
    )

    floor_upper = min(target_upper, gate1, source_upper)
    floor_branch = distribution.floor_probability * _gauss_integral(
        lambda target: source_density(target) * np.abs(gap(target)),
        boundary,
        floor_upper,
        integration_nodes,
    )
    cap_branch = distribution.cap_probability * _gauss_integral(
        lambda target: np.where(
            target / cap_factor < gate1,
            source_density(target / cap_factor)
            * np.abs(gap(target))
            / cap_factor,
            0.0,
        ),
        boundary,
        target_upper,
        integration_nodes,
    )

    atomic_target_loss = 0.0
    for atom_state, source_mass in source_atoms:
        if atom_state >= gate1:
            continue
        for factor, transition_mass in (
            (1.0, distribution.floor_probability),
            (cap_factor, distribution.cap_probability),
        ):
            target = atom_state * factor
            if boundary <= target <= target_upper:
                atomic_target_loss += (
                    source_mass
                    * transition_mass
                    * float(np.abs(gap(np.array([target])))[0])
                )

    initial_discount = contract.discount_factor * (
        1.0 - contract.termination_probability
    )
    date1 *= initial_discount
    date2 = initial_discount * contract.discount_factor * (
        1.0 - contract.termination_probability
    ) * (
        continuous_branch + floor_branch + cap_branch + atomic_target_loss
    )
    return RegretBreakdown(date1=float(date1), date2=float(date2))
