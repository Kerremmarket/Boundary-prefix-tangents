"""Deterministic audit evaluators for the literal uncapped accumulator.

The frozen pilot evaluates the sufficient capped state ``X=min(A,G)``.  That
state is exact for the optimal policy and for every supplied threshold at or
below ``G``.  It is not sufficient for a supplied threshold above ``G``:
``A>=k`` can be reached even though ``X`` never exceeds ``G``.  This module
keeps ``A`` on ``[0,Nc]`` and uses the exercise payoff ``min(A,G)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve

from cliquet import Contract, CreditLaw


def uncapped_grid(
    contract: Contract, intervals_per_local_cap: int
) -> tuple[np.ndarray, float]:
    contract.validate()
    if intervals_per_local_cap < 128 or intervals_per_local_cap % 2:
        raise ValueError("intervals_per_local_cap must be an even integer >=128")
    total_intervals = contract.resets * intervals_per_local_cap
    step = contract.local_cap / intervals_per_local_cap
    return np.linspace(0.0, contract.resets * contract.local_cap,
                       total_intervals + 1), step


def payoff(grid: np.ndarray, global_cap: float) -> np.ndarray:
    return np.minimum(np.asarray(grid, dtype=float), global_cap)


def _transition_lattice(values: np.ndarray, credit_mass: np.ndarray) -> np.ndarray:
    """Apply the additive-credit kernel, with a constant far-right tail."""

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
    return convolution[..., support : support + values.shape[-1]]


def _advance_mass(survivor_mass: np.ndarray, credit_mass: np.ndarray) -> np.ndarray:
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


def _crossing(grid: np.ndarray, gap: np.ndarray) -> float:
    indices = np.flatnonzero((gap[:-1] < 0) & (gap[1:] >= 0))
    if len(indices) != 1:
        raise ArithmeticError(
            f"expected one negative-to-positive root, found {len(indices)}"
        )
    index = int(indices[0])
    weight = -gap[index] / (gap[index + 1] - gap[index])
    return float(grid[index] + weight * (grid[index + 1] - grid[index]))


@dataclass
class UncappedLatticeReference:
    grid: np.ndarray
    credit_mass: np.ndarray
    exercise_payoff: np.ndarray
    optimal_issue_value: float
    optimal_gaps: dict[int, np.ndarray]
    grid_boundaries: dict[int, float]


def solve_uncapped_lattice_reference(
    law: CreditLaw,
    contract: Contract,
    *,
    intervals_per_local_cap: int,
) -> UncappedLatticeReference:
    if not np.isclose(law.local_cap, contract.local_cap, atol=1e-15, rtol=0):
        raise ValueError("law and contract local caps differ")
    grid, _ = uncapped_grid(contract, intervals_per_local_cap)
    credit_mass, _ = law.lattice_mass(intervals_per_local_cap)
    exercise = payoff(grid, contract.global_cap)
    next_optimal = exercise.copy()
    gaps: dict[int, np.ndarray] = {}
    boundaries: dict[int, float] = {}
    for date in range(contract.exercise_dates, 0, -1):
        continuation = contract.beta * _transition_lattice(
            next_optimal, credit_mass
        )
        gap = exercise - continuation
        gaps[date] = gap
        boundaries[date] = _crossing(grid, gap)
        next_optimal = np.maximum(exercise, continuation)
    optimal_issue = float(
        contract.beta * np.dot(credit_mass, next_optimal[: len(credit_mass)])
    )
    return UncappedLatticeReference(
        grid=grid,
        credit_mass=credit_mass,
        exercise_payoff=exercise,
        optimal_issue_value=optimal_issue,
        optimal_gaps=gaps,
        grid_boundaries=boundaries,
    )


def lattice_policy_values(
    reference: UncappedLatticeReference,
    contract: Contract,
    policy_thresholds: np.ndarray,
) -> np.ndarray:
    thresholds = np.asarray(policy_thresholds, dtype=float)
    if thresholds.ndim == 1:
        thresholds = thresholds[None, :]
    if thresholds.shape[1] != contract.exercise_dates:
        raise ValueError("one policy threshold is required per exercise date")
    next_policy = np.broadcast_to(
        reference.exercise_payoff, (thresholds.shape[0], len(reference.grid))
    ).copy()
    for date in range(contract.exercise_dates, 0, -1):
        continuation = contract.beta * _transition_lattice(
            next_policy, reference.credit_mass
        )
        stop = reference.grid[None, :] >= thresholds[:, date - 1, None]
        next_policy = np.where(
            stop, reference.exercise_payoff[None, :], continuation
        )
    return contract.beta * (
        next_policy[:, : len(reference.credit_mass)] @ reference.credit_mass
    )


@dataclass
class UncappedForwardResult:
    total_losses: np.ndarray
    date_losses: np.ndarray
    predecision_mass: np.ndarray
    survival_mass: np.ndarray
    mismatch_mass: np.ndarray


def forward_performance_difference_uncapped(
    reference: UncappedLatticeReference,
    contract: Contract,
    *,
    policy_thresholds: np.ndarray,
) -> UncappedForwardResult:
    thresholds = np.asarray(policy_thresholds, dtype=float)
    if thresholds.ndim == 1:
        thresholds = thresholds[None, :]
    policy_count, dates = thresholds.shape
    if dates != contract.exercise_dates:
        raise ValueError("threshold matrix has the wrong number of dates")

    survivor = np.zeros((policy_count, len(reference.grid)))
    survivor[:, 0] = 1.0
    date_losses = np.zeros((policy_count, dates))
    predecision_mass = np.zeros((policy_count, dates))
    survival_mass = np.zeros((policy_count, dates))
    mismatch_mass = np.zeros((policy_count, dates))
    for date in range(1, dates + 1):
        predecision = _advance_mass(survivor, reference.credit_mass)
        policy_stop = reference.grid[None, :] >= thresholds[:, date - 1, None]
        optimal_stop = reference.optimal_gaps[date] >= 0
        mismatch = policy_stop != optimal_stop[None, :]
        date_losses[:, date - 1] = (
            contract.beta**date
            * np.sum(
                predecision
                * mismatch
                * np.abs(reference.optimal_gaps[date])[None, :],
                axis=1,
            )
        )
        predecision_mass[:, date - 1] = predecision.sum(axis=1)
        mismatch_mass[:, date - 1] = np.sum(predecision * mismatch, axis=1)
        survivor = np.where(policy_stop, 0.0, predecision)
        survival_mass[:, date - 1] = survivor.sum(axis=1)
    return UncappedForwardResult(
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
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    result = np.zeros_like(values)
    state_maximum = float(grid[-1])
    for credit, weight in zip(credits, weights):
        query = np.minimum(grid + credit, state_maximum)
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
class UncappedQuadratureReference:
    grid: np.ndarray
    credits: np.ndarray
    weights: np.ndarray
    exercise_payoff: np.ndarray
    optimal_issue_value: float
    optimal_gaps: dict[int, np.ndarray]
    grid_boundaries: dict[int, float]


def solve_uncapped_quadrature_reference(
    law: CreditLaw,
    contract: Contract,
    *,
    intervals_per_local_cap: int,
    nodes_per_component: int,
) -> UncappedQuadratureReference:
    grid, _ = uncapped_grid(contract, intervals_per_local_cap)
    credits, weights = law.quadrature(nodes_per_component)
    exercise = payoff(grid, contract.global_cap)
    next_optimal = exercise.copy()
    gaps: dict[int, np.ndarray] = {}
    boundaries: dict[int, float] = {}
    for date in range(contract.exercise_dates, 0, -1):
        continuation = contract.beta * _transition_quadrature(
            next_optimal, grid, credits, weights
        )
        gap = exercise - continuation
        gaps[date] = gap
        boundaries[date] = _crossing(grid, gap)
        next_optimal = np.maximum(exercise, continuation)
    optimal_issue = contract.beta * float(
        np.dot(weights, np.interp(credits, grid, next_optimal))
    )
    return UncappedQuadratureReference(
        grid=grid,
        credits=credits,
        weights=weights,
        exercise_payoff=exercise,
        optimal_issue_value=optimal_issue,
        optimal_gaps=gaps,
        grid_boundaries=boundaries,
    )


def quadrature_policy_regret_uncapped(
    reference: UncappedQuadratureReference,
    contract: Contract,
    *,
    policy_thresholds: np.ndarray,
) -> np.ndarray:
    """Direct regret recursion, avoiding subtraction of close policy prices."""

    thresholds = np.asarray(policy_thresholds, dtype=float)
    if thresholds.ndim == 1:
        thresholds = thresholds[None, :]
    if thresholds.shape[1] != contract.exercise_dates:
        raise ValueError("threshold matrix has the wrong number of dates")
    loss_next = np.zeros((thresholds.shape[0], len(reference.grid)))
    for date in range(contract.exercise_dates, 0, -1):
        future_loss = contract.beta * _transition_quadrature(
            loss_next, reference.grid, reference.credits, reference.weights
        )
        gap = reference.optimal_gaps[date][None, :]
        stop = reference.grid[None, :] >= thresholds[:, date - 1, None]
        stop_loss = np.maximum(-gap, 0.0)
        continue_loss = np.maximum(gap, 0.0) + future_loss
        loss_next = np.where(stop, stop_loss, continue_loss)
    result = np.empty(thresholds.shape[0])
    for index, loss in enumerate(loss_next):
        result[index] = contract.beta * np.dot(
            reference.weights,
            np.interp(reference.credits, reference.grid, loss),
        )
    return result


def local_domain(
    boundary: float,
    global_cap: float,
    epsilon: float,
    direction: np.ndarray,
) -> tuple[bool, float, float]:
    thresholds = boundary + epsilon * np.asarray(direction, dtype=float)
    return (
        bool(np.all((thresholds >= 0.0) & (thresholds <= global_cap))),
        float(thresholds.min()),
        float(thresholds.max()),
    )


def reachable_threshold_dates(
    thresholds: np.ndarray, contract: Contract
) -> tuple[int, ...]:
    """Dates at which an uncapped threshold is inside the exact support."""

    values = np.asarray(thresholds, dtype=float)
    return tuple(
        date
        for date, threshold in enumerate(values, start=1)
        if 0.0 <= threshold <= date * contract.local_cap
    )

