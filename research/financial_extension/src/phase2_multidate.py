"""Phase II multidate indexed-annuity calculations.

The routines in this module deliberately remain one-dimensional.  They solve
time-dependent full-surrender Bellman equations, propagate the mixed credited
factor law from either issue or a local diagnostic state, and reconcile the
component-history coefficient with direct finite-perturbation repricing.

The continuous part of the forward law is represented as a density in log
account state.  The credited floor and cap atoms are propagated separately and
therefore remain exact rather than being smeared over the grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy.signal import fftconvolve

from indexed_annuity import (
    BellmanSolution,
    ContractSpec,
    DiscreteCreditedLaw,
    MixtureCreditedDistribution,
)


@dataclass(frozen=True)
class TimeSliceSolution:
    """Bellman objects and upper-boundary diagnostics at one decision date."""

    charge: float
    value: np.ndarray
    continuation: np.ndarray
    gap: np.ndarray
    stopping: np.ndarray
    high_state_value_slope: float
    high_state_margin: float
    boundary: float | None
    stopping_side_gap_slope: float | None


@dataclass(frozen=True)
class ScheduleSolution:
    """Backward solution for a deterministic surrender-charge schedule."""

    grid: np.ndarray
    slices: tuple[TimeSliceSolution, ...]
    terminal_value: np.ndarray
    terminal_high_state_value_slope: float

    @property
    def boundaries(self) -> tuple[float | None, ...]:
        return tuple(item.boundary for item in self.slices)


@dataclass
class LogStateMeasure:
    """Subprobability measure: log-density plus exact point atoms."""

    log_grid: np.ndarray
    continuous_density: np.ndarray
    atoms: dict[float, float]

    def copy(self) -> "LogStateMeasure":
        return LogStateMeasure(
            log_grid=self.log_grid,
            continuous_density=self.continuous_density.copy(),
            atoms=dict(self.atoms),
        )

    @property
    def continuous_mass(self) -> float:
        return max(0.0, float(np.trapz(self.continuous_density, self.log_grid)))

    @property
    def atom_mass(self) -> float:
        return float(sum(self.atoms.values()))

    @property
    def mass(self) -> float:
        return self.continuous_mass + self.atom_mass


@dataclass(frozen=True)
class ForwardAudit:
    """Reference-policy forward measures and fresh-arrival trace densities."""

    predecision: tuple[LogStateMeasure, ...]
    survivors: tuple[LogStateMeasure, ...]
    fresh_arrival_log_densities: tuple[np.ndarray, ...]
    propagation_mass_errors: tuple[float, ...]
    atom_boundary_collisions: tuple[tuple[int, float, float], ...]


@dataclass(frozen=True)
class CarrierContribution:
    source_date: int
    target_date: int
    target_boundary: float
    fresh_density_at_target: float
    floor_run_probability: float
    coefficient: float
    classification: str
    finite_gate_dates: tuple[int, ...]


@dataclass(frozen=True)
class MultidateCoefficient:
    ordinary: float
    resonant: float
    additive_pairwise: float
    coherent_pairwise: float
    carriers: tuple[CarrierContribution, ...]

    @property
    def full(self) -> float:
        return self.ordinary + self.resonant

    @property
    def copied_share(self) -> float:
        return self.resonant / self.full if self.full > 0 else 0.0


@dataclass(frozen=True)
class DirectRegret:
    losses_by_date: tuple[float, ...]
    predecision_masses: tuple[float, ...]
    survivor_masses: tuple[float, ...]
    propagation_mass_errors: tuple[float, ...]

    @property
    def total(self) -> float:
        return float(sum(self.losses_by_date))


@dataclass(frozen=True)
class LifetimeSequence:
    """Partial sums for the stationary common-direction identity carrier."""

    dates: tuple[int, ...]
    fresh_boundary_densities: tuple[float, ...]
    terminal_floor_run_densities: tuple[float, ...]
    coefficient_increments: tuple[float, ...]
    cumulative_coefficients: tuple[float, ...]
    predecision_masses: tuple[float, ...]
    survivor_masses: tuple[float, ...]


def _validate_grid(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid, dtype=float)
    if grid.ndim != 1 or len(grid) < 200 or np.any(np.diff(grid) <= 0):
        raise ValueError("grid must be a strictly increasing one-dimensional array")
    if np.any(grid <= 0):
        raise ValueError("state grid must be positive")
    log_grid = np.log(grid)
    steps = np.diff(log_grid)
    if not np.allclose(steps, steps[0], rtol=1e-9, atol=1e-13):
        raise ValueError("Phase II requires a log-spaced state grid")
    return grid


def _linear_tail_interpolation(
    query: np.ndarray,
    grid: np.ndarray,
    value: np.ndarray,
    high_state_slope: float,
) -> np.ndarray:
    query = np.asarray(query, dtype=float)
    result = np.interp(np.minimum(query, grid[-1]), grid, value, left=value[0])
    beyond = query > grid[-1]
    if np.any(beyond):
        result[beyond] = high_state_slope * query[beyond]
    return result


def _continuation_against_next_value(
    *,
    grid: np.ndarray,
    next_value: np.ndarray,
    next_high_state_slope: float,
    law: DiscreteCreditedLaw,
    discount_factor: float,
    termination_probability: float,
    death_guarantee: float,
) -> np.ndarray:
    expected_value = np.zeros_like(grid)
    expected_termination = np.zeros_like(grid)
    for factor, weight in zip(law.factors, law.weights):
        next_state = grid * factor
        expected_value += weight * _linear_tail_interpolation(
            next_state, grid, next_value, next_high_state_slope
        )
        expected_termination += weight * np.maximum(death_guarantee, next_state)
    return discount_factor * (
        termination_probability * expected_termination
        + (1.0 - termination_probability) * expected_value
    )


def _upper_root_and_slope(
    grid: np.ndarray,
    gap: np.ndarray,
) -> tuple[float, float] | None:
    candidates: list[tuple[float, float]] = []
    crossing = np.flatnonzero((gap[:-1] < 0.0) & (gap[1:] >= 0.0))
    for index in crossing:
        denominator = gap[index + 1] - gap[index]
        if denominator <= 0:
            continue
        weight = -gap[index] / denominator
        root = float(grid[index] + weight * (grid[index + 1] - grid[index]))
        local_step = grid[index + 1] - grid[index]
        window = max(12.0 * local_step, 0.005 * root)
        right_index = int(np.searchsorted(grid, root + window, side="right"))
        right_index = min(max(right_index, index + 4), len(grid))
        right_grid = grid[index + 1 : right_index]
        right_gap = gap[index + 1 : right_index]
        if len(right_grid) < 3:
            continue
        distance = right_grid - root
        design = np.column_stack((distance, distance**2))
        slope = float(np.linalg.lstsq(design, right_gap, rcond=None)[0][0])
        if slope > 0:
            candidates.append((root, slope))
    return max(candidates, key=lambda item: item[0]) if candidates else None


def stationary_high_state_slope(
    *,
    liquidation_slope: float,
    expected_factor: float,
    discount_factor: float,
    termination_probability: float,
) -> tuple[float, float]:
    """Return the stationary value slope and stop-versus-continue margin."""

    beta_ey = discount_factor * expected_factor
    continuation_at_stop_slope = beta_ey * (
        termination_probability
        + (1.0 - termination_probability) * liquidation_slope
    )
    margin = liquidation_slope - continuation_at_stop_slope
    if margin >= 0:
        return float(liquidation_slope), float(margin)
    denominator = 1.0 - beta_ey * (1.0 - termination_probability)
    if denominator <= 0:
        raise ValueError("the continuation high-state slope is not finite")
    continuation_slope = beta_ey * termination_probability / denominator
    return float(continuation_slope), float(margin)


def solve_stationary_with_tail(
    *,
    grid: np.ndarray,
    law: DiscreteCreditedLaw,
    contract: ContractSpec,
    tolerance: float = 1e-11,
    max_iterations: int = 20_000,
) -> tuple[BellmanSolution, float, float]:
    """Stationary Bellman solve with the correct linear asymptotic branch.

    Unlike the Phase I helper, this routine also handles a continuation tail.
    Such a solution can be used as a terminal value, but it has no regular
    upper surrender boundary when the high-state margin is nonpositive.
    """

    grid = _validate_grid(grid)
    law.validate()
    contract.validate()
    liquidation_slope = 1.0 - contract.surrender_haircut
    tail_slope, margin = stationary_high_state_slope(
        liquidation_slope=liquidation_slope,
        expected_factor=law.expected_factor,
        discount_factor=contract.discount_factor,
        termination_probability=contract.termination_probability,
    )
    liquidation = contract.surrender_value(grid)
    value = np.maximum(liquidation, tail_slope * grid)
    sup_error = np.inf
    for iteration in range(1, max_iterations + 1):
        continuation = _continuation_against_next_value(
            grid=grid,
            next_value=value,
            next_high_state_slope=tail_slope,
            law=law,
            discount_factor=contract.discount_factor,
            termination_probability=contract.termination_probability,
            death_guarantee=contract.death_guarantee,
        )
        updated = np.maximum(liquidation, continuation)
        sup_error = float(np.max(np.abs(updated - value)))
        value = updated
        if sup_error <= tolerance:
            break
    else:
        raise RuntimeError("stationary Phase II Bellman iteration did not converge")
    continuation = _continuation_against_next_value(
        grid=grid,
        next_value=value,
        next_high_state_slope=tail_slope,
        law=law,
        discount_factor=contract.discount_factor,
        termination_probability=contract.termination_probability,
        death_guarantee=contract.death_guarantee,
    )
    gap = liquidation - continuation
    roots: tuple[float, ...] = ()
    slopes: tuple[float, ...] = ()
    upper = _upper_root_and_slope(grid, gap) if margin > 0 else None
    if upper is not None:
        roots, slopes = (upper[0],), (upper[1],)
    solution = BellmanSolution(
        grid=grid,
        value=value,
        continuation=continuation,
        gap=gap,
        stopping=gap >= 0,
        roots=roots,
        root_slopes=slopes,
        iterations=iteration,
        sup_error=sup_error,
    )
    return solution, tail_slope, margin


def solve_charge_schedule(
    *,
    grid: np.ndarray,
    transition_laws: Sequence[DiscreteCreditedLaw],
    charges: Sequence[float],
    discount_factor: float,
    termination_probability: float,
    death_guarantee: float,
    terminal_value: np.ndarray,
    terminal_high_state_slope: float,
) -> ScheduleSolution:
    """Solve date-specific decisions backward from a supplied exact tail.

    ``transition_laws[t]`` is the credited law between decision ``t`` and
    decision ``t+1`` (or the supplied tail after the last listed decision).
    """

    grid = _validate_grid(grid)
    if len(charges) == 0 or len(transition_laws) != len(charges):
        raise ValueError("charges and transition_laws must have the same nonzero length")
    if np.asarray(terminal_value).shape != grid.shape:
        raise ValueError("terminal_value must align with grid")
    if not 0 < discount_factor < 1 or not 0 < termination_probability <= 1:
        raise ValueError("invalid discount or termination probability")
    for law in transition_laws:
        law.validate()

    next_value = np.asarray(terminal_value, dtype=float)
    next_slope = float(terminal_high_state_slope)
    reverse_slices: list[TimeSliceSolution] = []
    for charge, law in reversed(tuple(zip(charges, transition_laws))):
        if not 0 <= charge < 1:
            raise ValueError("charges must lie in [0,1)")
        continuation = _continuation_against_next_value(
            grid=grid,
            next_value=next_value,
            next_high_state_slope=next_slope,
            law=law,
            discount_factor=discount_factor,
            termination_probability=termination_probability,
            death_guarantee=death_guarantee,
        )
        liquidation = (1.0 - charge) * grid
        value = np.maximum(liquidation, continuation)
        continuation_slope = discount_factor * law.expected_factor * (
            termination_probability + (1.0 - termination_probability) * next_slope
        )
        liquidation_slope = 1.0 - charge
        margin = liquidation_slope - continuation_slope
        high_slope = max(liquidation_slope, continuation_slope)
        upper = _upper_root_and_slope(grid, liquidation - continuation)
        if margin <= 0:
            upper = None
        reverse_slices.append(
            TimeSliceSolution(
                charge=float(charge),
                value=value,
                continuation=continuation,
                gap=liquidation - continuation,
                stopping=liquidation >= continuation,
                high_state_value_slope=float(high_slope),
                high_state_margin=float(margin),
                boundary=None if upper is None else float(upper[0]),
                stopping_side_gap_slope=None if upper is None else float(upper[1]),
            )
        )
        next_value = value
        next_slope = high_slope
    return ScheduleSolution(
        grid=grid,
        slices=tuple(reversed(reverse_slices)),
        terminal_value=np.asarray(terminal_value, dtype=float),
        terminal_high_state_value_slope=float(terminal_high_state_slope),
    )


def _kernel_log_density(
    distribution: MixtureCreditedDistribution,
    log_factor: np.ndarray,
) -> np.ndarray:
    log_factor = np.asarray(log_factor, dtype=float)
    factor = np.exp(log_factor)
    return distribution.factor_density(factor) * factor


def _merge_atom(atoms: dict[float, float], location: float, mass: float) -> None:
    if mass <= 0:
        return
    key = round(float(location), 13)
    atoms[key] = atoms.get(key, 0.0) + float(mass)


def initial_issue_measure(
    *,
    log_grid: np.ndarray,
    initial_state: float,
    distribution: MixtureCreditedDistribution,
) -> tuple[LogStateMeasure, np.ndarray]:
    """One-credit predecision law from a deterministic state."""

    distribution.validate()
    if initial_state <= 0:
        raise ValueError("initial_state must be positive")
    log_grid = np.asarray(log_grid, dtype=float)
    shifted = log_grid - np.log(initial_state)
    density = _kernel_log_density(distribution, shifted)
    target_mass = 1.0 - distribution.floor_probability - distribution.cap_probability
    represented = float(np.trapz(density, log_grid))
    if target_mass > 0 and represented <= 0:
        raise ValueError("log grid misses the continuous credited-factor support")
    if represented > 0:
        density *= target_mass / represented
    atoms: dict[float, float] = {}
    _merge_atom(atoms, np.log(initial_state), distribution.floor_probability)
    _merge_atom(
        atoms,
        np.log(initial_state * distribution.cap_factor),
        distribution.cap_probability,
    )
    return LogStateMeasure(log_grid, density, atoms), density.copy()


def propagate_measure(
    measure: LogStateMeasure,
    distribution: MixtureCreditedDistribution,
) -> tuple[LogStateMeasure, np.ndarray, float]:
    """Apply one mixed transition and return total and fresh log densities."""

    distribution.validate()
    z = measure.log_grid
    dz = float(z[1] - z[0])
    if not np.allclose(np.diff(z), dz, rtol=1e-9, atol=1e-13):
        raise ValueError("log grid must be uniform")
    log_cap = float(np.log(distribution.cap_factor))
    kernel_steps = int(np.ceil(log_cap / dz))
    kernel_grid = np.arange(kernel_steps + 1, dtype=float) * dz
    kernel = _kernel_log_density(distribution, kernel_grid)
    interior_mass = 1.0 - distribution.floor_probability - distribution.cap_probability
    represented = float(kernel.sum() * dz)
    if represented <= 0:
        raise ValueError("transition grid misses the continuous factor support")
    kernel *= interior_mass / represented

    continuous = measure.continuous_density
    continuous_mass = measure.continuous_mass
    atom_mass = measure.atom_mass

    def rescale(component: np.ndarray, target_mass: float) -> np.ndarray:
        component = np.maximum(component, 0.0)
        if target_mass <= 1e-14:
            return np.zeros_like(component)
        represented_mass = float(np.trapz(component, z))
        if represented_mass <= 0.0:
            raise ValueError("log grid truncates a positive transition component")
        return component * (target_mass / represented_mass)

    interior_from_continuous = fftconvolve(continuous, kernel, mode="full")[: len(z)] * dz
    interior_from_continuous = rescale(
        interior_from_continuous, interior_mass * continuous_mass
    )
    floor_from_continuous = distribution.floor_probability * continuous
    cap_from_continuous = distribution.cap_probability * np.interp(
        z - log_cap, z, continuous, left=0.0, right=0.0
    )
    cap_from_continuous = rescale(
        cap_from_continuous,
        distribution.cap_probability * continuous_mass,
    )
    interior_from_atoms = np.zeros_like(z)
    for atom_location, atom_mass in measure.atoms.items():
        interior_from_atoms += atom_mass * _kernel_log_density(
            distribution, z - atom_location
        )
    interior_from_atoms = rescale(
        interior_from_atoms, interior_mass * measure.atom_mass
    )
    fresh = interior_from_continuous + cap_from_continuous + interior_from_atoms
    total_density = floor_from_continuous + fresh

    atoms: dict[float, float] = {}
    for atom_location, atom_mass in measure.atoms.items():
        _merge_atom(
            atoms,
            atom_location,
            atom_mass * distribution.floor_probability,
        )
        _merge_atom(
            atoms,
            atom_location + log_cap,
            atom_mass * distribution.cap_probability,
        )
    propagated = LogStateMeasure(z, total_density, atoms)
    mass_error = propagated.mass - measure.mass
    return propagated, fresh, float(mass_error)


def truncate_below(measure: LogStateMeasure, gate: float) -> LogStateMeasure:
    """Apply the strict continuation gate ``X < gate``."""

    if gate <= 0:
        return LogStateMeasure(
            measure.log_grid,
            np.zeros_like(measure.continuous_density),
            {},
        )
    log_gate = float(np.log(gate))
    density = measure.continuous_density.copy()
    density[measure.log_grid >= log_gate] = 0.0
    density[np.abs(density) < 1e-14] = 0.0
    density = np.maximum(density, 0.0)
    atoms = {
        location: mass
        for location, mass in measure.atoms.items()
        if location < log_gate
    }
    return LogStateMeasure(measure.log_grid, density, atoms)


def _density_in_state(measure_or_density, log_grid: np.ndarray, state: float) -> float:
    if state <= 0:
        return 0.0
    values = (
        measure_or_density.continuous_density
        if isinstance(measure_or_density, LogStateMeasure)
        else np.asarray(measure_or_density, dtype=float)
    )
    log_density = float(np.interp(np.log(state), log_grid, values, left=0.0, right=0.0))
    return log_density / state


def reference_forward_audit(
    *,
    log_grid: np.ndarray,
    initial_state: float,
    initial_distribution: MixtureCreditedDistribution,
    transition_distributions: Sequence[MixtureCreditedDistribution],
    boundaries: Sequence[float],
    collision_tolerance: float = 1e-8,
) -> ForwardAudit:
    """Propagate reference-policy occupancy and retain last-fresh traces."""

    if len(transition_distributions) != max(0, len(boundaries) - 1):
        raise ValueError("one transition distribution is required between decisions")
    current, first_fresh = initial_issue_measure(
        log_grid=log_grid,
        initial_state=initial_state,
        distribution=initial_distribution,
    )
    predecision: list[LogStateMeasure] = []
    survivors: list[LogStateMeasure] = []
    fresh: list[np.ndarray] = [first_fresh]
    mass_errors: list[float] = []
    collisions: list[tuple[int, float, float]] = []
    for index, boundary in enumerate(boundaries):
        predecision.append(current)
        for location, mass in current.atoms.items():
            atom_state = float(np.exp(location))
            if abs(atom_state - boundary) <= collision_tolerance * max(1.0, boundary):
                collisions.append((index + 1, atom_state, mass))
        survivor = truncate_below(current, boundary)
        survivors.append(survivor)
        if index < len(transition_distributions):
            current, next_fresh, mass_error = propagate_measure(
                survivor, transition_distributions[index]
            )
            fresh.append(next_fresh)
            mass_errors.append(mass_error)
    return ForwardAudit(
        predecision=tuple(predecision),
        survivors=tuple(survivors),
        fresh_arrival_log_densities=tuple(fresh),
        propagation_mass_errors=tuple(mass_errors),
        atom_boundary_collisions=tuple(collisions),
    )


def advance_issue_measure(
    *,
    log_grid: np.ndarray,
    initial_state: float,
    credit_distributions: Sequence[MixtureCreditedDistribution],
    preceding_boundaries: Sequence[float | None],
) -> tuple[LogStateMeasure, tuple[float, ...]]:
    """Advance from issue to the next decision under reference survival.

    ``credit_distributions`` contains the credit into each preceding decision
    and one final credit into the returned decision.  A ``None`` boundary is an
    inactive date with no upper stopping gate.
    """

    if len(credit_distributions) != len(preceding_boundaries) + 1:
        raise ValueError("one final credit beyond the preceding decisions is required")
    current, _ = initial_issue_measure(
        log_grid=log_grid,
        initial_state=initial_state,
        distribution=credit_distributions[0],
    )
    minimum_log_state = float(np.log(initial_state))
    survivor_masses: list[float] = []
    for boundary, distribution in zip(
        preceding_boundaries, credit_distributions[1:]
    ):
        if boundary is not None and np.log(boundary) <= minimum_log_state:
            # Every credited factor is at least one, so this is an exact support
            # statement rather than a numerical tolerance decision.
            survivor = LogStateMeasure(
                current.log_grid,
                np.zeros_like(current.continuous_density),
                {},
            )
        else:
            survivor = (
                current if boundary is None else truncate_below(current, boundary)
            )
        survivor_masses.append(survivor.mass)
        current, _, _ = propagate_measure(survivor, distribution)
        current.continuous_density[current.log_grid < minimum_log_state] = 0.0
        current.atoms = {
            location: mass
            for location, mass in current.atoms.items()
            if location >= minimum_log_state
        }
    return current, tuple(float(value) for value in survivor_masses)


def reference_forward_audit_from_measure(
    *,
    initial_predecision_measure: LogStateMeasure,
    transition_distributions: Sequence[MixtureCreditedDistribution],
    boundaries: Sequence[float],
    collision_tolerance: float = 1e-8,
) -> ForwardAudit:
    """Reference audit beginning from an already propagated predecision law.

    The first continuous source is collapsed into a single label because no
    earlier gate is perturbed in the selected active window.
    """

    if len(transition_distributions) != max(0, len(boundaries) - 1):
        raise ValueError("one transition distribution is required between decisions")
    current = initial_predecision_measure.copy()
    predecision: list[LogStateMeasure] = []
    survivors: list[LogStateMeasure] = []
    fresh: list[np.ndarray] = [current.continuous_density.copy()]
    mass_errors: list[float] = []
    collisions: list[tuple[int, float, float]] = []
    for index, boundary in enumerate(boundaries):
        predecision.append(current)
        for location, mass in current.atoms.items():
            atom_state = float(np.exp(location))
            if abs(atom_state - boundary) <= collision_tolerance * max(1.0, boundary):
                collisions.append((index + 1, atom_state, mass))
        survivor = truncate_below(current, boundary)
        survivors.append(survivor)
        if index < len(transition_distributions):
            current, next_fresh, mass_error = propagate_measure(
                survivor, transition_distributions[index]
            )
            fresh.append(next_fresh)
            mass_errors.append(mass_error)
    return ForwardAudit(
        predecision=tuple(predecision),
        survivors=tuple(survivors),
        fresh_arrival_log_densities=tuple(fresh),
        propagation_mass_errors=tuple(mass_errors),
        atom_boundary_collisions=tuple(collisions),
    )


def component_history_coefficient(
    *,
    audit: ForwardAudit,
    boundaries: Sequence[float],
    slopes: Sequence[float],
    shifts: Sequence[float],
    discount_survival: float,
    floor_probabilities: Sequence[float],
    alignment_groups: Sequence[str | None],
    boundary_tolerance: float = 2e-6,
    discount_start_power: int = 1,
) -> MultidateCoefficient:
    """Sum last-fresh/floor-run carriers for general deterministic boundaries.

    A floor run from source ``s`` to target ``t`` is killed if an intervening
    boundary lies strictly below the target boundary.  Intervening dates with
    the same structural alignment label impose finite prefix gates; dates with
    strictly higher boundaries contribute only a continuation-end coordinate.
    """

    length = len(boundaries)
    if not (
        length
        == len(slopes)
        == len(shifts)
        == len(alignment_groups)
        == len(audit.fresh_arrival_log_densities)
    ):
        raise ValueError("all date-indexed inputs must have equal length")
    if len(floor_probabilities) != max(0, length - 1):
        raise ValueError("floor_probabilities must index between-date transitions")
    if any(value <= 0 for value in slopes) or any(value < 0 for value in shifts):
        raise ValueError("positive slopes and nonnegative shifts are required")
    if discount_start_power < 1:
        raise ValueError("discount_start_power must be positive")

    ordinary = 0.0
    resonant = 0.0
    additive = 0.0
    coherent = 0.0
    carriers: list[CarrierContribution] = []
    z = audit.predecision[0].log_grid
    for target in range(length):
        b_target = float(boundaries[target])
        for source in range(target + 1):
            intervening = range(source, target)
            killed = any(
                b_target > boundaries[j] + boundary_tolerance * max(1.0, b_target)
                for j in intervening
            )
            if killed:
                continue
            density = _density_in_state(
                audit.fresh_arrival_log_densities[source], z, b_target
            )
            if density <= 0:
                continue
            floor_probability = 1.0
            for j in range(source, target):
                floor_probability *= floor_probabilities[j]
            finite_dates = [target]
            target_group = alignment_groups[target]
            if target_group is not None:
                for j in intervening:
                    if alignment_groups[j] == target_group:
                        finite_dates.append(j)
            finite_dates = sorted(set(finite_dates))
            base = (
                0.5
                * slopes[target]
                * discount_survival ** (discount_start_power + target)
                * density
                * floor_probability
            )
            gate = min(shifts[j] for j in finite_dates)
            coefficient = base * gate**2
            if len(finite_dates) == 1:
                classification = "ordinary"
                ordinary += coefficient
                additive += coefficient
                coherent += coefficient
            else:
                classification = "resonant"
                resonant += coefficient
                pair_terms = [
                    min(shifts[j], shifts[target]) ** 2
                    for j in finite_dates
                    if j != target
                ]
                additive += base * sum(pair_terms)
                coherent += base * min(pair_terms)
            carriers.append(
                CarrierContribution(
                    source_date=source + 1,
                    target_date=target + 1,
                    target_boundary=b_target,
                    fresh_density_at_target=float(density),
                    floor_run_probability=float(floor_probability),
                    coefficient=float(coefficient),
                    classification=classification,
                    finite_gate_dates=tuple(j + 1 for j in finite_dates),
                )
            )
    return MultidateCoefficient(
        ordinary=float(ordinary),
        resonant=float(resonant),
        additive_pairwise=float(additive),
        coherent_pairwise=float(coherent),
        carriers=tuple(carriers),
    )


def _band_integral(
    *,
    measure: LogStateMeasure,
    lower: float,
    upper: float,
    state_grid: np.ndarray,
    gap: np.ndarray,
    nodes: int = 64,
) -> float:
    if upper <= lower:
        return 0.0
    points, weights = np.polynomial.legendre.leggauss(nodes)
    z_lower, z_upper = np.log(lower), np.log(upper)
    z_points = 0.5 * (z_upper - z_lower) * points + 0.5 * (z_upper + z_lower)
    log_density = np.interp(
        z_points,
        measure.log_grid,
        measure.continuous_density,
        left=0.0,
        right=0.0,
    )
    state = np.exp(z_points)
    gap_values = np.abs(np.interp(state, state_grid, gap))
    result = 0.5 * (z_upper - z_lower) * float(
        np.dot(weights, log_density * gap_values)
    )
    for location, mass in measure.atoms.items():
        atom_state = float(np.exp(location))
        if lower <= atom_state < upper:
            atom_gap = abs(float(np.interp(atom_state, state_grid, gap)))
            result += mass * atom_gap
    return float(result)


def direct_multidate_regret(
    *,
    log_grid: np.ndarray,
    state_grid: np.ndarray,
    initial_state: float,
    initial_distribution: MixtureCreditedDistribution,
    transition_distributions: Sequence[MixtureCreditedDistribution],
    boundaries: Sequence[float],
    gaps: Sequence[np.ndarray],
    shifts: Sequence[float],
    epsilon: float,
    discount_survival: float,
    integration_nodes: int = 64,
    discount_start_power: int = 1,
    initial_predecision_measure: LogStateMeasure | None = None,
) -> DirectRegret:
    """Independently reprice the supplied upward-shifted multidate policy."""

    length = len(boundaries)
    if not (length == len(gaps) == len(shifts)):
        raise ValueError("boundary, gap, and shift sequences must align")
    if len(transition_distributions) != max(0, length - 1):
        raise ValueError("transition distributions must lie between decisions")
    if epsilon <= 0 or any(shift < 0 for shift in shifts):
        raise ValueError("epsilon must be positive and shifts nonnegative")
    if discount_start_power < 1:
        raise ValueError("discount_start_power must be positive")
    if initial_predecision_measure is None:
        current, _ = initial_issue_measure(
            log_grid=log_grid,
            initial_state=initial_state,
            distribution=initial_distribution,
        )
    else:
        if not np.array_equal(initial_predecision_measure.log_grid, log_grid):
            raise ValueError("initial predecision measure must use log_grid")
        current = initial_predecision_measure.copy()
    losses: list[float] = []
    pre_masses: list[float] = []
    survivor_masses: list[float] = []
    mass_errors: list[float] = []
    for index in range(length):
        pre_masses.append(current.mass)
        gate = boundaries[index] + epsilon * shifts[index]
        undiscounted = _band_integral(
            measure=current,
            lower=boundaries[index],
            upper=gate,
            state_grid=state_grid,
            gap=gaps[index],
            nodes=integration_nodes,
        )
        losses.append(
            discount_survival ** (discount_start_power + index) * undiscounted
        )
        survivor = truncate_below(current, gate)
        survivor_masses.append(survivor.mass)
        if index < len(transition_distributions):
            current, _, mass_error = propagate_measure(
                survivor, transition_distributions[index]
            )
            mass_errors.append(mass_error)
    return DirectRegret(
        losses_by_date=tuple(float(value) for value in losses),
        predecision_masses=tuple(float(value) for value in pre_masses),
        survivor_masses=tuple(float(value) for value in survivor_masses),
        propagation_mass_errors=tuple(float(value) for value in mass_errors),
    )


def stationary_lifetime_sequence(
    *,
    log_grid: np.ndarray,
    initial_state: float,
    distribution: MixtureCreditedDistribution,
    boundary: float,
    slope: float,
    shift: float,
    discount_survival: float,
    maximum_dates: int,
) -> LifetimeSequence:
    """Stream equation (8) without retaining every forward density array."""

    if boundary <= 0 or slope <= 0 or shift < 0 or maximum_dates < 1:
        raise ValueError("invalid lifetime-sequence inputs")
    current, fresh = initial_issue_measure(
        log_grid=log_grid,
        initial_state=initial_state,
        distribution=distribution,
    )
    floor_run_density = 0.0
    cumulative = 0.0
    dates: list[int] = []
    fresh_densities: list[float] = []
    run_densities: list[float] = []
    increments: list[float] = []
    cumulatives: list[float] = []
    pre_masses: list[float] = []
    survivor_masses: list[float] = []
    for index in range(maximum_dates):
        date = index + 1
        fresh_density = _density_in_state(fresh, log_grid, boundary)
        floor_run_density = (
            fresh_density + distribution.floor_probability * floor_run_density
        )
        increment = (
            0.5
            * slope
            * shift**2
            * discount_survival**date
            * floor_run_density
        )
        cumulative += increment
        pre_masses.append(current.mass)
        survivor = truncate_below(current, boundary)
        survivor_masses.append(survivor.mass)
        dates.append(date)
        fresh_densities.append(fresh_density)
        run_densities.append(floor_run_density)
        increments.append(increment)
        cumulatives.append(cumulative)
        if date < maximum_dates:
            current, fresh, _ = propagate_measure(survivor, distribution)
    return LifetimeSequence(
        dates=tuple(dates),
        fresh_boundary_densities=tuple(float(value) for value in fresh_densities),
        terminal_floor_run_densities=tuple(float(value) for value in run_densities),
        coefficient_increments=tuple(float(value) for value in increments),
        cumulative_coefficients=tuple(float(value) for value in cumulatives),
        predecision_masses=tuple(float(value) for value in pre_masses),
        survivor_masses=tuple(float(value) for value in survivor_masses),
    )


def all_boundaries_and_slopes(
    solution: ScheduleSolution,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Extract regular upper boundaries or fail with the offending date."""

    boundaries: list[float] = []
    slopes: list[float] = []
    for index, item in enumerate(solution.slices, start=1):
        if item.boundary is None or item.stopping_side_gap_slope is None:
            raise ValueError(f"decision date {index} has no regular upper boundary")
        boundaries.append(item.boundary)
        slopes.append(item.stopping_side_gap_slope)
    return tuple(boundaries), tuple(slopes)


def product(values: Iterable[float]) -> float:
    """Small explicit helper used in audit checks."""

    result = 1.0
    for value in values:
        result *= value
    return float(result)
