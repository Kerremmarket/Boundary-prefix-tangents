"""One-dimensional stationary indexed-annuity contract and Bellman solver."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.stats import norm

from market_models import LognormalMixtureParams


@dataclass(frozen=True)
class CreditingSpec:
    participation: float
    cap: float

    def validate(self) -> None:
        if not 0 < self.participation <= 2:
            raise ValueError("participation must lie in (0, 2]")
        if self.cap <= 0:
            raise ValueError("cap must be positive")


@dataclass(frozen=True)
class ContractSpec:
    discount_factor: float
    termination_probability: float
    surrender_haircut: float
    death_guarantee: float = 1.0
    minimum_surrender_value: float = 0.0

    def validate(self) -> None:
        if not 0 < self.discount_factor < 1:
            raise ValueError("discount_factor must lie in (0, 1)")
        if not 0 < self.termination_probability <= 1:
            raise ValueError("termination_probability must lie in (0, 1]")
        if not 0 <= self.surrender_haircut < 1:
            raise ValueError("surrender_haircut must lie in [0, 1)")
        if self.death_guarantee <= 0:
            raise ValueError("death_guarantee must be positive")
        if self.minimum_surrender_value < 0:
            raise ValueError("minimum_surrender_value cannot be negative")

    def surrender_value(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=float)
        return np.maximum(
            self.minimum_surrender_value,
            (1.0 - self.surrender_haircut) * state,
        )

    def termination_value(self, state: np.ndarray) -> np.ndarray:
        return np.maximum(self.death_guarantee, np.asarray(state, dtype=float))


@dataclass(frozen=True)
class DiscreteCreditedLaw:
    factors: np.ndarray
    weights: np.ndarray
    floor_probability: float
    cap_probability: float

    def validate(self) -> None:
        factors = np.asarray(self.factors, dtype=float)
        weights = np.asarray(self.weights, dtype=float)
        if factors.ndim != 1 or weights.ndim != 1 or factors.shape != weights.shape:
            raise ValueError("factors and weights must be one-dimensional and aligned")
        if len(factors) < 3:
            raise ValueError("law must include floor, interior, and cap support")
        if not np.all(np.isfinite(factors)) or not np.all(np.isfinite(weights)):
            raise ValueError("law contains nonfinite entries")
        if np.any(weights < 0) or not np.isclose(weights.sum(), 1.0, atol=1e-12):
            raise ValueError("weights must be nonnegative and sum to one")
        if not np.isclose(factors[0], 1.0, atol=1e-14):
            raise ValueError("first support point must be the copied floor factor one")
        if not np.isclose(weights[0], self.floor_probability, atol=1e-12):
            raise ValueError("first weight must equal floor_probability")

    @property
    def expected_factor(self) -> float:
        return float(np.dot(self.factors, self.weights))


@dataclass(frozen=True)
class MixtureCreditedDistribution:
    """Annual credited-factor law induced by a fitted lognormal mixture.

    ``gross_forward`` is the annual risk-neutral mean of the gross SPX return.
    The credited factor is ``1 + min(cap, max(0, alpha * (R - 1)))``.  Thus the
    return intervals below one and above the cap threshold become exact floor
    and cap atoms; only the interior has a Lebesgue density.
    """

    gross_forward: float
    mixture: LognormalMixtureParams
    crediting: CreditingSpec

    def validate(self) -> None:
        if not np.isfinite(self.gross_forward) or self.gross_forward <= 0:
            raise ValueError("gross_forward must be finite and positive")
        self.mixture.validate()
        self.crediting.validate()

    @property
    def cap_return(self) -> float:
        return 1.0 + self.crediting.cap / self.crediting.participation

    @property
    def cap_factor(self) -> float:
        return 1.0 + self.crediting.cap

    @property
    def component_weights(self) -> tuple[float, float]:
        return self.mixture.low_weight, 1.0 - self.mixture.low_weight

    @property
    def component_mean_returns(self) -> tuple[float, float]:
        return (
            self.gross_forward * self.mixture.low_forward_multiplier,
            self.gross_forward * self.mixture.high_forward_multiplier,
        )

    @property
    def component_volatilities(self) -> tuple[float, float]:
        return self.mixture.low_volatility, self.mixture.high_volatility

    def gross_return_density(self, gross_return) -> np.ndarray:
        self.validate()
        value = np.asarray(gross_return, dtype=float)
        density = np.zeros_like(value)
        positive = value > 0
        if not np.any(positive):
            return density
        x = value[positive]
        mixed = np.zeros_like(x)
        for weight, mean, volatility in zip(
            self.component_weights,
            self.component_mean_returns,
            self.component_volatilities,
        ):
            log_mean = np.log(mean) - 0.5 * volatility**2
            z = (np.log(x) - log_mean) / volatility
            mixed += weight * norm.pdf(z) / (x * volatility)
        density[positive] = mixed
        return density

    def gross_return_cdf(self, gross_return) -> np.ndarray:
        self.validate()
        value = np.asarray(gross_return, dtype=float)
        result = np.zeros_like(value)
        positive = value > 0
        if not np.any(positive):
            return result
        x = value[positive]
        mixed = np.zeros_like(x)
        for weight, mean, volatility in zip(
            self.component_weights,
            self.component_mean_returns,
            self.component_volatilities,
        ):
            log_mean = np.log(mean) - 0.5 * volatility**2
            z = (np.log(x) - log_mean) / volatility
            mixed += weight * norm.cdf(z)
        result[positive] = mixed
        return result

    @property
    def floor_probability(self) -> float:
        return float(self.gross_return_cdf(np.array([1.0]))[0])

    @property
    def cap_probability(self) -> float:
        return float(1.0 - self.gross_return_cdf(np.array([self.cap_return]))[0])

    def factor_density(self, factor) -> np.ndarray:
        """Lebesgue density on the open credited-factor interior."""

        value = np.asarray(factor, dtype=float)
        result = np.zeros_like(value)
        interior = (value > 1.0) & (value < self.cap_factor)
        if np.any(interior):
            gross_return = 1.0 + (
                value[interior] - 1.0
            ) / self.crediting.participation
            result[interior] = (
                self.gross_return_density(gross_return)
                / self.crediting.participation
            )
        return result

    def state_continuous_density(self, state, initial_state: float) -> np.ndarray:
        """Continuous density after one credit from a deterministic state."""

        if not np.isfinite(initial_state) or initial_state <= 0:
            raise ValueError("initial_state must be finite and positive")
        value = np.asarray(state, dtype=float)
        return self.factor_density(value / initial_state) / initial_state

    def state_atoms(self, initial_state: float) -> tuple[tuple[float, float], ...]:
        if not np.isfinite(initial_state) or initial_state <= 0:
            raise ValueError("initial_state must be finite and positive")
        return (
            (float(initial_state), self.floor_probability),
            (float(initial_state * self.cap_factor), self.cap_probability),
        )

    def discretize(self, quadrature_nodes: int = 96) -> DiscreteCreditedLaw:
        """Preserve both atoms exactly and quadrature-discretize the interior."""

        self.validate()
        if quadrature_nodes < 8:
            raise ValueError("quadrature_nodes must be at least eight")
        nodes, base_weights = leggauss(quadrature_nodes)
        interior_factors: list[np.ndarray] = []
        interior_weights: list[np.ndarray] = []
        for mixture_weight, mean, volatility in zip(
            self.component_weights,
            self.component_mean_returns,
            self.component_volatilities,
        ):
            log_mean = np.log(mean) - 0.5 * volatility**2
            lower_z = (0.0 - log_mean) / volatility
            upper_z = (np.log(self.cap_return) - log_mean) / volatility
            z = 0.5 * (upper_z - lower_z) * nodes + 0.5 * (
                upper_z + lower_z
            )
            weights = (
                mixture_weight
                * 0.5
                * (upper_z - lower_z)
                * base_weights
                * norm.pdf(z)
            )
            gross_returns = np.exp(log_mean + volatility * z)
            factors = 1.0 + self.crediting.participation * (
                gross_returns - 1.0
            )
            interior_factors.append(factors)
            interior_weights.append(weights)

        factors = np.concatenate(
            ([1.0], *interior_factors, [self.cap_factor])
        )
        weights = np.concatenate(
            ([self.floor_probability], *interior_weights, [self.cap_probability])
        )
        order = np.argsort(factors, kind="stable")
        factors = factors[order]
        weights = weights[order]
        weights = weights / weights.sum()
        law = DiscreteCreditedLaw(
            factors=factors,
            weights=weights,
            floor_probability=float(weights[0]),
            cap_probability=float(weights[-1]),
        )
        law.validate()
        return law


@dataclass(frozen=True)
class BellmanSolution:
    grid: np.ndarray
    value: np.ndarray
    continuation: np.ndarray
    gap: np.ndarray
    stopping: np.ndarray
    roots: tuple[float, ...]
    root_slopes: tuple[float, ...]
    iterations: int
    sup_error: float


def black_scholes_credited_law(
    *,
    annual_rate: float,
    dividend_yield: float,
    volatility: float,
    crediting: CreditingSpec,
    quadrature_nodes: int = 96,
) -> DiscreteCreditedLaw:
    """Discretize the annual credited factor under a risk-neutral BS return.

    The first and last support points are the exact floor and cap atoms.  Only
    the smooth interior return interval is quadrature-discretized.
    """

    crediting.validate()
    if volatility <= 0:
        raise ValueError("volatility must be positive")
    if quadrature_nodes < 8:
        raise ValueError("quadrature_nodes must be at least eight")

    mean_log_return = annual_rate - dividend_yield - 0.5 * volatility**2
    floor_z = -mean_log_return / volatility
    cap_return = 1.0 + crediting.cap / crediting.participation
    cap_z = (np.log(cap_return) - mean_log_return) / volatility
    if not cap_z > floor_z:
        raise ValueError("cap threshold must exceed the credited floor threshold")

    nodes, base_weights = leggauss(quadrature_nodes)
    interior_z = 0.5 * (cap_z - floor_z) * nodes + 0.5 * (cap_z + floor_z)
    interior_weights = (
        0.5 * (cap_z - floor_z) * base_weights * norm.pdf(interior_z)
    )
    gross_returns = np.exp(mean_log_return + volatility * interior_z)
    interior_factors = 1.0 + crediting.participation * (gross_returns - 1.0)

    floor_probability = float(norm.cdf(floor_z))
    cap_probability = float(norm.sf(cap_z))
    factors = np.concatenate(
        ([1.0], interior_factors, [1.0 + crediting.cap])
    )
    weights = np.concatenate(
        ([floor_probability], interior_weights, [cap_probability])
    )
    # The quadrature error in the truncated normal mass is tiny but explicitly
    # normalize it so downstream kernels remain probability kernels.
    weights = weights / weights.sum()
    floor_probability = float(weights[0])
    cap_probability = float(weights[-1])
    law = DiscreteCreditedLaw(
        factors=factors,
        weights=weights,
        floor_probability=floor_probability,
        cap_probability=cap_probability,
    )
    law.validate()
    return law


def log_state_grid(lower: float, upper: float, points: int) -> np.ndarray:
    if not 0 < lower < upper:
        raise ValueError("grid requires 0 < lower < upper")
    if points < 200:
        raise ValueError("grid requires at least 200 points")
    return np.exp(np.linspace(np.log(lower), np.log(upper), points))


def _interpolate_value(
    query: np.ndarray,
    grid: np.ndarray,
    value: np.ndarray,
    contract: ContractSpec,
) -> np.ndarray:
    """Interpolate value and impose surrender in the verified high-state tail."""

    clipped = np.minimum(query, grid[-1])
    interpolated = np.interp(clipped, grid, value, left=value[0])
    beyond = query > grid[-1]
    if np.any(beyond):
        interpolated[beyond] = contract.surrender_value(query[beyond])
    return interpolated


def continuation_value(
    grid: np.ndarray,
    value: np.ndarray,
    law: DiscreteCreditedLaw,
    contract: ContractSpec,
) -> np.ndarray:
    law.validate()
    contract.validate()
    expected_value = np.zeros_like(grid)
    expected_termination = np.zeros_like(grid)
    for factor, weight in zip(law.factors, law.weights):
        next_state = grid * factor
        expected_value += weight * _interpolate_value(
            next_state, grid, value, contract
        )
        expected_termination += weight * contract.termination_value(next_state)
    mortality = contract.termination_probability
    return contract.discount_factor * (
        mortality * expected_termination + (1.0 - mortality) * expected_value
    )


def _roots_and_slopes(grid: np.ndarray, gap: np.ndarray) -> tuple[tuple[float, ...], tuple[float, ...]]:
    roots: list[float] = []
    slopes: list[float] = []
    crossing_indices = np.flatnonzero(np.signbit(gap[:-1]) != np.signbit(gap[1:]))
    for index in crossing_indices:
        denominator = gap[index + 1] - gap[index]
        if denominator == 0:
            continue
        weight = -gap[index] / denominator
        root = grid[index] + weight * (grid[index + 1] - grid[index])
        # The Bellman gap can have different one-sided traces.  For an upper
        # negative-to-positive crossing the paper's upward perturbation uses
        # the stopping-side (right) trace.  Fit gap = slope*d + curvature*d^2
        # over a short physical window, constrained through the interpolated
        # root, instead of averaging across the kink.
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
        roots.append(float(root))
        slopes.append(float(slope))
    return tuple(roots), tuple(slopes)


def solve_stationary_bellman(
    *,
    grid: np.ndarray,
    law: DiscreteCreditedLaw,
    contract: ContractSpec,
    tolerance: float = 1e-11,
    max_iterations: int = 20_000,
    tail_check_fraction: float = 0.02,
) -> BellmanSolution:
    """Solve the stationary surrender problem by contraction value iteration."""

    grid = np.asarray(grid, dtype=float)
    if grid.ndim != 1 or len(grid) < 200 or np.any(np.diff(grid) <= 0):
        raise ValueError("grid must be a strictly increasing one-dimensional array")
    law.validate()
    contract.validate()
    if tolerance <= 0 or max_iterations < 1:
        raise ValueError("invalid iteration controls")

    liquidation = contract.surrender_value(grid)
    value = np.maximum(liquidation, contract.termination_value(grid))
    sup_error = np.inf
    continuation = np.empty_like(grid)
    for iteration in range(1, max_iterations + 1):
        continuation = continuation_value(grid, value, law, contract)
        updated = np.maximum(liquidation, continuation)
        sup_error = float(np.max(np.abs(updated - value)))
        value = updated
        if sup_error <= tolerance:
            break
    else:
        raise RuntimeError(
            f"value iteration did not converge after {max_iterations} iterations; "
            f"sup error={sup_error}"
        )

    # Recompute the continuation against the final fixed point.
    continuation = continuation_value(grid, value, law, contract)
    gap = liquidation - continuation
    stopping = gap >= 0
    tail_points = max(5, int(np.ceil(tail_check_fraction * len(grid))))
    if not np.all(stopping[-tail_points:]):
        raise ValueError(
            "upper grid does not lie in a verified stopping tail; expand the grid"
        )
    roots, root_slopes = _roots_and_slopes(grid, gap)
    return BellmanSolution(
        grid=grid,
        value=value,
        continuation=continuation,
        gap=gap,
        stopping=stopping,
        roots=roots,
        root_slopes=root_slopes,
        iterations=iteration,
        sup_error=sup_error,
    )


def upper_regular_boundary(solution: BellmanSolution) -> tuple[float, float]:
    """Return the highest negative-to-positive regular root and its slope."""

    candidates = [
        (root, slope)
        for root, slope in zip(solution.roots, solution.root_slopes)
        if slope > 0
    ]
    if not candidates:
        raise ValueError("no regular upper surrender boundary was found")
    return max(candidates, key=lambda item: item[0])
