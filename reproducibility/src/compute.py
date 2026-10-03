"""Deterministic formulas used by the revised manuscript.

The module intentionally uses only the Python standard library.  All numerical
integrals use fixed-order Gauss--Legendre quadrature, so a run is bitwise stable
on a fixed Python/libm platform and contains no Monte Carlo noise.
"""

from __future__ import annotations

import functools
import math
from typing import Callable, Iterable


SQRT2 = math.sqrt(2.0)
SQRT2PI = math.sqrt(2.0 * math.pi)


def phi(x: float) -> float:
    return math.exp(-0.5 * x * x) / SQRT2PI


def Phi(x: float) -> float:
    return 0.5 * math.erfc(-x / SQRT2)


@functools.lru_cache(maxsize=None)
def gauss_legendre(order: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Nodes and weights on [-1,1], computed by Newton iteration."""
    if order < 2:
        raise ValueError("order must be at least 2")
    nodes = [0.0] * order
    weights = [0.0] * order
    half = (order + 1) // 2
    for i in range(half):
        z = math.cos(math.pi * (i + 0.75) / (order + 0.5))
        for _ in range(100):
            p0, p1 = 1.0, z
            for j in range(2, order + 1):
                p0, p1 = p1, ((2 * j - 1) * z * p1 - (j - 1) * p0) / j
            derivative = order * (z * p1 - p0) / (z * z - 1.0)
            step = p1 / derivative
            z -= step
            if abs(step) <= 2e-16:
                break
        weight = 2.0 / ((1.0 - z * z) * derivative * derivative)
        nodes[i] = -z
        nodes[order - 1 - i] = z
        weights[i] = weight
        weights[order - 1 - i] = weight
    return tuple(nodes), tuple(weights)


def integrate(f: Callable[[float], float], a: float, b: float, order: int = 256) -> float:
    if a == b:
        return 0.0
    if b < a:
        return -integrate(f, b, a, order)
    nodes, weights = gauss_legendre(order)
    mid = 0.5 * (a + b)
    half = 0.5 * (b - a)
    return half * math.fsum(w * f(mid + half * x) for x, w in zip(nodes, weights))


def mismatch(u: float, h: float) -> float:
    return float((u >= h) != (u >= 0.0))


def all_sign_formula(
    h1: float,
    h2: float,
    a: float,
    lambda1: float = 1.0,
    lambda2: float = 1.0,
    rho: float = 1.0,
) -> float:
    if a == 0.0:
        raise ValueError("a must be nonzero")
    positive = lambda x: max(x, 0.0)
    negative = lambda x: max(-x, 0.0)
    ray_threshold = h2 / a
    later = min(positive(ray_threshold), positive(h1)) ** 2
    later += max(negative(ray_threshold) ** 2 - negative(h1) ** 2, 0.0)
    return 0.5 * rho * (lambda1 * h1 * h1 + lambda2 * abs(a) * later)


def all_sign_direct(
    h1: float,
    h2: float,
    a: float,
    lambda1: float = 1.0,
    lambda2: float = 1.0,
    rho: float = 1.0,
    order: int = 256,
) -> float:
    r = h2 / a
    lo, hi = sorted((0.0, r))
    cuts = [lo, hi]
    if lo < h1 < hi:
        cuts.insert(1, h1)
    later = 0.0
    for left, right in zip(cuts[:-1], cuts[1:]):
        midpoint = 0.5 * (left + right)
        if midpoint < h1 and mismatch(a * midpoint, h2):
            later += integrate(abs, left, right, order)
    return 0.5 * rho * lambda1 * h1 * h1 + rho * lambda2 * abs(a) * later


def gaussian_crossover(
    h1: float,
    h2: float,
    a: float,
    kappa: float,
    lambda1: float = 1.0,
    lambda2: float = 1.0,
    rho0: float = 1.0,
    order: int = 256,
) -> float:
    lo, hi = sorted((0.0, h2))
    sign_a = 1.0 if a > 0.0 else -1.0
    later = integrate(
        lambda v: abs(v)
        * Phi(sign_a * (a * h1 - v) / kappa),
        lo,
        hi,
        order,
    )
    return rho0 * (0.5 * lambda1 * h1 * h1 + lambda2 * later / abs(a))


def gaussian_diffuse_endpoint(
    h1: float,
    h2: float,
    a: float,
    lambda1: float = 1.0,
    lambda2: float = 1.0,
    rho0: float = 1.0,
) -> float:
    return rho0 * (0.5 * lambda1 * h1 * h1 + lambda2 * h2 * h2 / (4.0 * abs(a)))


def cubic_correction(epsilon: float, h1: float = 1.0, h2: float = 1.0) -> float:
    """Exact correction for independent standard-normal d1,d2 and D2=d2."""
    first = 0.5 * math.erf(epsilon * h1 / SQRT2)
    second = phi(0.0) * (-math.expm1(-0.5 * (epsilon * h2) ** 2))
    return first * second


def log_slope(xs: Iterable[float], ys: Iterable[float]) -> float:
    lx = [math.log(x) for x in xs]
    ly = [math.log(y) for y in ys]
    mx, my = math.fsum(lx) / len(lx), math.fsum(ly) / len(ly)
    num = math.fsum((x - mx) * (y - my) for x, y in zip(lx, ly))
    den = math.fsum((x - mx) ** 2 for x in lx)
    return num / den


def singular_tube_probability(delta: float) -> float:
    return math.erf(delta / SQRT2)


def diffuse_tube_probability(delta: float, sigma: float, order: int = 256) -> float:
    """P(|U|<=delta, |U+sigma Z|<=delta), independent standard normals."""
    return integrate(
        lambda u: phi(u)
        * (Phi((delta - u) / sigma) - Phi((-delta - u) / sigma)),
        -delta,
        delta,
        order,
    )


def running_max_coefficients(
    S0: float = 100.0,
    K: float = 120.0,
    t1: float = 0.5,
    t2: float = 1.0,
    r: float = 0.03,
    q: float = 0.0,
    sigma: float = 0.2,
    order: int = 1024,
    tail_sd: float = 12.0,
) -> tuple[float, float, float]:
    """Evaluate Appendix (A.8) by finite-interval Gauss--Legendre quadrature."""
    x0 = math.log(S0)
    k = math.log(K)
    mu = r - q - 0.5 * sigma * sigma
    Delta = t2 - t1
    lower = x0 + mu * t1 - tail_sd * sigma * math.sqrt(t1)

    def killed(x: float) -> float:
        scale = sigma * math.sqrt(t1)
        direct = phi((x - x0 - mu * t1) / scale)
        reflected = math.exp(2.0 * mu * (k - x0) / (sigma * sigma)) * phi(
            (x + x0 - 2.0 * k - mu * t1) / scale
        )
        return (direct - reflected) / scale

    def joint_max(x: float) -> float:
        arg = (x + x0 - 2.0 * k - mu * t1) / (sigma * math.sqrt(t1))
        return (
            2.0
            * (2.0 * k - x - x0)
            / (sigma**3 * t1**1.5)
            * math.exp(2.0 * mu * (k - x0) / (sigma * sigma))
            * phi(arg)
        )

    def no_hit(x: float) -> float:
        scale = sigma * math.sqrt(Delta)
        a = (k - x - mu * Delta) / scale
        c = (x - k - mu * Delta) / scale
        exponent = math.exp(2.0 * mu * (k - x) / (sigma * sigma))
        return Phi(a) - exponent * Phi(c)

    def no_hit_derivative(x: float) -> float:
        scale = sigma * math.sqrt(Delta)
        a = (k - x - mu * Delta) / scale
        c = (x - k - mu * Delta) / scale
        exponent = math.exp(2.0 * mu * (k - x) / (sigma * sigma))
        return (
            phi(a) / scale
            - exponent * (2.0 * mu / (sigma * sigma)) * Phi(c)
            + exponent * phi(c) / scale
        )

    rho1 = integrate(joint_max, lower, k, order) / K
    eta = integrate(lambda x: joint_max(x) * no_hit(x), lower, k, order) / K
    rho2 = integrate(lambda x: killed(x) * no_hit_derivative(x), lower, k, order) / K
    return rho1, rho2, eta


def full_recall_coefficients(
    a: float = 1.0,
    d: float = 1.0,
    boundary_cdf: float = 0.5,
    boundary_density: float = 1.0,
    stopping_slope: float = 0.5,
) -> tuple[float, float, float]:
    """Leading date-1, new-record, and copied-state search coefficients."""
    date1 = 0.5 * stopping_slope * boundary_density * a * a
    new_record = 0.5 * stopping_slope * boundary_cdf * boundary_density * d * d
    copied = (
        0.5
        * stopping_slope
        * boundary_cdf
        * boundary_density
        * min(a, d) ** 2
    )
    return date1, new_record, copied


def full_recall_uniform_channels(epsilon: float) -> tuple[float, float, float]:
    """Exact finite-scale channels for Uniform[0,1], c=1/8, unit shifts."""
    if not 0.0 <= epsilon <= 0.5:
        raise ValueError("the displayed local formula requires 0 <= epsilon <= 1/2")
    date1 = 0.25 * epsilon**2 - epsilon**3 / 6.0
    later = 0.125 * epsilon**2 + epsilon**3 / 12.0 - 0.125 * epsilon**4
    return date1, later, later


def full_recall_uniform_exact(epsilon: float) -> float:
    """Exact total regret for the same full-recall uniform instance."""
    if not 0.0 <= epsilon <= 0.5:
        raise ValueError("the displayed local formula requires 0 <= epsilon <= 1/2")
    return 0.5 * epsilon**2 - 0.25 * epsilon**4


def resurrection_coefficients(a: float, b: float) -> tuple[float, float]:
    model_a = a * a / 4.0 + b * b / 16.0
    model_b = model_a + min(a, b) ** 2 / 8.0
    return model_a, model_b


def error_coupling_checks() -> dict[str, float]:
    aligned = 0.5 * sum(0.5 * (h1 * h1 + min(h1, h2) ** 2) for h1, h2 in [(1, 1), (2, 2)])
    anti = 0.5 * sum(0.5 * (h1 * h1 + min(h1, h2) ** 2) for h1, h2 in [(1, 2), (2, 1)])
    a_pairs = list(zip((1, 2, 3, 4), (1, 2, 4, 3)))
    b_pairs = list(zip((1, 2, 3, 4), (1, 3, 2, 4)))

    def moment(pairs: list[tuple[int, int]], fn: Callable[[int, int], float]) -> float:
        return math.fsum(fn(x, y) for x, y in pairs) / len(pairs)

    return {
        "aligned": aligned,
        "anti_aligned": anti,
        "cross_moment_A": moment(a_pairs, lambda x, y: x * y),
        "cross_moment_B": moment(b_pairs, lambda x, y: x * y),
        "squared_min_A": moment(a_pairs, lambda x, y: min(x, y) ** 2),
        "squared_min_B": moment(b_pairs, lambda x, y: min(x, y) ** 2),
        "coefficient_A": moment(a_pairs, lambda x, y: 0.5 * (x * x + min(x, y) ** 2)),
        "coefficient_B": moment(b_pairs, lambda x, y: 0.5 * (x * x + min(x, y) ** 2)),
    }
