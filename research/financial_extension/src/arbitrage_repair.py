"""Sparse cross-strike repair for one European SPX expiry slice."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linprog


@dataclass(frozen=True)
class RepairResult:
    quotes: pd.DataFrame
    feasible_within_spread: bool
    weighted_l1_objective: float
    max_absolute_change: float
    rows_changed: int


def _constraint_matrix(strikes: np.ndarray, discount: float) -> tuple[np.ndarray, np.ndarray]:
    n = len(strikes)
    rows: list[np.ndarray] = []
    bounds: list[float] = []

    # Monotonicity and the European call vertical-spread lower slope bound.
    for index in range(n - 1):
        width = strikes[index + 1] - strikes[index]
        monotone = np.zeros(2 * n)
        monotone[index + 1] = 1.0
        monotone[index] = -1.0
        rows.append(monotone)
        bounds.append(0.0)

        lower_slope = np.zeros(2 * n)
        lower_slope[index] = 1.0
        lower_slope[index + 1] = -1.0
        rows.append(lower_slope)
        bounds.append(discount * width)

    # Convexity: adjacent call-price slopes are nondecreasing.
    for index in range(n - 2):
        left_width = strikes[index + 1] - strikes[index]
        right_width = strikes[index + 2] - strikes[index + 1]
        convex = np.zeros(2 * n)
        convex[index] = -1.0 / left_width
        convex[index + 1] = 1.0 / left_width + 1.0 / right_width
        convex[index + 2] = -1.0 / right_width
        rows.append(convex)
        bounds.append(0.0)

    return np.asarray(rows), np.asarray(bounds)


def _solve(
    *,
    strikes: np.ndarray,
    mids: np.ndarray,
    scales: np.ndarray,
    price_bounds: list[tuple[float, float | None]],
    discount: float,
) -> tuple[np.ndarray, float] | None:
    n = len(strikes)
    objective = np.concatenate((np.zeros(n), 1.0 / scales))
    shape_constraints, shape_bounds = _constraint_matrix(strikes, discount)

    absolute_rows = []
    absolute_bounds = []
    for index in range(n):
        positive = np.zeros(2 * n)
        positive[index] = 1.0
        positive[n + index] = -1.0
        absolute_rows.append(positive)
        absolute_bounds.append(mids[index])

        negative = np.zeros(2 * n)
        negative[index] = -1.0
        negative[n + index] = -1.0
        absolute_rows.append(negative)
        absolute_bounds.append(-mids[index])

    a_ub = np.vstack((shape_constraints, np.asarray(absolute_rows)))
    b_ub = np.concatenate((shape_bounds, np.asarray(absolute_bounds)))
    bounds = price_bounds + [(0.0, None)] * n
    result = linprog(
        objective,
        A_ub=a_ub,
        b_ub=b_ub,
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        return None
    return np.asarray(result.x[:n]), float(result.fun)


def repair_call_slice(quotes: pd.DataFrame) -> RepairResult:
    """Repair monotonicity/convexity with bid--ask feasibility attempted first.

    The input must be one `(date, expiry, settlement)` slice after OTM
    conversion to call-equivalent prices.  Calendar arbitrage is deliberately
    not claimed or repaired by this function.
    """

    required = {
        "strike",
        "call_equivalent_bid",
        "call_equivalent_offer",
        "call_equivalent_mid",
        "discount_factor",
        "matched_forward_price",
    }
    missing = sorted(required - set(quotes.columns))
    if missing:
        raise ValueError(f"quote slice is missing columns: {missing}")
    if len(quotes) < 3:
        raise ValueError("at least three strikes are required for convexity repair")

    frame = quotes.sort_values("strike").reset_index(drop=True).copy()
    if frame["strike"].duplicated().any():
        raise ValueError("quote slice must contain one selected quote per strike")
    strikes = frame["strike"].to_numpy(dtype=float)
    if not np.all(np.diff(strikes) > 0):
        raise ValueError("strikes must be finite and strictly increasing")
    discount_values = frame["discount_factor"].to_numpy(dtype=float)
    forward_values = frame["matched_forward_price"].to_numpy(dtype=float)
    if not np.allclose(discount_values, discount_values[0], rtol=0, atol=1e-12):
        raise ValueError("discount factor must be constant within an expiry slice")
    if not np.allclose(forward_values, forward_values[0], rtol=0, atol=1e-10):
        raise ValueError("forward must be constant within an expiry slice")
    discount = float(discount_values[0])
    forward = float(forward_values[0])
    if not 0 < discount <= 1.5 or forward <= 0:
        raise ValueError("invalid discount factor or forward")

    bids = frame["call_equivalent_bid"].to_numpy(dtype=float)
    offers = frame["call_equivalent_offer"].to_numpy(dtype=float)
    mids = frame["call_equivalent_mid"].to_numpy(dtype=float)
    if not (
        np.all(np.isfinite(bids))
        and np.all(np.isfinite(offers))
        and np.all(np.isfinite(mids))
        and np.all(offers > bids)
    ):
        raise ValueError("call-equivalent quote intervals are invalid")
    scales = np.maximum((offers - bids) / 2.0, 0.01)
    fundamental_lower = discount * np.maximum(forward - strikes, 0.0)
    fundamental_upper = np.full(len(strikes), discount * forward)

    within_bounds: list[tuple[float, float | None]] = []
    within_possible = True
    for lower, upper, bid, offer in zip(
        fundamental_lower, fundamental_upper, bids, offers
    ):
        interval = (max(lower, bid), min(upper, offer))
        if interval[0] > interval[1]:
            within_possible = False
        within_bounds.append(interval)

    solved = None
    if within_possible:
        solved = _solve(
            strikes=strikes,
            mids=mids,
            scales=scales,
            price_bounds=within_bounds,
            discount=discount,
        )
    feasible_within_spread = solved is not None
    if solved is None:
        solved = _solve(
            strikes=strikes,
            mids=mids,
            scales=scales,
            price_bounds=[
                (float(lower), float(upper))
                for lower, upper in zip(fundamental_lower, fundamental_upper)
            ],
            discount=discount,
        )
    if solved is None:
        raise RuntimeError("cross-strike repair LP is infeasible under fundamental bounds")

    repaired, objective = solved
    changes = repaired - mids
    frame["repaired_call_price"] = repaired
    frame["repair_change"] = changes
    frame["repair_outside_spread"] = (repaired < bids - 1e-10) | (
        repaired > offers + 1e-10
    )
    return RepairResult(
        quotes=frame,
        feasible_within_spread=feasible_within_spread,
        weighted_l1_objective=objective,
        max_absolute_change=float(np.max(np.abs(changes))),
        rows_changed=int(np.sum(np.abs(changes) > 1e-10)),
    )
