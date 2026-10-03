"""Finite one-date/additive and ordered-prefix comparator utilities.

The functions here evaluate the same fixed reference model throughout.  A
singleton policy changes exactly one supplied threshold and uses the reference
boundary at every other date.  The direct decomposition mirrors the exact
performance-difference identity: ``S`` uses reference-survival mass, whereas
``J`` uses the difference between joint-policy and reference-survival mass.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cliquet import Contract
from uncapped_cliquet import (
    UncappedLatticeReference,
    _advance_mass,
    forward_performance_difference_uncapped,
)


@dataclass(frozen=True)
class FiniteComparatorResult:
    loss: float
    singleton_sum: float
    interaction: float
    direct_loss: float
    direct_singleton_sum: float
    direct_interaction: float
    loss_identity_error: float
    singleton_identity_error: float
    interaction_identity_error: float
    native_loss: float
    date_loss: tuple[float, ...]
    date_singleton: tuple[float, ...]
    date_interaction: tuple[float, ...]


@dataclass(frozen=True)
class FiniteComparatorBatch:
    loss: np.ndarray
    singleton_sum: np.ndarray
    interaction: np.ndarray
    date_loss: np.ndarray
    date_singleton: np.ndarray
    date_interaction: np.ndarray
    native_loss: float


def singleton_thresholds(boundary: float, joint_thresholds: np.ndarray) -> np.ndarray:
    """Return one policy per date, changing only that date's threshold."""

    joint = np.asarray(joint_thresholds, dtype=float)
    if joint.ndim != 1:
        raise ValueError("joint_thresholds must be one-dimensional")
    dates = len(joint)
    matrix = np.full((dates, dates), float(boundary))
    matrix[np.arange(dates), np.arange(dates)] = joint
    return matrix


def assert_boundary_action_equivalence(
    reference,
    *,
    boundary: float,
    dates: int,
) -> None:
    """Require the supplied boundary to encode the evaluator's nodal action.

    Both lattice and quadrature/backward references used here expose a grid and
    date-indexed optimal gaps; no property specific to either evaluator enters
    this guard.
    """

    boundary_stop = reference.grid >= float(boundary)
    for date in range(1, dates + 1):
        optimal_stop = reference.optimal_gaps[date] >= 0
        if not np.array_equal(boundary_stop, optimal_stop):
            raise ValueError(
                "the supplied boundary and discrete Bellman action differ; "
                "the singleton and direct decompositions are then not equivalent"
            )


def finite_lattice_decomposition(
    reference: UncappedLatticeReference,
    contract: Contract,
    *,
    boundary: float,
    joint_thresholds: np.ndarray,
) -> FiniteComparatorResult:
    """Evaluate ``L``, ``S``, and ``J`` in two algebraically equivalent ways.

    Equivalence requires the supplied boundary threshold to reproduce the
    discrete Bellman-optimal action at every date; this is checked explicitly.
    """

    joint = np.asarray(joint_thresholds, dtype=float)
    dates = contract.exercise_dates
    if joint.shape != (dates,):
        raise ValueError("joint threshold vector has the wrong length")
    assert_boundary_action_equivalence(
        reference, boundary=boundary, dates=dates
    )

    native = np.full(dates, float(boundary))
    singletons = singleton_thresholds(boundary, joint)
    policies = np.vstack((native, joint, singletons))
    forward = forward_performance_difference_uncapped(
        reference, contract, policy_thresholds=policies
    )
    native_loss = float(forward.total_losses[0])
    loss = float(forward.total_losses[1])
    singleton_losses = np.asarray(forward.total_losses[2:], dtype=float)
    singleton_sum = float(singleton_losses.sum())
    interaction = loss - singleton_sum

    survivor_reference = np.zeros(len(reference.grid))
    survivor_joint = np.zeros(len(reference.grid))
    survivor_reference[0] = 1.0
    survivor_joint[0] = 1.0
    direct_date_loss: list[float] = []
    direct_date_singleton: list[float] = []
    direct_date_interaction: list[float] = []

    for date in range(1, dates + 1):
        pre_reference = _advance_mass(survivor_reference, reference.credit_mass)
        pre_joint = _advance_mass(survivor_joint, reference.credit_mass)
        optimal_stop = reference.optimal_gaps[date] >= 0
        joint_stop = reference.grid >= joint[date - 1]
        mismatch = joint_stop != optimal_stop
        weighted_gap = np.abs(reference.optimal_gaps[date])
        discount = contract.beta**date

        singleton_date = discount * float(
            np.sum(pre_reference * mismatch * weighted_gap)
        )
        interaction_date = discount * float(
            np.sum((pre_joint - pre_reference) * mismatch * weighted_gap)
        )
        joint_date = discount * float(
            np.sum(pre_joint * mismatch * weighted_gap)
        )
        direct_date_singleton.append(singleton_date)
        direct_date_interaction.append(interaction_date)
        direct_date_loss.append(joint_date)

        survivor_reference = np.where(optimal_stop, 0.0, pre_reference)
        survivor_joint = np.where(joint_stop, 0.0, pre_joint)

    direct_loss = float(sum(direct_date_loss))
    direct_singleton_sum = float(sum(direct_date_singleton))
    direct_interaction = float(sum(direct_date_interaction))
    return FiniteComparatorResult(
        loss=loss,
        singleton_sum=singleton_sum,
        interaction=interaction,
        direct_loss=direct_loss,
        direct_singleton_sum=direct_singleton_sum,
        direct_interaction=direct_interaction,
        loss_identity_error=loss - direct_loss,
        singleton_identity_error=singleton_sum - direct_singleton_sum,
        interaction_identity_error=interaction - direct_interaction,
        native_loss=native_loss,
        date_loss=tuple(direct_date_loss),
        date_singleton=tuple(direct_date_singleton),
        date_interaction=tuple(direct_date_interaction),
    )


def finite_lattice_decomposition_batch(
    reference: UncappedLatticeReference,
    contract: Contract,
    *,
    boundary: float,
    joint_thresholds: np.ndarray,
) -> FiniteComparatorBatch:
    """Vectorized direct gate-product decomposition for several policies."""

    thresholds = np.asarray(joint_thresholds, dtype=float)
    if thresholds.ndim == 1:
        thresholds = thresholds[None, :]
    policy_count, dates = thresholds.shape
    if dates != contract.exercise_dates:
        raise ValueError("joint threshold matrix has the wrong width")
    assert_boundary_action_equivalence(
        reference, boundary=boundary, dates=dates
    )

    survivor_reference = np.zeros(len(reference.grid))
    survivor_reference[0] = 1.0
    survivor_joint = np.zeros((policy_count, len(reference.grid)))
    survivor_joint[:, 0] = 1.0
    date_loss = np.zeros((policy_count, dates))
    date_singleton = np.zeros_like(date_loss)
    date_interaction = np.zeros_like(date_loss)
    native_loss = 0.0

    for date in range(1, dates + 1):
        pre_reference = _advance_mass(survivor_reference, reference.credit_mass)
        pre_joint = _advance_mass(survivor_joint, reference.credit_mass)
        optimal_stop = reference.optimal_gaps[date] >= 0
        native_stop = reference.grid >= float(boundary)
        joint_stop = reference.grid[None, :] >= thresholds[:, date - 1, None]
        mismatch = joint_stop != optimal_stop[None, :]
        weighted_gap = np.abs(reference.optimal_gaps[date])
        discount = contract.beta**date

        singleton = discount * np.sum(
            pre_reference[None, :] * mismatch * weighted_gap[None, :], axis=1
        )
        interaction = discount * np.sum(
            (pre_joint - pre_reference[None, :])
            * mismatch
            * weighted_gap[None, :],
            axis=1,
        )
        date_singleton[:, date - 1] = singleton
        date_interaction[:, date - 1] = interaction
        date_loss[:, date - 1] = discount * np.sum(
            pre_joint * mismatch * weighted_gap[None, :], axis=1
        )
        native_loss += discount * float(
            np.sum(
                pre_reference
                * (native_stop != optimal_stop)
                * weighted_gap
            )
        )

        survivor_reference = np.where(optimal_stop, 0.0, pre_reference)
        survivor_joint = np.where(joint_stop, 0.0, pre_joint)

    return FiniteComparatorBatch(
        loss=date_loss.sum(axis=1),
        singleton_sum=date_singleton.sum(axis=1),
        interaction=date_interaction.sum(axis=1),
        date_loss=date_loss,
        date_singleton=date_singleton,
        date_interaction=date_interaction,
        native_loss=float(native_loss),
    )


def quadrature_subtraction_decomposition(
    regrets: np.ndarray,
) -> tuple[float, float, float]:
    """Read joint and singleton losses from a backward-regret policy batch.

    The input order is joint first, followed by one singleton for every date.
    This is an independent-discretization check, not the primary gate-product
    evaluation of the finite interaction.
    """

    values = np.asarray(regrets, dtype=float)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("expected a joint loss followed by singleton losses")
    loss = float(values[0])
    singleton_sum = float(values[1:].sum())
    return loss, singleton_sum, loss - singleton_sum


def residual_components(
    *,
    loss: float,
    singleton_sum: float,
    interaction: float,
    additive_prediction: float,
    full_prediction: float,
) -> tuple[float, float, float, float]:
    """Return one-date, interaction, full residuals and their identity error."""

    predicted_interaction = full_prediction - additive_prediction
    one_date_residual = singleton_sum - additive_prediction
    interaction_residual = interaction - predicted_interaction
    full_residual = loss - full_prediction
    identity_error = full_residual - (one_date_residual + interaction_residual)
    return (
        float(one_date_residual),
        float(interaction_residual),
        float(full_residual),
        float(identity_error),
    )
