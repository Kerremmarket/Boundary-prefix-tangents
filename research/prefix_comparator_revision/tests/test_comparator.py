from __future__ import annotations

import numpy as np

from cliquet import (
    Contract,
    build_law,
    one_sided_slopes,
    prefix_coefficient,
    solve_boundary,
    trace_densities,
)
from comparator import (
    finite_lattice_decomposition,
    finite_lattice_decomposition_batch,
    residual_components,
    singleton_thresholds,
)
from uncapped_cliquet import solve_uncapped_lattice_reference


def _fixture():
    law = build_law(
        model="black_scholes_atm",
        gross_forward=1.02,
        atm_volatility=0.20,
        local_cap=0.08,
    )
    contract = Contract(0.08, 0.24, 6, 0.97)
    boundary_result = solve_boundary(law, contract)
    assert boundary_result.boundary is not None
    reference = solve_uncapped_lattice_reference(
        law, contract, intervals_per_local_cap=512
    )
    return law, contract, float(boundary_result.boundary), reference


def test_singleton_matrix_changes_only_one_date():
    boundary = 0.2
    joint = np.array([0.19, 0.21, 0.18])
    matrix = singleton_thresholds(boundary, joint)
    assert matrix.shape == (3, 3)
    assert np.allclose(np.diag(matrix), joint)
    assert np.allclose(matrix - np.diag(np.diag(matrix)),
                       np.full((3, 3), boundary) - np.eye(3) * boundary)


def test_finite_decomposition_and_common_signs():
    _, contract, boundary, reference = _fixture()
    for sign in (1.0, -1.0):
        thresholds = np.full(contract.exercise_dates, boundary + sign * 0.001)
        result = finite_lattice_decomposition(
            reference, contract, boundary=boundary, joint_thresholds=thresholds
        )
        assert abs(result.loss_identity_error) < 1e-12
        assert abs(result.singleton_identity_error) < 1e-12
        assert abs(result.interaction_identity_error) < 1e-12
        assert abs(result.loss - result.singleton_sum - result.interaction) < 1e-12
        if sign > 0:
            assert result.interaction >= -1e-14
        else:
            assert result.interaction <= 1e-14


def test_singleton_has_zero_interaction():
    _, contract, boundary, reference = _fixture()
    thresholds = np.full(contract.exercise_dates, boundary)
    thresholds[2] += 0.001
    result = finite_lattice_decomposition(
        reference, contract, boundary=boundary, joint_thresholds=thresholds
    )
    assert abs(result.interaction) < 1e-12


def test_residual_identity():
    values = residual_components(
        loss=1.2,
        singleton_sum=1.0,
        interaction=0.2,
        additive_prediction=0.9,
        full_prediction=1.15,
    )
    one_date, interaction, full, error = values
    assert np.isclose(one_date, 0.1)
    assert np.isclose(interaction, -0.05)
    assert np.isclose(full, 0.05)
    assert abs(error) < 1e-14


def test_batch_and_explicit_policy_evaluations_agree():
    _, contract, boundary, reference = _fixture()
    thresholds = np.vstack(
        (
            np.full(contract.exercise_dates, boundary + 0.001),
            np.full(contract.exercise_dates, boundary - 0.001),
        )
    )
    batch = finite_lattice_decomposition_batch(
        reference, contract, boundary=boundary, joint_thresholds=thresholds
    )
    for index, threshold in enumerate(thresholds):
        explicit = finite_lattice_decomposition(
            reference, contract, boundary=boundary, joint_thresholds=threshold
        )
        assert np.isclose(batch.loss[index], explicit.loss, atol=1e-13, rtol=0)
        assert np.isclose(
            batch.singleton_sum[index], explicit.singleton_sum, atol=1e-13, rtol=0
        )
        assert np.isclose(
            batch.interaction[index], explicit.interaction, atol=1e-13, rtol=0
        )
    assert batch.native_loss == 0.0


def test_prefix_coefficient_is_positive_two_homogeneous():
    law, contract, boundary, _ = _fixture()
    trace = trace_densities(
        law, contract, boundary=boundary, intervals_per_cap=1024
    )
    slopes = one_sided_slopes(law, contract, boundary)
    direction = np.linspace(0.25, 1.0, contract.exercise_dates)
    base = prefix_coefficient(
        direction=direction,
        contract=contract,
        law=law,
        trace=trace,
        slopes=slopes,
    )
    scaled = prefix_coefficient(
        direction=3.0 * direction,
        contract=contract,
        law=law,
        trace=trace,
        slopes=slopes,
    )
    assert np.isclose(scaled.full, 9.0 * base.full, atol=2e-12, rtol=1e-12)
    assert np.isclose(
        scaled.additive_single_date,
        9.0 * base.additive_single_date,
        atol=2e-12,
        rtol=1e-12,
    )


def test_boundary_action_mismatch_is_rejected():
    _, contract, boundary, reference = _fixture()
    thresholds = np.full(contract.exercise_dates, boundary + 0.001)
    with np.testing.assert_raises(ValueError):
        finite_lattice_decomposition(
            reference,
            contract,
            boundary=boundary + 0.05,
            joint_thresholds=thresholds,
        )
