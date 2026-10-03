from __future__ import annotations

import math

import numpy as np

from cliquet import (
    Contract,
    CreditLaw,
    LognormalComponent,
    _advance_mass_lattice,
    _transition_lattice,
    forward_performance_difference,
    gate_integral,
    make_directions,
    one_sided_slopes,
    prefix_coefficient,
    quadrature_policy_regret,
    solve_boundary,
    solve_lattice,
    solve_quadrature,
    trace_densities,
)


def example():
    law = CreditLaw(
        components=(LognormalComponent(1.0, 1.02, 0.20),),
        local_cap=0.08,
    )
    contract = Contract(
        local_cap=0.08,
        global_cap=0.24,
        resets=6,
        beta=math.exp(-0.03),
    )
    boundary_result = solve_boundary(law, contract)
    assert boundary_result.boundary is not None
    return law, contract, boundary_result.boundary


def test_analytic_moment_matches_quadrature():
    law, _, _ = example()
    credits, weights = law.quadrature(192)
    for level in (0.01, 0.04, 0.08, 0.12):
        observed = float(np.dot(weights, np.minimum(credits, level)))
        # The global Gauss rule sees a kink at ``C=level``; this is an
        # independent numerical check of the closed form, not an identity.
        assert abs(observed - law.expected_min_credit(level)) < 5e-8


def test_lattice_preserves_atoms_mass_and_credit_mean():
    law, _, _ = example()
    mass, step = law.lattice_mass(2048)
    support = np.arange(len(mass)) * step
    assert mass[0] == law.floor_probability
    assert mass[-1] == law.cap_probability
    assert abs(mass.sum() - 1.0) < 2e-13
    assert abs(np.dot(mass, support) - law.expected_credit) < 2e-8


def test_boundary_is_unique_and_beta_edge_has_no_interior_root():
    law, contract, boundary = example()
    assert 0 < boundary < contract.global_cap
    residual = (
        (1 - contract.beta) * boundary
        - contract.beta
        * law.expected_min_credit(contract.global_cap - boundary)
    )
    assert abs(residual) < 1e-13
    no_boundary = solve_boundary(
        law,
        Contract(0.08, 0.24, 6, beta=1.0),
    )
    assert no_boundary.status == "beta_at_least_one_no_interior_boundary"
    assert no_boundary.boundary is None
    negative_rate = solve_boundary(
        law,
        Contract(0.08, 0.24, 6, beta=1.01),
    )
    assert negative_rate.status == "beta_at_least_one_no_interior_boundary"
    assert negative_rate.boundary is None


def test_lattice_transition_and_forward_mass_match_brute_force():
    values = np.array([0.0, 1.0, 4.0, 9.0])
    weights = np.array([0.2, 0.3, 0.5])
    expected = np.array(
        [
            sum(weights[k] * values[min(i + k, len(values) - 1)] for k in range(3))
            for i in range(4)
        ]
    )
    assert np.allclose(_transition_lattice(values, weights), expected, atol=1e-13)
    initial = np.array([1.0, 0.0, 0.0, 0.0])
    advanced = _advance_mass_lattice(initial, weights)
    assert np.allclose(advanced, np.array([0.2, 0.3, 0.5, 0.0]), atol=1e-13)
    assert abs(advanced.sum() - 1.0) < 1e-13


def test_same_boundary_at_every_finite_horizon_date():
    law, contract, boundary = example()
    lattice = solve_lattice(law, contract, intervals_per_cap=1024)
    step = contract.local_cap / 1024
    assert max(abs(value - boundary) for value in lattice.grid_boundaries.values()) < 2 * step
    quadrature = solve_quadrature(
        law,
        contract,
        intervals_per_cap=1024,
        nodes_per_component=96,
    )
    assert max(abs(value - boundary) for value in quadrature.grid_boundaries.values()) < 2 * step
    assert abs(lattice.optimal_issue_value - quadrature.optimal_issue_value) < 2e-4


def test_slopes_are_positive_and_match_last_date_difference():
    law, contract, boundary = example()
    slopes = one_sided_slopes(law, contract, boundary)
    assert slopes.right > 0
    assert min(slopes.left_by_remaining_resets.values()) > 0
    contact = contract.global_cap - boundary
    step = 1e-7

    def last_gap(x):
        return x - contract.beta * (
            x + law.expected_min_credit(contract.global_cap - x)
        )

    right_numeric = (last_gap(boundary + step) - last_gap(boundary)) / step
    left_numeric = (last_gap(boundary) - last_gap(boundary - step)) / step
    assert abs(right_numeric - slopes.right) < 2e-5
    assert abs(left_numeric - slopes.left_by_remaining_resets[1]) < 2e-5
    assert 0 < contact < contract.global_cap


def test_trace_component_reconciliation_and_sign_gates():
    law, contract, boundary = example()
    slopes = one_sided_slopes(law, contract, boundary)
    trace = trace_densities(
        law,
        contract,
        boundary=boundary,
        intervals_per_cap=2048,
    )
    assert max(trace.component_reconciliation_error.values()) < 1e-8
    assert max(trace.fresh_density.values()) > 0
    directions = make_directions(contract.exercise_dates)
    up = prefix_coefficient(
        direction=directions["all_up"],
        contract=contract,
        law=law,
        trace=trace,
        slopes=slopes,
    )
    down = prefix_coefficient(
        direction=directions["all_down"],
        contract=contract,
        law=law,
        trace=trace,
        slopes=slopes,
    )
    assert up.full > up.additive_single_date
    assert down.full < down.additive_single_date
    assert np.isclose(down.full, down.fresh_diagonal, rtol=1e-10, atol=1e-12)
    assert gate_integral(np.array([1.0, 1.0]), 0, 1) == 0.5
    assert gate_integral(np.array([-1.0, -1.0]), 0, 1) == 0.0


def test_performance_difference_equals_policy_value_loss_on_same_lattice():
    law, contract, boundary = example()
    epsilon = 0.001
    directions = make_directions(contract.exercise_dates)
    thresholds = np.vstack(
        [
            boundary + epsilon * directions["all_up"],
            boundary + epsilon * directions["all_down"],
            boundary + epsilon * directions["alternating_up_down"],
        ]
    )
    solution = solve_lattice(
        law,
        contract,
        intervals_per_cap=1024,
        policy_thresholds=thresholds,
    )
    forward = forward_performance_difference(
        solution,
        contract,
        optimal_boundary=boundary,
        policy_thresholds=thresholds,
    )
    assert solution.policy_issue_values is not None
    backward_loss = solution.optimal_issue_value - solution.policy_issue_values
    assert np.allclose(forward.total_losses, backward_loss, atol=2e-12, rtol=2e-10)
    assert np.all(forward.total_losses >= -1e-14)


def test_direct_quadrature_regret_avoids_price_subtraction():
    law, contract, boundary = example()
    epsilon = 0.001
    directions = make_directions(contract.exercise_dates)
    thresholds = np.vstack(
        [
            boundary + epsilon * directions["all_up"],
            boundary + epsilon * directions["all_down"],
        ]
    )
    solution = solve_quadrature(
        law,
        contract,
        intervals_per_cap=2048,
        nodes_per_component=128,
        policy_thresholds=thresholds,
    )
    direct = quadrature_policy_regret(
        solution, contract, policy_thresholds=thresholds
    )
    assert solution.policy_issue_values is not None
    difference = solution.optimal_issue_value - solution.policy_issue_values
    assert np.all(direct >= 0)
    assert np.allclose(direct, difference, atol=3e-7, rtol=0.05)
