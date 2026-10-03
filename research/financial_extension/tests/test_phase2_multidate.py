"""Tests for the gated Phase II multidate calculations."""

from __future__ import annotations

import numpy as np

from indexed_annuity import (
    ContractSpec,
    CreditingSpec,
    MixtureCreditedDistribution,
    log_state_grid,
    solve_stationary_bellman,
    upper_regular_boundary,
)
from market_models import LognormalMixtureParams
from phase2_multidate import (
    ForwardAudit,
    LogStateMeasure,
    advance_issue_measure,
    all_boundaries_and_slopes,
    component_history_coefficient,
    direct_multidate_regret,
    initial_issue_measure,
    propagate_measure,
    reference_forward_audit,
    solve_charge_schedule,
    solve_stationary_with_tail,
)


def _distribution(cap: float = 0.08) -> MixtureCreditedDistribution:
    return MixtureCreditedDistribution(
        gross_forward=1.032,
        mixture=LognormalMixtureParams(
            low_weight=0.30,
            low_forward_multiplier=0.88,
            low_volatility=0.16,
            high_volatility=0.085,
        ),
        crediting=CreditingSpec(participation=1.0, cap=cap),
    )


def test_stationary_tail_solver_matches_stopping_tail_solver() -> None:
    distribution = _distribution()
    law = distribution.discretize(32)
    contract = ContractSpec(
        discount_factor=np.exp(-0.055),
        termination_probability=0.03,
        surrender_haircut=0.08,
    )
    grid = log_state_grid(0.03, 20.0, 4_000)
    inherited = solve_stationary_bellman(
        grid=grid,
        law=law,
        contract=contract,
        tolerance=1e-10,
    )
    phase2, tail_slope, margin = solve_stationary_with_tail(
        grid=grid,
        law=law,
        contract=contract,
        tolerance=1e-10,
    )
    inherited_boundary, _ = upper_regular_boundary(inherited)
    phase2_boundary, _ = upper_regular_boundary(phase2)
    assert margin > 0
    assert tail_slope == 1.0 - contract.surrender_haircut
    assert abs(phase2_boundary - inherited_boundary) < 2e-5
    assert np.max(np.abs(phase2.value - inherited.value)) < 2e-7


def test_log_measure_preserves_exact_atoms_and_total_mass() -> None:
    distribution = _distribution()
    state_grid = log_state_grid(0.5, 3.0, 12_000)
    initial, fresh = initial_issue_measure(
        log_grid=np.log(state_grid),
        initial_state=1.0,
        distribution=distribution,
    )
    assert abs(initial.mass - 1.0) < 2e-8
    assert len(initial.atoms) == 2
    assert np.max(np.abs(fresh - initial.continuous_density)) == 0
    propagated, _, mass_error = propagate_measure(initial, distribution)
    assert abs(propagated.mass - 1.0) < 5e-6
    assert abs(mass_error) < 5e-6
    assert len(propagated.atoms) == 3


def test_stationary_three_date_carriers_reconcile_and_pairwise_overcounts() -> None:
    distribution = _distribution()
    law = distribution.discretize(32)
    contract = ContractSpec(
        discount_factor=np.exp(-0.055),
        termination_probability=0.03,
        surrender_haircut=0.08,
    )
    state_grid = log_state_grid(0.03, 20.0, 8_000)
    stationary, tail_slope, _ = solve_stationary_with_tail(
        grid=state_grid,
        law=law,
        contract=contract,
        tolerance=1e-10,
    )
    boundary, slope = upper_regular_boundary(stationary)
    schedule = solve_charge_schedule(
        grid=state_grid,
        transition_laws=[law] * 3,
        charges=[0.08] * 3,
        discount_factor=contract.discount_factor,
        termination_probability=contract.termination_probability,
        death_guarantee=1.0,
        terminal_value=stationary.value,
        terminal_high_state_slope=tail_slope,
    )
    boundaries, slopes = all_boundaries_and_slopes(schedule)
    assert max(abs(value - boundary) for value in boundaries) < 2e-5
    audit = reference_forward_audit(
        log_grid=np.log(state_grid),
        initial_state=0.96 * boundary,
        initial_distribution=distribution,
        transition_distributions=[distribution, distribution],
        boundaries=boundaries,
    )
    shifts = tuple(boundaries)
    coefficients = component_history_coefficient(
        audit=audit,
        boundaries=boundaries,
        slopes=slopes,
        shifts=shifts,
        discount_survival=contract.discount_factor * 0.97,
        floor_probabilities=[distribution.floor_probability] * 2,
        alignment_groups=["stationary"] * 3,
    )
    assert coefficients.resonant > 0
    assert abs(coefficients.coherent_pairwise - coefficients.full) < 1e-12
    assert coefficients.additive_pairwise > coefficients.full

    regret = direct_multidate_regret(
        log_grid=np.log(state_grid),
        state_grid=state_grid,
        initial_state=0.96 * boundary,
        initial_distribution=distribution,
        transition_distributions=[distribution, distribution],
        boundaries=boundaries,
        gaps=[item.gap for item in schedule.slices],
        shifts=shifts,
        epsilon=0.001,
        discount_survival=contract.discount_factor * 0.97,
        integration_nodes=48,
    )
    scaled = regret.total / 0.001**2
    assert abs(scaled - coefficients.full) / coefficients.full < 0.08


def test_declining_schedule_has_no_false_alignment_label() -> None:
    distribution = _distribution()
    law = distribution.discretize(32)
    tail_contract = ContractSpec(
        discount_factor=np.exp(-0.055),
        termination_probability=0.03,
        surrender_haircut=0.0,
    )
    state_grid = log_state_grid(0.03, 20.0, 6_000)
    stationary, tail_slope, margin = solve_stationary_with_tail(
        grid=state_grid,
        law=law,
        contract=tail_contract,
        tolerance=1e-10,
    )
    assert margin > 0
    schedule = solve_charge_schedule(
        grid=state_grid,
        transition_laws=[law] * 5,
        charges=[0.08, 0.06, 0.04, 0.0, 0.0],
        discount_factor=tail_contract.discount_factor,
        termination_probability=tail_contract.termination_probability,
        death_guarantee=1.0,
        terminal_value=stationary.value,
        terminal_high_state_slope=tail_slope,
    )
    # A declining charge can eliminate the early upper stopping tail because
    # waiting unlocks a larger future liquidation slope.  The two dates already
    # in the exact zero-charge tail must nevertheless reproduce one boundary.
    assert schedule.slices[0].boundary is None
    assert schedule.slices[-1].boundary is not None
    assert schedule.slices[-2].boundary is not None
    assert abs(schedule.slices[-1].boundary - schedule.slices[-2].boundary) < 2e-5


def test_issue_support_below_one_is_exactly_empty() -> None:
    distribution = _distribution()
    state_grid = log_state_grid(0.03, 20.0, 8_000)
    before_target, masses = advance_issue_measure(
        log_grid=np.log(state_grid),
        initial_state=1.0,
        credit_distributions=[distribution] * 5,
        preceding_boundaries=[None, None, None, None],
    )
    assert np.allclose(masses, 1.0, atol=2e-14)
    assert np.max(before_target.continuous_density[state_grid < 1.0]) == 0.0
    killed, killed_masses = advance_issue_measure(
        log_grid=np.log(state_grid),
        initial_state=1.0,
        credit_distributions=[distribution, distribution],
        preceding_boundaries=[0.9],
    )
    assert killed_masses == (0.0,)
    assert killed.mass == 0.0


def test_time_varying_boundary_carrier_uses_only_structural_equalities() -> None:
    state_grid = log_state_grid(0.5, 2.0, 2_000)
    log_grid = np.log(state_grid)
    # log-density exp(z) corresponds to unit density in the state coordinate.
    density = state_grid.copy()
    measure = LogStateMeasure(log_grid, density, {})
    audit = ForwardAudit(
        predecision=(measure, measure),
        survivors=(measure, measure),
        fresh_arrival_log_densities=(density, density),
        propagation_mass_errors=(0.0,),
        atom_boundary_collisions=(),
    )

    lower_target = component_history_coefficient(
        audit=audit,
        boundaries=(1.2, 1.0),
        slopes=(1.0, 1.0),
        shifts=(0.3, 0.2),
        discount_survival=0.9,
        floor_probabilities=(0.4,),
        alignment_groups=(None, None),
    )
    transported = next(
        item
        for item in lower_target.carriers
        if item.source_date == 1 and item.target_date == 2
    )
    assert transported.classification == "ordinary"
    assert transported.finite_gate_dates == (2,)

    higher_target = component_history_coefficient(
        audit=audit,
        boundaries=(0.9, 1.0),
        slopes=(1.0, 1.0),
        shifts=(0.3, 0.2),
        discount_survival=0.9,
        floor_probabilities=(0.4,),
        alignment_groups=(None, None),
    )
    assert not any(
        item.source_date == 1 and item.target_date == 2
        for item in higher_target.carriers
    )

    aligned = component_history_coefficient(
        audit=audit,
        boundaries=(1.0, 1.0),
        slopes=(1.0, 1.0),
        shifts=(0.3, 0.2),
        discount_survival=0.9,
        floor_probabilities=(0.4,),
        alignment_groups=("same", "same"),
    )
    resonant = next(
        item
        for item in aligned.carriers
        if item.source_date == 1 and item.target_date == 2
    )
    assert resonant.classification == "resonant"
    assert resonant.finite_gate_dates == (1, 2)
