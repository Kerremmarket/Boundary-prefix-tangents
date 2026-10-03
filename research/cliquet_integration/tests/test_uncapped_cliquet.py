from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cliquet_feasibility" / "src"))
sys.path.insert(0, str(ROOT / "cliquet_integration" / "src"))

from cliquet import (  # noqa: E402
    Contract,
    build_law,
    forward_performance_difference,
    solve_boundary,
    solve_lattice,
)
from uncapped_cliquet import (  # noqa: E402
    forward_performance_difference_uncapped,
    lattice_policy_values,
    local_domain,
    reachable_threshold_dates,
    solve_uncapped_lattice_reference,
)


def frozen_style_case():
    contract = Contract(0.08, 0.24, 8, np.exp(-0.03))
    law = build_law(
        model="black_scholes_atm",
        gross_forward=1.04,
        atm_volatility=0.22,
        local_cap=contract.local_cap,
    )
    result = solve_boundary(law, contract)
    assert result.boundary is not None
    return law, contract, result.boundary


def test_optimal_capped_state_factorization_on_the_lattice():
    law, contract, boundary = frozen_style_case()
    capped = solve_lattice(
        law, contract, intervals_per_cap=512,
        policy_thresholds=np.full(contract.exercise_dates, boundary)
    )
    uncapped = solve_uncapped_lattice_reference(
        law, contract, intervals_per_local_cap=512
    )
    values = lattice_policy_values(
        uncapped, contract, np.full(contract.exercise_dates, boundary)
    )
    assert abs(uncapped.optimal_issue_value - capped.optimal_issue_value) < 2e-12
    assert abs(values[0] - capped.policy_issue_values[0]) < 2e-12
    assert max(abs(x - boundary) for x in uncapped.grid_boundaries.values()) < 2e-4


def test_local_supplied_policy_matches_capped_evaluator():
    law, contract, boundary = frozen_style_case()
    thresholds = np.full(contract.exercise_dates, boundary + 0.001)
    capped = solve_lattice(
        law, contract, intervals_per_cap=512, policy_thresholds=thresholds
    )
    capped_forward = forward_performance_difference(
        capped, contract, optimal_boundary=boundary,
        policy_thresholds=thresholds
    )
    uncapped = solve_uncapped_lattice_reference(
        law, contract, intervals_per_local_cap=512
    )
    uncapped_forward = forward_performance_difference_uncapped(
        uncapped, contract, policy_thresholds=thresholds
    )
    assert abs(
        uncapped_forward.total_losses[0] - capped_forward.total_losses[0]
    ) < 2e-12


def test_above_cap_threshold_is_reachable_and_not_never_exercise():
    law, contract, _ = frozen_style_case()
    threshold = contract.global_cap + 0.005
    thresholds = np.full(contract.exercise_dates, threshold)
    assert reachable_threshold_dates(thresholds, contract) == (4, 5, 6, 7)
    uncapped = solve_uncapped_lattice_reference(
        law, contract, intervals_per_local_cap=512
    )
    uncapped_policy = lattice_policy_values(uncapped, contract, thresholds)[0]
    never_thresholds = np.full(
        contract.exercise_dates, contract.resets * contract.local_cap + 1.0
    )
    never_policy = lattice_policy_values(uncapped, contract, never_thresholds)[0]
    assert uncapped_policy > never_policy


def test_local_domain_reports_physical_threshold_extrema():
    ok, low, high = local_domain(
        0.235, 0.24, 0.01, np.ones(7)
    )
    assert not ok
    assert low == high == 0.245

