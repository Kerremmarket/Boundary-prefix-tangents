from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE))

from indexed_annuity import ContractSpec  # noqa: E402
from run_full_annual_panel import _parameter_bound_hit  # noqa: E402
from run_market_contract_sensitivity import (  # noqa: E402
    CASES,
    _high_state_margin,
)


def test_full_panel_bound_diagnostic_includes_implied_high_mean() -> None:
    central = {
        "low_weight": 0.70,
        "low_forward_multiplier": 0.90,
        "low_volatility": 0.28,
        "high_volatility": 0.15,
        "high_forward_multiplier": 1.23,
    }
    assert not _parameter_bound_hit(central)
    assert _parameter_bound_hit({**central, "high_forward_multiplier": 3.0})
    assert _parameter_bound_hit({**central, "low_weight": 0.05})


def test_high_state_margin_matches_declared_asymptotic_formula() -> None:
    contract = ContractSpec(
        discount_factor=0.95,
        termination_probability=0.03,
        surrender_haircut=0.08,
    )
    liquidation_slope = 0.92
    expected_factor = 1.03
    expected = liquidation_slope - 0.95 * expected_factor * (
        0.03 + 0.97 * liquidation_slope
    )
    assert np.isclose(_high_state_margin(expected_factor, contract), expected)


def test_contract_cases_are_one_at_a_time_and_include_geometry_check() -> None:
    assert len(CASES) == 9
    assert len({case.name for case in CASES}) == len(CASES)
    benchmark = next(case for case in CASES if case.name == "benchmark")
    low_cap = next(case for case in CASES if case.name == "cap_04")
    changed = {
        field
        for field in (
            "participation",
            "cap",
            "surrender_haircut",
            "termination_probability",
        )
        if getattr(low_cap, field) != getattr(benchmark, field)
    }
    assert changed == {"cap"}
    assert 0.96 * (1.0 + low_cap.cap) < 1.0
    assert 0.96 * (1.0 + benchmark.cap) > 1.0
