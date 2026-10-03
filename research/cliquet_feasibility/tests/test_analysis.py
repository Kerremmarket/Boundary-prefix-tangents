from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_gate_verdict_and_required_rows():
    summary = json.loads((ROOT / "results/analysis_summary.json").read_text())
    assert summary["verdict"] == "LIMITED VALUE"
    assert summary["occupancy_alignment_gate"]["pass"]
    assert summary["mechanism_gate"]["pass"]
    assert not summary["finance_gate"]["adopt_pass"]
    assert summary["mechanism_gate"]["selected_rows"] == 40
    assert summary["counts"]["certified_finance_rows"] == 30


def test_unresolved_relative_claims_are_suppressed():
    mechanism = pd.read_csv(ROOT / "tables/mechanism_certified.csv")
    finance = pd.read_csv(ROOT / "tables/finance_certified.csv")
    assert mechanism.loc[
        ~mechanism["resolved"],
        ["full_relative_error", "additive_relative_error"],
    ].isna().all().all()
    assert finance.loc[
        ~finance["resolved"],
        [
            "loss_fraction_of_premium",
            "loss_fraction_of_early_exercise_value",
            "full_approximation_relative_error",
        ],
    ].isna().all().all()


def test_base_occupancy_is_from_inception_and_clean():
    occupancy = pd.read_csv(ROOT / "tables/occupancy_alignment.csv")
    assert len(occupancy) == 10
    assert occupancy["eligible_inception_trace"].all()
    assert occupancy["clean_alignment"].all()
    assert (occupancy["eligible_source_date"] == 3).all()
    assert (occupancy["eligible_target_date"] == 4).all()
    assert (occupancy["copied_coefficient"] > 0).all()


def test_reference_and_transferred_survival_are_exported():
    paths = pd.read_csv(ROOT / "results/policy_occupancy.csv")
    assert set(paths["policy_role"]) == {
        "reference_native",
        "perturbed_transferred",
    }
    reference = paths[paths["policy_role"] == "reference_native"]
    assert reference["mismatch_mass"].abs().max() < 2e-12


def test_model_price_difference_is_not_policy_transfer_loss():
    finance = pd.read_csv(ROOT / "tables/finance_certified.csv")
    prices = pd.read_csv(ROOT / "tables/model_price_differences.csv")
    base_finance = finance[finance["contract_id"] == "c08_g24_n8"]
    base_prices = prices[prices["contract_id"] == "c08_g24_n8"]
    assert base_prices["resolved"].all()
    assert base_prices["mixture_minus_black_bps"].abs().min() > 100
    assert base_finance["transfer_loss_bps_notional"].max() < 1
