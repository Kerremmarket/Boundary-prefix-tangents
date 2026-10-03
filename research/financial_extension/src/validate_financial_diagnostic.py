"""Validate the Bermudan control and read-only frozen annuity audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_DATES = {
    "2016-10-31",
    "2008-11-28",
    "2020-02-28",
    "2009-04-30",
    "2006-03-31",
}
EXPECTED_ETAS = {1.0, 0.5, 0.25, 0.125, 0.0625}


def validate(bermudan_dir: Path, annuity_dir: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    boundaries = pd.read_csv(bermudan_dir / "boundaries.csv")
    refinement = pd.read_csv(bermudan_dir / "reference_refinement.csv")
    eta = pd.read_csv(bermudan_dir / "eta_path_results.csv")
    gates = pd.read_csv(bermudan_dir / "gate_results.csv")
    native = pd.read_csv(bermudan_dir / "native_policy_evaluation.csv")
    coefficients = pd.read_csv(bermudan_dir / "diagonal_coefficients.csv")

    assert set(boundaries["date"]) == EXPECTED_DATES
    assert len(boundaries) == 490
    assert len(eta) == 275
    assert len(gates) == 55
    assert len(native) == 50
    assert len(coefficients) == 385
    assert set(eta["eta"].unique()) == EXPECTED_ETAS
    assert not eta[
        [
            "actual_loss",
            "diagonal_prediction",
            "actual_to_diagonal",
            "single_date_sum",
            "interaction",
        ]
    ].isna().any().any()
    assert (eta["actual_loss"] >= 0.0).all()
    assert (eta["diagonal_prediction"] > 0.0).all()
    assert np.allclose(
        eta["actual_loss"] / eta["diagonal_prediction"],
        eta["actual_to_diagonal"],
        rtol=2e-12,
        atol=2e-12,
    )
    assert np.allclose(
        eta["actual_loss"] - eta["single_date_sum"],
        eta["interaction"],
        rtol=2e-10,
        atol=2e-14,
    )
    assert (boundaries["crossing_count"] == 1).all()
    assert boundaries["orientation_ok"].all()
    assert gates["boundary_regular"].all()
    assert np.max(np.abs(eta["forward_minus_backward"])) < 1e-6
    assert np.max(np.abs(native["native_minus_threshold_loss"])) < 2e-6

    fft = refinement[refinement["method"] == "fft_convolution"]
    max_boundary_refinement = float(
        fft["max_boundary_difference_vs_65537"].max()
    )
    max_value_refinement = float(
        fft["value_difference_vs_65537"].abs().max()
    )
    assert max_boundary_refinement < 2e-4
    assert max_value_refinement < 2e-5
    coarse_gates = gates[gates["method"] == "coarse_fft_257"]
    lsmc_gates = gates[gates["method"] == "lsmc_linear"]
    assert len(coarse_gates) == 5 and coarse_gates["gate_pass"].all()
    assert len(lsmc_gates) == 50

    annuity_coefficients = pd.read_csv(annuity_dir / "multidate_coefficients.csv")
    annuity_occupancy = pd.read_csv(annuity_dir / "issue_occupancy.csv")
    reconciliation = pd.read_csv(
        annuity_dir / "carrier_density_reconciliation.csv"
    )
    renewal = pd.read_csv(annuity_dir / "renewal_cap_control.csv")
    local = annuity_coefficients[
        annuity_coefficients["initialization"] == "local_rho_boundary"
    ]
    issue = annuity_coefficients[
        annuity_coefficients["initialization"] == "issue_state"
    ]
    assert len(local) == 84 and len(issue) == 84
    assert (local["full_coefficient"] > 0.0).all()
    assert (issue["full_coefficient"] == 0.0).all()
    assert (annuity_occupancy["continuous_density_at_boundary"] == 0.0).all()
    assert (annuity_occupancy["survivor_mass_from_issue"] == 0.0).all()
    coherent_error = float(np.max(np.abs(local["coherent_minus_full"])))
    carrier_error = float(reconciliation["relative_difference"].max())
    additive_overcount_rows = int((local["additive_minus_full"] > 1e-15).sum())
    renewal_changes = renewal["boundary_change"].dropna().abs()
    assert coherent_error < 3e-17
    assert carrier_error < 3e-16
    assert additive_overcount_rows == 72
    assert (renewal_changes > 0.0).all()
    assert not renewal["structurally_aligned_with_previous"].any()

    lsmc_passes = int(lsmc_gates["gate_pass"].sum())
    mc_consistent = int(native["heldout_consistent"].sum())
    classification = (
        "FINANCIAL DIAGNOSTIC SECTION READY FOR INTEGRATION"
        if gates["gate_pass"].all()
        else "USEFUL BUT TOO WEAK FOR MAIN TEXT"
        if coarse_gates["gate_pass"].all()
        else "CONTROL STUDY FAILED — DO NOT INTEGRATE"
    )
    metrics = {
        "status": "PASS",
        "classification": classification,
        "bermudan": {
            "market_dates": len(EXPECTED_DATES),
            "approximate_policies": len(gates),
            "regular_policies": int(gates["boundary_regular"].sum()),
            "coarse_gate_passes": int(coarse_gates["gate_pass"].sum()),
            "coarse_gate_total": len(coarse_gates),
            "lsmc_gate_passes": lsmc_passes,
            "lsmc_gate_total": len(lsmc_gates),
            "heldout_mc_consistent": mc_consistent,
            "heldout_mc_total": len(native),
            "max_fft_boundary_refinement_error": max_boundary_refinement,
            "max_fft_value_refinement_error": max_value_refinement,
            "max_forward_backward_loss_difference": float(
                np.max(np.abs(eta["forward_minus_backward"]))
            ),
            "max_native_threshold_loss_difference": float(
                np.max(np.abs(native["native_minus_threshold_loss"]))
            ),
            "max_coarse_interaction_share_eta_one_sixteenth": float(
                coarse_gates["interaction_share_eta_one_sixteenth"].max()
            ),
            "max_lsmc_interaction_share_eta_one_sixteenth": float(
                lsmc_gates["interaction_share_eta_one_sixteenth"].max()
            ),
        },
        "annuity_read_only_audit": {
            "positive_local_rows": int((local["full_coefficient"] > 0).sum()),
            "local_rows": len(local),
            "positive_issue_rows": int((issue["full_coefficient"] > 0).sum()),
            "issue_rows": len(issue),
            "max_coherent_pairwise_error": coherent_error,
            "additive_overcount_rows": additive_overcount_rows,
            "max_carrier_reconciliation_relative_error": carrier_error,
            "renewal_boundary_change_min": float(renewal_changes.min()),
            "renewal_boundary_change_max": float(renewal_changes.max()),
            "renewal_aligned_rows": int(
                renewal["structurally_aligned_with_previous"].sum()
            ),
        },
    }
    (output_dir / "validation_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    report = f"""# Financial diagnostic validation report

## Status

All encoded evidence and cross-file invariants pass.

## Smooth Bermudan control

- Five prospectively selected OptionMetrics-derived market dates.
- 55 approximate policies; {int(gates['boundary_regular'].sum())}/55 have one
  correctly oriented native boundary at all seven nonterminal dates.
- Coarse-grid gate: {int(coarse_gates['gate_pass'].sum())}/5.
- Linear-LSMC gate: {lsmc_passes}/50.
- Held-out Monte Carlo consistency: {mc_consistent}/50.
- Maximum FFT boundary refinement error: {max_boundary_refinement:.3e}.
- Maximum FFT value refinement error: {max_value_refinement:.3e}.
- Maximum forward-versus-backward fixed-policy loss difference:
  {metrics['bermudan']['max_forward_backward_loss_difference']:.3e}.

The LSMC failures are retained. They arise because the supplied displacement
remains nonlocal at `eta=1/16` on three dates, not because of irregular native
signs or a hidden second-order prefix term.

## Frozen annuity audit (read only)

- Positive local rows: {len(local)}/{len(local)}.
- Positive issue-state rows: 0/{len(issue)}.
- Maximum coherent labelled-pairwise/full difference: {coherent_error:.3e}.
- Additive pairwise overcount rows: {additive_overcount_rows}.
- Maximum carrier-density reconciliation error: {carrier_error:.3e}.
- Renewal-cap boundary changes: {renewal_changes.min():.4f} to
  {renewal_changes.max():.4f}; exact prior-normal alignment is absent.

## Classification

**{classification}**
"""
    (output_dir / "validation_report.md").write_text(report, encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bermudan-dir",
        type=Path,
        default=Path("research/financial_extension/diagnostic/evidence/bermudan"),
    )
    parser.add_argument(
        "--annuity-dir",
        type=Path,
        default=Path("research/financial_extension/phase2/evidence/phase2a"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("research/financial_extension/diagnostic/audit"),
    )
    args = parser.parse_args()
    metrics = validate(args.bermudan_dir, args.annuity_dir, args.output_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
