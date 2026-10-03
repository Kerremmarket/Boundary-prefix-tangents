# Mapping generated results to the paper

All paths are relative to the repository root. These files are generated locally; they are not bundled market data.

| Paper content | Stage | Generated files |
|---|---|---|
| Stylized examples and numerical formula checks | `theory` | `reproducibility/outputs/results.json` |
| Main table: ten base-contract directed transfers | `prefix` | `research/prefix_comparator_revision/tables/transfer_actual_base.csv` |
| All 30 directed transfers; financial supplement transfer tables | `prefix` | `research/prefix_comparator_revision/tables/transfer_actual_all.csv` |
| Quadrature and selected-lattice comparisons | `prefix` | `research/prefix_comparator_revision/results/transfer_quadrature_actual.csv`, `transfer_lattice_levels.csv` in the same directory |
| Transfer homotopies | `prefix` | `research/prefix_comparator_revision/tables/transfer_homotopy_all.csv` |
| Main numerical figures | `prefix` | `research/prefix_comparator_revision/figures/transfer_error_comparison.pdf` and `transfer_residual_decomposition.pdf` |
| Controlled contract and volatility variations | `prefix` | `research/prefix_comparator_revision/tables/comparative_statics_all.csv` and `comparative_statics_summary.csv` |
| Separate mechanism experiment | `prefix` | `research/prefix_comparator_revision/tables/mechanism_comparator_all.csv` |
| Bermudan and learned-policy diagnostics | `bermudan` | `research/financial_extension/diagnostic/evidence/bermudan/` |
| Indexed-annuity coefficients, finite losses and controls | `annuity` | `research/financial_extension/phase2/evidence/phase2a/` |

For the original frozen sample, the actual-transfer comparison reports **20 prefix-closer resolved cases, 0 additive-closer resolved cases and 10 unresolved cases**. The controlled panel reports **241, 0 and 29**, respectively. These are numerical dispositions under the recorded resolution rules, not universal dominance claims. The original materiality gate did not pass.

The quadrature table contains 30 fine evaluations and one additional warning-case refinement. In the paper's combined discrepancy summaries, the refined evaluation replaces the fine evaluation for that case; it is not counted as a 31st independent observation. Loss discrepancies use the magnitude of the selected-lattice loss as denominator; interaction discrepancies use the magnitude of the predicted interaction. The reported medians are 4.8% and 5.2% after that substitution.

The original validators check numerical identities, tolerances, expected counts and output integrity. They may reject results from changed data rather than silently confirming the paper's original conclusions. The historical cliquet output tests (`research/cliquet_feasibility/tests/test_analysis.py`) require generated results and are deliberately separate from the data-free test command.
