# Validation of the public export

Checks performed locally on 3 October 2026 with Python 3.12.7:

- **76 tests passed**: 3 public integrity tests, 5 theoretical-example tests, 48 financial extraction/cleaning/calibration/solver tests, 9 cliquet kernel tests, 4 uncapped integration tests, and 7 comparator tests. These tests use synthetic inputs.
- The same 76-test command also passed in a fresh clone downloaded from the public GitHub repository, without local market files.
- The theory reproduction command completed successfully. Its all-sign formula maximum absolute error was approximately 1.33e-15.
- Recomputed the 12-date market-state selection using the existing licensed monthly surface/rate data. The first five dates matched the paper in the required order.
- Reran the annual calibration/preparation routines for the eight financial-experiment dates using the original licensed raw observations held locally: eight successful calibrations, no failures.
- Exported the five-snapshot solver inputs from those newly computed calibrations. All numeric fields matched the original frozen market and mixture input files exactly after CSV parsing (maximum absolute difference zero). This comparison does not claim file-byte equality or validate a new vendor download.

- Regenerated the 30-case cliquet pilot from the newly prepared inputs. All 22 mandatory validator checks and five additional data-dependent output tests passed. All 16 non-timing CSV result/table files were byte-identical to the original archived files; the separate runtime table differed as expected.
- The cliquet uncapped-state and warning-case refinement stages also completed. Sentinel rerun differences were zero; the warning-case cross-evaluator difference was approximately 2.40e-7 and passed the original tolerance. Both generated baseline tables consumed by the prefix stage matched their historical hashes.
- The prefix stage passed its source/input preflight and constructed all 30 cases from the newly prepared inputs and regenerated baseline.
- The decimal quadrature reconciliation script reproduced the manuscript's refined-warning-case summaries (approximately 4.7882% loss discrepancy and 5.1517% interaction discrepancy) and the 20–0–10 classification from the existing archived evaluator records. This is a records check, not a new prefix evaluator run.

The live WRDS connection/download was not repeated for this export. The full 248-date annual calibration, complete final prefix-grid experiment, Bermudan simulations and annuity refinement runs were not repeated as part of this export validation. The larger private archive's prior validation is distinct from validation of this public export. No licensed inputs or locally generated market outputs used in these checks are committed here.
