# Validation of the public export

Checks performed locally on 3 October 2026 with Python 3.12.7:

- **76 tests passed**: 3 public integrity tests, 5 theoretical-example tests, 48 financial extraction/cleaning/calibration/solver tests, 9 cliquet kernel tests, 4 uncapped integration tests, and 7 comparator tests. These tests use synthetic inputs.
- The theory reproduction command completed successfully. Its all-sign formula maximum absolute error was approximately 1.33e-15.
- Recomputed the 12-date market-state selection using the existing licensed monthly surface/rate data. The first five dates matched the paper in the required order.
- Reran the annual calibration/preparation routines for the eight financial-experiment dates using the original licensed raw observations held locally: eight successful calibrations, no failures.
- Exported the five-snapshot solver inputs from those newly computed calibrations. All numeric fields matched the original frozen market and mixture input files exactly after CSV parsing (maximum absolute difference zero). This comparison does not claim file-byte equality or validate a new vendor download.

The live WRDS connection/download was not repeated for this export. The full 248-date annual calibration and all financial experiments have not been rerun as part of these checks. The larger private archive's prior validation is distinct from validation of this public export. No licensed inputs or locally generated market outputs used in these checks are committed here.
