# Provenance and outcome-exposure log

## Repository lineage

- Immutable starting commit:
  `8671f64947b8afb46a86924748a48411bcde632f`.
- Starting commit description:
  `Audit capped cliquet and integrate finance illustration`.
- Parent integration-freeze commit:
  `f158ad4e9697bbe26f639c7f3d8add14f5570f29`.
- Frozen source-cliquet commit:
  `924f653c5a5ab6a09b738a46be8e2040455f49fb`.
- Revision branch: `codex/research/prefix-comparator-evidence`.
- Isolated worktree:
  `<original-research-checkout>/tmp/worktrees/prefix-comparator-evidence`.

The starting worktree was reported clean.  The original `main`, finance,
cliquet, annuity, RL, and StopDiag branches/worktrees are outside this worktree
and are not revision targets.  The authoritative input/output hashes at the
start are in `BASELINE_MANIFEST.md`.  No raw OptionMetrics data or credentials
are copied into this revision.

## Evidence categories

1. **Frozen imported evidence.**  The cliquet feasibility and integration
   packages predate this revision and are protected by recorded hashes.
2. **Pre-freeze prototype exposure.**  A parallel audit executed portions of
   the new comparator before `EXPERIMENT_SPECIFICATION.md` was written.
3. **Initial official run.**  A complete run under an earlier version of the
   revision scripts was subsequently superseded.
4. **Interrupted reruns.**  Two later executions were stopped after source
   changes made their launched generations obsolete.
5. **Final clean run.**  The coherent generation recorded in the current
   `run_metadata.json`; release claims point to this generation only.

## Exact pre-freeze prototype exposure

The reviewer observed the following before the experiment specification was
frozen.  These values must remain disclosed even if the final rerun differs.

1. For the 2016-10-31, `c08_g24_n8`, Black--Scholes-reference transfer using
   the mixture boundary:

   - at 2,048 lattice intervals per local cap,
     `L=2.6182590881e-6`, `S=1.6267268740e-6`, and
     `J=9.9153221412e-7`;
   - at 8,192 intervals,
     `L=2.6822674505e-6`, `S=1.6664849992e-6`, and
     `J=1.01578245135e-6`.

2. For all 30 actual transfers at 8,192 intervals, the prefix prediction was
   nominally closer in 30/30 rows, common-sign `J` had the predicted sign, the
   maximum residual-identity discrepancy was `2.03e-20`, and the two component
   residuals had opposite signs in 19/30 rows.  The prototype Gain range was
   `0.0010832`--`0.2200701` basis points and its median was `0.0233692` basis
   points.
3. A crude, subsequently rejected prototype gain-resolution rule classified only 4/30
   actual transfers as resolved.  It did not evaluate direct cross-level and
   cross-method changes in Gain and is not used in the final analysis.
4. Aggregate checks, but not a prospective controlled-outcome table, were seen
   for all 1,140 inherited mechanism rows.
5. Stored coefficient ranges from the earlier cliquet evidence were inspected.
6. One base-transfer scaling check showed that evaluating `R` on the physical
   displacement vector equals evaluating it on the sign vector and multiplying
   by the squared displacement, up to numerical precision.

The reviewer did not inspect any horizon, cap, valid zero-credit-probability,
or other controlled comparative-static outcome before the specification was
frozen.  The transfer and inherited-mechanism comparisons are therefore
confirmatory; the controlled design was prospectively specified.

## Initial official run

The superseded metadata recorded:

- start `2026-09-09T16:23:13.539840+00:00`;
- finish `2026-09-09T16:34:39.582990+00:00`;
- wall time `686.076971833827` seconds;
- Python 3.12.7, NumPy 1.26.4, pandas 2.2.3, SciPy 1.16.3;
- no Monte Carlo;
- Git head equal to the immutable starting commit;
- 305 transfer lattice rows, 30 transfer quadrature rows, 2,280 mechanism
  lattice rows, and 540 comparative-static level rows.

That generation is not the final evidence.  In particular, its quadrature
table lacks the separate 16,384/512 warning-row record now required in addition
to the 30 fine rows; the final expected quadrature count is 31.  Later changes
also tightened direct Gain/error discrepancy rules and local/nonlocal
classification.  Numerical summaries derived from that generation were
replaced by the final run.

## Interrupted mixed-generation attempts

Two rerun attempts after the initial official run were manually stopped with
`Ctrl-C` before completion because relevant source code changed after each
process had been launched.  Neither attempt is a successful run.  Partial
files from those attempts, if any, were treated as mixed-generation artifacts
and overwritten by the final clean run.  No process from those attempts was
active when the documentation draft began.

The interruption is scientifically important: a timestamp or a partially
updated CSV must not be mistaken for a validated coherent generation.  The
final validator requires the expected row counts, unique keys, source hashes,
raw-output hashes, and a single final run timestamp.

## Validation-invocation corrections

During baseline validation there were two non-result failures:

- a test invocation initially omitted the required `PYTHONPATH`; rerunning in
  the intended environment passed;
- a branch-sensitive source validator was initially invoked from the revision
  worktree rather than its authoritative frozen source worktree.  The imported
  generated file touched by that attempt was restored byte-for-byte, and the
  validator was rerun in the correct source worktree.

These were invocation/context mistakes, not favorable-result substitutions.
They remain disclosed in the final validation report, which must list the
successful commands as well as these corrected attempts.

Later release checks recorded two further environment-only failures: the first
Paper I reproducibility-test invocation omitted its required `PYTHONPATH` and
failed during collection before the corrected five-test pass, and an
incidental `shasum` call stopped because the local `C.UTF-8` locale was
unavailable.  Python `hashlib` checks in the run metadata and validator passed;
neither failed invocation changed an evidence file.

## Direction and scaling conventions

The generated table families do not use an interchangeable direction scale.

| Family | Direction stored/evaluated | Scale applied to coefficient | Physical threshold |
|---|---|---|---|
| Actual transfer / homotopy | `h=Delta b * 1`, where `Delta b=b_supplied-b_reference` is in accumulator units | `eta^2 R(h)`, `eta^2 A(h)`, `eta^2 I(h)` | `b+eta Delta b` |
| Inherited mechanism | Dimensionless pattern `h` (`all_up`, `all_down`, ramps, alternating, or singleton) | `epsilon^2 R(h)` with `epsilon=eta*c` | `b+eta*c*h` |
| Controlled horizon | Constant physical vector `h=delta_N*1`; either fixed `delta=0.08/128` per date or `delta_N=(0.08/128)sqrt(7/(N-1))` | The coefficient already absorbs the physical `delta_N`; no extra `epsilon^2` is applied | `b+h` |
| Controlled local/global cap and volatility stress | Constant physical vector `h=delta*1`, `delta=c/128` (with `c=0.08` for global-cap and volatility stresses) | The coefficient already absorbs physical `delta`; no extra scale is applied | `b+h` |

All coefficients are positively two-homogeneous, so equivalent normalizations
can be converted.  Reports must state which convention is used and must not
double-scale a physical direction.  In particular, a coefficient from the
mechanism tables is not numerically comparable to a physical-direction
coefficient in the transfer or comparative-static tables without applying the
stated scale.

## Final coherent generation

- Start: `2026-09-09T17:22:52.516312+00:00`.
- Finish: `2026-09-09T17:35:49.115160+00:00`.
- Wall time: `776.5979296660516` seconds, below the frozen 2,700-second limit.
- Peak resident memory: `0.5948028564453125` GiB, below the frozen 3 GiB limit.
- Runtime: Python 3.12.7, NumPy 1.26.4, pandas 2.3.3, SciPy 1.16.3.
- Monte Carlo: not used.
- Git head at run: the immutable base
  `8671f64947b8afb46a86924748a48411bcde632f`; the working-tree sources are
  identified separately by hashes because the revision was not yet committed.
- Configuration SHA-256 at start and finish:
  `64fefaa98bf1429c96335bbb0f6efd46f3bf629b399521356b6475b84cb2dede`.
- Source hashes were identical at start and finish:
  `analyze_results.py` `9be2c37a...17de`, `comparator.py`
  `29dfb27b...7086`, and `run_experiments.py` `dd55aab2...961a` (full hashes
  are in `results/run_metadata.json`).
- Observed row counts are exactly 305 transfer lattice, 31 transfer quadrature,
  2,145 transfer date-decomposition, 61 transfer singleton-crosscheck, 2,280
  mechanism lattice, 7,980 mechanism date-decomposition, 40 mechanism
  singleton-crosscheck, and 540 controlled level rows.
- `run_metadata.json` records SHA-256 hashes for all eight raw outputs.  The
  final validator recomputed those hashes and passed.
- `src/validate_outputs.py` passes its coverage, uniqueness, algebraic identity,
  sign, scaling, state-domain, resolution-blanking, figure-existence, and
  frozen-materiality checks.

The final revision commit is to be recorded after the single planned commit;
no uncommitted hash is invented here.  Generated paper tables and figures were
regenerated from this coherent run, so neither interrupted generation is cited
as final evidence.
