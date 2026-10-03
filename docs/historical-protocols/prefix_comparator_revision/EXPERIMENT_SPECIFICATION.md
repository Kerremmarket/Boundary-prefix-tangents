# Predeclared experiment specification

The controlled comparative-static design below was recorded before any of its
outcomes were computed.  A parallel read-only code audit had already performed
an in-memory prototype of the transfer and existing-mechanism decompositions
before this file was created.  Those checks are therefore confirmatory, not
prospectively unseen.

### Pre-freeze exploratory implementation log

The prototype wrote no file and made no source change, but the reviewer saw:

- one 2016-10-31 base-tuple Black--Scholes-reference transfer at 2,048 and
  8,192 intervals, including its `L`, `S`, and `J` values;
- the full 30-row actual-transfer decomposition at 8,192 intervals, including
  the fact that the prefix prediction was closer in all 30 prototype rows,
  the common-sign checks passed, and residual signs often opposed;
- a prototype gain-resolution classification based on existing loss
  discrepancies;
- aggregate (not row-level) decomposition summaries for all 1,140 existing
  mechanism rows, including direction-family gain shares and sign/identity
  checks; and
- the already-stored coefficient-normalization and boundary-displacement
  ranges.

No horizon, cap, valid zero-credit-probability stress, or other controlled-
comparative-static outcome was run or inspected.  The exact values seen in the
prototype and their relationship to the official rerun are preserved in the
final provenance report.  No claim below relies on pretending that the
confirmatory decomposition was blinded.

Before the official run, the audit also checked one base-transfer scaling
identity numerically: applying the coefficient to the physical displacement
direction agreed with applying it to a sign direction and multiplying by the
squared displacement.  This is logged as an implementation check, not new
evidence.

## Primary estimands and identities

For a supplied direction `h`, define

```text
A(h) = sum_t R(iota_t h_t),       I(h) = R(h)-A(h),
S_epsilon(h) = sum_t L_epsilon(iota_t h_t),
J_epsilon(h) = L_epsilon(h)-S_epsilon(h).
```

The main comparison uses

```text
E_add    = |L_epsilon-epsilon^2 A|,
E_prefix = |L_epsilon-epsilon^2 R|,
Gain     = E_add-E_prefix.
```

For the capped-credit implementation, `S_epsilon` is also computed directly
from reference-survival predecision mass, and `J_epsilon` directly from the
difference between joint-policy and reference survival mass.  These direct
gate-product calculations are checked against separate joint and singleton
policy evaluations.  A quadrature/backward implementation supplies a second
discretization on actual transfers.  Shared components and subtraction risk
will be disclosed.

The signed residual identity is checked row by row:

```text
L-epsilon^2 R = (S-epsilon^2 A) + (J-epsilon^2 I).
```

Predicted sign before outcomes: common upward shifts have `J>=0`; common
downward shifts have `J<=0`.  No definite sign is predeclared for mixed-sign
directions.

### Direction normalization by table family

The direction scale is fixed within, but not identical across, table families:

- **Transfers:** `h=Delta b*1` is the full physical boundary displacement in
  accumulator units; homotopy predictions are `eta^2 R(h)` and
  `eta^2 A(h)` for threshold `b+eta Delta b`.
- **Inherited mechanism:** `h` is a dimensionless sign/ramp/singleton pattern,
  `epsilon=eta*c`, and predictions are `epsilon^2 R(h)` and
  `epsilon^2 A(h)`.
- **Controlled statics:** the declared physical constant displacement `delta`
  is placed directly in the direction vector.  The reported coefficient
  already absorbs `delta^2`; it must not be scaled a second time.

Positive two-homogeneity supplies conversions between equivalent conventions,
but raw coefficient columns must not be pooled across families without that
conversion.

## Confirmatory experiments

1. Reconcile all 30 existing directed model transfers at the actual
   displacement (`eta=1`) and along the fixed homotopy
   `eta in {1,1/2,1/4,1/8,1/16}`.
2. For every row report `L`, `eta^2 A`, `eta^2 R`, `eta^2 I`, `S`, `J`, the two
   signed residual components, absolute additive/full errors, gain, numerical
   discrepancy scales, and resolution status.
3. Recompute the comparator/decomposition for the full existing 1,140-row
   mechanism set, including negative, singleton, mixed-sign, nonlocal, and
   unresolved rows.  No row is selected after its result is seen.
4. Use the literal accumulator for every nonlocal mechanism row.  Verify the
   capped/literal equivalence for all local rows and actual transfers.
5. Rerun the existing cliquet validators and focused tests before relying on
   the starting evidence.

## Numerical ladder and stopping rule

- Transfer homotopy: lattice levels 4,096 and 8,192 intervals per local cap;
  quadrature/backward evaluation at 8,192 intervals and 256 nodes per
  lognormal component for all 30 actual (`eta=1`) transfers and their
  singleton policies.
- The one prospectively selected warning row also uses the already-frozen
  16,384/512 refinement for the complete decomposition.
- Mechanism decomposition: reuse the audited 8,192 literal-accumulator
  reference and run the new joint/singleton decomposition at that level;
  compare with existing primary/fine evidence and the existing quadrature
  subset.
- No new adaptive refinement is allowed.  If gain or interaction remains
  unresolved at the finest predeclared level, retain it as unresolved.
- A loss is relatively reportable only above ten times its loss discrepancy
  scale.  An interaction is signed-resolved only when its magnitude exceeds
  ten times its interaction discrepancy scale.  Gain is evaluated separately
  at every completed grid/evaluator combination and is resolved only when
  `|Gain|` exceeds ten times the largest direct successive-grid or cross-
  evaluator change in Gain (with the common absolute floor).  This corrects an
  initial draft rule that counted the shared loss error only once; direct Gain
  discrepancies preserve the relevant correlation.  Otherwise report absolute
  values only.
- “Exact finite loss” denotes the mathematical estimand; numerical values are
  estimates.  No discrepancy scale is called a certified bound.

## Controlled comparative statics

All variations are synthetic exercises within the same capped annual-credit
family.  Boundaries, traces, slopes, and values are reoptimized under each
varied law/contract.  The five frozen dates are used throughout where the
required inputs exist.  These exercises do not alter the frozen materiality
gate and are not newly market-calibrated products.

### A. Horizon

- Hold `c=0.08`, `G=0.24`, law, forward, volatility/mixture parameters, rate,
  and initial state fixed; set `N in {6,8,10}`.
- Verify analytically/numerically the common-boundary invariance specific to
  Proposition 7.1.
- Evaluate common up/down errors with a fixed per-date physical displacement
  `delta=0.08/128`.
- Repeat with `delta_N=(0.08/128)*sqrt(7/(N-1))`, which holds the sum of squared
  date displacements fixed.  No monotonic prefix-share prediction is imposed:
  horizon also changes occupancy, discounting, slopes, and decision count.

### B. Caps

- Local-cap variation: hold `G=0.24`, `N=8`, and the annual gross-return law
  fixed; use `c in {0.06,0.08,0.12}`.  The law is re-credited at each cap.
- Global-cap variation: hold `c=0.08`, `N=8`, and the annual credited law
  fixed; use `G in {0.16,0.24,0.32}`.
- Use a fixed relative displacement `delta=c/128` for common up/down tests.
  There is no predeclared monotonicity claim because boundary location,
  headroom, densities, slopes, and occupancy all change.

### C. Valid zero-credit-probability stress

- Use the Black--Scholes lognormal gross-return family at each frozen date and
  base tuple `(0.08,0.24,8)`.
- Hold the annual gross forward and rate fixed; multiply lognormal volatility
  by `0.75`, `1.00`, and `1.25`.  Each law remains normalized and preserves its
  specified forward.  Recompute its boundary and all coefficient inputs.
- This changes the complete return law, not just one mixture weight.  It is a
  synthetic distribution stress and has no predeclared monotonic aggregate-
  regret or interaction-share prediction.

### D. Size and direction

- Reuse all predeclared common up/down, ramp up/down, alternating, and signed
  singleton directions and the existing six-scale mechanism grid.
- Distinguish upward delayed stopping from downward premature stopping.
- No new direction search is permitted.

For every controlled row record boundary/atom separation, `p0`, trace and
slope diagnostics, coefficient components, exact finite loss, comparator
errors, interaction, and validity/resolution status.  Boundary collisions or
failed clean assumptions are retained as a distinct regime.

The generated analysis has 270 row keys because baseline cases recur in
different panels and horizon normalizations.  These correspond to 200 distinct
physical scenarios, not 270 independent market observations; repeated baseline
rows remain useful internal consistency checks.

## Regression decision

No confirmatory regression is planned.  With five selected snapshots and
identities already governing several ratios, fitted coefficients would not add
credible market-level inference.  A regression may be generated only as an
explicitly exploratory audit if it answers a non-algebraic descriptive
question; it will not enter the manuscript by default.

## Output commitment

- Compact actual-transfer table, identical-case error-scale plot with
  `eta=1` marked, and residual-decomposition display.
- Complete machine-readable tables for every attempted transfer, mechanism,
  and comparative-static row, including exclusions and unresolved outcomes.
- Assumption-verification table, calibration specification, validation report,
  claim-to-evidence map, and revision memo.
- Revised main, technical-appendix, and financial-supplement PDFs.
