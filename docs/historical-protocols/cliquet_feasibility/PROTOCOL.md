# Frozen bounded-pilot protocol

Frozen before any cliquet outcome table was computed.

## Scope and nonclaims

Use exactly the five already selected market snapshots and the two compatible
annual marginal laws below.  Annual marginals are repeated iid only as a
proposed contract-model experiment.  There is no subannual construction, no
claim of observed cliquet trading or exercise, no claim that OptionMetrics
identifies future dynamics, and no assertion that either model is truth.

The manuscript, supplement, frozen financial results, and raw data are
read-only.  There is no full-panel expansion or manuscript integration in this
pilot.

## Frozen inputs and models

Dates, in their prior frozen order:

1. 2016-10-31
2. 2008-11-28
3. 2020-02-28
4. 2009-04-30
5. 2006-03-31

For each date:

- `black_scholes_atm`: one-year lognormal gross return with the frozen annual
  gross forward and ATM one-year implied volatility;
- `annual_lognormal_mixture`: the existing fitted annual two-lognormal
  marginal, used with the same annual gross forward.

The per-period discount is `exp(-zero_rate)` for one year.  Rates are never
modified.  The fitted mixture's calibration status and fit errors are carried
into all finance tables.

## Frozen proposed contracts

Participation is 100%; each tuple uses annual resets and `G=3c`:

| ID | Local annual cap `c` | Global cap `G` | Resets `N` | Exercise dates |
|---|---:|---:|---:|---:|
| `c06_g18_n6` | 6% | 18% | 6 | 5 |
| `c08_g24_n8` | 8% | 24% | 8 | 7 |
| `c10_g30_n10` | 10% | 30% | 10 | 9 |

These are proposed test tuples, not documented issued contracts.  No tuple may
be added, removed, or adjusted after outcomes are inspected.

## Frozen perturbations and directions

The physical boundary error is `delta=eta*c`, with

```text
eta in {1/8, 1/16, 1/32, 1/64, 1/128, 1/256}.
```

For every exercise date, test isolated `+1` and `-1` directions.  For every
contract test `all_up`, `all_down`, `ramp_up`, `ramp_down`, and an alternating
`+1,-1,+1,...` direction.  Thresholds outside `[0,G]` at a finite scale remain
valid never-stop/always-stop policies and are flagged; they are not deleted.

## Numerical design fixed in advance

All primary calculations are deterministic; no Monte Carlo check is planned,
so seed selection and multiple-comparison coverage are not applicable.

- root solve: Brent, absolute tolerance `1e-14`;
- expectation: analytic truncated-lognormal moments, cross-checked by
  Gauss--Legendre quadrature;
- backward evaluator: uniform state lattice with `4096` intervals per local
  cap and `128` return quadrature nodes per lognormal component;
- forward performance-difference evaluator: independent mass-conserving
  convolution lattice with `4096` intervals per local cap;
- all-case refinement: `2048/64`, `4096/128`, and `8192/256` lattice/quadrature;
- sentinel refinement: `16384/512` on the base contract for the first frozen
  date under both models;
- hard compute limit: 30 wall-clock minutes and 2 GiB working memory;
- atom collision tolerance: `1e-8` in accumulator units;
- positive numerical density tolerance: `1e-8`;
- absolute value reconciliation target: `2e-6` of notional;
- analytic/grid boundary target: two fine-grid widths;
- component reconciliation target: `1e-9` absolute.

For each row, define a certification floor as ten times the maximum of the
successive-refinement change, the forward/backward discrepancy when available,
and floating-point summation bound, with an absolute minimum of `1e-11`.
Relative ratios are suppressed when the exact loss is at or below that floor.

## Tests and decision gates

### Occupancy/alignment gate

For the base tuple and both laws, at least four of five snapshots must have an
inception-generated fresh trace followed by a positive zero-credit copy, with
the source date retaining at least two later exercise opportunities.  The
relevant one-sided gap slope must be positive, the boundary must avoid an atom
collision, and the direction-specific gates must leave positive weighted mass.

### Mechanism/accuracy gate

At the two smallest resolved perturbations on the base tuple, the full-prefix
coefficient must have median relative error at most 5% for `all_up` and
`all_down`, and must be closer than the additive single-date benchmark in at
least 80% of rows for which their predictions differ by more than the
certification floor.  Finite-scale failures and all exclusions remain visible.

### Finance gate

For each reference model, evaluate its native policy and the other model's
transferred threshold under the reference law.  Report exact transfer loss,
bps of notional, dollars per USD 1 million, fraction of premium, and fraction
of early-exercise value only when denominators exceed their floors.  Model
price differences are reported separately.

`ADOPT FOR EXPANSION` requires the occupancy and mechanism gates plus a median
directed-transfer loss of at least 1 bp on the base tuple and at least three
resolved base cases in which the prefix correction changes the finite-loss
assessment by at least 0.5 bp while the full approximation is within 25%.

`LIMITED VALUE` applies when the mathematics, occupancy, and asymptotic
mechanism pass but the finance gate does not, or materiality is isolated.

`REJECT` applies if eligible inception occupancy, clean alignment, or the
independently reconciled asymptotic mechanism fails materially, or if no
directed base transfer loss resolves above 0.1 bp.

The verdict is chosen mechanically from these frozen gates.  A shrinking-error
success is a theorem check, not evidence of practical materiality.
