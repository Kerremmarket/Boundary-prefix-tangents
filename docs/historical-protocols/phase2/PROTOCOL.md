# Phase II frozen protocol

**Frozen parent commit:** `0dd5841406973839a72e818f75c81618b70e4280`

**Phase II branch:** `codex/paper1-financial-extension-phase2`

**Protocol freeze date:** 26 August 2026

**Manuscript policy:** Paper I source and PDFs remain untouched until the final
integration recommendation has passed a fresh-context hostile audit.

## 1. Frozen Phase I evidence

Phase II treats the following as inherited evidence, not as outcomes to be
reselected:

- 248 monthly OptionMetrics dates, January 2005--August 2025;
- 248 converged annual two-lognormal-mixture calibrations;
- benchmark contract: participation `0.80`, cap `0.08`, surrender haircut
  `0.08`, annual independent termination probability `0.03`;
- 60 dates with a regular stationary upper surrender boundary;
- copied coefficient positive and prefix closer than the date-diagonal
  calculation on all 180 date/joint-direction comparisons at `epsilon=0.005`;
- median copied share about 14%, median prefix uplift about 16%, and small
  benchmark dollar losses.

The five licensed source Parquets remain in the existing project `output/`
directory. Their hashes are frozen in
`../evidence/optionmetrics_stage0_audit.json`; Phase II does not reconnect to
WRDS or replace the extraction.

## 2. Frozen product projections

The three projections below are fixed before any Phase II Bellman result is
examined. “Documented” means supported by the issuer materials in
`product_source_dossier.md`. “Projection assumption” means a deliberate model
choice and is never attributed to the issuer.

### A10 — Athene Accumulator 10, high premium band

- Documented S&P 500 one-year point-to-point cap: `10.00%` for the high band in
  the May 2026 product guide; participation is modeled as 100% for this capped
  strategy.
- Documented start-of-contract-year withdrawal charges:
  `9%, 9%, 8%, 7%, 6%, 5%, 4%, 3%, 2%, 1%, 0%`.
- Documented free withdrawal: 10% annually.
- Documented death-benefit basis: greater of accumulated value and minimum
  guaranteed contract value.
- Documented MVA: applies above the free-withdrawal amount during the charge
  period outside California.

### AA7 — Allianz Accumulation Advantage 7, $100,000+ band

- Documented S&P 500 annual point-to-point cap: `8.00%` in the rate sheet dated
  4 November 2025; the issuer states that participation is 100% unless noted.
- Documented start-of-contract-year withdrawal charges:
  `8.50%, 8.00%, 7.00%, 6.00%, 5.00%, 4.00%, 3.00%, 0%`.
- Documented free withdrawal: up to 10% under the brochure's conditions.
- Documented death benefit: greatest of accumulation value, guaranteed minimum
  value, or net premium.
- Documented MVA: applies to chargeable withdrawals during the first seven
  years and depends on the interest-rate environment.

### IP4 — MassMutual Ascend Index Protector 4

- Documented S&P 500 one-year point-to-point capped strategy with interest
  credited annually and never below 0%.
- Primary low-cap stress uses the documented guaranteed minimum cap of `3.00%`,
  not an assertion that 3% was the prevailing new-business cap on the study
  dates.
- Documented start-of-contract-year early-withdrawal charges:
  `5.60%, 5.60%, 5.60%, 5.60%, 0%`.
- Documented penalty-free withdrawal: 10% under the brochure's conditions.
- Documented guaranteed minimum surrender value: 87.5% of purchase payments,
  accumulated at a guaranteed rate and reduced by withdrawals.

### Common projection assumptions

- Full surrender is allowed only at annual crediting anniversaries; partial
  withdrawals, annuitization, optional riders, waivers, Index Lock, taxes, and
  additional premiums are excluded.
- The first modeled surrender decision is immediately after the first annual
  credit. It uses the published charge at the **start of contract year 2**.
- `X_0=1` is the issue-state account-to-normalized-guarantee ratio. The inherited
  `X_0=rho b` design remains only a local trace diagnostic.
- Annual mortality/termination probability is 3%. This is a research
  assumption, not an issuer term or an estimate of surrender behavior.
- The normalized termination benefit is `max(1,X)`. It is a common reduced-form
  guarantee mapping, not an exact reproduction of every documented minimum
  value formula.
- The valuation-date OptionMetrics annual marginal is reused independently at
  each contract year. This is an annual risk-neutral projection, not a physical
  forecast or identified multiperiod SPX path law.
- The initial declared cap is held constant in the main projection. Issuer
  documents state that renewal caps are reset annually; a time-varying-cap
  control must therefore be reported separately.
- The scalar primary projection sets MVA to zero. A rate-state/MVA negative
  control tests the consequence of this omission. No product-level conclusion
  may rely on silently treating the MVA as zero.
- Full-surrender value in the scalar calculation is `(1-kappa_t)X`. Documented
  guaranteed-minimum surrender values are disclosed but not modeled because
  their crediting rate and separate guarantee-base state are not identified in
  the OptionMetrics data. In particular, the IP4 `87.5%` base is not silently
  substituted for its complete contract formula.
- The renewal-cap negative control is fixed before its numerical run. A10 and
  AA7 alternate after the initial term between `75%` and `100%` of the declared
  initial cap; IP4 alternates between its documented `3%` minimum and `4%`.
  These are deterministic stress paths, not forecasts of issuer declarations.
  They test transport geometry only.

## 3. Phase II-A numerical design

### Market dates

The primary mathematical and product audit uses the three regular-boundary
dates inherited from the 12-date market-state pilot:

- 2006-03-31;
- 2022-10-31;
- 2023-05-31.

The no-boundary controls are 2008-11-28, 2016-10-31, and 2020-03-31, also from
the predeclared pilot. No date is selected using a new Phase II regret outcome.

### Horizons and directions

- Active surrender horizons: 3, 4, and 5 dates; the full published charge
  schedule plus five post-charge anniversaries is also evaluated.
- Relative upward directions for five dates:
  - common: `(1, 1, 1, 1, 1)`;
  - front-loaded: `(1.2, 1.1, 1.0, 0.9, 0.8)`;
  - back-loaded: `(0.8, 0.9, 1.0, 1.1, 1.2)`;
  - alternating: `(1.0, 0.6, 1.2, 0.8, 1.1)`.
- Epsilon grid: `(0.02, 0.01, 0.005, 0.0025, 0.00125)`.
- Reference grid: at least 120,000 log-spaced state points for final reported
  boundary/slopes; independent refinement uses 30,000 and 60,000 points.

### Required controls

1. the stationary Phase I contract;
2. each documented time-varying withdrawal-charge schedule;
3. a post-charge stationary window;
4. time-varying renewal caps;
5. a rate-state/MVA transition in which the credited floor copies the account
   coordinate but not the stopping normal;
6. the IP4 low-cap reachability control.

## 4. Ordered-prefix and pairwise tests

For `K=p_0 I+K^fresh`, every target history is labeled by the length of its
terminal all-floor run. The implementation must reconcile:

1. the component sum over labeled run lengths;
2. the aggregate causal-prefix integral;
3. direct finite-perturbation repricing.

The pairwise audit distinguishes:

- an **additive pairwise approximation**, which sums two-date corrections and
  can double count a long copied run;
- a **coherent labeled reconstruction**, which retains the carrier label and
  is allowed to take the minimum of its target-pair gates.

If the scalar all-floor prefix is exactly reconstructible by the second object,
that fact is a required result, not a failure to be hidden.

## 5. Phase II-A gate

Classify `PROCEED TO COMPUTATIONAL ROBUSTNESS` only if all of the following are
demonstrated:

1. at least one documented projection has three or more active decision dates
   with a theorem-valid nonzero ordered-prefix coefficient;
2. the contribution is present under issue-state occupancy, not only under the
   local `rho b` initialization;
3. direct repricing converges to the full coefficient under joint epsilon/grid
   refinement;
4. product schedules, renewal terms, and the MVA control are reported without
   disguising failures of exact transport;
5. the mathematical result adds verified multidate or lifetime content beyond
   the inherited two-date calculation.

If only the stationary/local construction survives, classify
`APPLICATION REMAINS ILLUSTRATIVE`. If documented terms remove every relevant
aligned carrier, classify `REALISTIC TERMS DESTROY THE MECHANISM`.

Only `PROCEED TO COMPUTATIONAL ROBUSTNESS` authorizes Phase II-B.

## 6. Predeclared Phase II-B protocol, conditional on the gate

- Coarse DP grids: 250, 500, 1,000, and 2,000 points.
- Regression method: independent-path least-squares Monte Carlo with a declared
  polynomial/spline basis selected without reference to prefix performance;
  20 fixed seeds and held-out policy repricing paths.
- Boundary interpolation path:
  `b_t^eta=b_t+eta*(bhat_t-b_t)` for
  `eta in {1, 1/2, 1/4, 1/8, 1/16}`.
- Predictions: date-diagonal, additive pairwise, coherent labeled-pairwise, and
  full ordered-prefix.
- Alternative annual marginal on the six frozen dates: a convex, arbitrage-
  consistent call-price interpolation followed by the implied discrete
  risk-neutral distribution. It is an annual marginal-law control only.
- Local or stochastic volatility is not used unless the Phase II-A MVA/rate
  control creates a specific state-dynamics question that the annual marginal
  comparison cannot answer.

## 7. Claims prohibited throughout Phase II

The extension does not observe surrender choices, insurer reserve errors,
future renewal caps, mortality, lapses, MVAs, or multiperiod SPX returns. It may
not claim true path-law identification, causal regime effects, actual insurer
losses, or portfolio materiality.
