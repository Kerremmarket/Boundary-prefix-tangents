# Frozen financial-diagnostic protocol

**Starting commit:** `e0ed840cca63f76dc6c16b958f8a9f3efa430c9a`

**Branch:** `codex/paper1-financial-diagnostic`

**Freeze date:** 27 August 2026

**Manuscript policy:** Paper I source and PDFs remain untouched. This is a
bounded control study and a condensation of already-frozen annuity evidence,
not a new application search or a redevelopment of the theorem.

## 1. Questions fixed before computation

The study asks only whether a standard smooth Bermudan control behaves as the
date-diagonal theory predicts, and whether the frozen indexed-annuity exercise
survives a stricter financial interpretation. It does not estimate a learned
continuation value, prove a random-tangent theorem, or establish an RL result.

The integration gate is passed only if all reported Bermudan boundaries are
regular, the supplied-boundary losses have stable quadratic scaling, the
date-diagonal coefficient is accurate at small scale, and the annuity failure
modes remain prominent. Otherwise the study is retained as an appendix-level
diagnostic or rejected.

## 2. Local OptionMetrics provenance and prospective dates

No WRDS connection or new extraction is used. The inputs are the committed,
compact diagnostics derived from the five previously validated local licensed
Parquets. The source hashes and cleaning record are frozen in
`../evidence/optionmetrics_stage0_audit.json`. The exact market-state columns
are read from `../evidence/full_market_panel/date_status.csv`.

The first five dates in the predeclared farthest-point market-state selection
are used in their original order. They were selected using only one-year ATM
IV, one-year 25-delta skew, the 182-to-730-day ATM-IV slope, and the one-year
zero rate; no Bermudan outcome enters selection:

1. 2016-10-31;
2. 2008-11-28;
3. 2020-02-28;
4. 2009-04-30;
5. 2006-03-31.

For each date, normalize spot and strike to one. Set the one-year constant
Black--Scholes volatility to `atm_iv_1y`, the continuously compounded rate to
`zero_rate`, and the dividend yield to

\[
q=r-\log(\texttt{annual\_gross\_forward}).
\]

This is a deliberately smooth, internally consistent control calibrated to a
small set of OptionMetrics-derived summaries. It is not a claim that the
Black--Scholes model fits the full smile or identifies the physical SPX law.

## 3. Frozen Bermudan contract and reference solver

The contract is a one-year at-the-money Bermudan put with payoff
`(1-S)^+`. There are eight equally spaced exercise dates
`t_i=i/8`, `i=1,...,8`; maturity exercise is compulsory. There is no exercise
at the calibration date.

The primary dynamic-programming reference uses log moneyness
`x=log(S/K)` on `[-4,2]`, 65,537 equally spaced nodes, and full Gaussian
convolution evaluated by a zero-padded FFT. The grid endpoints are far enough
from every decision boundary that the omitted Gaussian tails are numerically
negligible; this is checked directly. Reference audits use 32,769 and 131,073
nodes, and an independently coded 256-point Gauss--Hermite solve is retained as
a method audit. A reference boundary is admissible only when
the native payoff-minus-continuation signs have exactly one exercise-to-
continuation crossing below the strike at every nonterminal exercise date.

The primary coarse-grid policy is solved by the same convolution method on 257
nodes on the same interval. Its continuous supplied boundary is the midpoint
between the highest native exercise node and the adjacent continuation node;
no zero of a fitted continuation curve is imposed. Node counts 129 and 513 are
reported as solver refinements but do not replace the frozen primary policy.

## 4. Frozen LSMC policy

The regression control is ordinary least-squares Monte Carlo, trained on
60,000 independent Black--Scholes paths per market date for each of ten seeds
`2026082701,...,2026082710`. At each exercise date it regresses the discounted
realized future cash flow on the alive in-the-money paths using the two fixed
shifted-Legendre functions of spot moneyness,

\[
1,\quad 2S-1.
\]

This low-variance basis was frozen after a pre-market synthetic numerical
sanity check, before any selected-date Bermudan output was computed. The check
showed that quadratic and cubic global polynomials can create purely
extrapolative extra sign crossings. Reducing the basis, rather than deleting
unfavorable seeds or forcing a cutoff, preserves a genuinely native sign audit.
The resulting LSMC control is intentionally simple and is not presented as a
solver tournament winner.

The predicted continuation value is clipped below at zero. The native rule
exercises only with strictly positive intrinsic value and when intrinsic value
is at least the clipped regression estimate. Regression coefficients, sample
counts, ranks, and condition numbers are retained.

Native policy signs are scanned before any threshold representation is used.
An LSMC seed is boundary-regular only if every nonterminal date has exactly one
exercise-to-continuation crossing on `x in [-4,0]`, exercises on the low-price
side, and continues immediately below the strike. Only then is its unique
crossing used as a supplied boundary. Multiple crossings, missing crossings,
or reversed orientation are reported as failures rather than repaired.

Each trained native policy is repriced on 200,000 held-out paths generated from
a date-specific evaluation seed, respectively
`3026082701,...,3026082705` in selection order, disjoint from every training
seed. Common
held-out paths are used across the ten policies on a market date. The primary
small-loss calculation uses a separate deterministic fixed-policy forward
density engine so that Monte Carlo noise cannot manufacture or conceal the
quadratic limit.

## 5. Supplied-boundary path, coefficient, and actual loss

For each admissible approximate boundary vector `bhat` and reference boundary
vector `b`, use log-price interpolation

\[
b_i^\eta=b_i+\eta(\widehat b_i-b_i),\qquad
\eta\in\{1,1/2,1/4,1/8,1/16\}.
\]

Let `G_i(x)` be reference payoff minus reference continuation and let `f_i(x)`
be the predecision **subprobability** log-price density of paths that have
survived every earlier reference decision. The date-diagonal prediction for
the full displacement is

\[
C_{\rm diag}=\frac12\sum_{i=1}^7 e^{-rt_i}
 f_i(b_i)\,|G_i'(b_i)|\,(\widehat b_i-b_i)^2.
\]

Actual supplied-policy loss is computed by the exact performance-difference
identity on a separate 32,769-cell forward density grid with Gaussian
convolution and exact mismatch-strip localization. The same engine calculates
each one-date perturbation. The joint-minus-sum-of-one-date residual is the
finite-scale interaction diagnostic; under the smooth Black--Scholes kernel it
must be `o(eta^2)`, not a second-order prefix coefficient. Backward policy
repricing and held-out path repricing are independent audits of the forward
identity.

The predeclared numerical success criteria are:

- one correctly oriented crossing at all seven nonterminal dates;
- a log--log slope of actual loss on `eta <= 1/4` in `[1.90,2.10]`;
- `actual loss/(eta^2 C_diag)` within 10% of one at `eta=1/8` and `1/16`;
- absolute joint interaction no more than 10% of joint loss at `eta=1/16`;
- reference boundary changes below `2e-4` in log price and reference value
  changes below `2e-5` under the declared grid refinement, with the 256-point
  Gauss--Hermite audit separately disclosed rather than used to reselect the
  reference;
- deterministic native-policy loss consistent with the 95% held-out Monte
  Carlo interval, allowing an additional deterministic discretization tolerance
  of `2e-5` in normalized option value.

## 6. Frozen annuity falsification audit

The annuity experiment is not rerun. The study reads the committed Phase II-A
evidence and must preserve all four conclusions:

1. the local stationary-floor identity and finite-perturbation convergence are
   valid;
2. the scalar all-floor prefix is exactly reconstructible by coherent labelled
   pairwise minima, although the additive pairwise sum can overcount;
3. issue-state eligible occupancy is exactly zero for all 84 regular rows
   because the support has `X>=1` while the boundaries lie below one;
4. time-varying renewal caps move the boundary, and the rate-state/MVA control
   loses complete transport of the stopping normal.

The necessary condition distilled from this audit is: a singular component is
not enough; it must also transport the complete target normal exactly and carry
positive occupied eligible trace.

## 7. Output gate and prohibited claims

The compact synthesis must compare:

- smooth Bermudan put: no singular diagonal carrier;
- full-recall search: aligned and occupied copied-state prefix;
- stationary annuity local construction: valid local coefficient but zero
  issue-state eligible trace;
- realistic annuity controls: exact transport fails.

The intended main-text diagnostic is limited to three to five pages, with
solver and seed details in an appendix. It may claim only a supplied-boundary
validation under a fixed smooth model. It may not claim a learned-policy
theorem, RL sensitivity, Heston robustness, a new contract application, a
full OptionMetrics calibration exercise, or general empirical validity of the
boundary-prefix theorem.
