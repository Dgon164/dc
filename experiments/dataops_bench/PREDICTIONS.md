# DATA-OPS BENCH — PREDICTIONS, FILED BEFORE RUNNING
**2026-07-15. Per the segment plan (§3): predictions filed BEFORE the
first run so the results can surprise honestly. Nothing below has been
executed at filing time. PYTHONHASHSEED=0 everywhere. sklearn appears
as labeled BASELINE only — never the cognition.**

## B1 — ERROR DETECTION (cell-level precision/recall/F1 + false-flag rate)

- **P1 (in-range corruptions, the lane's native game):** corruptions
  planted WITHIN the target's marginal range are invisible to marginal
  flaggers (z-score, IQR see a healthy-looking value) but visible
  against the discovered law. Prediction: Isaac recall high (>0.8),
  z/IQR recall near zero, IsolationForest low.
- **P2 (gross corruptions):** far outside the marginal range — everyone
  catches them. Prediction: all methods recall ~1.0; the artifact
  difference (receipts vs bare scores) is the only separation.
- **P3 (clean Gaussian control):** Isaac false-flag rate ~0 (battery
  A2 already proved this shape); z/IQR flag at their nominal tail
  rates (>0).
- **P4 (heavy-tailed CLEAN world — fresh-eyes review target #4,
  measured):** the audit's hand-set lines (4x-floor flag, MAD<=3x-floor
  coverage gate) were only ever tested on Gaussian noise. On clean
  Student-t(3) noise, prediction: EITHER the coverage gate refuses
  (honest, zero flags) OR tail events false-flag at a materially
  higher rate than the Gaussian control. Either way the number goes on
  the record — this world exists to measure the review target, not to
  pass.
- **P5 (non-affine drag — fresh-eyes #4, second half):** corruptions
  planted correlated with x on a CURVED law (y ~ x^2). The consensus
  recalibration is affine-only by declared scope; prediction:
  precision/recall degrade vs the linear world W1. Size unknown —
  that's the point of measuring.

## B2 — IMPUTATION / COMPLETION (error in UNITS vs floor + refusal honesty)

- **P6 (real completion world):** B's columns complete A's law past
  the scrambled-pairing null. Prediction: Isaac imputes masked targets
  with error near the noise floor; clearly beats the mean imputer;
  competitive with kNN/iterative (BASELINE).
- **P7 (no-structure world):** B is unrelated to A. Prediction: Isaac
  says NO CLOSURE / no gain and refuses to impute beyond the honest
  mean-model, and SAYS so; baselines impute anyway with mean-level
  error and no warning. Refusal scored CORRECT.
- **P8 (ICA identifiability limit):** equal-variance Gaussian halves —
  unresolvable IN PRINCIPLE (triangulate's own documented limit).
  Prediction: NO CLOSURE, scored CORRECT-BY-DESIGN.
- **P9 (the refusal control — refusing everything is not honesty):**
  a resolvable HALVES world (unequal variances) must come back
  RESOLVED, not refused. A benchmark that scores refusals must include
  a world where refusing is WRONG. Prediction: RESOLVED — HALVES.

## Standing predictions from the plan, restated
B1 competitive (the lane's native game); B2 strong where joins exist,
honest refusals elsewhere; B3 mid until the verdict rewrite (§2b) —
the additive hole is the fix's price tag, counted not hidden; B4
respectable on seasonal series, alone on refusing raw returns.

## B3 — SYSTEM IDENTIFICATION (filed 2026-07-17, before the first
run; §2b has LANDED since the section above was written, so the
standing "mid until the rewrite" is superseded by these)

- **P10 (logistic map, r=3.7 chaotic):** x_{t+1} = r*x_t - r*x_t^2
  is inside the atom library (quadratic via lag). Prediction:
  SYMBOLIC RECOVERY — both coefficients said at the planted values
  (the additive hole is closed; the exact road should stamp this on
  the noiseless map, and near-exactly at small noise).
- **P11 (damped oscillator, lag form):** y_t = a*y_{t-1} + b*y_{t-2}
  (a=1.8*0.98, b=-0.98^2 planted). Prediction: RECOVERED via lag
  atoms with coefficients within noise; UNDERSTOOD.
- **P12 (predator–prey, log-linearized rates):** d(ln H)/dt = a - b*L
  sampled coarsely. Prediction: form recovered (linear in L), but
  coefficients degraded by discretization error; PARTIAL or
  UNDERSTOOD-with-degraded-constants both count as recovery; the
  failure mode to watch is the discretization error being minted as
  structure (that would be a WRONG-CONFIDENT, counted against).
- **P13 (outside the library — delay system, Mackey–Glass-shaped):**
  no combination of the atoms says it. Prediction: honest refusal
  (NO RULE / PARTIAL with the remainder named) — scored CORRECT.
- **P14 (scrambled-series control):** any series from P10-P12 with
  rows shuffled breaks the lag structure. Prediction: NO RULE, zero
  minted laws; anything else is a false discovery, counted.

## B4 — TREND / FORECAST (filed 2026-07-17, before the first run)

- **P15 (planted seasonal + trend):** level + slope + one seasonal
  wave + noise. Prediction: Isaac beats naive2 on MASE (the wave is
  in the library); equation readable.
- **P16 (M4 monthly subset, if the download road is open):**
  respectable but BEHIND the big statistical ensembles on aggregate
  sMAPE/MASE vs naive2 — the artifact difference (equations, scope,
  refusals) is the lane's claim, not the leaderboard. If the
  download fails, the leg is recorded BLOCKED, never quietly
  swapped for synthetic.
- **P17 (RAW RETURNS — the headline):** daily equity returns are
  pure noise to this library. Prediction: NO RULE / refusal on the
  overwhelming majority of return series; the refusal RATE is the
  deliverable ("the world's most expensive pure-noise battery").
  A minted return-forecast law would be the lane's r2-ghost —
  counted as WRONG-CONFIDENT if it survives the court.
- **P18 (REALIZED VOLATILITY — the pair):** |returns| / squared
  returns cluster (the lag atoms' home game). Prediction: structure
  FOUND past the null where raw returns refused — the pair is the
  proof the refusal is discrimination, not blindness.
