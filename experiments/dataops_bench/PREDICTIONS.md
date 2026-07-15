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
