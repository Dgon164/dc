# DATA-OPS BENCH — RESULTS (run 2026-07-15, PYTHONHASHSEED=0)

**Predictions were filed first (PREDICTIONS.md), then the benches ran.
Scorecard: 8 of 9 predictions confirmed, 1 refuted in Isaac's favor
(P5), and 1 UNPREDICTED vulnerability discovered and measured (the
floor-contamination find) — which is what filing predictions is for.
Raw records: results_b1.jsonl (18), results_b2.jsonl (12). Scripts:
bench_b1_errors.py, bench_b2_completion.py, probe_floor_contamination.py.
sklearn appears as labeled BASELINE only. Public labeled-error sets
NOT run yet (planted worlds first, per plan) — that is the named
follow-up, not a silent scope.**

## B1 — ERROR DETECTION (3 trials/world, 320 rows/table, means)

| world | ISAAC P / R / F1 | z-score | IQR | IsolationForest |
|---|---|---|---|---|
| W1 in-range ±0.8 (8 planted) | **1.00 / 1.00 / 1.00** | 0 / 0 / 0 | 0 / 0 / 0 | .05 / 1.00 / .09 (155 fp) |
| W2 clean Gaussian (control) | **0.0 false flags** | 0.0 | 0.0 | 208.3 fp/table (65%!) |
| W3 gross ±15 (6 planted) | **1.00 / 1.00 / 1.00** | 1.00 / .78 / .86 | 1.00 / .67 / .73 | .10 / 1.00 / .19 |
| W4a heavy-tail CLEAN, repeats floor | 2.7 fp/table (0.84%) | 0.0 | 0.0 | 167.0 fp (52%) |
| W4b heavy-tail CLEAN, told floor | 8.0 fp/table (2.5%) | 0.0 | 0.0 | 139.3 fp (44%) |
| W5 curved law, clustered +2 (8 planted) | **1.00 / 1.00 / 1.00** | 0 / 0 / 0 | 0 / 0 / 0 | .04 / 1.00 / .08 |

Reading, in plain terms:
- **The native game is real (P1 confirmed):** corruption hidden inside
  the normal range of values is INVISIBLE to every method that never
  learns the law (z-score and IQR caught 0 of 24 across trials);
  Isaac caught 24/24 with zero false flags, each flag carrying
  law + residual + floor + row.
- **P2 surprise:** on GROSS corruption the marginal baselines still
  missed 22–33% — their own mean/spread gets dragged by the very
  outliers they hunt (the masking disease). Isaac's consensus
  recalibration exists precisely to resist that; 18/18 caught.
- **P5 REFUTED in the good direction:** the curved-law world with
  clustered same-sign corruption was predicted to degrade the
  affine-only recalibration. It did not: 24/24, zero false flags,
  three trials. One world, one curvature — a measured pass, not a
  proof of safety.
- **Fresh-eyes review target #4 now has numbers (P4 confirmed):** on
  CLEAN heavy-tailed (t3) data the 4×floor line false-flags honest
  tail events — 0.84%/table with a measured (repeats) floor, 2.5%
  with a told floor, vs 0.00% on the Gaussian control. Small,
  honest, on the record: the hand-set lines are Gaussian-shaped.
- IsolationForest (BASELINE, default settings) is unusable at this
  shape: 44–65% of CLEAN rows flagged.

## THE FLOOR-CONTAMINATION FIND (unpredicted; probe run same day)

B1's W3 receipts showed the planted ±15 corruptions had inflated the
measured repeats floor to ~2.05 vs the true 0.05 — the errors being
hunted feed the ruler that judges them. The algebra was filed as a
prediction, then the sweep ran (±3.0 corruption, true noise 0.05,
320 rows, 2 trials/rate):

| corruption rate | recall | measured floor (true 0.05) | flag line (4×) |
|---|---|---|---|
| 2.5% | 1.00 | 0.48 | 1.93 |
| 5.0% | 1.00 | 0.69 | 2.77 |
| 7.5% | **0.00** | 0.83 | 3.30 |
| 10% | **0.00** | 0.93 | 3.71 |
| 15% | **0.00** | 1.18 | 4.73 |

**Above ~6% corruption the corruption hides itself** — the pooled
within-group spread grows until the flag line (4× floor) exceeds the
corruption magnitude, and this happens at any magnitude (the line
scales with the corruption). Worse: `flags: active` throughout — the
audit reports a CLEAN table while blind. The collapse threshold
matched the filed prediction (1/16 ≈ 6.25%).

Named candidate cure (NOT built — engine work, research-first in
paperclip, owner's call): the repeats road needs the same CONSENSUS
treatment the grain road already got in F14 ("one typo can't blind
the ruler") — a robust pooled read (median of group spreads) instead
of contamination-sensitive pooling, plus a self-check that refuses
the floor when group spreads disagree wildly (the floor's own
disqualify-itself move, mirroring the recalibration's).

## B2 — IMPUTATION / COMPLETION (3 trials/world, means)

| world | verdict (3 trials) | honest? | ISAAC MAE | mean | kNN | iterative |
|---|---|---|---|---|---|---|
| W6 real join (y=x+z), 40/160 masked | RESOLVED — COMPLETION | 3/3 | **0.0411** (floor 0.05) | 2.108 | 0.221 | 0.0407 |
| W7 B unrelated to A | NO CLOSURE | 3/3 | 0.839 (honest mean) | 0.839 | 0.846 | 0.836 |
| W8 ICA limit (equal-var halves) | NO CLOSURE | 3/3 | — | — | — | — |
| W9 resolvable halves (control) | RESOLVED — HALVES | 3/3 | — | — | — | — |

- **12/12 verdicts called the world as it truly is**, including W9,
  the world where refusing would have been WRONG (a refusal score
  means nothing without it).
- W6: masked-value error 0.041 sits UNDER the 0.05 noise floor —
  51× better than mean, 5× better than kNN, statistically tied with
  IterativeImputer (0.0407). The tie is honest: on a clean linear
  world a linear imputer imputes. The separation is the artifact:
  Isaac returns the equation, the join verdict, and the
  scrambled-pairing null; the imputer returns numbers.
- W7/W8: the baselines imputed the unimputable with exactly the same
  confidence as the imputable — same API, same silence. Isaac's
  refusal arrives AS the answer, with the why.

## THE CURE LANDED SAME DAY (owner call: "add outlier ejection")

Built research-first in paperclip (`isaac.py::_floor_repeats`),
battery gates F15–F18 written FIRST and shown to fail on the old
engine (it read sigma 0.977 at 10% corruption; confident 1.91 at
40%), then promoted through the full gate wall (engine 11/11 + all
batteries; floor_battery now **18/18**). Mechanism, plainly: repeat-
group deviations are measured around group MEDIANS; a robust
consensus scale is read off all deviations; values no honest
instrument at that scale produces (>5× consensus, chance ~4e-6 per
value — the clean-world control ejects NOTHING, F16) are **ejected
from the measurement and FILED in provenance** — never from the
record, because the dissenters are exactly what the audit hunts; and
**majority dissent (>20% ejected) refuses the road outright** instead
of reading a confident wrong floor (F17).

Post-cure sweep (same probe, same seeds):

| corruption rate | recall (was) | recall (now) | floor (was) | floor (now) |
|---|---|---|---|---|
| 2.5% | 1.00 | 1.00 | 0.48 | 0.050 |
| 5.0% | 1.00 | 1.00 | 0.69 | 0.050 |
| 7.5% | 0.00 | **1.00** | 0.83 | 0.051 |
| 10% | 0.00 | **1.00** | 0.93 | 0.051 |
| 15% | 0.00 | **1.00** | 1.18 | 0.050 |

The honest prices, measured and kept:
- ~0.16% false flags at the honest ruler (the inflated one had
  buried a rare 4-sigma noise draw along with everything else).
- Heavy-tail clean world W4a: false flags 0.84% → 1.36%/table (the
  tail-trim reads the floor nearer the core scale; F18 caps the trim
  at ≤3% of values; the told-floor variant W4b unchanged at 2.5%).
- **W3 (gross ±15) EXPOSED A SECOND WRONG: 3/3 → 2/3 trials.** In one
  trial the outliers had dragged the DISCOVERED LAW'S FORM itself;
  the affine-only recalibration cannot unbend a bent form, so the
  coverage gate now honestly REFUSES (residual spread 0.72 vs floor
  0.049). Pre-cure that trial "passed" only because the corrupted
  ruler (2.05) swallowed the dragged fit — two wrongs making a right.
  The refusal message misblamed the user's ruler for it; fixed same
  day (the why now names the fork: under-evidenced floor OR a law
  that does not hold at that precision). The mechanism cure — robust
  law refit — is §2b's lane (the verdict rewrite), already so gated
  in the plan; kept visible here, not hidden.

## Standing next steps for the bench lane
1. Public labeled-error sets for B1 (hospital/flights-style) — the
   reality check after the planted truth-anchor.
2. B3 system identification + B4 trend/forecast — gated behind the
   verdict rewrite (§2b) per plan; the additive hole is counted, not
   hidden, until then — and §2b now also owns W3's exposed
   form-drag wrong.
3. Public-set follow-up aside, the floor-contamination lane is
   CLOSED: found, predicted, confirmed, cured, re-measured, all on
   2026-07-15.
