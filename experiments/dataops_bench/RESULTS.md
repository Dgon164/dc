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

---

# RE-RUN + B3/B4 FIRST RUNS (2026-07-17, session 1 of the alpha push;
# engine = the promoted wall @ research 1daa5df: verdict rewrite +
# comparison words + outlier's law + rounding cure all live)

**Predictions P10-P18 filed first (PREDICTIONS.md, same file, before
any run). Raw records: results_b1.jsonl (18), results_b2.jsonl (12),
results_b3.jsonl (15), results_b4.jsonl.**

## B1 re-run vs the new judge — the 07-15 shape holds, one honest change
| world | ISAAC P / R / F1 (means, 3 trials) | note |
|---|---|---|
| W1 in-range ±0.8 | **1.00 / 1.00 / 1.00**, 0 fp | z/IQR still 0.00 recall — the lane's native game |
| W2 clean control | **0.0 false flags** of 320 | IsoForest 208 fp/table |
| W3 gross ±15 | 0.67 / 0.67 / 0.67 | **CHANGED vs 07-15 (was 18/18): trial 1 now REFUSES** (the two-fork message: floor vs dragged form) — refusal counted at recall 0, said, not hidden; z-score still misses 22% |
| W4a/b heavy-tail clean | 4.3 / 8.0 fp per 320 (1.4% / 2.5%) | the Gaussian-shaped flag line's known cost, on the record (fresh-eyes #4) |
| W5 curved + clustered +2 | **1.00 / 1.00 / 1.00**, 0 fp | P5's degradation prediction stays refuted |

## B2 re-run — 12/12 verdicts honest, unchanged
W6 completion MAE **0.0411** (floor 0.05; iterative 0.0407 — the
honest tie holds); W7 no-structure NO CLOSURE 3/3 (baselines imputed
anyway, no warning); W8 ICA-limit NO CLOSURE 3/3; W9 refusal-control
RESOLVED 3/3.

## B3 first run — SYMBOLIC SOLUTION RATE (predictions P10-P14)
| world | outcome (3 trials) | the number |
|---|---|---|
| logistic map r=3.7 | **RECOVERED 3/3** | `+3.70*x0 -3.70*x0*x0` fresh err 2.7e-15 — machine-exact; P10 CONFIRMED, the additive hole is closed |
| predator-prey (log-lin) | **RECOVERED 3/3** | `-0.40*x0 +1.10` = the planted truth exactly; PARTIAL verdict (discretization remainder honestly named) — recovery per filed P12 |
| damped oscillator AR(2) | MISSED 3/3 | engine says `+0.87*x0` (the AR(1) surrogate). DIAGNOSED: explicit 2-column feed recovers `+1.77*x0 -0.96*x1` (planted 1.7649/-0.9604) UNDERSTOOD — **the miss is entirely in _propose_lags, which proposes NOTHING on a near-unit-root oscillatory series** (its null-gate refuses). Named open door, not hidden |
| delay system (outside library) | **WRONG-CONFIDENT 3/3 — THE FIND** | UNDERSTOOD `+0.98*x0 +0.02`, fresh miss 0.033 on a DETERMINISTIC world. P13 predicted refusal — prediction FALSIFIED, kept. Mechanism: the residual is structured (a function of s[t-17]) but no lag-17 column was ever proposed, so the replay cannot interrogate it — the judge is honest only within the vocabulary the proposers hand it. Same class as the additive hole: count now, cure next, then a delay family joins the duel |
| scrambled control | REFUSED-CORRECT 3/3 | NO RULE; zero false lag discoveries (P14 confirmed) |

Totals: recovered 6/9 recoverable, refused-correct 3/6 refusable,
missed 3, wrong-confident 3 (the bench exits red on purpose until the
delay leak is cured).

## B4 first run — TREND / FORECAST (predictions P15-P18)
P15 planted seasonal+trend: **UNDERSTOOD
`+3.05*sin(0.5235*x0) +0.05*x0 +9.92`** (0.5235 = 2π/12 read off the
data) — **MASE 0.546 vs naive 3.26 vs seasonal-naive 1.607**: beats
the competition's own baseline family with a readable equation.
P16 M4: BLOCKED as filed (~300 MB train file; said, not swapped for
synthetic). P16 sunspots (public series): honest NO RULE refusal —
quasi-periodic, consistent with the gauntlet.

**P17 — THE HEADLINE, CONFIRMED: refusal 8/8, minted 0.** Real
daily market returns (Ken French daily factors — the third download
road; stooq robot-walled and FRED timed out, both recorded), 8
rolling 500-day windows: Isaac refused to mint a return-forecast law
in every window. Zero wrong-confident laws on the world's most
expensive pure noise.

**P18 — SUBSTANTIALLY FALSIFIED, kept.** The pair-claim ("structure
found on volatility where returns refused = discrimination, not
blindness") did NOT establish this run: one-step |r| found structure
in **0/8** windows (run 1); the literature's weekly realized
volatility (sqrt of 5-day squared-return sums, ~100 points/window)
found it in **1/8** (run 2). Both numbers on the record. Named
follow-up, not a re-roll: longer spans (weekly RV over 2000-day
windows gives n~400 where the clustering signal should clear the
selection-corrected null if it is real for this library) — until
then, the refusing-returns headline stands WITHOUT the pair's
support, and says so.
