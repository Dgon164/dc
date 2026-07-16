# RESEARCH WRITEUP — 2026-07-15: THE NEW JUDGE AND THE POOLS' THREE CLIMBS
**Plain-language research record (the writeup law: what was built, what
the structures look like, how information is represented, what was
found, how it performs, where it breaks). Companion to
POOL_BRANCH_EVAL_2026-07-15.md (the verdict sheet) and
verdict_rewrite_probe.py / the three pool_*.py probes (the code).
dc-only, like the testing-segment records.**

---

## PART 1 — THE NEW JUDGE (the §2b verdict rewrite, reference build)

### What was built
Isaac's judge is the piece of code that looks at a candidate equation
and decides: do we UNDERSTAND this data, or not. The old judge was one
number: if the equation matched 90% of the held-out variance
(r² ≥ 0.9), stamp UNDERSTOOD. Four measured attack families showed
that number can be earned by WRONG equations.

The new judge (reference v3, `experiments/verdict_rewrite_probe.py`,
23/23) is a court with three independent examiners and a metrology
rule:

1. **The exact road.** Before any statistics, try tiny formulas
   (1–3 terms). A hit must reproduce EVERY row to machine precision —
   on the training half AND the held-out half. Noise cannot fake
   machine-perfection on rows the fit never saw (measured: 0/20 fake
   stamps on noise at n=8). This is how Kepler's law with only 8
   planets gets stamped — where the OLD judge found nothing at all
   (r² = −2.94).

2. **The replay (leg 1).** After fitting, take the leftover error on
   held-out rows and ask every known building block: "can YOU predict
   this leftover?" The bar is selection-corrected: because we ask many
   candidates, the threshold is the maximum score over all of them on
   shuffled leftovers (the 119/120 lesson). If anything clears it,
   something is missing — never stamp UNDERSTOOD.

3. **The pool (leg 2).** A frozen random net — the kiddie pool — asks
   the grammar-free version of the same question: can ANY cheap
   function of the inputs predict this leftover better than its own
   shuffled self? This catches structure that no known building block
   expresses (a conditional rule, a bent form).

4. **The floor rule (the 13-inch foot).** UNDERSTOOD also requires the
   leftover to sit within 3× an EVIDENCED measurement floor — a floor
   read from the data's grain, from repeated measurements, or told by
   the owner. No floor evidenced and not machine-exact? The verdict is
   BEST-RESIDUAL — "here is the best equation so far, and I cannot
   certify it," never a confident stamp. Constants also snap to the
   coarsest ruler marks that survive held-out refit: 3.000154 measured
   on a coarse ruler honestly says 3.

### How it represents things
The verdict is no longer a number — it is a RECORD: the equation, the
floor and its provenance, each examiner's state (fired / quiet /
bypassed-with-reason), and the residual in units. Every leg either
fires or leaves a receipt saying why not (the consideration law).

### What was found, and how it performs
- The additive hole is cured: given a correlated pair where the old
  judge shipped `0.32*x0*x1` as UNDERSTOOD, the new judge recovers
  exactly `x0 + x1` (4/5 seeds; never confidently wrong 0/5).
- Poisoned data (±1 errors in 10% / 5% of cells): confidently-wrong
  0/10 across both rates — and in the 10% case the POOL was the only
  examiner that fired (replay quiet at 0.465 vs bar 0.629; pool scream
  0.890 vs 0.062). The second leg earns its seat.
- The judge does not chase the ruler's own artifacts: a true law on a
  coarse grain leaves a sawtooth-shaped leftover that belongs to the
  instrument, not the world — pool quiet 4/4, law stamped clean.
- Availability cost, filed: under poison the true form is recovered
  0/10 (an impostor term gets admitted first and cannot be displaced).
  Safety is solved; recovery needs the RIVAL-FORM DUEL (already proven
  3/3 elsewhere) — a port requirement, not a mystery.
- Cost: 0.77 s per full three-examiner verdict at n=400.

### Where it breaks / boundaries
Grain-only floors under-evidence real process noise (known, kept): an
honest PARTIAL results until a repeats road or a told sigma arrives.
The pool leg's competence starts at n≈48 rows (measured); below that
it bypasses with a receipt.

---

## PART 2 — RECURSIVE POOLS: relations between structures themselves

### The idea (owner): "what if we recursively applied these pools to
### find relational information about the structures themselves"
A matrix can hold pairwise relations between raw things — that is all
it can hold. The climb: make each discovered STRUCTURE a thing with
descriptors, and point the same instruments at the descriptors.
Relations among structures become ordinary channels one level up.

### What the structure actually looks like
```
LEVEL 0 — raw data, 10 groups per episode, each its own clean law:
  group A: 40 rows of (x, y) with y = a_A · x
  group B: 40 rows of (x, y) with y = a_B · x
  ...      (nothing relational is visible inside any group's rows)

LEVEL 1 — the DESCRIPTOR TABLE (one row per episode, 240 episodes):
  episode |  a_A    a_B    a_D  ...  (9 columns)   | target: a_C
     1    | +1.31  −0.62  +0.94 ...                |  −0.81
     2    | −1.75  +0.94  −1.22 ...                |  −1.65
    ...
  planted truth: a_C = a_A · a_B every episode — a relation between
  LAWS, not between data points. Marginally invisible (product of
  signed constants: correlation with either factor ≈ 0).
```
The pool is level-agnostic by construction — a frozen random
projection does not know what its channels mean — so the SAME
instrument that reads raw channels reads descriptor channels.

### What was found, with numbers
- Level 0 is honestly blind: 20/20 quiet on any group's own residual
  (the relation does not live in raw data).
- Level 1 detection: pool screams on the descriptor table (+0.536 vs
  null99 −0.431). Attribution names EXACTLY {A, B} of 9 candidates
  (0.93 / 1.00 vs threshold 0.10); plain correlation sees NOTHING
  (0.110 vs 0.189) — two channels that look unrelated to the target,
  told apart through the frozen net.
- GENESIS, pools removed: the engine mints the readable equation
  `a_C = 1.00 · a_A · a_B`, exact to 3e-15, predicts 12 UNSEEN
  episodes' constants to 2e-15, and composes down through the level-0
  law to predict raw y on fresh x to 4e-15. A relation between laws,
  discovered, named, minted readable, proven without the scaffolding.
- Scrambled control: alignment permuted → pool quiet AND the engine
  refuses (NO RULE). Noise robustness: at σ = 0.05 / 0.2 detection and
  naming hold exactly, mint errors scale linearly (3.96× measured vs
  4× predicted), no false exactness. Level 2+ follows by induction: a
  minted relation node has descriptors too.

### Where it breaks
- **The judge is the weak floor, not the pools** (BROKEN, verifier
  upheld): plant a relation OUTSIDE the equation grammar —
  a_C = max(a_A, a_B) — and the pools still point perfectly (scream;
  name {A,B}), but the engine's OLD judge stamps a confident
  UNDERSTOOD (r² = 0.99945) on a 9-term sin/cos/exp impostor hugging
  the max-surface, with error 7e-2 on an EXACT world. The single-lens
  pool gate is quiet on that residual (−0.740 vs −0.547): after a
  9-term fit the leftover is a thin high-frequency crease along the
  fold line a_A ≈ a_B — too sharp for smooth frozen features to grab.
  THE CURE IS ALREADY BUILT: v3's floor rule refuses this exact shape
  (exact world, floor unevidenced, residual astronomically above
  machine-zero → BEST-RESIDUAL, never UNDERSTOOD). Port urgency:
  every mint the recursion produces INHERITS THE JUDGE.
- Multi-target scanning needs a family-wise bar: scan all 10 possible
  targets instead of the planted one and you buy one confident false
  scream (survives 5/5 fresh nulls, p ≈ 0.003) — N lenses = N chances
  to hallucinate, the selection lesson at level 1.

---

## PART 3 — THE DROP ALGEBRA: how two contributors combine

### The idea (owner): "compare how parts get transformed compared to
### base to see how they abate"
Ablation through the pool measures NECESSITY: d_i = how much
prediction quality dies when channel i is scrambled. The finding
settled by the probe: the DEVIATION FROM ADDITIVITY of these death
tolls is itself a diagnostic. For a pair (i, j):

    S = d_i + d_j − d_ij      (solo drops minus the both-at-once drop)

```
ADDITIVE   (y = 1.2·x0 + 0.8·x2):  two separate pipes.
           Kill each alone: lose its share. Kill both: lose the sum.
           S ≈ 0                    measured S = −0.073, band ±0.075
SYNERGY    (y = 0.9·x1·x3):        two keys to one lock.
           Kill EITHER alone: everything dies. Kill both: same death.
           S >> 0                   measured S = +1.452, band ±0.206
REDUNDANCY (x4 = x1 + tiny noise): two copies of the same key.
           Kill one: twin covers. Kill both: everything dies.
           S << 0                   measured S = −0.435, band ±0.081
```
The instrument carries a measured self-correction: tanh features mix
channels slightly, so even a true additive pair shows a small S bias.
A planted KNOWN-additive pair inside the probe measures that bias
(0.0052 of pair scale) and the classification margin includes it —
measured, never hand-set.

### How it performs under attack
- Twins tightened from noise 0.05 to 0.005 (correlation 0.99999):
  REDUNDANCY holds 3/3 at 3.5–4.2× margin — and the feared
  solo-naming blindness NEVER OPENED (the solos symmetrize toward
  0.49/0.49 because tanh mixing means scrambling one twin still
  corrupts every hidden unit). The probe's own filed worry, falsified
  in its favor.
- Wrong pair hints invent nothing: non-contributing pairs land at
  0.10–0.52× band; naming is hint-independent.
- Mixed world (synergy pair + additive bystander in one law): all
  three junction calls correct, naming exact, 0 false names.

### Where it breaks / the vocabulary's edge
A pure THREE-WAY product saturates pairwise ablation: killing any one
channel kills everything, so solo ≈ pair ≈ triple drop (1.65 ± 0.09)
and the pairwise output is indistinguishable from genuine 2-way
synergy. Correct-but-incomplete, not misleading — and the cure is one
line: the order-3 statistic S3 = d_i + d_j + d_k − d_ijk reads +3.364
(≈ 2× capture) for a true triple vs ≈ 1× for a real 2-way. The kinds
ladder, honestly stated: pairs proven, order-3 named and cheap, not
yet claimed.

---

## PART 4 — E1, THE BRIDGE TO LANGUAGE

### The idea (owner): "how do we test this on non mathematical
### structures? like english?"
Pools never worked on math — they worked on CHANNELS. A
sentence-event's reading becomes a channel vector the moment each
feature value becomes an indicator column:

```
event: {voice: passive, tense: past, deco: comma, subj: cat, ...}
   →   [0 1 | 1 0 | 0 1 0 | 1 0 0 0 | ...]   (16 indicator columns)
```

### The world and why it matters
Planted rule: FAIL iff voice and tense DISAGREE with the planted
agreement pattern — fail = XOR(voice=passive, tense=perfect), plus 5%
verdict noise. At 50/50 base rates XOR has PERFECTLY FLAT marginals:
P(fail | passive) = P(fail | active) = 0.5. Isaac's own frontier miner
(mine_commonality) is a per-feature-value statistic, so it is blind
BY THEOREM — measured blind 3/3 seeds and 20/20 in the sweep. This is
not a toy: conjunctive frontiers in his real practice history are
invisible to the current self-assessment organ.

### The funnel, step by step, with the numbers
1. POOL SCREAMS on the indicator table: capture +0.64..0.70 vs null
   ≈ −0.2 → "a rule exists," grammar-free.
2. GROUPED ABLATION (scramble a whole feature's indicator block with
   one shared permutation) names exactly {voice, tense}: drops
   1.1–1.3 vs decoy threshold 0.03–0.05; subj/verb/junk ≤ 0.02.
3. DROP ALGEBRA on the named pair: S ≈ +1.2 vs band ±0.08 → SYNERGY,
   the conjunction's fingerprint (jointly necessary).
4. ARTICULATION collapses to a 2×2 fail-rate table on training rows:
   ```
                tense=past   tense=perfect
   voice=active     0.05          0.95
   voice=passive    0.95          0.05
   ```
   Cells > 0.5 read off the XOR rule EXACTLY; held-out verdict
   accuracy 0.91–0.96 = the 5% noise ceiling. The search that was
   2^k subsets became one pair and one small table — readable,
   evidenced, pools removed.

### How it performs under attack
- 20-seed sweep: 7/8 checks perfect; the one failure isolated to seed
  1004 where an innocent squeaked over a thin threshold at 1.4× (the
  real cause sat at ~90× threshold every seed).
- Skewed base rates (tense 70/30): funnel unaffected (names exact,
  SYNERGY at 9–13× band, articulation 0.947–0.973); the miner starts
  seeing again because marginal signal returns — correct behavior on
  both sides.
- Autonomy: remove the oracle pair-hint and choose the pair from the
  named set — identical results end to end. The scaffold is
  convenience, not load-bearing (production still owes the ~15-line
  chooser).

### Where it breaks
- CORRELATED INNOCENTS (the important one for language): make subj
  travel with voice (ρ ≈ 0.55, causally innocent) and the funnel
  over-names it 3/3 seeds at 1.6–2.7× threshold. Mechanism, measured:
  the decoy null calibrates the zero for an UNCORRELATED channel, but
  the readout genuinely leans a little on a correlated proxy, so
  scrambling it genuinely hurts. The true pair still towers 15–36×
  above the proxy, and the branch's claims were scoped to independent
  features (in-scope control clean) — but correlated features are the
  NORM in real language, so the null must grow correlated decoys (or
  a conditional permutation) BEFORE E3 runs on his real receipts.
- THE BASE-RATE TRAP (gate design): on a rare three-way rule (~8%
  fail rate), the articulation gate "accuracy ≥ 0.90" is passed by an
  empty rule that predicts "always passes" (0.911) while recovering
  NOTHING. The gate must demand BEATING THE ALL-PASS PREDICTOR, not
  an absolute score. (The honest 2×2×2 table closes the same world at
  0.96 vs base 0.88.)

---

## PART 5 — WHAT TONIGHT MEANS AT THE THEORY LEVEL
- The keystone held up as machinery: a structure (a law) was reified
  into a thing with descriptors, and the SAME unchanged instrument
  read relations among structures — representation → processing →
  new representation, one full turn of the crank, level 0 → level 1,
  with the mint proven and composed back down to raw prediction.
- The instruments (pools) point reliably across every world family
  tried — numeric, categorical, relational, mixed — and every
  deliverable stayed readable (an equation, a named pair, a small
  table). The scaffolding came down every time.
- The single deepest liability is now precisely located: THE JUDGE.
  The old r²-stamp fails identically at level 0 (additive hole) and
  level 1 (max relation), and recursion multiplies exposure — every
  minted level feeds the next. The v3 rewrite refuses every measured
  failure shape so far; porting it into the engine, with the
  rival-form duel, is the gate everything else queues behind.
- One statistical discipline showed up in four costumes: thresholds
  estimated from few samples + many simultaneous questions = rare
  confident false names. Family-wise bars, studentized/correlated
  decoys — one cure family, filed with numbers.

## THE QUEUE THIS SETS (owner's call on order)
1. §2b ENGINE PORT + gates (gauntlet, 100-world duel, confidence
   curve) + rival-form duel — the judge everything inherits.
2. Naming-null hardening (correlated decoys, family-wise scan bar,
   studentization) — cheap, unblocks E3.
3. E3 ON THE REAL ACCOUNT: the pool funnel over his actual practice
   receipts — first look at what marginal mining never saw.
4. S3 (order-3) + all-pass-beating articulation gate + the pair
   chooser — small, closes the measured holes.
