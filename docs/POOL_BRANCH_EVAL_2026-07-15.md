# POOL BRANCH — LOCAL EVALUATION & STRESS TEST (2026-07-15, night)

**What this is:** the local verification record for the cloud branch's
three pool probes (`pool_relations_probe.py`,
`pool_drop_algebra_probe.py`, `pool_conjunctive_frontier_probe.py`),
per the owner's instruction "new repo pushed for evaluation and stress
testing — run it." Method: (1) REPRODUCE the pushed claims verbatim;
(2) ATTACK with 13 adversarial probes (independent agents, predictions
filed before running, wrong predictions kept); (3) adversarially
VERIFY every BROKEN/surprising finding with an independent refuter.
All attack probes are deterministic (`eval_attack_*.py`, fixed seeds,
PYTHONHASHSEED=0) and committed alongside this record.

## PHASE 1 — REPRODUCTION: ALL GREEN
| probe | pushed claim | local run | numbers match sheet? |
|---|---|---|---|
| relations (recursive pools) | 10/10 | **10/10** (4.1s) | yes (capture +0.536 vs −0.431; mint held-out 1.78e-15; train 3.11e-15 vs sheet 2.2e-15 = platform epsilon) |
| drop algebra | 8/8 | **8/8** (13.4s) | yes (S +1.452 vs ±0.206; calib bias 0.0052) |
| E1 bridge | 8/8 ×3 seeds | **8/8 ×3** (13.8s) | yes (captures +0.635..0.696; acc 0.907–0.960) |

## PHASE 2 — THE VERDICT SHEET (13 attacks; verifier-checked where flagged)
Verdicts grade the PUSHED CLAIMS: WORKS / PARTIAL (measured boundary
worth filing) / BROKEN (confident wrong output) / HONEST-REFUSAL.

| # | attack | verdict | evidence (complete numbers in the eval_attack file docstrings/outputs) |
|---|---|---|---|
| 1 | R-A selection across targets | **PARTIAL** (verifier: upheld) | Scan all 10 targets instead of the planted one: no-relation world 1/10 hairline false scream; planted world 1 CONFIDENT false scream on independent distractor 4 (cap −0.381 clears all 20 measured thresholds; survives 5/5 fresh null seeds, pooled p=0.003). Single-target claims hold. CURE: family-wise (max-across-targets) null for scans. |
| 2 | R-B noise on the exact world | **WORKS** | σ∈{0.05,0.2}: L1 screams both; ablation names exactly {A,B} both (next-best 30–60× lower); mint keeps product form; held-out error scales linearly (3.96× measured vs 4× predicted); no false exactness. |
| 3 | R-C out-of-grammar relation | **BROKEN — charges the ENGINE's old verdict, not the pools** (verifier: upheld, refutation attempt failed) | a_C=max(a_A,a_B): pools generalize (scream; names {A,B} 6/6; ADD variant mints +1.00*x1+1.00*x0 at 2.66e-15). But engine discover() ships **confident UNDERSTOOD r²=0.99945 with a 9-term exp/sin/cos wrong form** (train maxerr 7.35e-2 on an EXACT world); survives fresh seed (r²=0.99907); and the pushed single-lens pool gate stays QUIET on that residual (−0.740 vs −0.547) — no shipped guard fires. This is §2b's case transplanted to level 1. NOTE: the verdict-rewrite v3 (main @ e49f1d5) refuses this exact shape by the floor discipline (exact road refuses; floor unevidenced ⇒ never UNDERSTOOD). |
| 4 | R-D second relation present | **PARTIAL** | F-target names exactly {D,E}; distractor target quiet; marginal blind. One marginal false name: F squeaks over for the C-target at 1.19× thr (0.063 vs 0.053). Same weak joint as #10/#13. |
| 5 | D-A tighter twins (probe's own filed attack) | **WORKS** (in the branch's favor) | Twin noise 0.02/0.01/0.005 (corr→0.99999): REDUNDANCY holds 3/3 at 3.5–4.2× margin; the feared solo-naming blindness NEVER OPENED (solos symmetrize ~0.49/0.49 — tanh mixing; original probe's worry falsified in its favor, kept). |
| 6 | D-B wrong pair hint | **WORKS** | Non-contributor pairs land deep in band ((0,2) at 0.10× band; (1,0) at 0.52× band); naming pair-independent, stays [1,3]; truth pair 7.6× band. Micro-guard filed: use max(pair_drop,0) in the margin formula (signed pair_drop can shrink margin below band). |
| 7 | D-C mixed junctions in one world | **WORKS** | y=0.9·x1·x3+1.0·x0: named exactly [0,1,3] every call; S(1,3)=+0.739 SYNERGY (margin ±0.108); S(0,1)=−0.026, S(0,3)=−0.039 ADDITIVE; 0 false names. |
| 8 | D-D three-way junction | **WORKS in scope; scope hole fingerprinted** | x0·x1·x2: names {0,1,2} exactly; all inside pairs SYNERGY (S≈1.3); distractor pair in-band. Drops SATURATE (solo≈pair≈triple≈1.65) so pairwise output cannot distinguish order-3 from order-2 — NOT actively misleading. CURE named: the order-3 statistic S3=d0+d1+d2−d012=+3.364 (≈2× capture for a pure triple) discriminates and is one line to compute. |
| 9 | E-A the probe's own 20-seed sweep | **PARTIAL (19/20 on one check)** | 7/8 checks 20/20. E8 innocents: 19/20 — forensic isolates seed 1004, W2 world: subj +0.0234 vs thr 0.0170 (1.4× squeak-over); real cause deco named at ~90× thr every seed. Thin decoy null (6 samples), nominal-rate borderline. |
| 10 | E-B skewed base rates | **WORKS** | tense 70/30: funnel names exactly {tense,voice} 2/2; SYNERGY 9–13× band; articulation 0.973/0.947; miner correctly sees the returning marginal signal. S shrinks 12–18% (sub-verdict). |
| 11 | E-C 3-way conjunction + base-rate trap | **PARTIAL — real gate-design finding** | Instruments cross (scream 2/2; names {voice,tense,deco} exactly; pairs SYNERGY). THE TRAP, measured: E5-style acc≥0.90 gate is base-rate-blind — an empty all-pass rule scores 0.9111 vs true labels and PASSES while recovering NOTHING (2×2 cells max 0.38 < 0.5). Honest 2×2×2 closes at 0.96 vs base 0.88. CURE: the gate must beat the all-pass predictor, not an absolute 0.90. |
| 12 | E-D correlated innocent | **PARTIAL** (verifier downgraded from BROKEN: out of pushed scope; in-scope control clean) | subj as voice-proxy (ρ≈0.55): funnel over-names subj 3/3 seeds at 1.6–2.7× thr (decoys are uncorrelated ⇒ wrong zero for correlated channels). True pair still tops proxy 15–36×; SYNERGY intact; miner blind-not-fooled. LIVE HAZARD for the language crossing (correlated features are the norm). CURE: correlated/matched decoys or conditional-permutation null. |
| 13 | E-E pair-hint scaffold autonomy | **WORKS** | Naming verified hint-independent (code + measurement); autonomous chooser (exactly-2-named) reproduces hinted results identically 2/2 (acc 0.907/0.947). DESIGN DEBT: production has no chooser; 0/1/3+-named branches unmeasured. |

## THE THREE SYSTEMIC FINDINGS (what the 13 attacks converge on)
1. **The deepest break is the engine's old judge, not the branch.**
   #3 (BROKEN, upheld): r²-graded UNDERSTOOD ships a confident wrong
   9-term equation on an exact out-of-grammar world, and the
   single-lens pool gate is quiet on that residual class. The
   verdict-rewrite v3 on main refuses this shape by floor discipline —
   the §2b port is now urgent for LEVEL-1 genesis too: **every mint
   the recursion produces inherits the judge.**
2. **Naming thresholds are the weak joint** — four independent
   manifestations (#1 multi-target scan, #4 second-relation squeak,
   #9 seed-1004 squeak, #12 correlated proxy): thin decoy nulls +
   multiple hypotheses ⇒ borderline false names at ~nominal rates.
   One cure family: family-wise nulls for scans, more/studentized
   decoy samples, correlated decoys for correlated features (the
   miner's rare-category cure shape, applied to the pool).
3. **Gates and vocabularies need their scope stamped:** the
   articulation accuracy gate must beat the all-pass predictor (#11);
   pairwise S needs the one-line S3 extension before anyone reads
   "SYNERGY" as "pairwise" (#8); production needs the pair chooser
   (#13).

## BOTTOM LINE
The three pushed claims REPRODUCE and SURVIVE adversarial stress
within their stated scopes; several attacks strengthened them (twins
to ρ=0.99999, noise, skew, autonomy, mixed worlds). No pushed pool
claim was broken. The one BROKEN charges the engine's pre-rewrite
verdict tier and is already cured in principle by main's verdict
rewrite v3 — measured, not assumed, before the recursion ships mints
through the old judge. Recommend: MERGE the branch (owner's call),
then land the three cure families above as the segment's next probes.

Evidence: 15 `eval_attack_*.py` files (deterministic, re-runnable) +
journal of the 13-agent run + verifier reports. Worktree: dc-eval;
branch @ 6e3560a atop main e49f1d5.
