# TESTING SEGMENT — 2026-07-15 (the attack campaign)
**The declared next step ("broader TESTING"), executed: the handoff's
attack list + FRESH EYES review targets turned into 12 adversarial
probes, every finding independently re-run and attacked by a reviewer,
plus a full reproduction sweep of the arc. Authored by a Claude Code
remote session on branch `claude/isaac-dc-local-setup-2e51bt`; probes
in `experiments/*_probe.py` (additive_boundary, gate6_unlabeled,
gate6_corruption, gate6_preconditions, election_mixedcause,
floor_heavytail, audit_adversarial, order_negatives,
commonality_adversarial, tools_ranking, dataops_b1, dataops_b2).
House discipline throughout: claim under test + prediction filed in
the docstring BEFORE first run; scrambled/null controls on every
positive claim; wrongs REPORTED, never patched; PYTHONHASHSEED=0,
pinned BLAS, fixed seeds; fresh throwaway minds only; no promoted
file edited; the account untouched (not present in this environment).
sklearn appears ONLY as labeled BASELINE. Rollback pointer:
`rollback-2026-07-13` (= 3855ee6, the last pre-experimental
promotion) pushed as a branch on origin.**

## ═══ ENVIRONMENT (what this repo runs on, measured) ═══
Cloud container, 4 cores, Python 3.11 default. Full promotion gauntlet
GREEN here (engine 11/11 + every battery) after two environment
findings, both cured in `.claude/hooks/session-start.sh`:
- **multicore battery** fails on low-core boxes from BLAS thread
  oversubscription (4 workers x 4 BLAS threads on 4 cores: 0.31x
  "speedup"); pinning OPENBLAS/OMP/MKL to 1 thread/process restores
  3.39x (7/7) and does NOT slow the 1-worker baseline (16.8s both
  ways) — Isaac's small-matrix work never used BLAS threads anyway.
- **test_intake.py** needs Python >= 3.12 (PEP 701 f-string at :62;
  the ONLY 3.12-only file in the repo, whole-tree sweep). 19/19 green
  under python3.13.

## ═══ REPRODUCTION SWEEP (the records' claims, re-measured) ═══
14/14 runnable arc probes GREEN under PYTHONHASHSEED=0, matching the
records' quoted numbers where quoted exactly:
- 07-14 reference probes 4/4 (surprise_verdict W8 floor **0.1243 to
  the digit**; failure_commonality excesses .496/.431 vs null .159,
  lockout .897 vs .140; taught_reduction poisoned pair refused;
  should_vs_can context-shoulds).
- Language ladder rungs 1-7 all green (scrambled controls bind 0
  everywhere; salad refusals 24/24; ambiguity 10/10; false gaps 0;
  idea_corrections ran WITHOUT a tutor — 8/8 word, 40/40 decoration).
- word_reelection_battery 12/12; frontier_battery 4/4.
- **BLOCKED here: gate6_operator_mining.py** — imports
  `isaac_interpreter` from the owner-machine staging
  (`Downloads/New folder (3)/interpreter_handoff/`), not in the repo.
  ALL interpreter-EXECUTION attack items (refraction/loops, rule
  interference, chunk correctness, Gate 1/2 transplant, WM/net
  vocabulary mapping) are untestable anywhere but the owner's machine
  until that package enters the repo. The mining-side attacks below
  need no interpreter and ran.

## ═══ THE VERDICT SHEET (new, strict; every row verifier-attacked) ═══
Verdicts: WORKS / PARTIAL / BROKEN / HONEST-REFUSAL, VERDICT_SHEET
style. "CONFIRMED" = an independent reviewer re-ran the probe
(byte-reproduced) and failed to refute the finding; two interpretive
claims were REFUTED by review and are recorded corrected.

### A. Gate 6 operator mining (the attack list)
| Attack | Verdict | Evidence |
|---|---|---|
| Unlabeled actions (labels stripped) | **WORKS**, qualified | Effect-shape clustering + partition mines both rules from 126 unlabeled episodes; scrambled mints 0 (top 10/126 vs threshold 25); mixed-cluster control NO RULE. Qualifier (reviewer): partition purity 1.000 is forced by world construction (the op symbol is the only categorical slot) — real observable, but do not read as statistical clustering power. |
| Constant distractor in preconditions | **BROKEN** (as 07-14 predicted) | ('bg','7') constant across all 48 episodes: presence 1.00, mult [1] → minted as REQUIRED LITERAL. Minted rule fires 6/6 with bg=7, **0/6 with the cell absent, 0/6 with bg=8** — transfer fully blocked by an irrelevant literal. Jitter distractors correctly die (presence 0.67, mult [0,1,2]). "Always co-present" is not "excess past a null"; the gate is structurally blind to constants. Cure shape: cross-world contrast or held-out ablation — the discipline lacks both. |
| Near-miss corruption vs the trim | **PARTIAL** — breaking rate measured | FAR garbage: wrong-kept 6/6, 12/12, 18/18 with whys through rate 0.3; breaks at **0.4** (keep-everything fixed point, cut 1.0→231). NEAR-MISS (±1): breaks at **0.1** — the ±1 residual sits on the trim's floor cut max(3·median, 1.0); 69/90 corrupted rows learned across the sweep. |
| Bare discover() vs near-miss | **BROKEN** (verdict tier) | SILENT POISONING at every rate 0.1-0.5: verdict UNDERSTOOD (r2 0.998+) while unseen predictions err 0.04-0.45. Reviewer re-derived independently (12/12 points, fresh seeds). Fair qualifiers: the organ honestly meets its documented r2>=0.9 contract (isaac.py:2745) — what is MISSING is the exactness tier (verdict rewrite §2b's case, now with numbers); round-correct consumers mostly survive (±1 self-cancels into intercept |b|<=0.23); a biased all-+1 attack (untested) would likely shift laws a full unit. |
| Bare discover() vs far garbage | HONEST-REFUSAL | NO RULE at all rates (held-out r2 -0.70..-1.98) — never a confident wrong law, but availability lost. **THE INVERSION (best finding of the batch): the trim kills far garbage and passes near-misses; bare discover refuses far garbage and absorbs near-misses. Neither layer covers the other's blind side.** |
| Multi-cell effects | **WORKS** / PARTIAL binding | Two-cell template recovered exactly (prod+sum, one slot each); both bindings unseen 3/3 — but only prod engine-exact; sum reproduces the additive hole (maxerr 1.63) and needs the labeled linear-duel BASELINE (maxerr 0). |
| Conditional effects (piecewise) | **WORKS** | flag re-mined as LITERAL in both partition preconditions; per-branch unseen 3/3; cross-fire 6/6-on-own-flag / 0/6-on-other. Flag-shuffled null mints 0 end-to-end — flag in preconditions is EARNED. Two traps found on the way: the collapse guard never fires (trim kept a 25%-contaminated consensus 32/48 >= min_keep 29 — the exactness gate is the actual defense), and raw discover() rates the blended conditional data UNDERSTOOD r2=0.9857 at train maxerr 4.234 — a confidently wrong blended law only an exactness demand catches. |
| 50% corruption collapse guard | HONEST-REFUSAL, wrong mechanism | Nothing mints at 0.5 (honest), but min_keep fired 0/24 times — the trim fails by KEEPING everything, never by collapsing; the coded guard defends a failure mode that does not occur. |

### B. The additive hole (engine discover()), regime-mapped
The 07-13 §6 claim splits three ways under measurement (reviewer
proved `_Enumerator`/`discover`/`apply_law` byte-identical to the
07-12 engine — no "engine changed" explanations):
| Claim | Verdict | Evidence |
|---|---|---|
| Product atom admitted first on positive sums | **CONFIRMED** | Engine's own round-1 on y=x0+x1 positive U(0.1,2.1): x0*x1 first. |
| "r2 ~0.96, any N" | **N-DEPENDENT** (both handoff and first probe overgeneralized) | Same engine: n=60 integer sums r2 **0.9600** (seed 99; 0.9956, 0.9982 others); N=2000 r2 0.99989. The hole is deep at gate6 scale (n~60) and nearly invisible at N=2000 — and operator mining lives at n~60. |
| Wrong-form articulation | **BROKEN** at every positive regime | 8-term product+trig surrogate claimed UNDERSTOOD (N=60/200/2000, integer and continuous). Also: y=x0-x1 → sin impostor r2 0.99996; y=x0/x1 → 10-term trig surrogate 0.99942; y=x0+x1+noise → wrong 5-term form at the noise ceiling. Exact wins: 2x0+3x1, x0*x1, x0*x1+0.5x0², x0², sin(x0). Shuffled-y control: NO RULE (calibration clean). |
| "Greedy never reconsiders" | **REFUTED as absolute** | y=2x0+3x1 admits x0*x1 first, final equation exactly +3.00*x1 +2.00*x0 — the weight-prune eviction (isaac.py:919-923) existed at handoff time. |
| Mixed-sign closes the hole | **CONFIRMED, mechanism corrected** | Closure reproduces on U(-2,2), asymmetric, and all-negative data — but product is STILL admitted first there (corr does not collapse); late eviction recovers. **The true failure boundary is all-positive data**, not sign-mixing. |
| Probe-side linear duel (the fix direction) | **WORKS as specified** | Accuracy TIES (gaps 1e-5..1e-4) satisfy the handoff's ">= accuracy" rule; duel + parsimony wins 3/3 additive failures, coefs (1,1,0) exact on unseen. First probe's "contradicts the handoff" framing was REFUTED — the cure fires as designed. |

### C. Word elections under mixed causes (FRESH EYES #1 — confirmed, sharpened)
| Finding | Verdict | Evidence |
|---|---|---|
| Good seat, judge-only channel | **BROKEN** | Word-substance-GOOD seat wrongly dethroned **30/60 trials at only 30% unrelated-fail share** (60/60 at 50%+; median first-dethrone use 13.5/5/4). Mechanism (code-verified): the "<50% rate" guard is a FIRST-PASSAGE check on the last-8 window re-run on EVERY banked use — it fires far below its stated boundary. Judge-only history = the remine transcript shape: **a verdict-only replay can retro-dethrone good words on the real account.** Deterministic knife edge at exactly 50%: pass-first alternation holds 40 uses; fail-first dethrones at use 5. |
| Tutor INSISTENCE | **WORKS** (stronger than the review feared) | 0/60 wrong dethrones at every share incl. 100%: the teaching correction banks a seat vote and every fail-correction banks another, so >=3 window fails mathematically implies vs>=INSIST. **Protection hinges entirely on correction TEXT flowing; judge-only channels get none.** |
| Bad word riding unrelated passes | **BROKEN** (boundary measured) | Poison survives 40 judged uses in **33/60 trials at 70% unrelated-pass share** (1/60 at 50%, 0/60 at 30%) — shielding becomes majority behavior between q=0.5 and q=0.7. Not permanent (first-passage catches it eventually). Told-road rescue: 3 identity votes dethrone 33/33 despite the shiny ledger — by design. |
| Re-adoption record hygiene | **PARTIAL** (bug shape) | `del words[concept]` never clears the outcome record: re-teach #1 re-seats then INSTANTLY re-dethrones inside the same learn_from_correction call (stale window [1,0,1,0,0], vs=2<3); sticks only at re-teach #2. |

### D. Floors and the audit under hostile noise (FRESH EYES #4, #2)
| Finding | Verdict | Evidence |
|---|---|---|
| Repeats road, Gaussian (control) | WORKS | 20 seeds: median floor/core 0.981 (0.886-1.067). |
| Repeats road, Student-t df=2 | **BROKEN** | Floor inflated 2.64x median (seed spread 1.79-10.11x) — pooled RMS (isaac.py:2117) has no defense against infinite second moment. |
| Repeats road, 95/5 contaminated Gaussian | **BROKEN** | Floor inflated **20.8x**; the 5% gross errors are swallowed INTO the ruler. |
| End-to-end r2-ghost (the plan's named risk, live) | **BROKEN** | Contaminated world: 4x flag line lands at **78 core-sigma**; planted 10-sigma wrongs recall **0/6** (Gaussian control 6/6) while the MAD<=3x-floor gate stays ACTIVE (inflated floor inflates the gate) — the audit reports confidently while laundering. Also structural: the ruler is estimated FROM the audited data (clean control floor already 1.57x core from its own 6 wrongs). |
| Near-dup 3x line under inflated floor | **BROKEN** | Honest pair 6 core-sigma apart flagged DUPLICATE under contamination (3x line 5.84 vs separation 0.6) — false merge candidates minted by the inflated ruler. |
| Telling the truth does NOT cure it (B1 probe) | **BROKEN** | With resolutions={'y': true 0.05} alongside contaminated repeats: read_floor's largest-evidenced-road-wins keeps the INFLATED sigma and marks the truthful told record **CONTRADICTED** (5/5 trials); recall unchanged. The owner cannot cure self-blinding by stating the truth — road-priority design finding, not a threshold tweak. |
| Grain road on continuous data (F4) | WORKS | 10/10 seeds refuse (floor None, ASK posture). |
| Grain consensus vs typos (F14 extension) | **PARTIAL** | Holds k<=3 typos; BLINDED at k=4 of 150 rows (2.7%; consensus 0.944 vs >0.95 gate at isaac.py:2084) — and blinded grain refuses the floor, so the typos become unflaggable: honest in form, self-defeating in consequence. |
| ONE half-grain typo rescales the ruler | **BROKEN** | A single value at grid+0.005 makes candidate 0.005 consensus 1.000 (every on-grid gap is a 0.005-multiple), beating true 0.01: floor UNDERSTATED 2.0x while looking maximally consistent. F14 survives literally; one typo can quietly RESCALE the ruler. |
| Affine recal vs 20% coordinated drag | **WORKS** | Recall 60/60, 0 false flags, judge line 2.0013x vs truth 2.0 (the uncured 07-15 analog was 1.89x+1.5). |
| Affine recal vs quadratic drag (25-30%) | **WORKS** (predicted defeat REFUTED, twice, in the audit's favor) | Both probes independently: recall 74/74 and 75/75, 0 false flags; the trim isolates the clean rows / the recal unbends the dragged affine fit (0.8547*2.34=2.0007). Caveat: survives because the fitted FORM stayed affine — V4 shows the failure when the form bends. |
| Heavy-tail false flags on a lawful table | **PARTIAL** | A2's ~0 promise is GAUSSIAN-SCOPED: t(3) noise yields false ERROR flags at exactly the t3 tail mass past 4x floor (21/960 told road; Gaussian twins 0/320). |
| Near-dup 3x..4x DEAD ZONE | **PARTIAL** | Pairs at 3.5x floor are invisible to BOTH classes (>3x dup line, <=4x error cut): neither DUPLICATE nor ERROR nor accounted — a silent-pass class against the charter. Boundary at exactly 3.0x decided by a csv %.10g float coin flip. |
| NEWS-vs-ERROR when discover absorbs the plant | HONEST-REFUSAL + **BROKEN diagnostics** | The planted regime got absorbed into x0*cos(x1) cross-atoms; clean-row residuals inflate; the audit self-disqualifies (0 false flags — closed failure). BUT: the refusal's why blames the floor that was truthfully told (its ASK is already answered — unblockable); the law card ships UNDERSTOOD on the plant-bent equation; the records ledger carries NO refusal line (a ledger consumer sees a clean pass). The A3 promise "a law cannot absorb the band" is world-fragile; iid twin refuses identically, so correlated noise is not the ingredient. The verdict-rewrite (§2b) case, now with numbers. |
| Door laws past A8 | WORKS | Garbage bytes and a 201MB over-budget file: clean whys, companion tables still audited, no crash. |

### E. Order organs (FRESH EYES #10, #11)
| Finding | Verdict | Evidence |
|---|---|---|
| order_matters beyond positives | **PARTIAL** (mis-badge, as feared) | posmul (a*b if both>0 else a-b) badged matters=FALSE though it fails to commute on **75.6%** of [-6,6]²; zerotrap likewise (witness at (0,1)). Probing never leaves U(0.5,6)². Badge wording is honest ("no counterexample in 50 probes; absence, not proof") — the domain is the hole, not the claim. |
| Custom-op entry road | **PARTIAL** | No per-mind road: order_matters reads the CLASS dict Isaac._ORDER_OPS (isaac.py:2491), takes no callable; instance copies are invisible; class rebinding is shared across all minds in a process. |
| Stock badges | WORKS | 7/7: -,/,^ carry witnesses that recompute <1e-12; +,*,max,min claim absence-only; unknown op refused with the known list. |
| compare_order duplicates | **WORKS** (record's doubts discharged) | 260 duplicate-heavy shuffles: 0 mapping-verification failures, 0 non-bijective, 0 untrue kind claims; naming priority 5/5 as recorded (swap > reversed > rotated); fixed-points-first minimality: exhaustive search over 167 pairs, **0 counterexamples** — the open doubt closes (mapping[i]=i possible only where b[i]=a[i]). |

### F. The commonality miner under selection pressure
| Finding | Verdict | Evidence |
|---|---|---|
| Perfect correlated twin | **WORKS** | Honest tie: both named at byte-equal excess 0.782; false exculpation did not occur; scramble and no-signal controls clean. |
| 90% near-correlate | **PARTIAL** | Innocent proxy named (0.650) alongside the cause (0.821, ranked first) — within the organ's associational v1 contract, but a caller reading signatures as causes is handed an innocent correlate. |
| Selection correction (82 candidates) | **WORKS** | 0/20 no-signal worlds falsely named (nominal alpha ~2/301); tiny-n 1/20 (within tolerance). |
| Rare-category cause | **BROKEN** (miss mode) | Hard worlds (12-in-600 category): **1/20 hit** — the category's excess is capped near n_cat/n_fail while the single unstudentized max-over-candidates null is set by high-variance common candidates. Easy worlds 17/20. Cure shape: per-candidate studentization. |
| Near-background cause (~95% of successes too) | **PARTIAL** | Scale-gated: not named at n=600 (honest nothing), named at n=6000; equal-rate control never named (frequency-immunity intact). |
| Tiny samples (n~10 fails) | WORKS | Deterministic cause 20/20; weak cause honest-nothing 18/20; false naming 1/20 at nominal alpha. |
| 07-14 reference through the ENGINE organ | **WORKS** | W1 {tense=perfect .496, voice=passive .431}, W2 word=slang alone .897 — the reference numbers, through mine_commonality itself. |

### G. The tools node (FRESH EYES #7, #3)
| Finding | Verdict | Evidence |
|---|---|---|
| catalog_tools auto-harvest | WORKS | 109 organs on a fresh mind (record said 107 — superset, organs added since); self-retrieval 110/110. |
| query_tools ranking | **PARTIAL** | Top-1 **8/17 (47%)**, real signal vs shuffled null (mean 0.013) but 9 misleads; 8/17 winners decided by ALPHABETICAL TIE-BREAK; 'solve for the unknown' ranks solve at 57 (no stemming: unknowns != unknown); pure stop-word queries return confident matches (uncalibrated). Catalog descriptions: 'Convenience wrapper.', 'THE CHALLENGE LOOP.' say nothing an agent can rank on. |
| run_tool sandbox (fresh-eyes #3) | **BROKEN** (hazard demonstrated once, no sweep) | run_tool('teach',...) on a throwaway mind mutated it live: doors 109→111, node minted, fact_index written, use counter banked. No fork, no undo — "run them all" on the account would scribble on his life exactly as the review warned. fork-first remains unbuilt. |

### H. Data-ops bench B1/B2, planted worlds (predictions filed first)
| Finding | Verdict | Evidence |
|---|---|---|
| B1 "competitive" prediction | **PARTIAL** (holds 4/6 cells) | Told-floor: F1 **1.000 at 10x (= ORACLE)**, 0.678 at 4x (= oracle ceiling exactly), 0 false flags on clean (baselines: IForest 0.42-0.67 false-flag rate). Fails at 2x in both designs — but 2x-under-1-sigma is below the detectability wall (oracle recall 0.025); IForest's "edge" there is bought with 42-67% false flags. |
| B1 repeats-floor self-blinding | **BROKEN** | Corruption inflates the measured floor (0.0555→0.1238 across magnitudes vs true 0.05); recall drops 1.000→0.562 at 10x; **and telling the true resolution does not rescue** (road-priority keeps the inflated ruler, truthful told marked CONTRADICTED 5/5). |
| B1 refusal + classification integrity | WORKS | Scrambled null: NO RULE, 0 flags (IForest invented 753); 0 NEWS misfires in 30 corrupted worlds; precision 1.000 everywhere; every flag carries law+residual+floor+value+predicted receipts. |
| B2 completion (join exists) | **WORKS** | MAE 0.0395 units = **1.0x the noise floor** (oracle 0.0380); joint law +2.99*x1 +0.50*x0 vs planted 3x+0.5a1; baselines 20-67x floor. |
| B2 refusal honesty (no join path) | **WORKS** | Isaac fabrications **0/540**; baselines fabricated **100%** everywhere (each an error by the spec) — the lane's product asymmetry, measured. |
| B2 claim-level gate | **PARTIAL** (wrong-kept) | 1/10 seeds: false RESOLVED-HALVES on unrelated tables (corr 0.156 vs null95 0.118) — the two-half confirmation gate mostly works but is not tight; reported, kept. |

## ═══ WHAT THIS MEANS (the shortest honest summary) ═══
- **The epistemics hold up remarkably under attack**: 14/14 arc probes
  reproduce; the miner's selection correction is calibrated; the audit
  survives drag attacks it was never promised to survive; refusals are
  everywhere honest-in-form; B1/B2 mostly land the plan's predictions,
  and the zero-false-flag / zero-fabrication asymmetry vs baselines is
  real and large.
- **The breaks cluster in exactly two shapes**: (1) RULERS ESTIMATED
  FROM THE DATA THEY JUDGE (repeats floor, grain consensus, election
  windows) — every one can be captured by the contamination it exists
  to catch, and road-priority can lock the capture in against a
  truthful told; (2) THE VERDICT TIER (r2-graded UNDERSTOOD) — near
  -miss poisoning, blended conditional laws, plant-absorbing forms,
  additive impostors all ship confident wrong equations that only an
  exactness demand catches. Both shapes are already named in the
  repo's own plans (§2b verdict rewrite; robust-floor roads) — the
  campaign turns them from named risks into measured, reproducible
  exhibits with breaking points.
- **Two review predictions came true verbatim** (constant-distractor
  preconditions; election credit assignment), one was falsified in
  the design's favor (INSISTENCE is mathematically airtight when
  correction text flows), and one open doubt discharged
  (compare_order minimality, exhaustively).

## ═══ COVERAGE vs THE HANDOFF'S OWN LISTS ═══
Attack list (07-15 open item 1 / 07-13 next item 1):
- COVERED: unlabeled actions; near-miss corruption; constant
  distractors; multi-cell/conditional effects; noise rates vs the
  consensus trim; data-ops B1/B2 (planted worlds per the plan's
  order; public sets still ahead).
- BLOCKED HERE (interpreter package absent): refraction/loops; rule
  interference; chunk correctness; chunking admission gating; WM/net
  vocabulary mapping; Gate 1/2 transplant.
- NOT DONE (build work, not probe work): the verdict rewrite §2b
  itself — but this campaign measured its CASE from four independent
  directions (near-miss silent poisoning, blended-conditional
  UNDERSTOOD, plant-absorbed forms with clean ledgers, additive
  impostors). §2b is now the single highest-leverage build item by
  measured evidence.
FRESH EYES 1-12: probed and settled: 1 (confirmed, sharpened),
3 (hazard demonstrated), 4 (confirmed with breaking points),
7 (confirmed, mechanism measured), 10 (doubt discharged),
11 (confirmed mis-badge). Partially: 2 (knife-edge and window
mechanics measured; the constants themselves remain owner calls).
Not probeable here / owner-side: 5 (_order_kind duplicate copies),
6 (remine equivalence), 8 (doc sync process), 9 (session-95 sample
caution), 12 (theory sharpening confirmation).
Weak-evidence flags (say so, per the standard): tools_ranking rates
are phrasing-dependent (17 hand-built queries; mechanism findings
phrasing-independent); corruption breaking rates are single-seed per
sweep point (reviewer's spot-seeds moved r2 by ±0.04); B2's false
RESOLVED-HALVES is 1 world in 10 (a ~5%-level gate behaving like
one).

## ═══ NAMED NEXT PROBES (from the reviewers; deduplicated) ═══
1. BIASED near-miss corruption (all +1, no self-cancel) — the sharper
   poisoning attack; expected to shift minted laws a full unit.
2. Seed-averaged breaking rates (single seed per sweep point today).
3. A second categorical distractor slot in the unlabeled world
   (partition-key selection is currently forced); a world where NO
   before-cell distinguishes operators.
4. Near-miss corruption THROUGH the unlabeled pipeline (dimensions
   only tested separately).
5. Per-candidate studentized null for mine_commonality (the
   rare-category miss's cure shape).
6. Robust floor roads (MAD/trimmed pooled spread) + road-priority
   that lets a truthful told beat a contaminated repeats road.
7. Election record hygiene: clear/age the outcome window on dethrone;
   a rate test that is actually a rate, not first-passage.
8. Interpreter-execution attack list — BLOCKED here until
   `interpreter_handoff/` enters the repo.

## Hygiene
This record + 12 probe files committed on
`claude/isaac-dc-local-setup-2e51bt`; nothing promoted was edited;
promote.py/PROVENANCE untouched. Full agent-verified evidence trail
(re-runs, independent recomputations, refutation notes) lives in the
session transcripts; every probe is deterministic and re-runnable:
`cd experiments && PYTHONHASHSEED=0 python3 <probe>.py`.
