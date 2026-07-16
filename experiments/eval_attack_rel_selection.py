"""eval_attack_rel_selection.py — ATTACK R-A: SELECTION ACROSS TARGETS
(segment attack list: N lenses = N chances to hallucinate).

TARGET CLAIM (pool_relations_probe.py, pushed): the level-1 pool —
scream_verdict on a descriptor table (rows=episodes, channels=other
groups' fitted constants) — SCREAMS where a relation is planted and
stays QUIET on relational distractors / scrambled alignment. But the
pushed ladder only ever points the pool at group C, where the truth
was planted. Real use scans MANY targets: 10 groups = 10 lenses = 10
chances for the 99th-pct null to hallucinate a relation.

THE ATTACK
  World 1 (NO-RELATION): identical episode machinery (same seed 42,
    same constant draws, same raw rows) but the plant line
    a_C = a_A * a_B is REMOVED — all 10 groups independent. Scan all
    10 groups as target (channels = the other 9 groups' constants,
    240 episodes). Any scream is a FALSE ALARM.
  World 2 (PLANTED, verbatim probe world): a_C = a_A * a_B, seed 42,
    240 episodes. Scan all 10 targets. The relation is SYMMETRIC —
    a_A = a_C / a_B and a_B = a_C / a_A are deterministic given the
    channels, so screams on targets A, B, C are all LEGITIMATE and are
    NOT graded as false alarms. The 7 distractor targets (3..9) must
    stay quiet.

PREDICTIONS (filed BEFORE the first run; kept verbatim whatever runs
measure — falsified ones stay as wrong-kept):
  E1 NO-RELATION world: 0/10 false screams (design tolerates <=1;
     family-wise chance of >=1 at a per-target 99th-pct null across
     10 targets ~ 1 - 0.99^10 ~ 9.6%).
  E2 PLANTED world, target C: screams, capture > +0.5 (the pushed Q2
     leg, reproduced under the scan).
  E3 PLANTED world, targets A and B: ALSO scream — the inverse
     surface a_A = a_C / a_B is smooth and bounded on this world
     (|a| >= 0.3 after the small-constant adjust, so |ratio| <= ~13)
     and n=240 rows feed a 400-feature frozen tanh pool; predicted
     captures positive but LOWER than C's (ratio is harder than
     product for tanh features).
  E4 PLANTED world, 7 distractor targets: 0/7 screams.
  E5 full scan (20 scream_verdicts, n=240, m_eff=400) < 60 s.

RUN-1 MEASURED (2026-07-15, one run, deterministic seeds — falsified
predictions KEPT above as filed):
  E1 held at the tolerance edge: 1/10 false screams in the no-relation
     world — target 6, capture -0.488 vs null99 -0.504, a HAIRLINE
     margin (+0.016, 3% of the null spread 0.499); the other 9 quiet
     at captures -0.657..-1.258.
  E2 held: C capture +0.536 vs -0.490.
  E3 held: A +0.319 vs -0.518, B +0.062 vs -0.511 — both symmetric
     legs scream, as predicted (legitimate, not false alarms).
  E4 FALSIFIED, WRONG-KEPT: 1/7 distractors screamed — target 4,
     capture -0.381 vs null99 -0.503, margin +0.122 (25% of the null
     spread), and -0.381 sits ABOVE ALL 20 thresholds measured across
     both worlds (range -0.575..-0.467) — not threshold noise. a_4 is
     independent of every channel by construction: a confident false
     scream. Combined quiet-truth false-scream rate under the scan:
     2/17 (~12% per lens) vs the ~1-1.5% nominal per-target rate.
     SELECTION ACROSS TARGETS IS REAL: N lenses need a family-wise
     (max-across-targets) null, which the pushed ladder never tested
     because it only ever pointed the pool at C.
  E5 FALSIFIED, WRONG-KEPT: 418.3 s total (~21 s/verdict on this box;
     Windows Store python BLAS is slow — the pushed probes' cost
     claims were measured elsewhere). DISCLOSED: exceeds the ~180 s
     budget; run once at full spec, no shrink applied.

CONVENTIONS MATCHED TO THE PUSHED PROBE: constants extracted from raw
rows via fit_constant (the probe's disclosed scaffolding), episodes
seed 42, M_POOL=800, PYTHONHASHSEED=0, instrument imported unchanged.
No existing file is modified; the no-relation episode maker replicates
make_episode verbatim minus the single plant line, so the RNG stream
(and therefore every non-C constant) is IDENTICAL to the probe's.

Run: cd experiments && PYTHONHASHSEED=0 python eval_attack_rel_selection.py
"""
import time

import numpy as np

import pool_relations_probe as prp
from reservoir_surprise_probe import scream_verdict

N_EP = 240  # matches the pushed probe's training-table size


def make_episode_norel(rng):
    """Verbatim prp.make_episode MINUS the plant line — identical RNG
    stream, so every constant except C's overwrite matches the probe's
    draws."""
    a = rng.uniform(-2.0, 2.0, prp.N_GROUPS)
    a[np.abs(a) < 0.3] += 0.6 * np.sign(a[np.abs(a) < 0.3]) + 0.3
    # (plant line a[C] = a[A]*a[B] removed — that is the whole edit)
    rows = []
    for g in range(prp.N_GROUPS):
        x = rng.uniform(-2.0, 2.0, prp.ROWS_PER_GROUP)
        rows.append((x, a[g] * x))
    return a, rows


def descriptor_table(episodes):
    return np.array([[prp.fit_constant(*ep[1][g])
                      for g in range(prp.N_GROUPS)] for ep in episodes])


def scan_all_targets(A_mat, rng_base, label):
    """Point the level-1 pool at EVERY group in turn."""
    print(f"\n{label}: scan all {prp.N_GROUPS} targets "
          f"({A_mat.shape[0]} episodes, m={prp.M_POOL})")
    results = []
    for t in range(prp.N_GROUPS):
        X = np.delete(A_mat, t, axis=1)
        y = A_mat[:, t]
        cap, thr, scream, spread = scream_verdict(
            X, y, prp.M_POOL, rng_seed=rng_base + t)
        results.append((t, cap, thr, bool(scream), spread))
        tag = "SCREAM" if scream else "quiet "
        print(f"  target {t} ({'ABC'[t] if t < 3 else 'distractor'}): "
              f"capture {cap:+.3f} vs null99 {thr:+.3f} "
              f"(spread {spread:.3f}) -> {tag}")
    return results


def main():
    t0 = time.perf_counter()
    checks = []

    def check(name, ok, detail):
        checks.append(ok)
        print(f"  [{'ok ' if ok else 'FAIL'}] {name} — {detail}")

    # World 1 — NO-RELATION (same seed, plant removed)
    rng1 = np.random.default_rng(42)
    eps_norel = [make_episode_norel(rng1) for _ in range(N_EP)]
    A_norel = descriptor_table(eps_norel)

    res1 = scan_all_targets(A_norel, 1000, "WORLD 1 (NO RELATION)")
    false_screams = [t for t, _, _, s, _ in res1 if s]
    check("E1 no-relation world false screams <=1/10",
          len(false_screams) <= 1,
          f"{len(false_screams)}/10 screamed (targets {false_screams})")

    # World 2 — PLANTED, verbatim probe world
    rng2 = np.random.default_rng(42)
    eps_plant = [prp.make_episode(rng2) for _ in range(N_EP)]
    A_plant = descriptor_table(eps_plant)

    res2 = scan_all_targets(A_plant, 2000, "WORLD 2 (PLANTED a_C=a_A*a_B)")
    by_t = {t: (cap, thr, s) for t, cap, thr, s, _ in res2}
    check("E2 planted world: target C screams",
          by_t[prp.C_IDX][2],
          f"capture {by_t[prp.C_IDX][0]:+.3f} vs null99 "
          f"{by_t[prp.C_IDX][1]:+.3f}")
    ab = [by_t[prp.A_IDX][2], by_t[prp.B_IDX][2]]
    # E3 is a filed prediction, NOT a claim of the pushed probe — A/B
    # screams (or their absence) are reported, never graded as false
    # alarms; the symmetric relation makes them legitimate either way.
    check("E3 (prediction, not probe claim) targets A and B also scream",
          all(ab),
          f"A: cap {by_t[prp.A_IDX][0]:+.3f} vs {by_t[prp.A_IDX][1]:+.3f} "
          f"scream={ab[0]}; "
          f"B: cap {by_t[prp.B_IDX][0]:+.3f} vs {by_t[prp.B_IDX][1]:+.3f} "
          f"scream={ab[1]}")
    distractor_screams = [t for t, _, _, s, _ in res2
                          if s and t not in (prp.A_IDX, prp.B_IDX,
                                             prp.C_IDX)]
    check("E4 planted world: 7 distractor targets all quiet",
          distractor_screams == [],
          f"{len(distractor_screams)}/7 distractors screamed "
          f"(targets {distractor_screams})")

    dt = time.perf_counter() - t0
    check("E5 full 20-target scan cost", dt < 60, f"{dt:.1f}s")
    npass = sum(checks)
    print(f"\n  ATTACK R-A: {npass} passed, {len(checks) - npass} failed "
          f"of {len(checks)}  ({dt:.1f}s)")


if __name__ == "__main__":
    main()
