"""eval_attack_rel_distractor.py — ATTACK R-D: independence of naming
under a SECOND relation in the same world.

TARGET CLAIM (pool_relations_probe.py, pushed): the level-1 pool +
ablation names EXACTLY the parent groups of a planted product relation
({A,B} among 9 candidates) and marginal correlation stays blind. The
pushed probe plants ONE relation. This attack plants TWO in the same
world and asks whether naming stays independent:

  a_C = a_A * a_B   (A=0, B=1, C=2)
  a_F = a_D * a_E   (D=3, E=4, F=5)
  groups 6..9 = independent relational distractors.

240 episodes, exact world (same draw law as the probe's make_episode:
uniform(-2,2), small magnitudes pushed away from zero, then the two
overwrites). Constants extracted with the probe's own disclosed
scaffold fit_constant; the instrument is imported UNCHANGED from
reservoir_surprise_probe.py. No existing file modified.

THE THREE TESTS
  T1 target a_C, channels = the 9 other groups (INCLUDING D, E, F):
     pool must scream; ablation must name exactly {A, B}; D/E/F must
     not leak in despite F being a product channel itself.
  T2 target a_F, channels = the 9 other groups (INCLUDING A, B, C):
     pool must scream; ablation must name exactly {D, E}.
  T3 target a distractor (group 7), channels = the 9 other groups:
     pool must stay QUIET (no relation exists for this target).

PREDICTIONS (filed before the first run; wrong-kept if falsified)
  P1 T1 screams (capture > null99, expect capture > +0.5 as in the
     parent probe's Q2 at n=240) and names exactly {A, B} = original
     groups {0, 1}; no leak of D/E/F/distractors.
  P2 T2 screams and names exactly {D, E} = original groups {3, 4};
     no leak of A/B/C/distractors.
  P3 T3 quiet: capture <= null99 threshold.
  P4 marginal baseline names NOTHING on T1 and T2 (product relations,
     near-zero-mean channels — the probe's Q3b logic carries over).
  P5 runtime < 120 s total.

INTEGRITY: pools never write into a mind; stdlib + numpy;
PYTHONHASHSEED=0; fixed seeds (world 42; instrument rng_seeds 300..).
Run: cd experiments && PYTHONHASHSEED=0 python eval_attack_rel_distractor.py

RUN-1 MEASURED (2026-07-15) — P1b FALSIFIED, WRONG-KEPT: target C's
ablation named [0, 1, 5] — group F (the OTHER relation's child,
statistically independent of a_C) leaked in at contrib 0.063 vs decoy
thr 0.053; D and E sat just below (0.049, 0.042) while true
distractors read <= 0.029. Suspected mechanism: F = D*E is exactly
dependent on other CHANNELS, and the matched-decoy null is built from
channels independent of everything — permuting F breaks the D-E-F
manifold the readout leaned on, a drop no independent decoy can model.
P5 also falsified (435 s, partly under CPU contention from a duplicate
run; wrong-kept). RECHECK MODE (added after run 1, disclosed): argv
'recheck' reruns ONLY target-C attribution at two fresh instrument
seeds (401, 501) to test whether the F leak is systematic or one-seed
luck.
"""
import sys
import time

import numpy as np

from reservoir_surprise_probe import (attribution, marginal_names,
                                      scream_verdict)

N_EP = 240
N_GROUPS = 10
ROWS_PER_GROUP = 40
M_POOL = 800                    # probe's request; auto-capped to 400 at n=240
A_IDX, B_IDX, C_IDX = 0, 1, 2
D_IDX, E_IDX, F_IDX = 3, 4, 5
DISTRACTOR_TARGET = 7


def make_episode_two_relations(rng):
    """Probe's make_episode draw law + a SECOND planted relation."""
    a = rng.uniform(-2.0, 2.0, N_GROUPS)
    a[np.abs(a) < 0.3] += 0.6 * np.sign(a[np.abs(a) < 0.3]) + 0.3
    a[C_IDX] = a[A_IDX] * a[B_IDX]
    a[F_IDX] = a[D_IDX] * a[E_IDX]
    rows = []
    for g in range(N_GROUPS):
        x = rng.uniform(-2.0, 2.0, ROWS_PER_GROUP)
        rows.append((x, a[g] * x))
    return a, rows


def fit_constant(x, y):
    """Level-0 scaffold, identical to the probe's (disclosed)."""
    return float(np.dot(x, y) / np.dot(x, x))


def run_target(A_desc, target, rng_base, want_names, label):
    """Point the level-1 pool at one group's constant; report everything."""
    X = np.delete(A_desc, target, axis=1)
    y = A_desc[:, target]
    orig_ids = [g for g in range(N_GROUPS) if g != target]
    cap, thr, scream, _ = scream_verdict(X, y, M_POOL, rng_seed=rng_base)
    print(f"  {label}: capture {cap:+.3f} vs null99 {thr:+.3f} -> "
          f"{'SCREAM' if scream else 'quiet'}")
    named_orig, contrib, athr = None, None, None
    mnamed_orig = None
    if want_names:
        contrib, athr, named = attribution(X, y, M_POOL, rng_seed=rng_base + 1)
        named_orig = [orig_ids[j] for j in named]
        mobs, mthr, mnamed = marginal_names(X, y)
        mnamed_orig = [orig_ids[j] for j in mnamed]
        print(f"    contrib {np.round(contrib, 3)} thr {athr:.3f}")
        print(f"    ablation named (orig group ids) = {named_orig}")
        print(f"    marginal named = {mnamed_orig} "
              f"(max|corr| {mobs.max():.3f} vs {mthr:.3f})")
    return cap, thr, scream, named_orig, mnamed_orig


def build_world():
    rng = np.random.default_rng(42)
    episodes = [make_episode_two_relations(rng) for _ in range(N_EP)]
    return np.array([[fit_constant(*ep[1][g]) for g in range(N_GROUPS)]
                     for ep in episodes])


def recheck():
    """Post-run-1 robustness leg: is the F leak into C seed-luck?"""
    t0 = time.perf_counter()
    A_desc = build_world()
    X = np.delete(A_desc, C_IDX, axis=1)
    y = A_desc[:, C_IDX]
    orig_ids = [g for g in range(N_GROUPS) if g != C_IDX]
    for seed in (401, 501):
        contrib, athr, named = attribution(X, y, M_POOL, rng_seed=seed)
        named_orig = [orig_ids[j] for j in named]
        print(f"  rng_seed {seed}: contrib {np.round(contrib, 3)} "
              f"thr {athr:.3f} named {named_orig}")
    print(f"  recheck done ({time.perf_counter() - t0:.1f}s)")


def main():
    t0 = time.perf_counter()
    rng = np.random.default_rng(42)
    episodes = [make_episode_two_relations(rng) for _ in range(N_EP)]
    A_desc = np.array([[fit_constant(*ep[1][g]) for g in range(N_GROUPS)]
                       for ep in episodes])

    # world sanity: both relations exact at the descriptor level
    errC = np.max(np.abs(A_desc[:, C_IDX] - A_desc[:, A_IDX] * A_desc[:, B_IDX]))
    errF = np.max(np.abs(A_desc[:, F_IDX] - A_desc[:, D_IDX] * A_desc[:, E_IDX]))
    print(f"world sanity: |a_C - a_A*a_B| max {errC:.2e}, "
          f"|a_F - a_D*a_E| max {errF:.2e} over {N_EP} episodes")

    results = {}
    print("T1. target a_C (channels include D,E,F)")
    results["C"] = run_target(A_desc, C_IDX, 300, True, "target C")
    print("T2. target a_F (channels include A,B,C)")
    results["F"] = run_target(A_desc, F_IDX, 310, True, "target F")
    print("T3. target distractor group 7 (no relation planted)")
    results["7"] = run_target(A_desc, DISTRACTOR_TARGET, 320, False,
                              "target g7")

    capC, thrC, scrC, namedC, mnamedC = results["C"]
    capF, thrF, scrF, namedF, mnamedF = results["F"]
    cap7, thr7, scr7, _, _ = results["7"]

    checks = [
        ("P1a T1 screams", scrC),
        ("P1b T1 names exactly {A,B}={0,1}", namedC == [A_IDX, B_IDX]),
        ("P2a T2 screams", scrF),
        ("P2b T2 names exactly {D,E}={3,4}", namedF == [D_IDX, E_IDX]),
        ("P3  T3 distractor target quiet", not scr7),
        ("P4  marginal blind on both", mnamedC == [] and mnamedF == []),
    ]
    dt = time.perf_counter() - t0
    checks.append(("P5  runtime < 120 s", dt < 120))
    npass = 0
    for name, ok in checks:
        npass += ok
        print(f"  [{'ok ' if ok else 'FAIL'}] {name}")
    print(f"\nATTACK R-D: {npass} passed, {len(checks) - npass} failed of "
          f"{len(checks)}  ({dt:.1f}s)")
    print(f"summary: C cap {capC:+.3f}/{thrC:+.3f} named {namedC}; "
          f"F cap {capF:+.3f}/{thrF:+.3f} named {namedF}; "
          f"g7 cap {cap7:+.3f}/{thr7:+.3f} "
          f"{'SCREAM' if scr7 else 'quiet'}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "recheck":
        recheck()
    else:
        main()
