"""pool_relations_probe.py — RECURSIVE POOLS: relational information
about the structures themselves (owner, 2026-07-15, the representation
arc): "building group and sibling nets. that's what the matrix was
supposed to be but it can only represent 2d relations. what if we
recursively applied these pools to find relational information about
the structures themselves."

THE CLAIM UNDER TEST: a pool is level-agnostic — the same frozen
instrument that finds structure in raw data (pool level 0, proven
13/13 in reservoir_surprise_probe.py) finds structure in STRUCTURE
DESCRIPTORS (pool level 1): which groups relate to which, through
what function. A matrix can only hold pairwise relations; reifying
structures as descriptor channels and recursing covers higher-order
relations with the same pairwise machinery, one level up.

THE WORLD (truth planted, invisible below level 1):
  10 groups per episode; each group g is a clean level-0 law
  y = a_g * x on its own rows (nothing relational visible in any
  group's raw data). Across 60 episodes the constants redraw fresh —
  EXCEPT the planted relation: a_C = a_A * a_B every episode (a pure
  product of signed constants: marginally invisible by construction,
  the proven interaction world shape). Other groups are relational
  distractors (independent redraws).

THE LADDER UNDER TEST:
  L0 (control): group C's own residual after its own law — the level-0
     pool must be QUIET (the relation does not live in raw data).
  L1 (detection): rows = episodes, channels = the other groups'
     fitted constants, target = a_C. The level-1 pool must SCREAM.
  L1 (attribution): ablation must name exactly {A, B} among 9
     candidate groups; the marginal baseline must name NOTHING
     (product relation, zero-mean channels).
  GENESIS (the mint, pools removed): engine discover() on the
     descriptor rows must mint the relation as a readable equation
     (+1.00*aA*aB shape) and the mint must predict HELD-OUT episodes:
     a_C from (a_A, a_B), then group C's raw y on fresh x through the
     composed chain (level-1 relation -> level-0 law). Exactness at
     machine precision (the world is exact).
  CONTROLS: scrambled episode alignment (a_C permuted across
     episodes) — pool quiet AND discover refuses; distractor groups
     never named.

PREDICTIONS (filed before the first run)
  Q1 level-0 pool quiet on group C's own-law residual (20 episodes
     tested, 0 screams).
  Q2 level-1 pool screams on the descriptor table.
  Q3 ablation names exactly {A, B}; marginal baseline names nothing.
  Q4 scrambled alignment: pool quiet, discover NO RULE / inexact.
  Q5 genesis mint is the product form, engine-exact on train.
  Q6 held-out: a_C predicted to machine precision (<1e-9) on 12
     unseen episodes; composed raw-y prediction <1e-9 max err.
  Q7 full ladder cost < 60 s here.

SCAFFOLDING DISCLOSED: level-0 constant extraction for the bulk of
episodes uses least squares — the level-0 road (discover finds these
laws) is already battery-proven and is NOT what this probe tests;
spot-checks prove lstsq == discover's articulated constant on 3
(group, episode) samples. All level-1 COGNITION that ships is engine
discover()/apply_law(); pools point only. The pool instrument is
imported unchanged from reservoir_surprise_probe.py (13/13).

INTEGRITY: pools never write into a mind, never feed an equation;
stdlib + numpy; PYTHONHASHSEED=0; fixed seeds; fresh throwaway minds.
Run: cd experiments && PYTHONHASHSEED=0 python3 pool_relations_probe.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import isaac as isaac_mod  # noqa: E402
from reservoir_surprise_probe import (  # noqa: E402  (13/13 instrument, unchanged)
    attribution, marginal_names, scream_verdict)

Isaac = isaac_mod.Isaac

N_EP = 240         # episodes (level-1 rows; the cheap axis — run 1's
                   # 60 under-sampled the 9-channel interaction:
                   # capture -0.67, the overfit symptom, vs +0.92 at
                   # n~400 in the parent probe. WRONG-KEPT run 1 note:
                   # naming needs rows; rows here cost lstsq, so buy
                   # them.)
N_HOLD = 12        # held-out episodes for the mint's proof
N_GROUPS = 10      # A, B, C=A*B, plus 7 relational distractors
ROWS_PER_GROUP = 40
M_POOL = 800       # instrument capacity request (auto-capped by n)
A_IDX, B_IDX, C_IDX = 0, 1, 2


def make_episode(rng):
    """One episode: fresh constants, the planted relation, raw rows."""
    a = rng.uniform(-2.0, 2.0, N_GROUPS)
    a[np.abs(a) < 0.3] += 0.6 * np.sign(a[np.abs(a) < 0.3]) + 0.3
    a[C_IDX] = a[A_IDX] * a[B_IDX]
    rows = []
    for g in range(N_GROUPS):
        x = rng.uniform(-2.0, 2.0, ROWS_PER_GROUP)
        rows.append((x, a[g] * x))
    return a, rows


def fit_constant(x, y):
    """Level-0 extraction (scaffolding, disclosed): a = argmin |y-ax|."""
    return float(np.dot(x, y) / np.dot(x, x))


def main():
    t0 = time.perf_counter()
    checks = []

    def check(name, ok, detail):
        checks.append(ok)
        print(f"  [{'ok ' if ok else 'FAIL'}] {name} — {detail}")

    rng = np.random.default_rng(42)
    episodes = [make_episode(rng) for _ in range(N_EP + N_HOLD)]
    train_eps, hold_eps = episodes[:N_EP], episodes[N_EP:]

    # Level-0 descriptor extraction + the lstsq==discover spot-check
    print("L0. structure extraction (scaffolding) + engine spot-check x3")
    A_train = np.array([[fit_constant(*ep[1][g]) for g in range(N_GROUPS)]
                        for ep in train_eps])
    ok_spot, details = 0, []
    for k in range(3):
        a_true, rows = train_eps[k]
        x, y = rows[C_IDX]
        mind = Isaac()
        res = mind.discover(x.reshape(-1, 1), y, name="spot")
        pred = mind.apply_law("spot", x.reshape(-1, 1))
        a_probe = fit_constant(x, y)
        agree = (pred is not None
                 and np.max(np.abs(pred - a_probe * x)) < 1e-6)
        ok_spot += agree
        details.append(f"ep{k} discover={res['verdict']} agree={agree}")
    check("L0 spot-check: engine law == scaffold constant (3/3)",
          ok_spot == 3, "; ".join(details))

    # Q1 — the relation is invisible at level 0. RUN-1 WRONG-KEPT: the
    # first run pointed a pool at a MACHINE-ZERO residual (this world
    # is exact) — capture on zeros is ill-posed and screamed 11/20 on
    # numerical dust. The verdict ladder's own regime split applies
    # (exact road BEFORE surprise road, as in verdict_rewrite): pools
    # only run when a residual exists to test.
    print("Q1. level-0: group C's rows after group C's own law x20 "
          "(exact road first; pool only if residual exists)")
    quiet0, road = 0, {"exact": 0, "pool": 0}
    for k in range(20):
        a_true, rows = train_eps[k]
        x, y = rows[C_IDX]
        resid = y - fit_constant(x, y) * x
        if np.max(np.abs(resid)) < 1e-9:
            road["exact"] += 1
            quiet0 += 1            # nothing left — quiet by exactness
            continue
        road["pool"] += 1
        _, _, s, _ = scream_verdict(x.reshape(-1, 1), resid, M_POOL,
                                    rng_seed=100 + k)
        quiet0 += (not s)
    check("Q1 level-0 quiet (relation not in raw data)", quiet0 == 20,
          f"{quiet0}/20 quiet (exact road {road['exact']}, pool "
          f"{road['pool']})")

    # Q2 — level-1 detection on the descriptor table
    print("Q2. level-1 pool: rows=episodes, channels=other groups' constants,"
          " target=a_C")
    X1 = np.delete(A_train, C_IDX, axis=1)          # 9 candidate channels
    y1 = A_train[:, C_IDX]
    cap, thr, scream, _ = scream_verdict(X1, y1, M_POOL, rng_seed=200)
    check("Q2 level-1 pool screams", scream,
          f"capture {cap:+.3f} vs null99 {thr:+.3f}")

    # Q3 — attribution names exactly {A, B}; marginal baseline blind
    contrib, athr, named = attribution(X1, y1, M_POOL, rng_seed=201)
    mobs, mthr, mnamed = marginal_names(X1, y1)
    print(f"       contrib {np.round(contrib, 3)} thr {athr:.3f} "
          f"named={named}; marginal named={mnamed} "
          f"(max|corr| {mobs.max():.3f} vs {mthr:.3f})")
    check("Q3a ablation names exactly {A,B}", named == [A_IDX, B_IDX],
          f"named {named} (A={A_IDX}, B={B_IDX} after C-column removal)")
    check("Q3b marginal baseline blind to the product relation",
          mnamed == [], f"named {mnamed}")

    # Q4 — scrambled alignment control
    print("Q4. scrambled episode alignment (a_C permuted across episodes)")
    y_scr = np.random.default_rng(7).permutation(y1)
    cap_s, thr_s, scream_s, _ = scream_verdict(X1, y_scr, M_POOL,
                                               rng_seed=202)
    mind_s = Isaac()
    res_s = mind_s.discover(X1[:, [A_IDX, B_IDX]], y_scr, name="scr")
    pred_s = mind_s.apply_law("scr", X1[:, [A_IDX, B_IDX]])
    scr_confident = (res_s["verdict"] == "UNDERSTOOD"
                     and pred_s is not None
                     and np.max(np.abs(pred_s - y_scr)) < 1e-6)
    check("Q4 scrambled: pool quiet AND no exact mint",
          (not scream_s) and (not scr_confident),
          f"capture {cap_s:+.3f} vs {thr_s:+.3f}; discover verdict "
          f"{res_s['verdict']}")

    # Q5 — GENESIS: the engine mints the relation (pools removed)
    print("Q5. genesis mint: discover() on descriptor rows (pools removed)")
    mind = Isaac()
    res = mind.discover(X1, y1, name="relation")
    pred = mind.apply_law("relation", X1)
    train_err = np.max(np.abs(pred - y1)) if pred is not None else np.inf
    eq = mind.findings[-1].get("equation", "?") if mind.findings else "?"
    print(f"       verdict={res['verdict']} eq='{eq}' train maxerr "
          f"{train_err:.2e}")
    check("Q5 mint engine-exact on train", train_err < 1e-9,
          f"verdict={res['verdict']} maxerr {train_err:.2e} eq {eq}")

    # Q6 — the mint proves itself on held-out episodes, down to raw y
    print("Q6. held-out proof: 12 unseen episodes, constant + composed raw y")
    A_hold = np.array([[fit_constant(*ep[1][g]) for g in range(N_GROUPS)]
                       for ep in hold_eps])
    Xh = np.delete(A_hold, C_IDX, axis=1)
    pred_c = mind.apply_law("relation", Xh)
    err_c = np.max(np.abs(pred_c - A_hold[:, C_IDX])) \
        if pred_c is not None else np.inf
    raw_errs = []
    for i, (a_true, rows) in enumerate(hold_eps):
        x_new = np.random.default_rng(900 + i).uniform(-2, 2, 25)
        raw_errs.append(np.max(np.abs(pred_c[i] * x_new
                                      - a_true[C_IDX] * x_new)))
    check("Q6a held-out constants exact", err_c < 1e-9,
          f"max |a_C_pred - a_C| = {err_c:.2e} over {N_HOLD} episodes")
    check("Q6b composed raw-y prediction exact (level-1 mint -> level-0 law)",
          max(raw_errs) < 1e-9, f"max raw err {max(raw_errs):.2e}")

    dt = time.perf_counter() - t0
    check("Q7 full ladder cost", dt < 60, f"{dt:.1f}s")
    npass = sum(checks)
    print(f"\n  PROBE: {npass} passed, {len(checks) - npass} failed of "
          f"{len(checks)}  ({dt:.1f}s)")


if __name__ == "__main__":
    main()
