"""eval_attack_rel_othertypes.py — ATTACK R-C: RELATION-TYPE GENERALITY.

TARGET: pool_relations_probe.py claims the level-1 pool ladder (detect ->
name -> mint) on structure descriptors, but its only planted relation is
a PRODUCT a_C = a_A * a_B — a shape the engine's atom pool happens to
contain (degree-2 var*var term). This attack replants the SAME world
with two other relation types and runs the same ladder unchanged:

  WORLD-ADD  a_C = a_A + a_B   (in-grammar: two linear terms; channels
             are mixed-sign, so the historic positive-data product-first
             additive hole — additive_boundary_probe.py W1 vs W4 — should
             NOT bite; verify, don't assume)
  WORLD-MAX  a_C = max(a_A, a_B)  (OUT-OF-GRAMMAR: no max atom exists;
             max(a,b) = (a+b)/2 + |a-b|/2 and |a-b| is only APPROXIMABLE
             by cosh(k*(a-b)) atoms — the honest engine outcome is
             PARTIAL/CAPTURED/refusal, never a machine-exact claim)

Same conventions as the parent probe: 240 train episodes + 12 held-out,
10 groups, 40 rows/group, lstsq descriptor scaffolding (disclosed there,
spot-proven there — not re-tested here), C-column removed so candidate
channels are A=0, B=1, distractors 2..8. Pool instrument imported
unchanged from reservoir_surprise_probe.py; engine = fresh throwaway
Isaac() minds. PYTHONHASHSEED=0; world seeds 4242 (ADD) / 4243 (MAX);
pool seeds 300.. (ADD) / 400.. (MAX).

PREDICTIONS (filed BEFORE the first run; wrong ones kept below as-is)
  P1 ADD detection: level-1 pool SCREAMS (predict capture > +0.90 vs
     null99 ~ +0.1).
  P2 ADD naming: ablation names exactly [0, 1]; the marginal baseline
     ALSO names [0, 1] (additive relations are marginally visible —
     expected, not a failure; corr per arm ~ 0.7).
  P3 ADD genesis: discover() verdict UNDERSTOOD, equation the additive
     form (+1.00*x0 + 1.00*x1 shape, no distractor terms), train max
     err < 1e-9, held-out constant max err < 1e-9 over 12 episodes,
     composed raw-y max err < 1e-9. The additive hole does NOT bite
     (mixed-sign channels kill corr(x0*x1, y)).
  P4 MAX detection: level-1 pool SCREAMS (capture > +0.85 vs null99).
  P5 MAX naming: ablation names exactly [0, 1]; marginal baseline also
     names [0, 1] (max is monotone in each arm, corr ~ +0.5-0.65 each).
  P6 MAX genesis — THE ENGINE'S OLD VERDICT TIER FAILS HONESTY HERE:
     predict verdict UNDERSTOOD (r2 between 0.95 and 0.999, past the
     0.9 tier gate) with a wrong-form equation (cosh/sind difference
     atoms + linear terms approximating |a-b|), train max err in the
     0.02-0.5 band (NOT machine-exact), held-out constant max err same
     band — i.e. a CONFIDENT WRONG MINT (the additive-hole failure mode
     transplanted to level 1). The pool legs stay honest pointers; the
     dishonesty, if it appears, is the engine tier's r2>=0.9 gate.
  P7 total runtime < 180 s.
     >>> FALSIFIED BY RUN 1, KEPT: measured 313.1 s (170.9 s ADD +
     142.2 s MAX). The two attribution passes (8 decoy rounds x 10
     channels x 10 repeats at n=240, m_eff=400) dominate; the parent
     probe pays this once, this attack pays it twice. P1-P6 all
     confirmed as filed, including P6's confident-wrong-mint band
     (measured train maxerr 7.35e-02, inside the predicted 0.02-0.5).

VERDICT GRADING KEY (per attack brief): ADD ladder exact = the claim
generalizes in-grammar. MAX: pool screams + names, engine PARTIAL/
CAPTURED/NO-RULE = honest scope limit (PARTIAL); engine UNDERSTOOD with
non-exact train/held-out error = confident wrong mint = BROKEN, charged
to the ENGINE's verdict tier (the probe's pool legs graded separately).

Run: cd experiments && PYTHONHASHSEED=0 python eval_attack_rel_othertypes.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import isaac as isaac_mod  # noqa: E402
from reservoir_surprise_probe import (  # noqa: E402  (13/13 instrument, unchanged)
    attribution, marginal_names, scream_verdict)
from pool_relations_probe import (  # noqa: E402  (parent conventions, unchanged)
    N_GROUPS, ROWS_PER_GROUP, fit_constant)

Isaac = isaac_mod.Isaac

N_EP = 240
N_HOLD = 12
M_POOL = 800
A_IDX, B_IDX, C_IDX = 0, 1, 2


def make_episode(rng, rel):
    """Parent probe's episode builder, relation injected."""
    a = rng.uniform(-2.0, 2.0, N_GROUPS)
    a[np.abs(a) < 0.3] += 0.6 * np.sign(a[np.abs(a) < 0.3]) + 0.3
    a[C_IDX] = rel(a[A_IDX], a[B_IDX])
    rows = []
    for g in range(N_GROUPS):
        x = rng.uniform(-2.0, 2.0, ROWS_PER_GROUP)
        rows.append((x, a[g] * x))
    return a, rows


def run_world(tag, rel, world_seed, pool_seed):
    print(f"\n════════ WORLD {tag} (world seed {world_seed}) ════════")
    t0 = time.perf_counter()
    rng = np.random.default_rng(world_seed)
    episodes = [make_episode(rng, rel) for _ in range(N_EP + N_HOLD)]
    train_eps, hold_eps = episodes[:N_EP], episodes[N_EP:]

    A_train = np.array([[fit_constant(*ep[1][g]) for g in range(N_GROUPS)]
                        for ep in train_eps])
    X1 = np.delete(A_train, C_IDX, axis=1)          # A=0, B=1, distractors 2..8
    y1 = A_train[:, C_IDX]

    # ── detection ──
    cap, thr, scream, spread = scream_verdict(X1, y1, M_POOL,
                                              rng_seed=pool_seed)
    print(f"  detection: capture {cap:+.3f} vs null99 {thr:+.3f} "
          f"(spread {spread:.3f}) -> {'SCREAM' if scream else 'quiet'}")

    # ── naming ──
    contrib, athr, named = attribution(X1, y1, M_POOL, rng_seed=pool_seed + 1)
    mobs, mthr, mnamed = marginal_names(X1, y1)
    print(f"  ablation: contrib {np.round(contrib, 3)} thr {athr:.3f} "
          f"named={named} (truth = [0, 1])")
    print(f"  marginal baseline: named={mnamed} "
          f"(per-channel |corr| {np.round(mobs, 3)} vs thr {mthr:.3f})")

    # ── genesis: engine discover() on the descriptor rows ──
    mind = Isaac()
    res = mind.discover(X1, y1, name=tag)
    pred = mind.apply_law(tag, X1)
    train_err = float(np.max(np.abs(pred - y1))) if pred is not None else np.inf
    eq = mind.findings[-1].get("equation", "?") if mind.findings else "?"
    print(f"  genesis: verdict={res['verdict']} r2={res['r2']:.5f} "
          f"train maxerr {train_err:.2e}")
    print(f"  equation: {eq[:220]}")

    # ── held-out: 12 unseen episodes, constants + composed raw y ──
    A_hold = np.array([[fit_constant(*ep[1][g]) for g in range(N_GROUPS)]
                       for ep in hold_eps])
    Xh = np.delete(A_hold, C_IDX, axis=1)
    pred_c = mind.apply_law(tag, Xh)
    err_c = float(np.max(np.abs(pred_c - A_hold[:, C_IDX]))) \
        if pred_c is not None else np.inf
    raw_err = np.inf
    if pred_c is not None:
        raw_errs = []
        for i, (a_true, rows) in enumerate(hold_eps):
            x_new = np.random.default_rng(900 + i).uniform(-2, 2, 25)
            raw_errs.append(np.max(np.abs(pred_c[i] * x_new
                                          - a_true[C_IDX] * x_new)))
        raw_err = float(max(raw_errs))
    print(f"  held-out ({N_HOLD} eps): constant maxerr {err_c:.2e}, "
          f"composed raw-y maxerr {raw_err:.2e}")
    dt = time.perf_counter() - t0
    print(f"  world wall time {dt:.1f}s")
    return dict(tag=tag, cap=cap, thr=thr, scream=scream, named=named,
                mnamed=mnamed, verdict=res["verdict"], r2=res["r2"],
                eq=eq, train_err=train_err, hold_err=err_c,
                raw_err=raw_err, secs=dt)


def main():
    t_all = time.perf_counter()
    add = run_world("add_rel", lambda a, b: a + b,
                    world_seed=4242, pool_seed=300)
    mx = run_world("max_rel", lambda a, b: max(a, b),
                   world_seed=4243, pool_seed=400)

    print("\n════════ VERDICT SHEET (attack R-C) ════════")
    rows = []

    def sheet(cap_, verdict, ev):
        rows.append((cap_, verdict, ev))
        print(f"  {cap_:38s} {verdict:16s} {ev}")

    sheet("ADD pool detection", "WORKS" if add["scream"] else "BROKEN",
          f"capture {add['cap']:+.3f} vs null99 {add['thr']:+.3f}")
    sheet("ADD ablation names {A,B}",
          "WORKS" if add["named"] == [0, 1] else "BROKEN",
          f"named {add['named']}")
    sheet("ADD marginal-also-sees (expected)",
          "AS-FILED" if add["mnamed"] == [0, 1] else "SURPRISE",
          f"marginal named {add['mnamed']}")
    add_exact = (add["verdict"] == "UNDERSTOOD"
                 and add["train_err"] < 1e-9 and add["hold_err"] < 1e-9)
    sheet("ADD genesis exact (hole must not bite)",
          "WORKS" if add_exact else "BROKEN",
          f"{add['verdict']} train {add['train_err']:.2e} "
          f"hold {add['hold_err']:.2e} raw {add['raw_err']:.2e}")

    sheet("MAX pool detection", "WORKS" if mx["scream"] else "BROKEN",
          f"capture {mx['cap']:+.3f} vs null99 {mx['thr']:+.3f}")
    sheet("MAX ablation names {A,B}",
          "WORKS" if mx["named"] == [0, 1] else "BROKEN",
          f"named {mx['named']}")
    mx_exact = mx["train_err"] < 1e-9
    if mx["verdict"] in ("PARTIAL", "CAPTURED", "NO RULE"):
        mx_v = "HONEST-REFUSAL"
    elif mx["verdict"] == "UNDERSTOOD" and not mx_exact:
        mx_v = "BROKEN"          # confident wrong mint — old verdict tier
    else:
        mx_v = "SURPRISE-EXACT"  # would contradict the grammar analysis
    sheet("MAX genesis honesty (engine tier)", mx_v,
          f"{mx['verdict']} r2={mx['r2']:.5f} train {mx['train_err']:.2e} "
          f"hold {mx['hold_err']:.2e}")

    dt = time.perf_counter() - t_all
    print(f"\n  TOTAL {dt:.1f}s  (budget 180s)")


if __name__ == "__main__":
    main()
