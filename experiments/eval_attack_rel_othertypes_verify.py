"""eval_attack_rel_othertypes_verify.py — ADVERSARIAL VERIFICATION of
the R-C finding (BROKEN: engine verdict tier mints confident-wrong
UNDERSTOOD on the out-of-grammar MAX relation).

REFUTATION ANGLES TESTED
  V1 seed robustness: replant WORLD-MAX with a DIFFERENT world seed
     (5243 vs the finding's 4243). If the confident-wrong UNDERSTOOD
     was a one-seed accident, the break dies here.
  V2 the pushed system's OWN GATE: the finding ran discover() BARE.
     The same pushed branch ships reservoir_surprise_probe claim A —
     "if the residual screams, the verdict must not say UNDERSTOOD,
     whatever r2 says" — as the instrument for exactly this failure
     class. Point scream_verdict (unchanged) at the engine's residual
     y1 - apply_law(...) on both MAX worlds. If the pool SCREAMS, the
     pushed branch as a system catches the wrong mint it is already
     documented to produce (additive-hole class, Sec-2b rewrite
     queued) -> the finding grades a KNOWN, FILED, INSTRUMENTED
     boundary, i.e. PARTIAL, not BROKEN. If the pool stays QUIET, the
     wrong mint is invisible to the whole pushed system -> BROKEN
     stands, stronger.

SHRINK DISCLOSED: no attribution/marginal passes here (naming is not
in dispute and it dominated the attack's 313s); only genesis + the
residual scream leg. World construction, fit_constant, N_GROUPS,
ROWS_PER_GROUP, M_POOL=800, C-column removal identical to the attack
probe / parent probe conventions. PYTHONHASHSEED=0.

PREDICTIONS (filed BEFORE the first run; wrong ones kept)
  PV1 MAX @ new seed 5243: discover() again returns UNDERSTOOD with
      r2 in (0.99, 0.9999) and train maxerr in the 0.02-0.5 band —
      the break SURVIVES the seed change (not an accident).
  PV2 residual scream: on BOTH max worlds (4243 and 5243) the frozen
      pool SCREAMS on the engine residual — capture > +0.5 vs null99
      < +0.2 (the |a-b| wiggle the atoms could not say is structured
      and n=240 is far past the naming boundary, let alone the
      detection one).
  PV3 exact-road control: on the finding's ADD world (4242) the
      engine residual is machine-zero (max|resid| < 1e-9), so the
      regime split (exact road first, parent probe Q1) never wakes
      the pool — no scream possible on the in-grammar relation.
  PV4 runtime < 150 s total.

Run: cd experiments && PYTHONHASHSEED=0 python eval_attack_rel_othertypes_verify.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import isaac as isaac_mod  # noqa: E402
from reservoir_surprise_probe import scream_verdict  # noqa: E402  (unchanged)
from pool_relations_probe import (  # noqa: E402  (parent conventions, unchanged)
    N_GROUPS, ROWS_PER_GROUP, fit_constant)

Isaac = isaac_mod.Isaac

N_EP = 240
N_HOLD = 12
M_POOL = 800
A_IDX, B_IDX, C_IDX = 0, 1, 2


def make_episode(rng, rel):
    """Attack probe's episode builder, verbatim (relation injected)."""
    a = rng.uniform(-2.0, 2.0, N_GROUPS)
    a[np.abs(a) < 0.3] += 0.6 * np.sign(a[np.abs(a) < 0.3]) + 0.3
    a[C_IDX] = rel(a[A_IDX], a[B_IDX])
    rows = []
    for g in range(N_GROUPS):
        x = rng.uniform(-2.0, 2.0, ROWS_PER_GROUP)
        rows.append((x, a[g] * x))
    return a, rows


def build_world(rel, world_seed):
    rng = np.random.default_rng(world_seed)
    episodes = [make_episode(rng, rel) for _ in range(N_EP + N_HOLD)]
    train_eps, hold_eps = episodes[:N_EP], episodes[N_EP:]
    A_train = np.array([[fit_constant(*ep[1][g]) for g in range(N_GROUPS)]
                        for ep in train_eps])
    X1 = np.delete(A_train, C_IDX, axis=1)
    y1 = A_train[:, C_IDX]
    A_hold = np.array([[fit_constant(*ep[1][g]) for g in range(N_GROUPS)]
                       for ep in hold_eps])
    Xh = np.delete(A_hold, C_IDX, axis=1)
    yh = A_hold[:, C_IDX]
    return X1, y1, Xh, yh


def genesis_plus_gate(tag, rel, world_seed, pool_seed):
    print(f"\n──── {tag} (world seed {world_seed}) ────")
    t0 = time.perf_counter()
    X1, y1, Xh, yh = build_world(rel, world_seed)
    mind = Isaac()
    res = mind.discover(X1, y1, name=tag)
    pred = mind.apply_law(tag, X1)
    train_err = float(np.max(np.abs(pred - y1))) if pred is not None else np.inf
    pred_h = mind.apply_law(tag, Xh)
    hold_err = float(np.max(np.abs(pred_h - yh))) if pred_h is not None else np.inf
    eq = mind.findings[-1].get("equation", "?") if mind.findings else "?"
    print(f"  genesis: verdict={res['verdict']} r2={res['r2']:.5f} "
          f"train maxerr {train_err:.2e} held-out maxerr {hold_err:.2e}")
    print(f"  equation: {eq[:200]}")
    resid_max = np.inf
    cap = thr = spread = float("nan")
    scream = None
    if pred is not None:
        resid = y1 - pred
        resid_max = float(np.max(np.abs(resid)))
        if resid_max < 1e-9:
            scream = False
            print(f"  residual: machine-zero (max {resid_max:.2e}) — exact "
                  f"road, pool never wakes (parent Q1 regime split)")
        else:
            cap, thr, scream, spread = scream_verdict(X1, resid, M_POOL,
                                                      rng_seed=pool_seed)
            print(f"  RESIDUAL GATE: capture {cap:+.3f} vs null99 {thr:+.3f} "
                  f"(spread {spread:.3f}) -> "
                  f"{'SCREAM — pushed gate catches the mint' if scream else 'QUIET — mint invisible to the pushed system'}")
    dt = time.perf_counter() - t0
    print(f"  wall {dt:.1f}s")
    return dict(verdict=res["verdict"], r2=res["r2"], train_err=train_err,
                hold_err=hold_err, resid_max=resid_max, cap=cap, thr=thr,
                scream=scream, secs=dt)


def main():
    t_all = time.perf_counter()

    # V1+V2a: the finding's own MAX world (seed 4243) — tie to its numbers
    mx0 = genesis_plus_gate("max_rel_4243", lambda a, b: max(a, b),
                            world_seed=4243, pool_seed=500)
    # V1: MAX at a NEW seed — does the break survive?
    mx1 = genesis_plus_gate("max_rel_5243", lambda a, b: max(a, b),
                            world_seed=5243, pool_seed=510)
    # V3: ADD exact-road control at the finding's seed
    ad0 = genesis_plus_gate("add_rel_4242", lambda a, b: a + b,
                            world_seed=4242, pool_seed=520)

    print("\n──── VERIFIER SHEET ────")
    pv1 = (mx1["verdict"] == "UNDERSTOOD" and 0.99 < mx1["r2"] < 0.9999
           and 0.02 < mx1["train_err"] < 0.5)
    print(f"  PV1 break survives new seed: {pv1} "
          f"(verdict {mx1['verdict']}, r2 {mx1['r2']:.5f}, "
          f"train {mx1['train_err']:.2e})")
    pv2 = bool(mx0["scream"]) and bool(mx1["scream"])
    print(f"  PV2 pushed residual gate screams on both MAX worlds: {pv2} "
          f"(4243: {mx0['cap']:+.3f} vs {mx0['thr']:+.3f} "
          f"scream={mx0['scream']}; 5243: {mx1['cap']:+.3f} vs "
          f"{mx1['thr']:+.3f} scream={mx1['scream']})")
    pv3 = ad0["resid_max"] < 1e-9
    print(f"  PV3 ADD exact road (resid machine-zero): {pv3} "
          f"(max resid {ad0['resid_max']:.2e})")
    dt = time.perf_counter() - t_all
    print(f"  TOTAL {dt:.1f}s")


if __name__ == "__main__":
    main()
