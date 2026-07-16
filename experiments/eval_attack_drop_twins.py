"""eval_attack_drop_twins.py — ATTACK D-A: TIGHTER TWINS.

The drop-algebra probe's OWN filed attack ("tighter twins go on the
attack list"). Target claim (pool_drop_algebra_probe.py, P3):
REDUNDANCY world (x4 = x1 + twin_noise; y = x1 + 0.05*eps, N=400)
classifies as REDUNDANCY via S = d_1 + d_4 - d_14 < -margin, where
margin = decoy band + 2 * calibrated bias fraction * pair scale.
At the pushed twin noise 0.05 the probe measured solo visibility
0.92/0.21 (both twins named). Here we tighten: noise in
{0.02, 0.01, 0.005}.

INSTRUMENT: drop_algebra imported UNCHANGED from
pool_drop_algebra_probe (full ROUNDS=10, REPEATS=8 — no shrink was
needed; runtime measured and reported). Calibration additive pair
replicated exactly as the probe does it (rng stream seeded 11, cal
world drawn the same way, rng_seed=999) so the margin rule matches
the pushed classifier.

PREDICTIONS (filed BEFORE the first run — kept verbatim, wrong or not)
  Q1 classification: REDUNDANCY (S < -margin) at ALL three levels;
     tightening the twin should make S MORE negative (solos shrink,
     pair drop stays ~capture), so the junction call gets stronger,
     not weaker. Misclassification as ADDITIVE/SYNERGY = BROKEN.
  Q2 solo naming: drops shrink as the twin tightens; at 0.02 both
     twins still named; by 0.005 at least one twin falls below
     name_thr (solo-naming BLINDNESS — the hole the original probe
     predicted as possible). The boundary is the finding.
  Q3 pair drop stays ~ capture (> 0.5 * capture) at every level.
  Q4 distractors {0, 2, 3}: never named at any level (0 false names).
  Q5 runtime < 180 s (4 drop_algebra calls, same count as the
     original probe's <90 s run).

Run: cd experiments && PYTHONHASHSEED=0 python eval_attack_drop_twins.py

MEASURED (run 1, 2026-07-15) — predictions above kept verbatim:
  Q1 HELD: REDUNDANCY at all levels. S = -0.460 / -0.457 / -0.489 vs
     -margin -0.117 / -0.120 / -0.132 (band 0.027/0.034/0.042; cal
     bias fraction 0.0302 on pair scale 1.887). Band-only rule agrees.
  Q2 HELD at 0.02 (x1 0.685 / x4 0.348 > thr 0.001, both named).
  Q2b WRONG-KEPT: NO solo blindness down to noise 0.005 (corr(x1,x4)
     0.999990): solos 0.517/0.486 vs name_thr 0.001 — the twins'
     solo drops SYMMETRIZE (~0.49/0.49) instead of vanishing, because
     tanh units mix all channels, so permuting one twin still corrupts
     every feature. The original probe's "possible blind" hole did NOT
     open in this range. Pair drops 1.494/1.433/1.492 vs capture
     0.996/0.996/0.997 (subadditive: solo sums 1.033/0.977/1.003).
  Q3, Q4 HELD (0 false names; named=[1,4] at every level).
  Q5 WRONG-KEPT: 966.9 s on this box (predicted <180 s). No shrink was
     applied (full ROUNDS=10/REPEATS=8). This box is ~10x slower than
     the original probe's "<90 s here" machine — its P5 would also
     fail HERE (machine-speed scope note, not an instrument fault).
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..")))

from pool_drop_algebra_probe import drop_algebra, N, NOISE, M_POOL  # noqa: E402
from reservoir_surprise_probe import scream_verdict                 # noqa: E402

TWIN_LEVELS = (0.02, 0.01, 0.005)


def main():
    t0 = time.perf_counter()
    checks = []

    def check(name, ok, detail):
        checks.append(ok)
        print(f"  [{'ok ' if ok else 'FAIL'}] {name} — {detail}")

    # ── CALIBRATION: same recipe as the pushed probe (measured margin) ──
    rng = np.random.default_rng(11)
    Xc = rng.uniform(-2, 2, (N, 4))
    X_cal = np.column_stack([Xc, rng.uniform(-2, 2, N)])
    y_cal = 1.0 * X_cal[:, 0] + 1.0 * X_cal[:, 2] + rng.normal(0, NOISE, N)
    r_cal = drop_algebra(X_cal, y_cal, (0, 2), rng_seed=999)
    bias_frac = abs(r_cal["S"]) / max(r_cal["pair_drop"], 1e-9)
    print(f"CALIBRATION additive pair: S {r_cal['S']:+.3f} on pair scale "
          f"{r_cal['pair_drop']:.3f} -> bias fraction {bias_frac:.4f}")

    results = {}
    for k, tw in enumerate(TWIN_LEVELS):
        wrng = np.random.default_rng(20260715 + k)
        Xw = wrng.uniform(-2, 2, (N, 4))
        extra = Xw[:, 1] + wrng.normal(0.0, tw, N)
        X = np.column_stack([Xw, extra])
        y = 1.0 * X[:, 1] + wrng.normal(0, NOISE, N)
        corr = float(np.corrcoef(X[:, 1], X[:, 4])[0, 1])

        cap, thr, scream, _ = scream_verdict(X, y, M_POOL, rng_seed=50 + k)
        r = drop_algebra(X, y, (1, 4), rng_seed=400 + k)
        margin = r["band"] + 2.0 * bias_frac * r["pair_drop"]
        kind_margin = ("SYNERGY" if r["S"] > margin else
                       "REDUNDANCY" if r["S"] < -margin else "ADDITIVE")
        results[tw] = (r, cap, margin, kind_margin, corr)

        print(f"\nTWIN NOISE {tw}: corr(x1,x4) {corr:.6f}  capture "
              f"{cap:+.3f} (null99 {thr:+.3f}, "
              f"{'SCREAM' if scream else 'quiet'})")
        print(f"   contrib {np.round(r['contrib'], 3)}  name_thr "
              f"{r['name_thr']:.3f}  named={r['named']}")
        print(f"   pair(1,4): d1={r['contrib'][1]:+.3f} d4={r['contrib'][4]:+.3f} "
              f"d14={r['pair_drop']:+.3f}  S={r['S']:+.3f}  band ±{r['band']:.3f} "
              f"margin ±{margin:.3f}")
        print(f"   kind(band-only)={r['kind']}  kind(margin)={kind_margin}")

    # ── CHECKS against the filed predictions ──
    all_red = all(results[tw][3] == "REDUNDANCY" for tw in TWIN_LEVELS)
    check("Q1 classification REDUNDANCY at all levels", all_red,
          "; ".join(f"{tw}: S {results[tw][0]['S']:+.3f} vs -margin "
                    f"{-results[tw][2]:.3f} -> {results[tw][3]}"
                    for tw in TWIN_LEVELS))

    blind = {tw: (results[tw][0]["contrib"][1] <= results[tw][0]["name_thr"],
                  results[tw][0]["contrib"][4] <= results[tw][0]["name_thr"])
             for tw in TWIN_LEVELS}
    check("Q2 solo naming at 0.02 (both twins named, per prediction)",
          not any(blind[0.02]),
          f"x1 {results[0.02][0]['contrib'][1]:.3f} / x4 "
          f"{results[0.02][0]['contrib'][4]:.3f} vs thr "
          f"{results[0.02][0]['name_thr']:.3f}")
    check("Q2b solo blindness reached by 0.005 (per prediction)",
          any(blind[0.005]),
          f"x1 {results[0.005][0]['contrib'][1]:.3f} / x4 "
          f"{results[0.005][0]['contrib'][4]:.3f} vs thr "
          f"{results[0.005][0]['name_thr']:.3f}")
    for tw in TWIN_LEVELS:
        b1, b4 = blind[tw]
        print(f"   [boundary] twin {tw}: x1 "
              f"{'BLIND' if b1 else 'visible'}, x4 "
              f"{'BLIND' if b4 else 'visible'}")

    check("Q3 pair drop ~ capture at all levels",
          all(results[tw][0]["pair_drop"] > 0.5 * results[tw][1]
              for tw in TWIN_LEVELS),
          "; ".join(f"{tw}: d14 {results[tw][0]['pair_drop']:.3f} vs cap "
                    f"{results[tw][1]:.3f}" for tw in TWIN_LEVELS))

    false_names = sum(len([j for j in results[tw][0]["named"]
                           if j not in (1, 4)]) for tw in TWIN_LEVELS)
    check("Q4 distractors never named", false_names == 0,
          f"{false_names} false names across "
          f"{[results[tw][0]['named'] for tw in TWIN_LEVELS]}")

    dt = time.perf_counter() - t0
    check("Q5 cost", dt < 180, f"{dt:.1f}s")
    npass = sum(checks)
    print(f"\n  ATTACK D-A: {npass} passed, {len(checks) - npass} failed "
          f"of {len(checks)}  ({dt:.1f}s)")


if __name__ == "__main__":
    main()
