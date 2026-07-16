"""eval_attack_drop_threeway.py — ATTACK D-D: HIGHER-ORDER JUNCTION
(the kinds-rung overclaim guard). Target: pool_drop_algebra_probe.py.

QUESTION: pairwise S = d_i + d_j - d_ij has no vocabulary for a 3-way
junction. On y = 0.8*x0*x1*x2 + eps, does the drop algebra MISLEAD
(false flags, wrong names) or degrade honestly (correct-but-incomplete:
all three inside pairs flagged SYNERGY, distractor pair in band)?

WORLD: N=400, 5 channels U(-2,2); y = 0.8*x0*x1*x2 + N(0, 0.05).
Protocol mirrors the pushed main(): scream_verdict gate, known-additive
calibration world -> bias_frac, margin = band + 2*bias_frac*pair_drop,
kind re-derived with the margin. Full pushed constants (ROUNDS=10,
REPEATS=8, M_POOL=800 -> m_eff 400, N=400); NO shrink.

PREDICTIONS (filed before the first run):
  A0 the pool captures the triple product: SCREAM (capture > null99);
     capture guessed 0.3-0.8 (triple products are hard for tanh
     features, but noise is tiny).
  A1 naming: named == [0, 1, 2] in every drop_algebra call (each solo
     ablation kills a pure triple product); 0 false names among {3,4}.
  A2 pairs (0,1), (0,2), (1,2): S > margin for all three -> SYNERGY.
     Correct-but-incomplete is the honest outcome.
  A3 distractor pair (0,3): S inside +/-margin -> ADDITIVE (in band);
     no false flag on a distractor pair.
  A4 the pairwise output CANNOT distinguish one 3-way junction from a
     genuine 2-way synergy: saturation fingerprint d_i ~= d_ij ~=
     d_012 ~= capture (triple ablation measured here with
     capture_ablate_cols, OUTSIDE the pushed classifier). The pushed
     classifier reports no order-3 statistic, so a reader of its output
     alone sees three pairwise SYNERGY verdicts and nothing naming the
     triple -> scope finding (PARTIAL), not active misleading, unless
     A1 or A3 fail.
  A5 runtime < 180 s.

Run: cd experiments && PYTHONHASHSEED=0 python eval_attack_drop_threeway.py
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from reservoir_surprise_probe import CaptureMeter, scream_verdict  # noqa: E402
from pool_drop_algebra_probe import (  # noqa: E402
    drop_algebra, capture_ablate_cols, M_POOL, NOISE, ROUNDS, REPEATS)

N = 400
D = 5
REAL = (0, 1, 2)


def main():
    t0 = time.perf_counter()
    checks = []

    def check(name, ok, detail):
        checks.append(ok)
        print(f"  [{'ok ' if ok else 'FAIL'}] {name} — {detail}")

    print(f"constants: N={N} D={D} M_POOL={M_POOL} ROUNDS={ROUNDS} "
          f"REPEATS={REPEATS} NOISE={NOISE} (full pushed constants, no shrink)")

    rng = np.random.default_rng(11)
    X = rng.uniform(-2, 2, (N, D))
    y = 0.8 * X[:, 0] * X[:, 1] * X[:, 2] + rng.normal(0, NOISE, N)

    # A0 — scream gate (pushed protocol step 1)
    cap, thr, scream, spread = scream_verdict(X, y, M_POOL, rng_seed=7)
    print(f"A0 capture {cap:+.3f} vs null99 {thr:+.3f} "
          f"(spread {spread:.4f}) -> {'SCREAM' if scream else 'quiet'}")
    check("A0 triple product captured (SCREAM)", bool(scream),
          f"capture {cap:+.3f} vs null99 {thr:+.3f}")

    # CALIBRATION (mirrors pushed main(): known-additive pair -> bias)
    Xc = rng.uniform(-2, 2, (N, D))
    yc = 1.0 * Xc[:, 0] + 1.0 * Xc[:, 2] + rng.normal(0, NOISE, N)
    rc = drop_algebra(Xc, yc, (0, 2), rng_seed=999)
    bias_frac = abs(rc["S"]) / max(rc["pair_drop"], 1e-9)
    print(f"CALIBRATION additive pair: S {rc['S']:+.3f} on pair scale "
          f"{rc['pair_drop']:.3f} -> bias fraction {bias_frac:.4f} "
          f"(cal kind {rc['kind']}, band {rc['band']:.3f})")

    # THE FOUR PAIRS
    pairs = [(0, 1), (0, 2), (1, 2), (0, 3)]
    results = {}
    for i, pr in enumerate(pairs):
        t1 = time.perf_counter()
        r = drop_algebra(X, y, pr, rng_seed=300 + i)
        margin = r["band"] + 2.0 * bias_frac * r["pair_drop"]
        kind = ("SYNERGY" if r["S"] > margin else
                "REDUNDANCY" if r["S"] < -margin else "ADDITIVE")
        results[pr] = (r, margin, kind)
        dsum = r["contrib"][pr[0]] + r["contrib"][pr[1]]
        print(f"pair {pr}: contrib {np.round(r['contrib'], 3)} "
              f"name_thr {r['name_thr']:.3f} named={r['named']}")
        print(f"   d_i+d_j={dsum:+.3f} d_ij={r['pair_drop']:+.3f} "
              f"S={r['S']:+.3f} band ±{r['band']:.3f} margin ±{margin:.3f} "
              f"-> {kind}   ({time.perf_counter() - t1:.1f}s)")

    # A1 — naming, every call
    named_ok = all(results[pr][0]["named"] == [0, 1, 2] for pr in pairs)
    false_names = sum(len([j for j in results[pr][0]["named"]
                           if j not in REAL]) for pr in pairs)
    check("A1 naming == [0,1,2] in all 4 calls, 0 false names",
          named_ok and false_names == 0,
          f"named per call {[results[pr][0]['named'] for pr in pairs]}, "
          f"{false_names} false names")

    # A2 — three real pairs all SYNERGY
    real_pairs = [(0, 1), (0, 2), (1, 2)]
    kinds = {pr: results[pr][2] for pr in real_pairs}
    check("A2 all three inside pairs -> SYNERGY",
          all(k == "SYNERGY" for k in kinds.values()),
          "; ".join(f"S{pr}={results[pr][0]['S']:+.3f} vs margin "
                    f"±{results[pr][1]:.3f} {results[pr][2]}"
                    for pr in real_pairs))

    # A3 — distractor pair in band
    r03, m03, k03 = results[(0, 3)]
    check("A3 distractor pair (0,3) in band -> ADDITIVE", k03 == "ADDITIVE",
          f"S(0,3)={r03['S']:+.3f} vs margin ±{m03:.3f} -> {k03}")

    # A4 — saturation fingerprint + vocabulary audit (OUTSIDE the pushed
    # classifier: triple ablation via the same exported ablation kernel)
    col_sets = [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]
    drops = {cs: [] for cs in col_sets}
    for s in range(3):  # 3 meter seeds for stability
        meter = CaptureMeter(X, M_POOL, seed=777 + s)
        cap_m = meter.capture(y)
        rng_a = np.random.default_rng(42 + s)
        for cs in col_sets:
            vals = [cap_m - capture_ablate_cols(meter, X, y, list(cs), rng_a)
                    for _ in range(REPEATS)]
            drops[cs].append(float(np.mean(vals)))
    dmean = {cs: float(np.mean(v)) for cs, v in drops.items()}
    d012 = dmean[(0, 1, 2)]
    print("A4 fingerprint (mean over 3 meter seeds x 8 repeats):")
    print("   " + "  ".join(f"d{cs}={dmean[cs]:+.3f}" for cs in col_sets))
    solo_sat = max(abs(dmean[(j,)] - d012) for j in REAL)
    pair_sat = max(abs(dmean[cs] - d012) for cs in [(0, 1), (0, 2), (1, 2)])
    S3 = dmean[(0,)] + dmean[(1,)] + dmean[(2,)] - d012
    print(f"   saturation: max|d_solo - d_012| = {solo_sat:.3f}, "
          f"max|d_pair - d_012| = {pair_sat:.3f} (scale d_012 {d012:+.3f})")
    print(f"   order-3 statistic the classifier never computes: "
          f"S3 = d0+d1+d2-d012 = {S3:+.3f} (~2x capture expected for a "
          f"pure triple; a genuine single 2-way synergy would give "
          f"S3 ~= 1x its pair scale)")
    saturated = solo_sat < 0.25 * max(d012, 1e-9) \
        and pair_sat < 0.25 * max(d012, 1e-9)
    check("A4 drops saturate (d_i ~= d_ij ~= d_012): pairwise output "
          "indistinguishable from genuine 2-way synergy", saturated,
          f"solo_sat {solo_sat:.3f}, pair_sat {pair_sat:.3f} vs d_012 "
          f"{d012:+.3f}; classifier vocabulary = {{ADDITIVE, SYNERGY, "
          f"REDUNDANCY}} on PAIRS only, no order-3 term")

    misleads = (not (named_ok and false_names == 0)) or k03 != "ADDITIVE"
    print(f"   ACTIVE-MISLEADING verdict: "
          f"{'YES (false name or distractor pair flagged)' if misleads else 'NO (correct-but-incomplete; scope hole only)'}")

    dt = time.perf_counter() - t0
    check("A5 cost < 180 s", dt < 180, f"{dt:.1f}s")
    npass = sum(checks)
    print(f"\n  ATTACK D-D: {npass} passed, {len(checks) - npass} failed "
          f"of {len(checks)}  ({dt:.1f}s)")


if __name__ == "__main__":
    main()
