"""eval_attack_drop_mixed.py — ATTACK D-C: MIXED JUNCTIONS IN ONE WORLD.

Target: pool_drop_algebra_probe.py (drop_algebra; S = d_i + d_j - d_ij
against a calibrated margin). All three pushed worlds were PURE (one
junction type per world). This attack plants BOTH junction types in a
single world — the realistic composite case:

    y = 0.9*x1*x3 + 1.0*x0 + eps    (N=400, NOISE=0.05, 5 channels,
                                     no twin channel)

so pair (1,3) is a true SYNERGY junction while pairs (0,1) and (0,3)
are true ADDITIVE junctions (x0 does not interact with either factor).
Distractors are x2 and x4.

MARGIN CALIBRATION mirrors the pushed main() arithmetic exactly: a
separate known-additive world (y = 1.0*x0 + 1.0*x2 + eps, drawn from
the same rng stream) measures bias_frac = |S_cal| / pair_drop_cal, and
each tested pair's margin = its own decoy band
+ 2 * bias_frac * its own pair_drop.

PREDICTIONS (filed BEFORE the first run):
  P1 naming: named == [0, 1, 3] in every drop_algebra call; x2/x4
     never named (0 false names).
  P2 pair (1,3): S > margin -> SYNERGY.
  P3 pair (0,1): |S| <= margin -> ADDITIVE.
  P4 pair (0,3): |S| <= margin -> ADDITIVE.
  P5 runtime < 180 s (4 drop_algebra calls — the same count as the
     pushed probe's own main()).
  RISK FILED: the mixed world's tanh saturation/mixing bias could
  exceed the pure-world calibration (two signal terms of comparable
  variance share the same features); if a mixed pair's |S| lands just
  outside its margin the classifier mislabels an ADDITIVE junction ->
  BROKEN per the grading rules.

Deterministic: PYTHONHASHSEED=0; world rng seed 11 (probe convention),
calibration drop seed 999, pair drop seeds 300/301/302 (mirroring the
pushed main()'s 300 + index).
Run: cd experiments && PYTHONHASHSEED=0 python eval_attack_drop_mixed.py
"""
import time

import numpy as np

from pool_drop_algebra_probe import M_POOL, N, NOISE, drop_algebra
from reservoir_surprise_probe import scream_verdict


def main():
    t0 = time.perf_counter()
    rng = np.random.default_rng(11)

    def make_X():
        Xw = rng.uniform(-2, 2, (N, 4))
        extra = rng.uniform(-2, 2, N)
        return np.column_stack([Xw, extra])

    # THE MIXED WORLD: synergy pair (1,3) + additive lone channel x0.
    X = make_X()
    y = 0.9 * X[:, 1] * X[:, 3] + 1.0 * X[:, 0] + rng.normal(0, NOISE, N)

    # CALIBRATION (mirrors the pushed main(): known-additive pair at
    # full strength measures the instrument's S bias fraction).
    X_cal = make_X()
    y_cal = 1.0 * X_cal[:, 0] + 1.0 * X_cal[:, 2] + rng.normal(0, NOISE, N)
    r_cal = drop_algebra(X_cal, y_cal, (0, 2), rng_seed=999)
    bias_frac = abs(r_cal["S"]) / max(r_cal["pair_drop"], 1e-9)
    print(f"CALIBRATION additive pair: S {r_cal['S']:+.3f} on pair scale "
          f"{r_cal['pair_drop']:.3f} -> bias fraction {bias_frac:.4f}")

    cap, thr, scream, _ = scream_verdict(X, y, M_POOL, rng_seed=5)
    print(f"MIXED world capture {cap:+.3f} (null99 {thr:+.3f}, "
          f"{'SCREAM' if scream else 'quiet'})")

    expected = {(1, 3): "SYNERGY", (0, 1): "ADDITIVE", (0, 3): "ADDITIVE"}
    checks = []

    def check(name, ok, detail):
        checks.append(ok)
        print(f"  [{'ok ' if ok else 'FAIL'}] {name} — {detail}")

    results = {}
    for i, pair in enumerate([(1, 3), (0, 1), (0, 3)]):
        r = drop_algebra(X, y, pair, rng_seed=300 + i)
        margin = r["band"] + 2.0 * bias_frac * r["pair_drop"]
        r["kind"] = ("SYNERGY" if r["S"] > margin else
                     "REDUNDANCY" if r["S"] < -margin else "ADDITIVE")
        r["margin"] = margin
        results[pair] = r
        di, dj = r["contrib"][pair[0]], r["contrib"][pair[1]]
        print(f"PAIR {pair}: contrib {np.round(r['contrib'], 3)} "
              f"name_thr {r['name_thr']:.3f} named={r['named']}")
        print(f"   d_i+d_j={di + dj:+.3f} d_ij={r['pair_drop']:+.3f} "
              f"S={r['S']:+.3f} margin ±{r['margin']:.3f} "
              f"(band {r['band']:.3f}) -> {r['kind']}")
        check(f"P junction {pair} = {expected[pair]}",
              r["kind"] == expected[pair],
              f"S {r['S']:+.3f} vs margin ±{margin:.3f} -> {r['kind']}")
        check(f"P1 naming at pair {pair}: named == [0,1,3]",
              r["named"] == [0, 1, 3], f"named {r['named']}")

    false_names = sum(len([j for j in r["named"] if j not in {0, 1, 3}])
                      for r in results.values())
    check("P1b distractors x2/x4 never named (all calls)",
          false_names == 0, f"{false_names} false names")

    dt = time.perf_counter() - t0
    check("P5 runtime < 180 s", dt < 180, f"{dt:.1f}s")
    npass = sum(checks)
    print(f"\nATTACK D-C: {npass} passed, {len(checks) - npass} failed "
          f"of {len(checks)}  ({dt:.1f}s)")


if __name__ == "__main__":
    main()
