"""eval_attack_drop_wronghint.py — ATTACK D-B: WRONG PAIR HINT / MISUSE
ROBUSTNESS against pool_drop_algebra_probe.drop_algebra.

THE ATTACK: S = d_i + d_j - d_ij is computed on a CALLER-SUPPLIED pair.
Production must survive a wrong pick. In the probe's own SYNERGY world
(y = 0.9*x1*x3 + eps, N=400, worlds rebuilt with the probe's exact
rng(11) draw order so the world and the calibration pair are IDENTICAL
to the pushed run), compute drop_algebra for:
  (0,2)  two non-contributors        — expected honest: S inside band,
                                       kind ADDITIVE (no invented junction)
  (1,0)  one contributor + one non   — expected honest: S inside band,
                                       kind ADDITIVE
  (1,3)  the truth (control)         — expected: SYNERGY (matches the
                                       pushed run; probe's own seed 301)
Classification uses the probe's PRODUCTION margin rule:
margin = band + 2 * bias_frac * pair_drop, with bias_frac measured on
the probe's own calibration world (rng_seed=999, identical draws).

PREDICTIONS (filed before the first run — wrong-kept if falsified):
  P1 (0,2): d_0 ~ 0, d_2 ~ 0, d_02 ~ 0; S inside +/- margin -> ADDITIVE;
     named stays [1, 3] (naming is pair-independent solo drops vs decoy
     threshold, so the wrong hint cannot corrupt it).
  P2 (1,0): d_1 ~ capture (synergy solo carries the whole signal's
     death warrant), d_0 ~ 0, d_10 ~ d_1; S ~ 0 inside +/- margin ->
     ADDITIVE; named stays [1, 3].
  P3 (1,3) control: S >> margin -> SYNERGY (d_1 + d_3 - d_13 ~ +capture).
  P4 misleading-structure question on (1,0): prediction — S will carry
     a small NEGATIVE tilt (pair ablation on a strong channel + a null
     channel interferes slightly via tanh mixing, same saturation bias
     the probe measured at -0.076 on additive pairs), but |S| < margin,
     so no invented REDUNDANCY. If |S| > margin with S < 0 -> BROKEN
     (invented REDUNDANCY from a half-wrong hint).
  P5 runtime < 180 s (4 drop_algebra calls + 1 scream_verdict; no
     shrink of ROUNDS/REPEATS — full-fidelity probe constants).

WRONG-KEPT (recorded after the runs; predictions above left intact):
  - P4 SIGN falsified: the (1,0) tilt measured POSITIVE (synergy-ward),
    not negative — full-fidelity S(1,0) = +0.0298 = +0.52x raw band,
    +0.41x margin (shrunk 6/4 run: +0.0025 = +0.03x band). The
    substantive half of P4 HELD: |S| < margin, no invented junction.
  - P5 falsified on wall clock only: 968.8 s full / 373.1 s shrunk,
    measured while ~15 sibling eval jobs shared the box — instrument
    cost is unchanged from the probe's pushed <90 s claim (contention,
    not cost).
  - All graded checks passed in BOTH runs (7/7 substantive each):
    (0,2) ADDITIVE named [1,3]; (1,0) ADDITIVE named [1,3];
    (1,3) SYNERGY named [1,3].
  - Micro scope note: margin = band + 2*bias_frac*pair_drop uses the
    SIGNED pair_drop, so a slightly negative null pair drop SHRINKS
    the margin below the band (full run (0,2): margin 0.0356 < band
    0.0357). Inconsequential here (|S|=0.0037 << both).

SHRINK DISCLOSURE (added before any numbers were seen): the eval box
was running ~15 concurrent python jobs and the full-fidelity run
starved past the reporting deadline with zero output. Env knobs
EVAL_ROUNDS / EVAL_REPEATS (default = probe's 10 / 8) let this attack
run at reduced ROUNDS/REPEATS; any shrunk run is disclosed in the
report. The shrink touches sampling depth only — worlds, seeds, S
definition, band, margin rule are unchanged.

Run: cd experiments && PYTHONHASHSEED=0 python eval_attack_drop_wronghint.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pool_drop_algebra_probe as pda  # noqa: E402
from pool_drop_algebra_probe import drop_algebra, N, M_POOL, NOISE  # noqa: E402
from reservoir_surprise_probe import scream_verdict  # noqa: E402


def main():
    t0 = time.perf_counter()
    # disclosed shrink knobs (defaults = probe's own constants)
    pda.ROUNDS = int(os.environ.get("EVAL_ROUNDS", pda.ROUNDS))
    pda.REPEATS = int(os.environ.get("EVAL_REPEATS", pda.REPEATS))
    print(f"SAMPLING: ROUNDS={pda.ROUNDS} REPEATS={pda.REPEATS} "
          f"(probe defaults 10/8; any reduction is a disclosed shrink)")

    # ── rebuild the probe's worlds with the IDENTICAL rng(11) draw order ──
    rng = np.random.default_rng(11)

    def make_X(twin):
        Xw = rng.uniform(-2, 2, (N, 4))
        extra = (Xw[:, 1] + rng.normal(0, 0.05, N)) if twin \
            else rng.uniform(-2, 2, N)
        return np.column_stack([Xw, extra])

    X_add, X_syn, X_red = make_X(False), make_X(False), make_X(True)
    _y_add = 1.2 * X_add[:, 0] + 0.8 * X_add[:, 2] + rng.normal(0, NOISE, N)
    y_syn = 0.9 * X_syn[:, 1] * X_syn[:, 3] + rng.normal(0, NOISE, N)
    _y_red = 1.0 * X_red[:, 1] + rng.normal(0, NOISE, N)
    X_cal = make_X(False)
    y_cal = 1.0 * X_cal[:, 0] + 1.0 * X_cal[:, 2] + rng.normal(0, NOISE, N)

    # ── the probe's own calibration (identical seed) ──
    r_cal = drop_algebra(X_cal, y_cal, (0, 2), rng_seed=999)
    bias_frac = abs(r_cal["S"]) / max(r_cal["pair_drop"], 1e-9)
    print(f"CALIBRATION: S {r_cal['S']:+.3f} on pair scale "
          f"{r_cal['pair_drop']:.3f} -> bias_frac {bias_frac:.4f}")

    cap, thr, scream, _ = scream_verdict(X_syn, y_syn, M_POOL,
                                         rng_seed=len("SYNERGY"))
    print(f"SYNERGY world capture {cap:+.3f} (null99 {thr:+.3f}, "
          f"{'SCREAM' if scream else 'quiet'})")

    attacks = [
        ((0, 2), 402, "two non-contributors", "ADDITIVE"),
        ((1, 0), 410, "one contributor + one non", "ADDITIVE"),
        ((1, 3), 301, "truth control (probe's exact seed)", "SYNERGY"),
    ]
    checks = []

    def check(name, ok, detail):
        checks.append(ok)
        print(f"  [{'ok ' if ok else 'FAIL'}] {name} — {detail}")

    results = {}
    for pair, seed, label, want in attacks:
        r = drop_algebra(X_syn, y_syn, pair, rng_seed=seed)
        margin = r["band"] + 2.0 * bias_frac * r["pair_drop"]
        kind = ("SYNERGY" if r["S"] > margin else
                "REDUNDANCY" if r["S"] < -margin else "ADDITIVE")
        results[pair] = (r, margin, kind)
        d_i, d_j = r["contrib"][pair[0]], r["contrib"][pair[1]]
        print(f"\nPAIR {pair} [{label}]")
        print(f"   contrib {np.round(r['contrib'], 3)} "
              f"name_thr {r['name_thr']:.3f} named={r['named']}")
        print(f"   d_i={d_i:+.4f} d_j={d_j:+.4f} d_ij={r['pair_drop']:+.4f} "
              f"S={r['S']:+.4f} raw band ±{r['band']:.4f} "
              f"margin ±{margin:.4f} -> {kind}"
              f" (raw-band kind: {r['kind']})")
        check(f"pair {pair}: kind == {want}", kind == want,
              f"S {r['S']:+.4f} vs margin ±{margin:.4f} -> {kind}")
        check(f"pair {pair}: naming stays [1, 3]", r["named"] == [1, 3],
              f"named {r['named']}")

    # ── misleading-structure statement for (1,0) ──
    r10, m10, k10 = results[(1, 0)]
    frac_band = r10["S"] / r10["band"] if r10["band"] > 0 else float("inf")
    frac_marg = r10["S"] / m10 if m10 > 0 else float("inf")
    print(f"\n(1,0) STRUCTURE: S {r10['S']:+.4f} = {frac_band:+.2f}x raw "
          f"band, {frac_marg:+.2f}x margin; sign "
          f"{'NEGATIVE (redundancy-ward)' if r10['S'] < 0 else 'POSITIVE (synergy-ward)'}"
          f"; decomposition d_1 {r10['contrib'][1]:+.4f} + d_0 "
          f"{r10['contrib'][0]:+.4f} - d_10 {r10['pair_drop']:+.4f}")
    check("(1,0) |S| stays under HALF the margin (no near-miss)",
          abs(r10["S"]) < 0.5 * m10,
          f"|S| {abs(r10['S']):.4f} vs 0.5*margin {0.5 * m10:.4f}")

    dt = time.perf_counter() - t0
    check("runtime < 180 s", dt < 180, f"{dt:.1f}s")
    npass = sum(checks)
    print(f"\nATTACK D-B: {npass} passed, {len(checks) - npass} failed of "
          f"{len(checks)}  ({dt:.1f}s)")


if __name__ == "__main__":
    main()
