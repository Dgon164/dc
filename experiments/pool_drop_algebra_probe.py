"""pool_drop_algebra_probe.py — THE DROP ALGEBRA: junction types from
ablation patterns (owner framing, 2026-07-15: "from an algebra
structure this makes sense... in different things with different
combinations you can tell which things are more important and
reliable... the pools are just functions that transform the answers,
and you compare how parts get transformed compared to base to see how
they abate"; assistant sharpening, offered not settled until this
probe: ablation drops measure NECESSITY, not ownership — and the
deviation from additivity is itself the diagnostic).

THE CLAIM UNDER TEST: for a pair of contributing channels (i, j),
define the junction statistic S = d_i + d_j - d_ij, where d_i is the
solo ablation capture-drop and d_ij the pair drop (both permuted
together). Then, against a matched-decoy null band:
  S ~ 0        -> ADDITIVE junction (independent contributions)
  S >> 0       -> SYNERGY (jointly necessary; solos each carry the
                  whole signal's death warrant — superadditive solos)
  S << 0       -> REDUNDANCY (either suffices; solos small because
                  the twin covers, pair drop large — subadditive)
The classifier runs BEFORE enumeration and can only inform search
ORDER, never admission (the Fisher law: informers do not judge).

PREDICTIONS (filed before the first run)
  P1 ADDITIVE world (y = 1.2*x0 + 0.8*x2 + eps): {x0, x2} named by
     solo drops; S(0,2) inside the null band -> ADDITIVE.
  P2 SYNERGY world (y = 0.9*x1*x3 + eps): solo drops each ~ capture;
     S(1,3) above the band -> SYNERGY.
  P3 REDUNDANCY world (x4 = x1 + tiny noise; y = x1 + eps): solo
     drops of the twins SMALL (prediction: solo ablation may be
     BLIND to twins — the known correlated-importance hole, filed as
     a wanted finding either way), pair drop ~ capture; S(1,4) below
     the band -> REDUNDANCY.
  P4 distractor channels: never named, in any world (0 false names).
  P5 full probe < 90 s here.

NULL (matched, the scrambled-pairing move): per round, two seeded
decoy channels ride the same fit; the null band for S pools the
decoy-decoy pair and each planted-channel x decoy pair (a real
channel and a decoy cannot interact, so their S is a true zero
sample). Band = max |S_null| (conservative; thin-sample caveat filed).

RUN-1 WRONG-KEPT (probe design bugs, cured in run 2 — recorded):
  - a shared twin channel (x4 = x1 + noise) contaminated ALL three
    worlds: ghost namings at the 0.01 level in the additive world and
    a twin-shielded solo drop in the synergy world. Worlds are now
    ISOLATED (each draws its own X; the twin exists only in the
    redundancy world).
  - S carries a small SATURATION/MIXING bias on strong additive
    signals (measured -0.076 on a ~1.9 pair scale, ~4%: tanh features
    mix channels, so pair ablation interferes slightly), invisible to
    the no-signal decoy null. Cure is MEASURED, not hand-set: a
    planted known-additive CALIBRATION pair inside the probe measures
    the bias fraction, and the classification margin = decoy band +
    2x the calibrated bias at the tested pair's scale.

INTEGRITY: instrument imported unchanged from
reservoir_surprise_probe.py (13/13); pools point, never answer;
stdlib + numpy; PYTHONHASHSEED=0; fixed seeds.
Run: cd experiments && PYTHONHASHSEED=0 python3 pool_drop_algebra_probe.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from reservoir_surprise_probe import CaptureMeter, scream_verdict  # noqa: E402

N = 400
M_POOL = 800
ROUNDS = 10
REPEATS = 8
NOISE = 0.05


def capture_ablate_cols(meter, X, r, cols, rng):
    """Cross-fit capture with the given columns permuted in each held
    fold (readout fit on intact data) — the multi-column ablation."""
    pred = np.empty(meter.n)
    for tr, ho, mu, sd, _H, M in meter.parts:
        Xp = X[ho].copy()
        for j in cols:
            Xp[:, j] = Xp[rng.permutation(len(ho)), j]
        H = np.tanh(((Xp - mu) / sd) @ meter.net.W + meter.net.b)
        c = r[tr].mean()
        pred[ho] = H @ (M @ (r[tr] - c)) + c
    ss = np.sum((r - pred) ** 2)
    tot = np.sum((r - r.mean()) ** 2)
    return 1.0 - ss / tot if tot > 0 else 0.0


def drop_algebra(X, y, pair, rng_seed):
    """Measure solo drops for all real channels, the pair drop for the
    given pair, and the S null band from matched decoys."""
    rng = np.random.default_rng(rng_seed)
    d_real = X.shape[1]
    solo = np.zeros((ROUNDS, d_real))
    pair_drops, s_null, decoy_solos = [], [], []
    for t in range(ROUNDS):
        D = rng.normal(0.0, 1.0, (len(y), 2))
        Xa = np.column_stack([X, D])
        meter = CaptureMeter(Xa, M_POOL, seed=777 + t)
        cap = meter.capture(y)

        def drop(cols):
            vals = [cap - capture_ablate_cols(meter, Xa, y, cols, rng)
                    for _ in range(REPEATS)]
            return float(np.mean(vals))

        for j in range(d_real):
            solo[t, j] = drop([j])
        d_i, d_j = solo[t, pair[0]], solo[t, pair[1]]
        d_ij = drop(list(pair))
        pair_drops.append(d_ij)
        dec1, dec2 = drop([d_real]), drop([d_real + 1])
        decoy_solos += [dec1, dec2]
        d_dd = drop([d_real, d_real + 1])
        s_null.append(dec1 + dec2 - d_dd)                     # dec x dec
        for p in pair:                                        # real x dec
            d_pd = drop([p, d_real])
            s_null.append(solo[t, p] + dec1 - d_pd)
    contrib = solo.mean(0)
    name_thr = np.percentile(np.abs(decoy_solos), 99)
    named = [j for j in range(d_real) if contrib[j] > name_thr]
    S = float(contrib[pair[0]] + contrib[pair[1]] - np.mean(pair_drops))
    band = float(np.max(np.abs(s_null)))
    kind = ("SYNERGY" if S > band else
            "REDUNDANCY" if S < -band else "ADDITIVE")
    return dict(contrib=contrib, named=named, name_thr=name_thr,
                pair_drop=float(np.mean(pair_drops)), S=S, band=band,
                kind=kind)


def main():
    t0 = time.perf_counter()
    checks = []

    def check(name, ok, detail):
        checks.append(ok)
        print(f"  [{'ok ' if ok else 'FAIL'}] {name} — {detail}")

    rng = np.random.default_rng(11)

    def make_X(twin):
        Xw = rng.uniform(-2, 2, (N, 4))
        extra = (Xw[:, 1] + rng.normal(0, 0.05, N)) if twin \
            else rng.uniform(-2, 2, N)
        return np.column_stack([Xw, extra])

    X_add, X_syn, X_red = make_X(False), make_X(False), make_X(True)
    worlds = {
        "ADDITIVE": (X_add, 1.2 * X_add[:, 0] + 0.8 * X_add[:, 2]
                     + rng.normal(0, NOISE, N), (0, 2)),
        "SYNERGY": (X_syn, 0.9 * X_syn[:, 1] * X_syn[:, 3]
                    + rng.normal(0, NOISE, N), (1, 3)),
        "REDUNDANCY": (X_red, 1.0 * X_red[:, 1]
                       + rng.normal(0, NOISE, N), (1, 4)),
    }

    # CALIBRATION (measured, not hand-set): a known-additive pair at
    # full strength measures the S bias fraction of the instrument.
    X_cal = make_X(False)
    y_cal = 1.0 * X_cal[:, 0] + 1.0 * X_cal[:, 2] \
        + rng.normal(0, NOISE, N)
    r_cal = drop_algebra(X_cal, y_cal, (0, 2), rng_seed=999)
    bias_frac = abs(r_cal["S"]) / max(r_cal["pair_drop"], 1e-9)
    print(f"CALIBRATION additive pair: S {r_cal['S']:+.3f} on pair scale "
          f"{r_cal['pair_drop']:.3f} -> bias fraction {bias_frac:.4f}")

    results = {}
    for wname, (Xw, y, pair) in worlds.items():
        cap, thr, scream, _ = scream_verdict(Xw, y, M_POOL,
                                             rng_seed=len(wname))
        r = drop_algebra(Xw, y, pair, rng_seed=300 + len(results))
        margin = r["band"] + 2.0 * bias_frac * r["pair_drop"]
        r["kind"] = ("SYNERGY" if r["S"] > margin else
                     "REDUNDANCY" if r["S"] < -margin else "ADDITIVE")
        r["margin"] = margin
        results[wname] = (r, cap)
        print(f"W {wname}: capture {cap:+.3f} (null99 {thr:+.3f}, "
              f"{'SCREAM' if scream else 'quiet'})")
        print(f"   contrib {np.round(r['contrib'], 3)} name_thr "
              f"{r['name_thr']:.3f} named={r['named']}")
        print(f"   pair{pair}: d_i+d_j={r['contrib'][pair[0]] + r['contrib'][pair[1]]:+.3f} "
              f"d_ij={r['pair_drop']:+.3f} S={r['S']:+.3f} "
              f"margin ±{r['margin']:.3f} -> {r['kind']}")

    r, cap = results["ADDITIVE"]
    check("P1a additive: parts {0,2} named", r["named"] == [0, 2],
          f"named {r['named']}")
    check("P1b additive: S inside band -> ADDITIVE", r["kind"] == "ADDITIVE",
          f"S {r['S']:+.3f} vs margin ±{r['margin']:.3f}")

    r, cap = results["SYNERGY"]
    check("P2a synergy: S above band -> SYNERGY", r["kind"] == "SYNERGY",
          f"S {r['S']:+.3f} vs margin ±{r['margin']:.3f}")
    check("P2b synergy: solos each carry ~the whole signal",
          min(r["contrib"][1], r["contrib"][3]) > 0.5 * cap,
          f"solos {r['contrib'][1]:.3f}/{r['contrib'][3]:.3f} vs capture "
          f"{cap:.3f}")

    r, cap = results["REDUNDANCY"]
    check("P3a redundancy: S below band -> REDUNDANCY",
          r["kind"] == "REDUNDANCY",
          f"S {r['S']:+.3f} vs margin ±{r['margin']:.3f}")
    check("P3b redundancy: pair drop ~ capture, solos subadditive",
          r["pair_drop"] > 0.5 * cap and
          (r["contrib"][1] + r["contrib"][4]) < r["pair_drop"],
          f"pair {r['pair_drop']:.3f} vs capture {cap:.3f}; solos sum "
          f"{r['contrib'][1] + r['contrib'][4]:.3f}")
    twin_blind = (r["contrib"][1] <= r["name_thr"]
                  or r["contrib"][4] <= r["name_thr"])
    print(f"   [finding] solo-ablation twin visibility: x1 "
          f"{r['contrib'][1]:.3f}, x4 {r['contrib'][4]:.3f} vs name_thr "
          f"{r['name_thr']:.3f} -> {'BLIND (as predicted possible)' if twin_blind else 'visible'}"
          f"; pair ablation recovers them either way")

    false_names = 0
    for wname, (Xw, y, pair) in worlds.items():
        r, _ = results[wname]
        planted = set(pair) if wname != "ADDITIVE" else {0, 2}
        if wname == "REDUNDANCY":
            planted = {1, 4}
        false_names += len([j for j in r["named"] if j not in planted])
    check("P4 distractors never named (all worlds)", false_names == 0,
          f"{false_names} false names")

    dt = time.perf_counter() - t0
    check("P5 cost", dt < 90, f"{dt:.1f}s")
    npass = sum(checks)
    print(f"\n  PROBE: {npass} passed, {len(checks) - npass} failed of "
          f"{len(checks)}  ({dt:.1f}s)")


if __name__ == "__main__":
    main()
