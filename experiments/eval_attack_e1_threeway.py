"""eval_attack_e1_threeway.py — ATTACK E-C: THREE-WAY CONJUNCTION +
THE ACCURACY-VS-BASE-RATE TRAP, against pool_conjunctive_frontier_probe.

WORLD (per seed): fail iff (voice=passive AND tense=perfect AND
deco=exclaim), then FLIP=5% verdict noise (probe convention).
N=900 (bigger than the probe's 600 because the positive class is rare:
true fail rate = 1/2 * 1/2 * 1/3 ~ 8.33%, observed after flips ~12.5%
— DISCLOSED). Two seeds (2000, 2001). Funnel shrunk from the probe's
ROUNDS=6/REPEATS=6 to ROUNDS=4/REPEATS=4 for runtime — DISCLOSED; the
naming threshold (99th pct of |decoy drops|) therefore sits on 4
values ~ the max, and the S band on 12 s_null values instead of 12+.
All instruments imported UNCHANGED (CaptureMeter, scream_verdict,
ablate_groups, onehot, make_events); no existing file modified.

PREDICTIONS (filed before the first run):
  P1 SCREAM: the pool screams on the one-hot table both seeds
     (capture well past the permuted-y null99).
  P2 NAMING: grouped ablation names exactly {deco, tense, voice} both
     seeds; a miss on any of the three = boundary finding.
  P3 PAIRWISE ALGEBRA: all three pairs (voice,tense), (voice,deco),
     (tense,deco) come out SYNERGY (S > band) both seeds —
     correct-but-incomplete: the true story is 3-way, the pairwise
     instrument can only say "jointly necessary" per pair.
  P4 THE TRAP (2x2 on voice,tense ONLY — the probe's E5 road):
     cell (passive,perfect) train fail rate ~0.35 < 0.5, all other
     cells ~0.05, so rule_cells = EMPTY -> the articulated rule is
     all-pass, recovering 0 truth cells. Held-out accuracy of that
     empty rule: ~0.92 vs TRUE labels (BEATS the 0.90 gate while
     recovering NOTHING — the base-rate trap is real vs truth);
     ~0.875 vs NOISY labels (the probe's own convention) — I predict
     the noisy-label acc lands BELOW 0.90 both seeds, so the pushed
     gate as-implemented does NOT close this world at FLIP=0.05, but
     only by ~2.5 points, ~1.1 sd of a 225-row holdout — sampling
     luck could flip it (if it flips, that is the trap firing in
     full, kept either way).
  P5 HONEST 2x2x2 (voice,tense,deco): recovers exactly the single
     AND cell {(passive,perfect,exclaim)} (train rate ~0.95) and
     held-out acc ~0.95 vs noisy labels, beating the all-pass base
     rate ~0.875, both seeds.
  P6 MINER NOT BLIND HERE: unlike XOR, a rare AND has marginal
     footprints (P(fail|passive)~16.7% vs P(fail|active)~0);
     mine_commonality names voice=passive, tense=perfect, and
     deco=exclaim among its named list both seeds.
  P7 INNOCENTS: subj/verb/junk never named by the funnel either seed.

WRONG-KEPT / MEASURED (run 1, 2026-07-15; the filed text above is
never edited): NO prediction falsified — 8/8 checks passed both
seeds. The trap numbers, filed hard: the 2x2 (voice,tense) road
recovered 0/1 truth cells (max cell rate 0.380/0.349 < 0.5 -> empty
rule = all-pass) yet scored 0.9111/0.9067 vs TRUE labels — past the
0.90 gate while recovering NOTHING. Vs the probe's own noisy-label
convention it scored 0.8800/0.8667 (< 0.90) ONLY because 5% flips
inflate the observed fail rate to 0.1156/0.1311 — a ~2-3 point
base-rate accident, not a designed defense: the gate is an absolute
threshold with no base-rate comparison, and an all-pass rule ties it
exactly (acc == all-pass base in both seeds, 0.8800==0.8800,
0.8667==0.8667). The honest 2x2x2 closed it properly: rule_cells ==
{(passive,perfect,exclaim)} (train rate 0.966 n=58 / 0.954 n=65),
held-out 0.9600 noisy / 1.0000 true, beating the base. Scream 2/2
(+0.5416 vs -0.1295; +0.5376 vs -0.1546); named exactly
{deco,tense,voice} 2/2 (contribs 0.47-0.76 vs thr ~0.015; innocents
<=0.013); all 3 pairs SYNERGY 2/2 (S +0.2325..+0.3214 vs band
+-0.047/0.049); S3 (triple) +0.7224/+0.7478 — 2-3x any pairwise S,
the 3-way footprint the pairwise algebra cannot name. Miner NOT
blind on AND (named all 3 cause values, excess 0.34-0.55 vs thr
~0.15) — the blindness theorem is XOR/flat-marginal-specific.
Wall-clock 523s under heavy box contention (a dozen concurrent
sibling probes); uncontended CPU well under the 180s budget.

RUN:  cd experiments && PYTHONHASHSEED=0 python eval_attack_e1_threeway.py
"""
import os
import sys
import time
from itertools import product

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import isaac as isaac_mod  # noqa: E402
from reservoir_surprise_probe import CaptureMeter, scream_verdict  # noqa: E402
from pool_conjunctive_frontier_probe import (  # noqa: E402
    make_events, onehot, ablate_groups, FEATURES, M_POOL, FLIP)

Isaac = isaac_mod.Isaac

N_EVAL = 900          # disclosed: probe uses 600; rare positive needs rows
ROUNDS_E = 4          # disclosed shrink from 6
REPEATS_E = 4         # disclosed shrink from 6
SEEDS = (2000, 2001)
CAUSE = ("voice", "tense", "deco")
PAIRS = [("voice", "tense"), ("voice", "deco"), ("tense", "deco")]


def funnel_multi(events, y, rng_seed):
    """pool_funnel's exact recipe (same decoy null, same drop algebra),
    extended to score ALL THREE cause pairs + the triple in one pass."""
    rng = np.random.default_rng(rng_seed)
    feats = list(FEATURES)
    drops = {f: [] for f in feats}
    decoy_drops, s_null, triple_drops = [], [], []
    pair_drops = {p: [] for p in PAIRS}
    for t in range(ROUNDS_E):
        X, spans = onehot(events, feats)
        dec = np.zeros((len(events), 3))
        for i in range(len(events)):
            dec[i, rng.integers(3)] = 1.0
        Xa = np.column_stack([X, dec])
        dec_cols = list(range(X.shape[1], X.shape[1] + 3))
        meter = CaptureMeter(Xa, M_POOL, seed=777 + t)
        cap = meter.capture(y)

        def drop(groups):
            vals = [cap - ablate_groups(meter, Xa, y, groups, rng)
                    for _ in range(REPEATS_E)]
            return float(np.mean(vals))

        d_t = {f: drop([spans[f]]) for f in feats}
        for f in feats:
            drops[f].append(d_t[f])
        d_dec = drop([dec_cols])
        decoy_drops.append(d_dec)
        for p in PAIRS:
            pair_drops[p].append(drop([spans[p[0]], spans[p[1]]]))
        for f in CAUSE:
            s_null.append(d_t[f] + d_dec - drop([spans[f], dec_cols]))
        triple_drops.append(drop([spans[f] for f in CAUSE]))
    contrib = {f: float(np.mean(drops[f])) for f in feats}
    name_thr = float(np.percentile(np.abs(decoy_drops), 99))
    named = sorted([f for f in feats if contrib[f] > name_thr])
    band = float(np.max(np.abs(s_null)))
    pairs_out = {}
    for p in PAIRS:
        S = float(contrib[p[0]] + contrib[p[1]] - np.mean(pair_drops[p]))
        kind = ("SYNERGY" if S > band else
                "REDUNDANCY" if S < -band else "ADDITIVE")
        pairs_out[p] = dict(S=S, kind=kind,
                            pair_drop=float(np.mean(pair_drops[p])))
    S3 = float(sum(contrib[f] for f in CAUSE) - np.mean(triple_drops))
    return dict(contrib=contrib, name_thr=name_thr, named=named, band=band,
                pairs=pairs_out, S3=S3,
                triple_drop=float(np.mean(triple_drops)))


def articulate(ev, y, truth, feats_used, cut):
    """The probe's E5 recipe generalized to k features: majority-fail
    cells from the TRAIN table, verdicts on the held-out tail."""
    combos = list(product(*[FEATURES[f] for f in feats_used]))
    table = {}
    for combo in combos:
        m = [i for i in range(cut)
             if all(ev[i][f] == v for f, v in zip(feats_used, combo))]
        table[combo] = (float(np.mean(y[m])) if m else 0.0, len(m))
    rule_cells = {k for k, (r, _n) in table.items() if r > 0.5}
    pred = np.array([tuple(ev[i][f] for f in feats_used) in rule_cells
                     for i in range(cut, len(ev))])
    yo = y[cut:].astype(bool)
    to = truth[cut:]
    acc_noisy = float(np.mean(pred == yo))
    acc_true = float(np.mean(pred == to))
    allpass = np.zeros(len(yo), dtype=bool)
    base_noisy = float(np.mean(allpass == yo))
    base_true = float(np.mean(allpass == to))
    return dict(table=table, rule_cells=rule_cells, acc_noisy=acc_noisy,
                acc_true=acc_true, base_noisy=base_noisy,
                base_true=base_true)


def run_seed(seed):
    rng = np.random.default_rng(seed)
    ev = make_events(rng, n=N_EVAL)
    truth = np.array([e["voice"] == "passive" and e["tense"] == "perfect"
                      and e["deco"] == "exclaim" for e in ev])
    y = np.where(rng.uniform(size=N_EVAL) < FLIP, ~truth,
                 truth).astype(float)
    print(f"\nseed {seed}: true fails {int(truth.sum())}/{N_EVAL} "
          f"({truth.mean():.4f}), observed fail rate {y.mean():.4f}")

    # (0) the marginal miner — P6 says NOT blind on AND
    miner = Isaac().mine_commonality(ev, list(y.astype(bool)))
    named_vals = [(k, v) for (k, v, _e) in miner["named"]]
    print(f"  miner named {miner['named']} (thr {miner['threshold']:.4f})")

    # (1) pool scream
    X, _ = onehot(ev, list(FEATURES))
    cap, thr, scream, spread = scream_verdict(X, y, M_POOL, rng_seed=seed)
    print(f"  scream: capture {cap:+.4f} vs null99 {thr:+.4f} "
          f"(spread {spread:.4f}) -> {'SCREAM' if scream else 'quiet'}")

    # (2)+(3) grouped naming + all-pairs drop algebra
    fun = funnel_multi(ev, y, rng_seed=seed + 1)
    print(f"  contrib {{"
          + ", ".join(f"{f}:{fun['contrib'][f]:+.4f}" for f in FEATURES)
          + f"}} name_thr {fun['name_thr']:.4f} -> named {fun['named']}")
    for p, d in fun["pairs"].items():
        print(f"  S({p[0]},{p[1]}) = {d['S']:+.4f} vs band "
              f"±{fun['band']:.4f} -> {d['kind']} "
              f"(pair_drop {d['pair_drop']:+.4f})")
    print(f"  triple_drop {fun['triple_drop']:+.4f}, "
          f"S3(sum singles - triple) {fun['S3']:+.4f}")

    # (4) THE TRAP: the probe's own 2x2 road on (voice,tense) only
    cut = int(0.75 * N_EVAL)
    a2 = articulate(ev, y, truth, ("voice", "tense"), cut)
    cells2 = {k: f"{r:.3f}(n={n})" for k, (r, n) in a2["table"].items()}
    print(f"  2x2 (voice,tense) train fail rates: {cells2}")
    print(f"  2x2 rule_cells recovered: {a2['rule_cells'] or 'NONE'} "
          f"(truth cells at this projection: NONE are majority-fail; "
          f"AND cell recovered: 0/1)")
    print(f"  2x2 held-out acc: {a2['acc_noisy']:.4f} vs noisy labels "
          f"(gate 0.90 {'PASSED' if a2['acc_noisy'] >= 0.90 else 'failed'})"
          f", {a2['acc_true']:.4f} vs true labels "
          f"({'PASSED' if a2['acc_true'] >= 0.90 else 'failed'}); "
          f"all-pass base: {a2['base_noisy']:.4f} noisy / "
          f"{a2['base_true']:.4f} true")

    # honest 2x2x2 on (voice,tense,deco)
    a3 = articulate(ev, y, truth, ("voice", "tense", "deco"), cut)
    print(f"  2x2x2 rule_cells: {a3['rule_cells']} — held-out acc "
          f"{a3['acc_noisy']:.4f} noisy / {a3['acc_true']:.4f} true "
          f"(beats all-pass base {a3['base_noisy']:.4f}: "
          f"{a3['acc_noisy'] > a3['base_noisy']})")

    return dict(
        scream=bool(scream), cap=cap, thr=thr,
        named=fun["named"], contrib=fun["contrib"],
        name_thr=fun["name_thr"], band=fun["band"], pairs=fun["pairs"],
        S3=fun["S3"], triple_drop=fun["triple_drop"],
        miner_named=named_vals,
        a2=a2, a3=a3,
        fail_rate=float(y.mean()), n_true=int(truth.sum()))


def main():
    t0 = time.perf_counter()
    print(f"ATTACK E-C: 3-way AND (voice=passive & tense=perfect & "
          f"deco=exclaim), FLIP={FLIP}, N={N_EVAL}, seeds {SEEDS}")
    res = [run_seed(s) for s in SEEDS]

    print("\n===== CHECKS vs PREDICTIONS =====")
    def check(name, ok, detail):
        print(f"  [{'ok ' if ok else 'FAIL'}] {name} — {detail}")

    check("P1 scream 2/2", all(r["scream"] for r in res),
          "; ".join(f"{r['cap']:+.4f} vs {r['thr']:+.4f}" for r in res))
    check("P2 named exactly {deco,tense,voice} 2/2",
          all(r["named"] == ["deco", "tense", "voice"] for r in res),
          "; ".join(str(r["named"]) for r in res))
    syn = all(all(d["kind"] == "SYNERGY" for d in r["pairs"].values())
              for r in res)
    check("P3 all 3 pairs SYNERGY 2/2", syn,
          "; ".join(",".join(d["kind"] for d in r["pairs"].values())
                    for r in res))
    trap_true = all(r["a2"]["rule_cells"] == set()
                    and r["a2"]["acc_true"] >= 0.90 for r in res)
    check("P4a trap vs TRUE labels (empty rule >= gate) 2/2", trap_true,
          "; ".join(f"cells={len(r['a2']['rule_cells'])} "
                    f"acc_true={r['a2']['acc_true']:.4f}" for r in res))
    noisy_below = all(r["a2"]["acc_noisy"] < 0.90 for r in res)
    check("P4b noisy-label acc < 0.90 both seeds (as filed)", noisy_below,
          "; ".join(f"{r['a2']['acc_noisy']:.4f}" for r in res))
    p5 = all(r["a3"]["rule_cells"] ==
             {("passive", "perfect", "exclaim")}
             and r["a3"]["acc_noisy"] > r["a3"]["base_noisy"]
             and r["a3"]["acc_noisy"] >= 0.90 for r in res)
    check("P5 honest 2x2x2 recovers AND cell + beats base 2/2", p5,
          "; ".join(f"{sorted(r['a3']['rule_cells'])} "
                    f"acc {r['a3']['acc_noisy']:.4f} vs base "
                    f"{r['a3']['base_noisy']:.4f}" for r in res))
    want = {("voice", "passive"), ("tense", "perfect"),
            ("deco", "exclaim")}
    p6 = all(want <= set(r["miner_named"]) for r in res)
    check("P6 miner names all three cause values 2/2", p6,
          "; ".join(str(r["miner_named"]) for r in res))
    bad = {"subj", "verb", "junk"}
    p7 = all(not (bad & set(r["named"])) for r in res)
    check("P7 innocents unnamed 2/2", p7,
          "; ".join(str(r["named"]) for r in res))
    print(f"\n  done in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
