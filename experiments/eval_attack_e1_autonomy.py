"""eval_attack_e1_autonomy.py — ATTACK E-E: THE PAIR-HINT SCAFFOLD,
END-TO-END AUTONOMY CHECK against pool_conjunctive_frontier_probe.py.

THE ATTACK: the pushed probe's pool_funnel(events, y, pair_hint, ...)
receives the TRUE causal pair ("voice","tense") as an argument, and
the E5 articulation hardcodes voice/tense in its 2x2 table. The
pushed claim is a funnel that "hands articulation a 2x2 table" —
but production has no oracle. Here the SAME W1 XOR worlds (seeds
1000, 1001, exactly as run_seed builds them) are pushed through an
AUTONOMOUS chain:
  (1) naming pass with NO use of any hint (code-verified: in the
      pushed pool_funnel, `named` derives only from per-feature
      drops and decoy drops; pair_hint touches only pair_drops and
      s_null — coupling to naming is via the shared rng stream ONLY);
  (2) pair chosen FROM the named set — CHOOSER RULE IMPLEMENTED:
      if exactly 2 features are named, that is the pair; otherwise
      the top-2 features by contrib (rule disclosed in output);
  (3) S + junction kind + the 2x2 table computed from the CHOSEN
      pair (generic over the chosen features' value lists — no
      hardcoded voice/tense);
  (4) held-out accuracy exactly as E5 (train = first 75%, cells with
      fail-rate > 0.5, verdicts on the last 25%).
The hinted funnel (pushed code imported UNCHANGED) runs on the same
worlds for direct comparison.

PREDICTIONS (filed before the first run)
  P-A the hint-free naming pass names exactly ['tense','voice'] on
      both seeds (contribs above the decoy 99th-pct threshold),
      matching the pushed E3 claim.
  P-B the chooser fires its exactly-2-named branch on both seeds and
      selects the pair {voice, tense}.
  P-C S on the CHOSEN pair exceeds the s_null band -> SYNERGY on
      both seeds (matching E4).
  P-D the 2x2 from the chosen pair recovers the XOR fail cells
      {voice=passive & tense=past, voice=active & tense=perfect}
      and held-out accuracy >= 0.90 on both seeds (noise ceiling
      ~0.95 at FLIP=0.05).
  P-E the hinted funnel on the same worlds agrees with the
      autonomous one: same named set, same kind; accuracy identical
      whenever the same pair is chosen (articulation depends only on
      the world and the pair). Numeric contribs/S differ slightly
      (rng-stream interleaving differs) but verdicts do not.
  If any of P-A..P-D fails, the scaffold is load-bearing and the
  autonomy gap is the finding.

SHRINK DISCLOSURE: 2 seeds (1000, 1001) as specified by the attack
order (the pushed probe default is 3); ROUNDS/REPEATS/M_POOL taken
UNCHANGED from the pushed probe (6/6/800). Deterministic:
PYTHONHASHSEED=0, fixed seeds throughout.

RUN: cd experiments && PYTHONHASHSEED=0 python eval_attack_e1_autonomy.py
"""
import sys
import time

import numpy as np

from pool_conjunctive_frontier_probe import (
    make_events, onehot, pool_funnel, ablate_groups,
    FEATURES, N, FLIP, M_POOL, ROUNDS, REPEATS)
from reservoir_surprise_probe import CaptureMeter


def autonomous_funnel(events, y, rng_seed):
    """The pushed funnel re-staged in three hint-free passes.
    Pass 1 (naming) is a faithful copy of the pushed per-feature +
    decoy machinery with every pair_hint-dependent call REMOVED.
    Pass 2 chooses the pair from the named set. Pass 3 computes
    pair_drops and s_null for the CHOSEN pair on the SAME meters."""
    rng = np.random.default_rng(rng_seed)
    feats = list(FEATURES)
    drops = {f: [] for f in feats}
    decoy_drops, stash = [], []

    # ── pass 1: naming, no hint anywhere ─────────────────────────
    for t in range(ROUNDS):
        X, spans = onehot(events, feats)
        dec = np.zeros((len(events), 3))
        for i in range(len(events)):
            dec[i, rng.integers(3)] = 1.0
        Xa = np.column_stack([X, dec])
        dec_cols = list(range(X.shape[1], X.shape[1] + 3))
        meter = CaptureMeter(Xa, M_POOL, seed=777 + t)
        cap = meter.capture(y)

        def drop(groups, meter=meter, Xa=Xa, cap=cap):
            vals = [cap - ablate_groups(meter, Xa, y, groups, rng)
                    for _ in range(REPEATS)]
            return float(np.mean(vals))

        d_t = {f: drop([spans[f]]) for f in feats}
        for f in feats:
            drops[f].append(d_t[f])
        d_dec = drop([dec_cols])
        decoy_drops.append(d_dec)
        stash.append((meter, Xa, spans, dec_cols, cap, d_t, d_dec))

    contrib = {f: float(np.mean(drops[f])) for f in feats}
    name_thr = float(np.percentile(np.abs(decoy_drops), 99))
    named = sorted([f for f in feats if contrib[f] > name_thr])

    # ── pass 2: choose the pair FROM the named set ───────────────
    if len(named) == 2:
        pair, rule = tuple(named), "exactly-2-named"
    else:
        top2 = sorted(feats, key=lambda f: contrib[f], reverse=True)[:2]
        pair, rule = tuple(sorted(top2)), \
            f"top-2-by-contrib (|named|={len(named)})"

    # ── pass 3: S on the CHOSEN pair, same meters ────────────────
    pair_drops, s_null = [], []
    for meter, Xa, spans, dec_cols, cap, d_t, d_dec in stash:
        def drop(groups, meter=meter, Xa=Xa, cap=cap):
            vals = [cap - ablate_groups(meter, Xa, y, groups, rng)
                    for _ in range(REPEATS)]
            return float(np.mean(vals))

        pair_drops.append(drop([spans[pair[0]], spans[pair[1]]]))
        s_null.append(d_t[pair[0]] + d_dec
                      - drop([spans[pair[0]], dec_cols]))
        s_null.append(d_t[pair[1]] + d_dec
                      - drop([spans[pair[1]], dec_cols]))
    S = float(contrib[pair[0]] + contrib[pair[1]] - np.mean(pair_drops))
    band = float(np.max(np.abs(s_null)))
    kind = ("SYNERGY" if S > band else
            "REDUNDANCY" if S < -band else "ADDITIVE")
    return dict(contrib=contrib, name_thr=name_thr, named=named,
                pair=pair, rule=rule, S=S, band=band, kind=kind,
                pair_drop=float(np.mean(pair_drops)))


def articulate(ev, y, pair, cut):
    """E5 articulation, generic over the CHOSEN pair (no hardcoded
    voice/tense): cell fail-rate table on train, rule = cells > 0.5,
    verdicts on held-out rows."""
    f1, f2 = pair
    table = {}
    for v1 in FEATURES[f1]:
        for v2 in FEATURES[f2]:
            m = [i for i in range(cut)
                 if ev[i][f1] == v1 and ev[i][f2] == v2]
            table[(v1, v2)] = float(np.mean(y[m])) if m else 0.0
    rule_cells = {k for k, r in table.items() if r > 0.5}
    pred = np.array([(ev[i][f1], ev[i][f2]) in rule_cells
                     for i in range(cut, len(ev))])
    acc = float(np.mean(pred == y[cut:].astype(bool)))
    return table, rule_cells, acc


def canon(pair, cells):
    """Orientation-free cell identity: set of frozensets of
    (feature, value) pairs."""
    f1, f2 = pair
    return {frozenset({(f1, v1), (f2, v2)}) for (v1, v2) in cells}


XOR_CANON = {frozenset({("voice", "passive"), ("tense", "past")}),
             frozenset({("voice", "active"), ("tense", "perfect")})}


def main():
    t0 = time.perf_counter()
    print("ATTACK E-E: pair-hint scaffold autonomy check "
          f"(N={N}, FLIP={FLIP}, ROUNDS={ROUNDS}, REPEATS={REPEATS}, "
          f"M_POOL={M_POOL})")
    results = []
    for seed in (1000, 1001):
        # W1 world EXACTLY as run_seed builds it
        rng = np.random.default_rng(seed)
        ev = make_events(rng)
        truth = np.array([(e["voice"] == "passive")
                          ^ (e["tense"] == "perfect") for e in ev])
        y1 = np.where(rng.uniform(size=N) < FLIP, ~truth,
                      truth).astype(float)
        cut = int(0.75 * N)

        auto = autonomous_funnel(ev, y1, rng_seed=seed + 1)
        table_a, cells_a, acc_a = articulate(ev, y1, auto["pair"], cut)
        xor_ok_a = canon(auto["pair"], cells_a) == XOR_CANON \
            if set(auto["pair"]) == {"voice", "tense"} else False

        hint = pool_funnel(ev, y1, ("voice", "tense"),
                           rng_seed=seed + 1)
        table_h, cells_h, acc_h = articulate(ev, y1,
                                             ("voice", "tense"), cut)
        xor_ok_h = canon(("voice", "tense"), cells_h) == XOR_CANON

        c = auto["contrib"]
        print(f"\nseed {seed}")
        print("  AUTONOMOUS (hint-free):")
        print("    contrib " + " ".join(f"{f}={c[f]:+.3f}"
                                        for f in FEATURES))
        print(f"    name_thr {auto['name_thr']:.3f} -> named "
              f"{auto['named']}")
        print(f"    chosen pair {auto['pair']} via [{auto['rule']}]")
        print(f"    S {auto['S']:+.3f} vs band ±{auto['band']:.3f} "
              f"(pair_drop {auto['pair_drop']:+.3f}) -> {auto['kind']}")
        print(f"    2x2 table (train {cut} rows): "
              + " ".join(f"{k}={v:.3f}" for k, v in table_a.items()))
        print(f"    rule cells {sorted(cells_a)} XOR-match={xor_ok_a} "
              f"| held-out acc {acc_a:.3f} on {N - cut} rows")
        print("  HINTED (pushed code unchanged, hint=('voice','tense')):")
        print(f"    named {hint['named']} thr {hint['name_thr']:.3f} | "
              f"S {hint['S']:+.3f} vs band ±{hint['band']:.3f} -> "
              f"{hint['kind']}")
        print(f"    rule cells {sorted(cells_h)} XOR-match={xor_ok_h} "
              f"| held-out acc {acc_h:.3f}")
        same_pair = set(auto["pair"]) == {"voice", "tense"}
        agree = (auto["named"] == hint["named"]
                 and auto["kind"] == hint["kind"] and same_pair
                 and acc_a == acc_h)
        print(f"  AGREEMENT auto-vs-hinted: same_named="
              f"{auto['named'] == hint['named']} same_pair={same_pair} "
              f"same_kind={auto['kind'] == hint['kind']} "
              f"same_acc={acc_a == acc_h} -> {'AGREE' if agree else 'DIVERGE'}")
        results.append(dict(seed=seed, named=auto["named"],
                            pair=auto["pair"], rule=auto["rule"],
                            S=auto["S"], band=auto["band"],
                            kind=auto["kind"], acc=acc_a,
                            xor_ok=xor_ok_a, agree=agree))

    print("\nSUMMARY")
    ok = 0
    for r in results:
        passed = (r["named"] == ["tense", "voice"]
                  and set(r["pair"]) == {"voice", "tense"}
                  and r["kind"] == "SYNERGY" and r["xor_ok"]
                  and r["acc"] >= 0.90 and r["agree"])
        ok += passed
        print(f"  seed {r['seed']}: named={r['named']} pair={r['pair']} "
              f"[{r['rule']}] S={r['S']:+.3f}/band {r['band']:.3f} "
              f"{r['kind']} acc={r['acc']:.3f} xor={r['xor_ok']} "
              f"agree={r['agree']} -> "
              f"{'AUTONOMY HOLDS' if passed else 'AUTONOMY GAP'}")
    print(f"\n  ATTACK: autonomy holds on {ok}/{len(results)} seeds "
          f"({time.perf_counter() - t0:.1f}s)")


if __name__ == "__main__":
    main()
