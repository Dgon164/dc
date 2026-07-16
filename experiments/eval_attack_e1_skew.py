"""eval_attack_e1_skew.py — ATTACK E-B: SKEWED BASE RATES against
pool_conjunctive_frontier_probe.py.

THE ATTACK: the probe's W1 XOR world is flat-marginal ONLY because both
features are 50/50 (the probe itself calls the miner's blindness "a
THEOREM about its statistic"). Here the same rule
fail = XOR(voice=passive, tense=perfect) + 5% flip noise, N=600, but
tense is drawn 70/30 (past/perfect). Then, pre-noise:
    P(fail | voice=passive) = P(tense=past)    = 0.7
    P(fail | voice=active)  = P(tense=perfect) = 0.3
so the VOICE marginal now carries signal (tense marginals stay flat at
0.5 because voice is still 50/50). Both instruments run on the skewed
world, 2 seeds (2000, 2001). A flat 50/50 control funnel runs at the
same seeds to QUANTIFY any pool degradation (extra measurement, not a
pushed claim).

PREDICTIONS (filed before the first run):
  P1  miner NAMES on the skewed world: ('voice','passive') with excess
      ~ +0.36 (0.4 attenuated by the 5% flips), far past a threshold
      ~0.10-0.15; both seeds. This is the marginal signal RETURNING,
      not a failure. ('voice','active') has excess ~ -0.36 and is not
      named (one-sided). Nothing else named.
  P2  pool funnel still names exactly ['tense','voice'] both seeds
      (d_voice ~ cap ~ 0.8 because dropping voice kills main effect +
      interaction; d_tense ~ cap - R2(voice alone) ~ 0.8 - 0.13 ~ 0.67;
      both >> decoy threshold).
  P3  kind = SYNERGY both seeds: S = d_voice + d_tense - d_pair
      ~ 0.8 + 0.67 - 0.8 ~ +0.67 > band (band expected < 0.1).
  P4  articulation (E5 arithmetic mirrored: 75/25 split, 2x2 train
      fail-rate table, cells > 0.5) recovers exactly the XOR cells
      {(passive,past),(active,perfect)} and held-out accuracy ~ 0.95
      >= 0.90 both seeds. Rarest cell (active,perfect) still has
      ~ 0.5*0.3*450 ~ 67 train rows.
  P5  degradation is REAL but sub-verdict: S_skew < S_flat at the same
      seed (the voice main effect survives a tense-drop, shrinking
      d_tense), roughly 0.65-0.7 vs 0.8; no named-set or kind change.
  P6  scream_verdict still screams on the skewed world (capture ~0.8
      vs permuted-y threshold ~0.05).

RUN: cd experiments && PYTHONHASHSEED=0 python eval_attack_e1_skew.py
Runtime budget: 2 seeds x (2 funnels + 1 scream + 1 miner) — same
per-seed cost as the original probe's "cloud-friendly" setting.
"""
import sys
import time

import numpy as np

from pool_conjunctive_frontier_probe import (FEATURES, FLIP, M_POOL, N,
                                             pool_funnel)
from reservoir_surprise_probe import scream_verdict
import isaac as isaac_mod

Isaac = isaac_mod.Isaac

SKEW_PAST = 0.70   # P(tense=past); perfect gets 0.30


def make_skewed_events(rng, n=N, p_past=SKEW_PAST):
    """Same event dicts as the probe's make_events, but tense 70/30."""
    ev = []
    for _ in range(n):
        e = {f: vals[rng.integers(len(vals))]
             for f, vals in FEATURES.items()}
        e["tense"] = "past" if rng.uniform() < p_past else "perfect"
        ev.append(e)
    return ev


def make_flat_events(rng, n=N):
    return [{f: vals[rng.integers(len(vals))]
             for f, vals in FEATURES.items()} for _ in range(n)]


def xor_labels(rng, ev):
    truth = np.array([(e["voice"] == "passive") ^ (e["tense"] == "perfect")
                      for e in ev])
    return np.where(rng.uniform(size=len(ev)) < FLIP,
                    ~truth, truth).astype(float)


def articulate(ev, y):
    """Mirror of the probe's E5 arithmetic: 2x2 train table, cells with
    fail rate > 0.5 are the rule, verdicts scored on the held 25%."""
    cut = int(0.75 * len(ev))
    table = {}
    for v in FEATURES["voice"]:
        for t in FEATURES["tense"]:
            m = [i for i in range(cut) if ev[i]["voice"] == v
                 and ev[i]["tense"] == t]
            table[(v, t)] = (float(np.mean(y[m])) if m else 0.0, len(m))
    rule_cells = {k for k, (r, _) in table.items() if r > 0.5}
    xor_cells = {("passive", "past"), ("active", "perfect")}
    pred = np.array([(ev[i]["voice"], ev[i]["tense"]) in rule_cells
                     for i in range(cut, len(ev))])
    acc = float(np.mean(pred == y[cut:].astype(bool)))
    return table, rule_cells, rule_cells == xor_cells, acc


def onehot_local(events):
    cols = []
    for f, vals in FEATURES.items():
        block = np.zeros((len(events), len(vals)))
        for i, e in enumerate(events):
            block[i, vals.index(e[f])] = 1.0
        cols.append(block)
    return np.column_stack(cols)


def run_seed(seed):
    rng = np.random.default_rng(seed)
    out = {"seed": seed}

    # --- skewed world -------------------------------------------------
    ev = make_skewed_events(rng)
    y = xor_labels(rng, ev)
    n_past = sum(e["tense"] == "past" for e in ev)
    n_pass = sum(e["voice"] == "passive" for e in ev)
    fail_pass = float(np.mean([y[i] for i in range(N)
                               if ev[i]["voice"] == "passive"]))
    fail_act = float(np.mean([y[i] for i in range(N)
                              if ev[i]["voice"] == "active"]))
    out["world"] = dict(n_past=n_past, n_passive=n_pass,
                        fail_rate_passive=fail_pass,
                        fail_rate_active=fail_act,
                        n_fail=int(y.sum()))

    # instrument 1: the real engine organ
    miner = Isaac().mine_commonality(ev, list(y.astype(bool)))
    out["miner"] = dict(named=miner["named"], thr=miner["threshold"],
                        top=miner["top"])

    # scream (does the pool still see structure at all?)
    cap, thr, scream, _ = scream_verdict(onehot_local(ev), y, M_POOL,
                                         rng_seed=seed)
    out["scream"] = dict(cap=float(cap), thr=float(thr),
                         scream=bool(scream))

    # instrument 2: the pool funnel, unchanged
    fun = pool_funnel(ev, y, ("voice", "tense"), rng_seed=seed + 1)
    out["funnel"] = fun

    # articulation (E5 mirrored)
    table, cells, cells_ok, acc = articulate(ev, y)
    out["articulation"] = dict(table=table, cells=cells,
                               cells_ok=cells_ok, acc=acc)

    # --- flat 50/50 control at the same seed (degradation yardstick) ---
    rng2 = np.random.default_rng(seed)
    ev_f = make_flat_events(rng2)
    y_f = xor_labels(rng2, ev_f)
    fun_f = pool_funnel(ev_f, y_f, ("voice", "tense"), rng_seed=seed + 1)
    out["funnel_flat"] = fun_f
    return out


def main():
    t0 = time.perf_counter()
    seeds = [2000, 2001]
    if len(sys.argv) > 1:
        seeds = seeds[: int(sys.argv[1])]
    print(f"ATTACK E-B skewed base rates — seeds {seeds}")
    results = []
    for s in seeds:
        r = run_seed(s)
        results.append(r)
        w, mi, sc = r["world"], r["miner"], r["scream"]
        f, ff, a = r["funnel"], r["funnel_flat"], r["articulation"]
        print(f"\nseed {s}: tense past {w['n_past']}/{N}, "
              f"passive {w['n_passive']}/{N}, fails {w['n_fail']}/{N}, "
              f"P(fail|passive)={w['fail_rate_passive']:.3f} "
              f"P(fail|active)={w['fail_rate_active']:.3f}")
        print(f"  miner named: {mi['named']}  thr {mi['thr']:.3f} "
              f"top {mi['top']:.3f}")
        print(f"  scream: cap {sc['cap']:+.3f} vs thr {sc['thr']:+.3f} "
              f"-> {sc['scream']}")
        print(f"  funnel SKEW: named {f['named']} thr {f['name_thr']:.4f} "
              f"contrib { {k: round(v, 3) for k, v in f['contrib'].items()} }")
        print(f"    S {f['S']:+.3f} band ±{f['band']:.3f} -> {f['kind']} "
              f"(pair_drop {f['pair_drop']:.3f})")
        print(f"  funnel FLAT: named {ff['named']} S {ff['S']:+.3f} "
              f"band ±{ff['band']:.3f} -> {ff['kind']}")
        cells = {k: (round(v[0], 3), v[1])
                 for k, v in a['table'].items()}
        print(f"  articulation: cells {cells}")
        print(f"    rule {sorted(a['cells'])} xor-match {a['cells_ok']} "
              f"held-out acc {a['acc']:.3f}")

    print("\n--- verdict lines (branch claims under skew) ---")
    checks = {
        "pool_names_exactly_tense_voice":
            [r["funnel"]["named"] == ["tense", "voice"] for r in results],
        "pool_kind_SYNERGY":
            [r["funnel"]["kind"] == "SYNERGY" for r in results],
        "articulation_xor_cells_and_acc>=0.90":
            [r["articulation"]["cells_ok"] and r["articulation"]["acc"] >= 0.90
             for r in results],
        "miner_names_something (marginal signal returned)":
            [r["miner"]["named"] != [] for r in results],
        "miner_top_hit_is_voice=passive":
            [bool(r["miner"]["named"]) and
             tuple(r["miner"]["named"][0][:2]) == ("voice", "passive")
             for r in results],
        "scream_on_skew":
            [r["scream"]["scream"] for r in results],
        "S_skew < S_flat (degradation, P5)":
            [r["funnel"]["S"] < r["funnel_flat"]["S"] for r in results],
    }
    for k, vals in checks.items():
        print(f"  [{'ok ' if all(vals) else 'FAIL'}] {k} — "
              f"{sum(vals)}/{len(vals)} seeds")
    dS = [r["funnel_flat"]["S"] - r["funnel"]["S"] for r in results]
    print(f"  degradation Delta-S (flat minus skew) per seed: "
          f"{[round(x, 3) for x in dS]}")
    print(f"\n  done in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
