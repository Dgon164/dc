"""pool_conjunctive_frontier_probe.py — E1, THE BRIDGE TO LANGUAGE:
conjunctive failure causes in categorical (language-shaped) worlds
(owner question, 2026-07-15: "how do we test this on non mathematical
structures? like english?").

THE CLAIM UNDER TEST: the pool funnel crosses to language unchanged,
because pools never worked on math — they worked on CHANNELS, and a
sentence-event's reading (voice, tense, decoration, word seats) is a
channel vector the moment features become indicators. And the
crossing is URGENT, not optional: the engine's own self-assessment
organ (mine_commonality, the frontier habit's miner) is MARGINAL —
per-feature=value excess — the exact statistic the campaign proved
blind to interactions. A CONJUNCTIVE frontier (fails only when two
features co-occur in a pattern, each innocent alone) is invisible to
it by construction. The pool + grouped ablation + drop algebra must
name the pair, classify the junction, and hand articulation a 2x2
table instead of a 2^k search.

THE WORLDS (per seed; N=600 sentence-events; features: voice
{active,passive}, tense {past,perfect}, deco {none,comma,exclaim},
subj {cat,dog,bird,girl}, verb {chase,hold,see}, junk {a,b}):
  W1 HIDDEN AGREEMENT (the conjunctive frontier): fail iff voice and
     tense DISAGREE with the planted agreement rule — fail =
     XOR(voice=passive, tense=perfect) — plus 5% verdict noise.
     XOR at 50/50 base rates has PERFECTLY FLAT marginals:
     P(fail|passive) = P(fail|active) = 0.5. The marginal miner sees
     nothing, and that is a THEOREM about its statistic, not bad luck.
  W2 MARGINAL CAUSE (the agreement control): fail iff deco=exclaim
     (+5% noise) — BOTH instruments must find this one.
  W3 SCRAMBLED verdicts — both instruments must go quiet.

PREDICTIONS (filed before the first run)
  E1 miner blind on W1: mine_commonality names NOTHING (flat
     marginals) — the measured organ gap, every seed.
  E2 pool screams on W1 (capture past the permuted-verdict null).
  E3 grouped ablation names exactly {voice, tense} (feature-level,
     one shared permutation per feature group — one-hot consistency).
  E4 drop algebra on (voice, tense): S above the decoy band ->
     SYNERGY (jointly necessary — the conjunction's fingerprint).
  E5 articulation closes cheap: with the pair named, a 2x2
     fail-rate table on train recovers the XOR cells and predicts
     held-out verdicts at the noise ceiling (>=90%).
  E6 W2 agreement: miner names deco=exclaim AND pool names {deco}.
  E7 W3: miner nothing AND pool quiet.
  E8 subj/verb/junk never named by the pool in any world.

INTEGRITY: mine_commonality is the REAL engine organ (the baseline
under test, not a reimplementation); the pool instrument is imported
unchanged from reservoir_surprise_probe.py (13/13); pools point,
never answer — what ships is the articulated rule, proven on held-out
events with pools removed. Categorical channels enter as INDICATORS
(never ordinals — the re-encoding-invariance law). stdlib + numpy;
PYTHONHASHSEED=0; fixed seeds.

RUN:  cd experiments && PYTHONHASHSEED=0 python3 pool_conjunctive_frontier_probe.py [n_seeds]
      n_seeds defaults to 3 (cloud-friendly); on a big local box run
      20+ for the aggregated stability sweep.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import isaac as isaac_mod  # noqa: E402
from reservoir_surprise_probe import CaptureMeter, scream_verdict  # noqa: E402

Isaac = isaac_mod.Isaac

N = 600
M_POOL = 800
ROUNDS = 6
REPEATS = 6
FLIP = 0.05

FEATURES = {
    "voice": ["active", "passive"],
    "tense": ["past", "perfect"],
    "deco": ["none", "comma", "exclaim"],
    "subj": ["cat", "dog", "bird", "girl"],
    "verb": ["chase", "hold", "see"],
    "junk": ["a", "b"],
}


def make_events(rng, n=N):
    return [{f: vals[rng.integers(len(vals))]
             for f, vals in FEATURES.items()} for _ in range(n)]


def onehot(events, features):
    cols, spans, at = [], {}, 0
    for f in features:
        vals = FEATURES[f]
        block = np.zeros((len(events), len(vals)))
        for i, e in enumerate(events):
            block[i, vals.index(e[f])] = 1.0
        cols.append(block)
        spans[f] = list(range(at, at + len(vals)))
        at += len(vals)
    return np.column_stack(cols), spans


def ablate_groups(meter, X, r, groups, rng):
    """Cross-fit capture with each feature GROUP permuted by one
    shared row permutation per group (one-hot consistency kept)."""
    pred = np.empty(meter.n)
    for tr, ho, mu, sd, _H, M in meter.parts:
        Xp = X[ho].copy()
        for cols in groups:
            perm = rng.permutation(len(ho))
            Xp[:, cols] = Xp[perm][:, cols]
        H = np.tanh(((Xp - mu) / sd) @ meter.net.W + meter.net.b)
        c = r[tr].mean()
        pred[ho] = H @ (M @ (r[tr] - c)) + c
    ss = np.sum((r - pred) ** 2)
    tot = np.sum((r - r.mean()) ** 2)
    return 1.0 - ss / tot if tot > 0 else 0.0


def pool_funnel(events, y, pair_hint, rng_seed):
    """Grouped attribution + drop algebra over feature groups, decoy
    feature as the matched null."""
    rng = np.random.default_rng(rng_seed)
    feats = list(FEATURES)
    drops = {f: [] for f in feats}
    decoy_drops, s_null, pair_drops = [], [], []
    for t in range(ROUNDS):
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
                    for _ in range(REPEATS)]
            return float(np.mean(vals))

        d_t = {f: drop([spans[f]]) for f in feats}
        for f in feats:
            drops[f].append(d_t[f])
        d_dec = drop([dec_cols])
        decoy_drops.append(d_dec)
        d_pair = drop([spans[pair_hint[0]], spans[pair_hint[1]]])
        pair_drops.append(d_pair)
        s_null.append(d_t[pair_hint[0]] + d_dec
                      - drop([spans[pair_hint[0]], dec_cols]))
        s_null.append(d_t[pair_hint[1]] + d_dec
                      - drop([spans[pair_hint[1]], dec_cols]))
    contrib = {f: float(np.mean(drops[f])) for f in feats}
    name_thr = float(np.percentile(np.abs(decoy_drops), 99))
    named = sorted([f for f in feats if contrib[f] > name_thr])
    S = float(contrib[pair_hint[0]] + contrib[pair_hint[1]]
              - np.mean(pair_drops))
    band = float(np.max(np.abs(s_null)))
    kind = ("SYNERGY" if S > band else
            "REDUNDANCY" if S < -band else "ADDITIVE")
    return dict(contrib=contrib, name_thr=name_thr, named=named,
                S=S, band=band, kind=kind,
                pair_drop=float(np.mean(pair_drops)))


def run_seed(seed, verbose):
    rng = np.random.default_rng(seed)
    ok = {}

    # W1 — the conjunctive frontier
    ev = make_events(rng)
    truth = np.array([(e["voice"] == "passive") ^ (e["tense"] == "perfect")
                      for e in ev])
    y1 = np.where(rng.uniform(size=N) < FLIP, ~truth, truth).astype(float)
    miner = Isaac().mine_commonality(ev, list(y1.astype(bool)))
    ok["E1_miner_blind_W1"] = miner["named"] == []
    X1, _ = onehot(ev, list(FEATURES))
    cap, thr, scream, _ = scream_verdict(X1, y1, M_POOL, rng_seed=seed)
    ok["E2_pool_screams_W1"] = bool(scream)
    fun = pool_funnel(ev, y1, ("voice", "tense"), rng_seed=seed + 1)
    ok["E3_names_voice_tense"] = fun["named"] == ["tense", "voice"]
    ok["E4_synergy"] = fun["kind"] == "SYNERGY"
    # E5 articulation: 2x2 table on train, verdicts on held-out
    cut = int(0.75 * N)
    table = {}
    for v in FEATURES["voice"]:
        for t in FEATURES["tense"]:
            m = [i for i in range(cut) if ev[i]["voice"] == v
                 and ev[i]["tense"] == t]
            table[(v, t)] = float(np.mean(y1[m])) if m else 0.0
    rule_cells = {k for k, r in table.items() if r > 0.5}
    xor_cells = {("passive", "past"), ("active", "perfect")}
    pred = np.array([(ev[i]["voice"], ev[i]["tense"]) in rule_cells
                     for i in range(cut, N)])
    acc = float(np.mean(pred == y1[cut:].astype(bool)))
    ok["E5_articulated_rule"] = (rule_cells == xor_cells and acc >= 0.90)

    # W2 — marginal cause: both instruments must agree
    ev2 = make_events(rng)
    t2 = np.array([e["deco"] == "exclaim" for e in ev2])
    y2 = np.where(rng.uniform(size=N) < FLIP, ~t2, t2).astype(float)
    miner2 = Isaac().mine_commonality(ev2, list(y2.astype(bool)))
    miner2_hit = any(tuple(c[:2]) == ("deco", "exclaim")
                     for c in miner2["named"])
    fun2 = pool_funnel(ev2, y2, ("deco", "voice"), rng_seed=seed + 2)
    ok["E6_agreement_W2"] = miner2_hit and ("deco" in fun2["named"])

    # W3 — scrambled verdicts: both quiet
    y3 = np.random.default_rng(seed + 9).permutation(y1)
    miner3 = Isaac().mine_commonality(ev, list(y3.astype(bool)))
    X3, _ = onehot(ev, list(FEATURES))
    _, _, scream3, _ = scream_verdict(X3, y3, M_POOL, rng_seed=seed + 3)
    ok["E7_scrambled_quiet"] = (miner3["named"] == []) and (not scream3)

    # E8 — innocents never named by the pool
    bad = {"subj", "verb", "junk"}
    ok["E8_innocents_unnamed"] = (not (bad & set(fun["named"]))
                                  and not (bad & set(fun2["named"])))
    if verbose:
        print(f"  seed {seed}: W1 miner named {miner['named']} | pool "
              f"capture {cap:+.3f} vs {thr:+.3f} | named {fun['named']} "
              f"thr {fun['name_thr']:.3f} | S {fun['S']:+.3f} band "
              f"±{fun['band']:.3f} -> {fun['kind']} | rule acc {acc:.3f}")
    return ok


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    t0 = time.perf_counter()
    print(f"E1 conjunctive-frontier bridge probe — {n_seeds} seed(s)")
    agg = {}
    for s in range(n_seeds):
        for k, v in run_seed(1000 + s, verbose=True).items():
            agg.setdefault(k, []).append(v)
    print()
    npass = 0
    for k, vals in agg.items():
        good = sum(vals)
        ok = good == len(vals)
        npass += ok
        print(f"  [{'ok ' if ok else 'FAIL'}] {k} — {good}/{len(vals)} seeds")
    dt = time.perf_counter() - t0
    print(f"\n  PROBE: {npass} passed, {len(agg) - npass} failed of "
          f"{len(agg)}  ({dt:.1f}s, {n_seeds} seeds)")


if __name__ == "__main__":
    main()
