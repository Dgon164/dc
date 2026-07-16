"""eval_attack_e1_innocent.py — ATTACK E-D: CORRELATED INNOCENT
(the innocent-proxy hazard, pool edition).

TARGET: pool_conjunctive_frontier_probe.py's naming claims (E3: the
funnel names exactly {tense, voice}; E8: subj/verb/junk never named).
The probe's worlds draw every feature independently. Here subj is made
a PROXY for voice: when voice=passive, subj='cat' with prob 0.8
(otherwise uniform over {dog,bird,girl}); when voice=active, subj is
uniform over all four (the probe's default draw). Rule unchanged:
fail = XOR(voice=passive, tense=perfect) + 5% verdict noise, N=600,
seeds 1000 and 1001 (matching the probe's seed convention).

WHY THIS COULD BREAK THE FUNNEL: the ridge readout over random tanh
features has no reason to concentrate weight on voice alone when subj
carries correlated information — L2 spreads weight across correlated
channels, so permuting subj on holdout could produce a drop above the
decoy null even though subj is causally innocent. The probe's defense
(as pushed): ablating subj leaves voice intact, so the fit's cleaner
channel keeps predicting and subj's drop stays inside the decoy band.

PREDICTIONS (filed before the first run):
  P1 MINER: mine_commonality names NOTHING on both seeds — including
     subj=cat. Reason: XOR at 50/50 base rates has flat marginals in
     voice, and subj is conditionally independent of tense given
     voice, so P(fail | subj=cat) ~ 0.5 too. The proxy inherits the
     flatness; the miner is BLIND here, not fooled. (The known miner
     innocent-proxy hazard needs a cause with a nonflat marginal;
     this world does not provide one.)
  P2 FUNNEL NAMES: named == ['tense', 'voice'] exactly, both seeds;
     contrib[subj] < name_thr (the decoy-null defense holds because
     ablation is leave-the-rest-intact).
  P3 SOFTENING: contrib[voice] may come in lower than contrib[tense]
     (the fit can lean on subjxtense as a partial backup when voice is
     permuted), but still clears name_thr on both seeds.
  P4 ALGEBRA: S > band -> SYNERGY on the (voice, tense) pair, both
     seeds (the correlation does not fake redundancy on the true pair).

MEASURED OUTCOMES (recorded after run 1; predictions above kept):
  P1 CORRECT, both seeds: miner named [] (top +0.063 vs thr +0.123;
     +0.101 vs +0.145); subj=cat excess -0.028 / +0.016 — flat, the
     proxy inherits XOR's flat marginals; the miner is blind here,
     not fooled (P(fail|cat) 0.494/0.541 vs P(fail|~cat) 0.522/0.525).
  P2 FALSIFIED — WRONG-KEPT, both seeds: the funnel NAMED subj.
     seed 1000: contrib[subj] +0.0349 vs name_thr 0.0216 (1.6x);
     seed 1001: contrib[subj] +0.0809 vs name_thr 0.0295 (2.7x);
     named = ['subj','tense','voice'], never exactly {tense,voice}.
     The pushed defense ("ablating subj leaves voice intact") does
     NOT hold: the ridge readout spreads weight onto the correlated
     channel, so permuting subj on holdout costs real capture — and
     the decoy null cannot see this, because decoys are UNCORRELATED
     noise: a proxy's drop is calibrated against the wrong zero.
     This is the innocent-proxy failure, pool edition: 2/2 seeds.
  P3 DIRECTIONALLY CORRECT: voice softened slightly (1.1439 vs tense
     1.2492; 1.0558 vs 1.2330) but far above name_thr — still named.
  P4 CORRECT, both seeds: S +1.160 vs band 0.079, +1.119 vs 0.085 ->
     SYNERGY on (voice, tense); the true pair's junction survives.
  Pool scream: +0.648 vs null99 -0.201 and +0.621 vs -0.168 (2/2).
  deco/verb/junk (uncorrelated innocents) stayed under the bar every
  seed (max +0.0184 vs 0.0295) — the failure is SPECIFIC to
  correlation, not a loose threshold.

RUNTIME/SHRINK DISCLOSURE: no shrink — full N=600, ROUNDS=6,
REPEATS=6 as in the target probe, 2 seeds as ordered; measured
664.4s on this box (the target probe's own funnel is the cost;
disclosed as over the ~180s budget rather than shrunk, since the
run had already completed at full fidelity).

RUN: cd experiments && PYTHONHASHSEED=0 python eval_attack_e1_innocent.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import isaac as isaac_mod  # noqa: E402
from pool_conjunctive_frontier_probe import (  # noqa: E402
    FEATURES, FLIP, N, onehot, pool_funnel,
)
from reservoir_surprise_probe import scream_verdict  # noqa: E402

Isaac = isaac_mod.Isaac

M_POOL = 800          # same capacity the target probe hands its pools
P_CAT_GIVEN_PASSIVE = 0.8


def make_correlated_events(rng, n=N):
    """The probe's make_events, except subj is planted as a voice proxy:
    passive -> subj='cat' w.p. 0.8 else uniform over the other three;
    active -> uniform over all four (default draw)."""
    events = []
    rest = [s for s in FEATURES["subj"] if s != "cat"]
    for _ in range(n):
        e = {f: vals[rng.integers(len(vals))] for f, vals in FEATURES.items()}
        if e["voice"] == "passive":
            if rng.uniform() < P_CAT_GIVEN_PASSIVE:
                e["subj"] = "cat"
            else:
                e["subj"] = rest[rng.integers(len(rest))]
        events.append(e)
    return events


def run_seed(seed):
    rng = np.random.default_rng(seed)
    ev = make_correlated_events(rng)
    truth = np.array([(e["voice"] == "passive") ^ (e["tense"] == "perfect")
                      for e in ev])
    y = np.where(rng.uniform(size=N) < FLIP, ~truth, truth).astype(float)

    # plant verification — the correlation actually exists
    is_pas = np.array([e["voice"] == "passive" for e in ev])
    is_cat = np.array([e["subj"] == "cat" for e in ev])
    p_cat_pas = float(is_cat[is_pas].mean())
    p_cat_act = float(is_cat[~is_pas].mean())
    fail_cat = float(y[is_cat].mean())
    fail_ncat = float(y[~is_cat].mean())

    # instrument 1: the engine's marginal miner (the comparison row)
    miner = Isaac().mine_commonality(ev, list(y.astype(bool)))
    subj_cat_excess = None
    for (k, v, e) in miner["named"]:
        pass  # named entries checked below
    # pull subj=cat's raw excess for the record even if unnamed
    n_f = int(y.sum())
    n_s = N - n_f
    exc_cat = (float(is_cat[y.astype(bool)].sum()) / n_f
               - float(is_cat[~y.astype(bool)].sum()) / n_s)
    miner_names_cat = any(k == "subj" and v == "cat"
                          for (k, v, _e) in miner["named"])

    # instrument 2: the pool — scream then funnel
    X, _ = onehot(ev, list(FEATURES))
    cap, thr, scream, _sp = scream_verdict(X, y, M_POOL, rng_seed=seed)
    fun = pool_funnel(ev, y, ("voice", "tense"), rng_seed=seed + 1)

    named_exact = fun["named"] == ["tense", "voice"]
    subj_named = "subj" in fun["named"]

    print(f"\nseed {seed}  (N={N}, flip={FLIP})")
    print(f"  plant: P(cat|passive)={p_cat_pas:.3f}  "
          f"P(cat|active)={p_cat_act:.3f}")
    print(f"  marginals: P(fail|cat)={fail_cat:.3f}  "
          f"P(fail|~cat)={fail_ncat:.3f}   (flat = proxy inherits XOR)")
    print(f"  MINER: named={miner['named']}  top={miner['top']:+.3f} "
          f"vs thr {miner['threshold']:+.3f}  "
          f"(subj=cat excess {exc_cat:+.3f}) "
          f"-> names subj=cat? {miner_names_cat}")
    print(f"  POOL scream: capture {cap:+.3f} vs null99 {thr:+.3f} "
          f"-> scream={scream}")
    order = sorted(fun["contrib"], key=lambda f: -fun["contrib"][f])
    for f in order:
        c = fun["contrib"][f]
        mark = "NAMED" if f in fun["named"] else "     "
        print(f"    contrib[{f:5s}] = {c:+.4f}  vs name_thr "
              f"{fun['name_thr']:.4f}  {mark}"
              f"{'   <-- the planted proxy' if f == 'subj' else ''}")
    print(f"  named={fun['named']}  exact {{tense,voice}}? {named_exact}  "
          f"subj named? {subj_named}")
    print(f"  algebra: S {fun['S']:+.3f} vs band +-{fun['band']:.3f} "
          f"-> {fun['kind']}  (pair_drop {fun['pair_drop']:+.3f})")

    return dict(named_exact=named_exact, subj_named=subj_named,
                miner_blind=miner["named"] == [],
                miner_names_cat=miner_names_cat,
                scream=bool(scream), kind=fun["kind"],
                subj_contrib=fun["contrib"]["subj"],
                name_thr=fun["name_thr"])


def main():
    t0 = time.perf_counter()
    print("ATTACK E-D: correlated innocent (subj as 0.8-proxy for voice)"
          " on the XOR(voice,tense) world")
    res = [run_seed(1000 + s) for s in range(2)]
    n_exact = sum(r["named_exact"] for r in res)
    n_subj = sum(r["subj_named"] for r in res)
    n_blind = sum(r["miner_blind"] for r in res)
    n_cat = sum(r["miner_names_cat"] for r in res)
    n_scream = sum(r["scream"] for r in res)
    n_syn = sum(r["kind"] == "SYNERGY" for r in res)
    print(f"\nSUMMARY over {len(res)} seeds:")
    print(f"  funnel named exactly {{tense,voice}}: {n_exact}/{len(res)}")
    print(f"  funnel named subj (innocent-proxy FAILURE): "
          f"{n_subj}/{len(res)}")
    print(f"  miner blind (named nothing): {n_blind}/{len(res)}; "
          f"miner named subj=cat: {n_cat}/{len(res)}")
    print(f"  pool screamed: {n_scream}/{len(res)}; "
          f"SYNERGY on (voice,tense): {n_syn}/{len(res)}")
    print(f"  ({time.perf_counter() - t0:.1f}s)")


if __name__ == "__main__":
    main()
