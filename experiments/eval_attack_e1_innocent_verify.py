"""eval_attack_e1_innocent_verify.py — ADVERSARIAL VERIFICATION of the
e1-correlated-innocent BROKEN finding (eval_attack_e1_innocent.py).

ROLE: refute-or-confirm. The finding says the pool funnel names the
causally innocent proxy subj above its decoy-null threshold on 2/2
seeds (1000, 1001) when subj correlates with voice at 0.8. The
original numbers REPRODUCED exactly on this box before this file was
written (subj +0.0349 vs thr 0.0216; +0.0809 vs 0.0295).

THIS FILE'S TWO LEGS (seed 1002, previously untested):
  LEG A (break survival, new seed): the attack's correlated world,
     generator copied verbatim from eval_attack_e1_innocent.py
     (P(cat|passive)=0.8), full fidelity (N=600, ROUNDS=6, REPEATS=6,
     M_POOL=800), funnel rng_seed=seed+1 per the branch convention.
  LEG B (in-scope control, same seed): the branch probe's OWN
     independent-feature world (make_events unchanged), same rule,
     same noise, same funnel call — does the pushed claim (named ==
     exactly {tense, voice}; subj under thr) hold in its stated scope?

PREDICTIONS (filed before the first run):
  V1 LEG A: the break SURVIVES — contrib[subj] > name_thr at seed
     1002; named != ['tense','voice']. Basis: the mechanism is
     structural (L2 readout spreads weight across correlated channels;
     the decoy null calibrates an UNCORRELATED zero), and the prior
     margins were 1.6x and 2.7x, not knife-edge.
  V2 LEG B: the pushed claim HOLDS in scope — named == ['tense',
     'voice'] exactly and contrib[subj] < name_thr on the independent
     world at the same seed.
  V3 miner blind (named []) in both legs — flat XOR marginals; the
     proxy inherits them.
  V4 SYNERGY on (voice, tense) in both legs (S > band).

MEASURED OUTCOMES (recorded after run 1, verbatim from the output;
  note: an earlier draft of this block briefly contained placeholder
  numbers written before any run — an integrity error, caught and
  removed before the first run; these are the real measurements):
  V1 CORRECT — break survives seed 1002 (3/3 seeds total with the
     attack's 1000/1001): contrib[subj] +0.0678 vs name_thr 0.0273
     (2.5x); named = ['subj','tense','voice']; plant
     P(cat|passive)=0.781 vs P(cat|active)=0.217.
  V2 CORRECT — pushed claim holds in its stated scope at the same
     seed: named == ['tense','voice'] exactly; contrib[subj] -0.0197
     vs thr 0.0477; all innocents under (junk -0.0024, verb -0.0033,
     deco -0.0143).
  V3 CORRECT both legs: miner named [] (A top +0.059 vs thr +0.144,
     subj=cat excess -0.053; B top +0.054 vs thr +0.128, excess
     +0.001).
  V4 CORRECT both legs: SYNERGY (A: S +1.215 vs band 0.102;
     B: S +1.316 vs band 0.075).
  Runtime 4.7s on this box (the attack's disclosed 664.4s did not
  reproduce as a cost — likely CPU contention on its original run;
  the numbers themselves reproduced exactly).

RUNTIME/SHRINK DISCLOSURE: no shrink — both legs at the target
probe's full fidelity; run in background and disclosed if over the
~180s budget (two funnels ~ the cost of one attack seed x2).

RUN: cd experiments && PYTHONHASHSEED=0 python eval_attack_e1_innocent_verify.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import isaac as isaac_mod  # noqa: E402
from pool_conjunctive_frontier_probe import (  # noqa: E402
    FEATURES, FLIP, N, make_events, pool_funnel,
)
from eval_attack_e1_innocent import make_correlated_events  # noqa: E402

Isaac = isaac_mod.Isaac

SEED = 1002


def run_leg(label, gen, seed):
    rng = np.random.default_rng(seed)
    ev = gen(rng)
    truth = np.array([(e["voice"] == "passive") ^ (e["tense"] == "perfect")
                      for e in ev])
    y = np.where(rng.uniform(size=N) < FLIP, ~truth, truth).astype(float)

    is_pas = np.array([e["voice"] == "passive" for e in ev])
    is_cat = np.array([e["subj"] == "cat" for e in ev])
    p_cat_pas = float(is_cat[is_pas].mean())
    p_cat_act = float(is_cat[~is_pas].mean())

    miner = Isaac().mine_commonality(ev, list(y.astype(bool)))
    n_f = int(y.sum())
    exc_cat = (float(is_cat[y.astype(bool)].sum()) / n_f
               - float(is_cat[~y.astype(bool)].sum()) / (N - n_f))

    fun = pool_funnel(ev, y, ("voice", "tense"), rng_seed=seed + 1)
    named_exact = fun["named"] == ["tense", "voice"]
    subj_named = "subj" in fun["named"]

    print(f"\n[{label}] seed {seed}  P(cat|passive)={p_cat_pas:.3f}  "
          f"P(cat|active)={p_cat_act:.3f}")
    print(f"  MINER: named={miner['named']}  top={miner['top']:+.3f} vs "
          f"thr {miner['threshold']:+.3f}  (subj=cat excess {exc_cat:+.3f})")
    for f in sorted(fun["contrib"], key=lambda f: -fun["contrib"][f]):
        mark = "NAMED" if f in fun["named"] else "     "
        print(f"    contrib[{f:5s}] = {fun['contrib'][f]:+.4f}  vs "
              f"name_thr {fun['name_thr']:.4f}  {mark}"
              f"{'   <-- proxy seat' if f == 'subj' else ''}")
    print(f"  named={fun['named']}  exact {{tense,voice}}? {named_exact}  "
          f"subj named? {subj_named}")
    print(f"  algebra: S {fun['S']:+.3f} vs band +-{fun['band']:.3f} "
          f"-> {fun['kind']}")
    return dict(named_exact=named_exact, subj_named=subj_named,
                miner_blind=miner["named"] == [], kind=fun["kind"],
                subj_contrib=fun["contrib"]["subj"],
                name_thr=fun["name_thr"])


def main():
    t0 = time.perf_counter()
    print("VERIFY e1-correlated-innocent: seed 1002, correlated leg + "
          "in-scope control leg")
    a = run_leg("LEG A correlated", make_correlated_events, SEED)
    b = run_leg("LEG B independent (branch's own world)", make_events, SEED)
    print(f"\nVERDICT INPUTS:")
    print(f"  A break survives (subj named, correlated): {a['subj_named']}  "
          f"({a['subj_contrib']:+.4f} vs {a['name_thr']:.4f})")
    print(f"  B pushed claim holds in scope (exact naming, independent): "
          f"{b['named_exact']}  (subj {b['subj_contrib']:+.4f} vs "
          f"{b['name_thr']:.4f})")
    print(f"  miner blind: A={a['miner_blind']} B={b['miner_blind']}; "
          f"SYNERGY: A={a['kind']} B={b['kind']}")
    print(f"  ({time.perf_counter() - t0:.1f}s)")


if __name__ == "__main__":
    main()
