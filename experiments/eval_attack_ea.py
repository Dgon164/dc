"""eval_attack_ea.py — ATTACK E-A follow-up: identify WHICH seed failed
E8_innocents_unnamed in the probe's own 20-seed stability sweep, and why.

CONTEXT (measured first, 2026-07-15): PYTHONHASHSEED=0 python
pool_conjunctive_frontier_probe.py 20 -> 7/8 checks 20/20;
E8_innocents_unnamed 19/20. Every verbose line printed W1's funnel
named == ['tense','voice'], so the one E8 failure MUST be in the
unprinted W2 funnel (fun2) naming one of {subj, verb, junk}.

PREDICTIONS (filed before this script's first run)
  P1 exactly ONE seed in 1000..1019 has an innocent in fun2['named'].
  P2 the culprit is a decoy-threshold artifact: name_thr is the 99th
     percentile of only ROUNDS=6 |decoy drops| (~= max of 6 samples),
     so on the failing seed the innocent's mean contrib is only
     MARGINALLY above thr (within ~2x), not a large real effect.
  P3 deco is still correctly named on the failing seed (E6 passed
     20/20, so this is guaranteed by construction — sanity check).

METHOD: reproduce run_seed's rng consumption exactly up to W2
(make_events, uniform(N) for y1 noise, make_events, uniform(N) for y2
noise), then call the probe's own pool_funnel(ev2, y2,
("deco","voice"), rng_seed=seed+2) unchanged. Takes start/end seed
args so four chunks can run in parallel (runtime disclosure: the full
20-seed sweep took 1186s; this reruns only the W2 leg, chunked).
"""
import sys

import numpy as np

from pool_conjunctive_frontier_probe import (FLIP, N, make_events,
                                             pool_funnel)

BAD = {"subj", "verb", "junk"}


def w2_for_seed(seed):
    rng = np.random.default_rng(seed)
    make_events(rng)                 # ev  (W1) — consume
    rng.uniform(size=N)              # y1 noise — consume
    ev2 = make_events(rng)
    t2 = np.array([e["deco"] == "exclaim" for e in ev2])
    y2 = np.where(rng.uniform(size=N) < FLIP, ~t2, t2).astype(float)
    return pool_funnel(ev2, y2, ("deco", "voice"), rng_seed=seed + 2)


def main():
    lo = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    hi = int(sys.argv[2]) if len(sys.argv) > 2 else 1019
    for seed in range(lo, hi + 1):
        fun2 = w2_for_seed(seed)
        hit = sorted(BAD & set(fun2["named"]))
        tag = "E8-FAIL" if hit else "ok"
        cont = " ".join(f"{f}={fun2['contrib'][f]:+.4f}"
                        for f in fun2["contrib"])
        print(f"seed {seed}: [{tag}] named={fun2['named']} "
              f"innocents={hit} thr={fun2['name_thr']:.4f} | {cont}",
              flush=True)


if __name__ == "__main__":
    main()
