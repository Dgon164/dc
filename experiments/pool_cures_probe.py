"""pool_cures_probe.py — THE POOL CURE LIST (the pool-branch eval's
five filed cures, built as instrument v2 and re-measured against the
EXACT exhibits that earned them). E3's preflight: these are the
conditions under which the pool funnel may read the live account's
receipts.

THE FIVE CURES UNDER TEST
  1. FAMILY-WISE SCAN NULL — multi-target scans buy false screams at
     the family rate (eval: 1 confident false scream, p=0.003,
     survived 5/5 null re-estimations). Cure: the null is the MAX
     CAPTURE ACROSS ALL SCANNED TARGETS per permutation; every
     target's scream is judged against the family bar.
  2. CONDITIONAL-PERMUTATION NAMING NULL — uncorrelated decoys
     calibrate the wrong zero for a correlated innocent (eval: subj
     as a voice-proxy named 3/3 seeds at 1.6-2.7x thr). Cure: each
     feature's naming null is ITS OWN drop distribution under
     WITHIN-STRATA permutation (strata = its most-associated other
     feature, association measured) — correlation to the partner
     preserved, link to the target broken; per-candidate thresholds
     (the studentization cure shape, applied).
  3. S3, THE ORDER-3 STATISTIC — pairwise S saturates on triples.
     >>> RUN-1 FALSIFIED THE RAW FORM, KEPT: S3 alone does not
     discriminate (ablated captures go negative by convention, so
     drops exceed capture and BOTH worlds inflate: triple 3.5x cap,
     2-way 1.9x cap). THE TRUE SIGNATURE IS THE EXCESS OVER THE BEST
     PAIR: S3 - max pairwise S — a triple's synergy is exactly the
     part no pair explains (theory: ~0 for 2-way + spectator,
     ~pair-scale for a pure triple). Measured after the fix in K5.
  4. BEAT-ALL-PASS ARTICULATION GATE — an absolute acc>=0.90 gate is
     base-rate-blind (eval: an EMPTY rule scored 0.911 past it on a
     rare conjunction). Cure: the rule must beat the all-pass
     predictor by a binomial margin: acc > base + 2*sqrt(base(1-base)/n).
  5. THE PAIR CHOOSER — production has no oracle hint: exactly-2
     named -> that pair; k>=3 named -> the k-way table road (composing
     with cure 4); <2 named -> articulation REFUSED with a receipt.

PREDICTIONS (filed before the first run; wrong ones kept)
  K1 no-relation world, 10-target family-wise scan: 0 false screams
     (the old per-target bar bought 1 confident + 1 borderline).
  K2 power kept: the planted product world's C target still screams
     PAST THE FAMILY BAR (margin reported).
  K3 correlated-proxy world (rho~0.55): subj named 0/2 seeds under
     the conditional null (was 3/3); voice+tense still named 2/2.
  K4 no regression: the independent-features world names exactly
     {tense, voice} under the conditional null.
  K5 S3 flags the pure triple (~2x capture) and stays at pair scale
     on a genuine 2-way synergy.
  K6 the trap world's EMPTY 2x2 rule is REJECTED by the
     beat-all-pass gate; the honest 2x2x2 PASSES it; the original
     XOR world's 2x2 still passes (power kept).
  K7 the chooser: 2-named -> pair; 3-named -> k-way road; 0-named ->
     refusal receipt.
  K8 cost < 300 s (shrunk ROUNDS/REPEATS disclosed inline).

Run: cd experiments && PYTHONHASHSEED=0 python pool_cures_probe.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from reservoir_surprise_probe import CaptureMeter  # noqa: E402
from pool_conjunctive_frontier_probe import (  # noqa: E402
    FEATURES, ablate_groups, make_events, onehot)

M_POOL = 800
FAM_NULLS = 100        # family-wise permutations (shrunk, disclosed)
ROUNDS = 3             # funnel rounds (shrunk from 6, disclosed)
REPEATS = 4
COND_REPS = 8          # within-strata permuted copies per feature


# ── cure 1: the family-wise scan ────────────────────────────────────
def family_scan(tables, rng_seed=1):
    """tables: list of (X, y) target scans. Null = MAX capture across
    ALL targets per permutation; every scream judged at the family
    bar. Returns per-target (capture, family_thr, scream)."""
    meters = [CaptureMeter(X, M_POOL, seed=777) for X, _ in tables]
    caps = [m.capture(y) for m, (_, y) in zip(meters, tables)]
    rng = np.random.default_rng(rng_seed)
    fam = np.empty(FAM_NULLS)
    for p in range(FAM_NULLS):
        fam[p] = max(m.capture(rng.permutation(y))
                     for m, (_, y) in zip(meters, tables))
    thr = float(np.percentile(fam, 99.0))
    return [(float(c), thr, c > thr) for c in caps]


# ── cure 2: conditional-permutation naming null ─────────────────────
def _assoc_partner(X, spans, f, feats):
    """Most-associated OTHER feature (mean |corr| between one-hot
    blocks — measured, not assumed)."""
    best, arg = -1.0, None
    A = X[:, spans[f]]
    for g in feats:
        if g == f:
            continue
        B = X[:, spans[g]]
        Za = (A - A.mean(0)) / (A.std(0) + 1e-12)
        Zb = (B - B.mean(0)) / (B.std(0) + 1e-12)
        a = float(np.abs(Za.T @ Zb / len(A)).mean())
        if a > best:
            best, arg = a, g
    return arg


def funnel_v2(events, y, rng_seed=1):
    """Grouped attribution with PER-FEATURE conditional-permutation
    nulls: feature f's null drops come from permuting f WITHIN the
    strata of its most-associated partner — correlation kept, target
    link broken. Names = drop > that feature's OWN null 99pct."""
    rng = np.random.default_rng(rng_seed)
    feats = list(FEATURES)
    X, spans = onehot(events, feats)
    drops = {f: [] for f in feats}
    null_drops = {f: [] for f in feats}
    partners = {f: _assoc_partner(X, spans, f, feats) for f in feats}
    for t in range(ROUNDS):
        meter = CaptureMeter(X, M_POOL, seed=777 + t)
        cap = meter.capture(y)

        def drop_of(cols):
            vals = [cap - ablate_groups(meter, X, y, [cols], rng)
                    for _ in range(REPEATS)]
            return float(np.mean(vals))

        for f in feats:
            drops[f].append(drop_of(spans[f]))
            # conditional null: f permuted within partner's strata
            part = partners[f]
            strata = X[:, spans[part]].argmax(1)
            for _ in range(COND_REPS // ROUNDS + 1):
                Xn = X.copy()
                for s in np.unique(strata):
                    idx = np.where(strata == s)[0]
                    Xn[np.ix_(idx, spans[f])] = \
                        Xn[np.ix_(rng.permutation(idx), spans[f])]
                mn = CaptureMeter(Xn, M_POOL, seed=777 + t)
                capn = mn.capture(y)
                null_drops[f].append(
                    capn - ablate_groups(mn, Xn, y, [spans[f]], rng))
    contrib = {f: float(np.mean(drops[f])) for f in feats}
    thr = {f: float(np.percentile(np.abs(null_drops[f]), 99))
           for f in feats}
    named = sorted([f for f in feats if contrib[f] > thr[f]])
    return dict(contrib=contrib, thr=thr, named=named,
                partners=partners)


# ── cure 3: S3 ──────────────────────────────────────────────────────
def s3_statistic(X, y, trip, rng_seed=1):
    """Order-3 EXCESS: S3 - max pairwise S. A triple's synergy is
    the part no pair explains (~0 for 2-way + spectator; ~pair-scale
    for a pure triple). Returns (excess, S3, best_pair_S, capture)."""
    rng = np.random.default_rng(rng_seed)
    meter = CaptureMeter(X, M_POOL, seed=777)
    cap = meter.capture(y)

    def drop(cols):
        from pool_drop_algebra_probe import capture_ablate_cols
        vals = [cap - capture_ablate_cols(meter, X, y, cols, rng)
                for _ in range(REPEATS)]
        return float(np.mean(vals))

    i, j, k = trip
    d = {c: drop([c]) for c in trip}
    d123 = drop(list(trip))
    s3 = d[i] + d[j] + d[k] - d123
    pair_S = []
    for (a, b) in ((i, j), (i, k), (j, k)):
        pair_S.append(d[a] + d[b] - drop([a, b]))
    excess = s3 - max(pair_S)
    return float(excess), float(s3), float(max(pair_S)), float(cap)


# ── cure 4: the beat-all-pass gate ──────────────────────────────────
def articulation_gate(acc, base, n):
    """The rule must beat the all-pass predictor by a binomial
    margin — never an absolute score."""
    bar = base + 2.0 * np.sqrt(max(base * (1 - base), 1e-9) / n)
    return bool(acc > bar), float(bar)


# ── cure 5: the chooser ─────────────────────────────────────────────
def choose_articulation(named, contrib):
    if len(named) == 2:
        return dict(road="pair", pair=tuple(named),
                    receipt="exactly two named -> the pair")
    if len(named) >= 3:
        return dict(road="k-way", features=tuple(named),
                    receipt=f"{len(named)} named -> the {len(named)}-"
                            f"way table (gated by beat-all-pass)")
    return dict(road="refused", receipt="fewer than two named — "
                "articulation refused, dossier kept")


def main():
    t0 = time.perf_counter()
    checks = []

    def check(name, ok, detail):
        checks.append(ok)
        print(f"  [{'ok ' if ok else 'FAIL'}] {name} — {detail}")

    # K1/K2 — family-wise scan on descriptor worlds
    print("K1/K2. family-wise scan: 10 targets, no-relation vs planted")
    rng = np.random.default_rng(42)

    def episodes(plant, n_ep=120):
        A = np.empty((n_ep, 10))
        for e in range(n_ep):
            a = rng.uniform(-2, 2, 10)
            small = np.abs(a) < 0.3
            a[small] += 0.6 * np.sign(a[small]) + 0.3
            if plant:
                a[2] = a[0] * a[1]
            A[e] = a
        return A

    A0 = episodes(plant=False)
    tables0 = [(np.delete(A0, t, axis=1), A0[:, t]) for t in range(10)]
    res0 = family_scan(tables0, rng_seed=11)
    fp = sum(s for _, _, s in res0)
    check("K1 family-wise: 0/10 false screams on no-relation",
          fp == 0, f"{fp}/10 (caps "
          f"{[round(c, 2) for c, _, _ in res0[:4]]}... vs family thr "
          f"{res0[0][1]:+.3f})")
    A1 = episodes(plant=True)
    tables1 = [(np.delete(A1, t, axis=1), A1[:, t]) for t in range(10)]
    res1 = family_scan(tables1, rng_seed=12)
    c_cap, c_thr, c_scream = res1[2]
    distr = [s for t, (_, _, s) in enumerate(res1) if t >= 3]
    check("K2 power kept at the family bar: planted C screams, "
          "distractor targets quiet",
          c_scream and sum(distr) == 0,
          f"C {c_cap:+.3f} vs family thr {c_thr:+.3f} (margin "
          f"{c_cap - c_thr:+.3f}); distractor screams {sum(distr)}/7")

    # K3/K4 — conditional naming null
    print("K3/K4. conditional-permutation naming (the correlated "
          "innocent)")
    def corr_events(rng2, n=600):
        evs = []
        for _ in range(n):
            e = {f: v[rng2.integers(len(v))]
                 for f, v in FEATURES.items()}
            if e["voice"] == "passive":
                e["subj"] = "cat" if rng2.uniform() < 0.8 \
                    else e["subj"]
            evs.append(e)
        return evs

    named_subj = 0
    named_true = 0
    for s in range(2):
        r2 = np.random.default_rng(1000 + s)
        ev = corr_events(r2)
        truth = np.array([(e["voice"] == "passive")
                          ^ (e["tense"] == "perfect") for e in ev])
        y = np.where(r2.uniform(size=len(ev)) < 0.05, ~truth,
                     truth).astype(float)
        fu = funnel_v2(ev, y, rng_seed=2000 + s)
        named_subj += int("subj" in fu["named"])
        named_true += int({"tense", "voice"} <= set(fu["named"]))
        if s == 0:
            print(f"       seed 0: named {fu['named']}; subj contrib "
                  f"{fu['contrib']['subj']:+.4f} vs ITS null thr "
                  f"{fu['thr']['subj']:.4f} (partner "
                  f"{fu['partners']['subj']})")
    check("K3 correlated proxy NOT named (was 3/3 under decoy null)",
          named_subj == 0 and named_true == 2,
          f"subj named {named_subj}/2; true pair named {named_true}/2")
    r4 = np.random.default_rng(4000)
    ev4 = make_events(r4)
    truth4 = np.array([(e["voice"] == "passive")
                       ^ (e["tense"] == "perfect") for e in ev4])
    y4 = np.where(r4.uniform(size=len(ev4)) < 0.05, ~truth4,
                  truth4).astype(float)
    fu4 = funnel_v2(ev4, y4, rng_seed=4001)
    check("K4 independent world: names exactly {tense, voice}",
          fu4["named"] == ["tense", "voice"], f"named {fu4['named']}")

    # K5 — S3
    print("K5. the order-3 statistic")
    r5 = np.random.default_rng(500)
    X5 = r5.uniform(-2, 2, (400, 5))
    y_tri = 0.8 * X5[:, 0] * X5[:, 1] * X5[:, 2] \
        + r5.normal(0, 0.05, 400)
    ex_t, s3_t, ps_t, cap_t = s3_statistic(X5, y_tri, (0, 1, 2),
                                            rng_seed=51)
    y_pair = 0.9 * X5[:, 1] * X5[:, 3] + r5.normal(0, 0.05, 400)
    ex_p, s3_p, ps_p, cap_p = s3_statistic(X5, y_pair, (1, 3, 0),
                                           rng_seed=52)
    check("K5 order-3 EXCESS separates: triple >> 0, 2-way ~ 0",
          ex_t > 0.5 * cap_t and ex_p < 0.25 * cap_p,
          f"triple excess {ex_t:+.3f} (S3 {s3_t:+.3f}, best pair "
          f"{ps_t:+.3f}, cap {cap_t:+.3f}); 2-way excess {ex_p:+.3f} "
          f"(S3 {s3_p:+.3f}, best pair {ps_p:+.3f})")

    # K6 — the beat-all-pass gate on the trap world
    print("K6. the articulation gate vs the base-rate trap")
    r6 = np.random.default_rng(600)
    ev6 = make_events(r6, n=900)
    t6 = np.array([e["voice"] == "passive" and e["tense"] == "perfect"
                   and e["deco"] == "exclaim" for e in ev6])
    y6 = np.where(r6.uniform(size=900) < 0.05, ~t6, t6).astype(float)
    cut = 675
    base = float(1 - y6[cut:].mean())          # the all-pass score
    # the empty 2x2 rule (recovers nothing -> predicts all-pass)
    acc_empty = base
    ok_e, bar = articulation_gate(acc_empty, base, 900 - cut)
    # the honest 2x2x2
    tab = {}
    for v in FEATURES["voice"]:
        for t in FEATURES["tense"]:
            for d in FEATURES["deco"]:
                m = [i for i in range(cut)
                     if ev6[i]["voice"] == v and ev6[i]["tense"] == t
                     and ev6[i]["deco"] == d]
                tab[(v, t, d)] = float(np.mean(y6[m])) if m else 0.0
    cells = {k for k, r in tab.items() if r > 0.5}
    pred = np.array([(ev6[i]["voice"], ev6[i]["tense"],
                      ev6[i]["deco"]) in cells
                     for i in range(cut, 900)])
    acc3 = float(np.mean(pred == y6[cut:].astype(bool)))
    ok_3, _ = articulation_gate(acc3, base, 900 - cut)
    # the XOR power check
    r6b = np.random.default_rng(601)
    ev7 = make_events(r6b)
    t7 = np.array([(e["voice"] == "passive") ^ (e["tense"] == "perfect")
                   for e in ev7])
    y7 = np.where(r6b.uniform(size=600) < 0.05, ~t7, t7).astype(float)
    cut7 = 450
    tab7 = {}
    for v in FEATURES["voice"]:
        for t in FEATURES["tense"]:
            m = [i for i in range(cut7) if ev7[i]["voice"] == v
                 and ev7[i]["tense"] == t]
            tab7[(v, t)] = float(np.mean(y7[m])) if m else 0.0
    cells7 = {k for k, r in tab7.items() if r > 0.5}
    pred7 = np.array([(ev7[i]["voice"], ev7[i]["tense"]) in cells7
                      for i in range(cut7, 600)])
    acc7 = float(np.mean(pred7 == y7[cut7:].astype(bool)))
    base7 = float(max(y7[cut7:].mean(), 1 - y7[cut7:].mean()))
    ok_7, bar7 = articulation_gate(acc7, base7, 600 - cut7)
    check("K6 gate: empty rule REJECTED, 2x2x2 passes, XOR passes",
          (not ok_e) and ok_3 and ok_7,
          f"empty {acc_empty:.3f} vs bar {bar:.3f} -> {ok_e}; 2x2x2 "
          f"{acc3:.3f} -> {ok_3}; XOR {acc7:.3f} vs bar {bar7:.3f} "
          f"-> {ok_7}")

    # K7 — the chooser
    c2 = choose_articulation(["tense", "voice"], {})
    c3 = choose_articulation(["deco", "tense", "voice"], {})
    c0 = choose_articulation([], {})
    check("K7 chooser: pair / k-way / refusal with receipts",
          c2["road"] == "pair" and c3["road"] == "k-way"
          and c0["road"] == "refused",
          f"{c2['road']} / {c3['road']} / {c0['road']}")

    dt = time.perf_counter() - t0
    check("K8 cost < 300 s", dt < 300, f"{dt:.1f}s")
    npass = sum(checks)
    print(f"\n  PROBE: {npass} passed, {len(checks) - npass} failed of "
          f"{len(checks)}  ({dt:.1f}s)")


if __name__ == "__main__":
    main()
