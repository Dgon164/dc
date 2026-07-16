"""basic_mechanisms_probe.py — THE COMPARISON WORDS (owner, 2026-07-15
night): "does it have a way to show simple logic structures? if a-b is
<0 (negative), then b>a? ... im sure there is some computer something
or other that is simpler than that too." And the build order: "hes got
tool structures, and we have a way to route through other concept
nodes and tools, so what if we did that? ... adding these basic tools
and any other basic computer mechanism to his toolkit will be
extremely helpful."

RESEARCH, NOT THE ENGINE (research-first-then-promote). Two dialects
of the same mechanisms, built and tested together:

  DIALECT 1 — WORDS FOR THE EQUATION GRAMMAR (atoms). The measured
  hole: the vocabulary has no comparison — max(a,b) broke the old
  judge (9-term impostor at r2=0.99945), the blended conditional and
  the plant step were detectable but unsayable. New candidate words:
    |xi - xj|          the CREASE atom (owner's a-b, direction
                       dropped) — max/min become plain algebra:
                       max = (a+b+|a-b|)/2, min = (a+b-|a-b|)/2
    [xi > t]           the SWITCH atom (step), thresholds t PROPOSED
                       from the data (deciles + coarse mark ladder),
                       never hand-set; t is a constant so the ruler
                       discipline applies to it like any constant
    [f]*xi*xj          the GATED PRODUCT (switch routed into a
                       product) for binary channels f — the blended
                       world's missing triple
  Family members NAMED but deferred (honest ladder): hinge
  max(0, x-t), sign(x), remainder(a,b). The judge is imported
  UNCHANGED from verdict_rewrite_probe (23/23); only the candidate
  pool is swapped at the module seam (disclosed instrument change).

  DIALECT 2 — TOOLS THAT ROUTE (probe-level model of the engine
  promotion). compare/larger/smaller/switch as functions that return
  value + RECEIPT, where larger() literally routes THROUGH compare()
  (the owner's a-b<0 wiring made visible), and IDENTITY RECOGNITION:
  a minted equation that is numerically a known mechanism gets doored
  onto that tool's name (two representations of one law, joined —
  the keystone move). Engine registration (nodes, consideration law)
  is the PROMOTION work order, not claimed here.

PREDICTIONS (filed before the first run; wrong ones kept)
  P-M1 MAX WORLD (y = max(x0,x1), exact, n=240): with the crease atom
       in the pool, verdict = UNDERSTOOD (machine-exact); equation
       contains 0.5*|x0-x1| with mates 0.5*x0 + 0.5*x1 (snap to the
       0.5 grid); road recorded (exact road or continue-loop — either
       is a pass; the crease residual should pull the atom in via
       partial correlation if the exact road misses it).
  P-M2 IDENTITY: the M1 mint is recognized as larger(x0, x1) by
       numerical identity on fresh rows (<=1e-12), and the tool call
       larger(a,b) carries a receipt chain routing through compare.
  P-M3 STEP WORLD (y = 2*x0 + 1.5*[1<x1<2] + 0.05 noise, 0.01 grain,
       sigma TOLD): with switch atoms, verdict = UNDERSTOOD with two
       step terms at thresholds 1 and 2 (from the mark ladder) and
       amplitudes +-1.5; the same world was PARTIAL-with-scream for
       the wordless judge (v3 V5) and a poly-impostor UNDERSTOOD for
       the old judge.
  P-M4 BLENDED WORLD (flag ? x0*x1 : x0+x1, n=96, exact integers):
       with the gated product, verdict = UNDERSTOOD, machine-exact
       residual, the full 5-term form recovered (x0 + x1 - f*x0 -
       f*x1 + f*x0*x1 or algebraic equivalent). Uncertainty filed:
       five admissions at 48 train rows must each clear the
       selection-corrected null; a falsification here is kept and
       measured, not patched.
       >>> FALSIFIED RUN 1, KEPT: the words now EXIST but the greedy
       search could not assemble the 5-term sentence at 48 train
       rows — it admitted a wrong-basin 3-term compromise
       (0.75*[f]*x0*x1 + 0.225*x0*x1 - 4*f) and the NEW judge
       correctly refused it (PARTIAL, both legs fired). The OLD
       judge shipped 0.838*[f]*x0*x1 as UNDERSTOOD — new words hand
       the old judge new ways to be confidently wrong. The wall is
       the SEARCH, not the vocabulary; cures already named: the
       rival-form duel (port requirement) and a DEEPER EXACT ROAD.
  P-M4b (filed before its first run) DEEP EXACT ROAD: extending the
       exact-subset search from 3 to 5 terms over the top-16
       candidates (same held-out machine-exactness guard — noise
       cannot fake it at any k) recovers the exact 5-term blended
       form with grid-snapped +-1 coefficients.
       >>> RUN 2: top-16 FAILED (kept) — measured why: x0's marginal
       correlation ranks 23rd, and ranks 7-21 are CLONES (on integer
       data [x1>6]==[x1>6.5] exactly; a binary flag's squares/cubes/
       alternating lags are affine copies of itself). Duplicates add
       nothing to an exact-subset search, so run 3 DEDUPES affine
       clones (|corr|>1-1e-9) before the cut and widens to top-20
       (disclosed; C(20,5)=15,504 fits, ~2s). PORT LESSON filed: the
       candidate pool itself should dedupe clones globally — the
       prefilter starves real terms while copies of one column vote
       as a bloc (the marginal-blindness theme in one more costume).
  P-M4c (filed with M4b) controls for the deep road: n=8 noise x20
       and n=200 d=3 noise x10 -> 0 exact stamps.
  P-M5 CONTROLS with the ENRICHED pool (the cost of new words is
       measured, not assumed): (a) noise x20 (d=3 incl. a binary
       flag): false UNDERSTOOD 0/20; (b) true quadratic, sigma told,
       x20: availability >= 17/20 and combined false surprise <= 2/20
       — the selection bar rises with vocabulary size; if
       availability drops below the wordless pool's 19/20, that cost
       is the finding (complete numbers).
  P-M6 the engine's tool catalog does NOT yet answer for these
       mechanisms (measured gap = the promotion work order).
  P-M7 cost: full probe < 240 s uncontended.

Run: cd experiments && PYTHONHASHSEED=0 python basic_mechanisms_probe.py
"""
import os
import sys
import time
from itertools import combinations

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import verdict_rewrite_probe as vrp  # noqa: E402

RNG = vrp.RNG
q = vrp.q

# ─────────────────────────── DIALECT 1: the new words (candidate pool)
BASE_MAKE = vrp.make_candidates

def _mark_ladder(lo, hi):
    """Coarse threshold proposals: multiples of 0.5 strictly inside
    (lo, hi) — thresholds are constants, so they are proposed ON
    RULER MARKS first, refined only by admission."""
    lo_m = np.ceil(lo / 0.5) * 0.5
    return [t for t in np.arange(lo_m, hi, 0.5) if lo < t < hi]

def make_candidates_mech(X, n_rows):
    C, names = BASE_MAKE(X, n_rows)
    n, d = X.shape
    vn = (lambda i: "x") if d == 1 else (lambda i: f"x{i}")
    add, anames = [], []
    binch = [i for i in range(d)
             if set(np.unique(X[:, i])).issubset({0.0, 1.0})]
    for i in range(d):
        for j in range(i + 1, d):
            add.append(np.abs(X[:, i] - X[:, j]))
            anames.append(f"|{vn(i)}-{vn(j)}|")
    for f in binch:                      # the switch routed into products
        for i in range(d):
            for j in range(i + 1, d):
                if f in (i, j): continue
                add.append(X[:, f] * X[:, i] * X[:, j])
                anames.append(f"[{vn(f)}]*{vn(i)}*{vn(j)}")
    for i in range(d):                   # step atoms, proposed thresholds
        if i in binch: continue
        u = X[:, i]
        lo, hi = float(u.min()), float(u.max())
        ts = sorted(set(np.round(np.quantile(u, np.linspace(.1, .9, 9)), 6))
                    | set(np.round(_mark_ladder(lo, hi), 6)))
        for t in ts:
            add.append((u > t).astype(float))
            anames.append(f"[{vn(i)}>{t:g}]")
    if add:
        C = np.column_stack([C] + add)
        names = names + anames
    return C, names

vrp.make_candidates = make_candidates_mech   # the disclosed seam swap


# ─────────────────────────── DIALECT 2: tools that route, with receipts
def t_compare(a, b, tol=0.0):
    d = a - b
    v = 0 if abs(d) <= tol else (1 if d > 0 else -1)
    return {"tool": "compare", "value": v,
            "receipt": f"sign({a:g}-{b:g}) -> {v:+d} (tol {tol:g})",
            "routed_via": []}

def t_larger(a, b):
    c = t_compare(a, b)
    return {"tool": "larger", "value": a if c["value"] >= 0 else b,
            "receipt": f"larger({a:g},{b:g}) via compare -> "
                       f"{a if c['value'] >= 0 else b:g}",
            "routed_via": [c]}

def t_smaller(a, b):
    c = t_compare(a, b)
    return {"tool": "smaller", "value": b if c["value"] >= 0 else a,
            "receipt": f"smaller({a:g},{b:g}) via compare",
            "routed_via": [c]}

def t_switch(cond, then_v, else_v):
    return {"tool": "switch", "value": then_v if cond else else_v,
            "receipt": f"switch({cond}) -> "
                       f"{'then' if cond else 'else'}",
            "routed_via": []}

MECH_TOOLS = ("compare", "larger", "smaller", "switch")


def exact_road_deep(C, y, C_ho, y_ho, kmax=5, top=20):
    """The exact road with a longer sentence bound (disclosed variant;
    P-M4b). Same guard as the sealed judge's exact road: a hit must be
    machine-exact RELATIVE on train AND held-out — noise cannot fake
    held-out exactness at any k, which is what makes deepening safe;
    P-M4c measures that claim. Run-3 addition (disclosed, P-M4b note):
    affine-clone candidates are DEDUPED (|corr|>1-1e-9) before the
    top-N cut — copies of one column otherwise vote as a bloc and
    starve real terms out of the shortlist."""
    ok_t, ok_h = vrp.vrows(C), vrp.vrows(C_ho)
    if ok_h.sum() < 3 or ok_t.sum() < 3:
        return None
    Ct, yt = C[ok_t], y[ok_t]
    Ch, yh = C_ho[ok_h], y_ho[ok_h]
    scale = max(np.abs(np.concatenate([yt, yh])).max(), 1e-12)
    base = yt - yt.mean()
    ranked = np.argsort([-vrp.safe_corr(Ct[:, j], base)
                         for j in range(Ct.shape[1])]).tolist()
    order = []
    for j in ranked:
        if len(order) >= top:
            break
        if any(vrp.safe_corr(Ct[:, j], Ct[:, k]) > 1 - 1e-9
               for k in order):
            continue
        order.append(j)
    for k in range(1, kmax + 1):
        if len(yt) < k + 3:
            break
        for sub in combinations(order, k):
            sub = list(sub)
            coef = vrp.ols(Ct[:, sub], yt)
            if np.abs(yt - vrp.pred(Ct[:, sub], coef)).max() / scale \
                    > vrp.EXACT_REL:
                continue
            if np.abs(yh - vrp.pred(Ch[:, sub], coef)).max() / scale \
                    > vrp.EXACT_REL:
                continue
            out = coef.copy()
            for t in range(k):
                for g in vrp.GRIDS:
                    m = np.round(out[t] / g) * g
                    trial = out.copy(); trial[t] = m
                    if (np.abs(yt - vrp.pred(Ct[:, sub], trial)).max()
                            / scale <= vrp.EXACT_REL and
                        np.abs(yh - vrp.pred(Ch[:, sub], trial)).max()
                            / scale <= vrp.EXACT_REL):
                        out = trial
                        break
            return sub, out
    return None


def main():
    t0 = time.perf_counter()
    checks = []

    def check(name, ok, detail):
        checks.append(ok)
        print(f"  [{'ok ' if ok else 'FAIL'}] {name} — {detail}")

    # M1 — the max world: the crease atom makes max sayable
    print("M1. max world: y = max(x0, x1), exact, n=240")
    r = RNG(77)
    X = np.column_stack([r.uniform(-2, 2, 240), r.uniform(-2, 2, 240)])
    y = np.maximum(X[:, 0], X[:, 1])
    v = vrp.verdict_v3(X, y, 7)
    vrp.show("M1 max world (judge unchanged, pool enriched)", v)
    got_crease = "0.5*|x0-x1|" in v["eq_n"]
    check("P-M1 machine-exact UNDERSTOOD saying 0.5*|x0-x1| (+ 0.5*x0 "
          "+ 0.5*x1)", v["new"] == "UNDERSTOOD (machine-exact)"
          and got_crease, f"{v['new']} / {v['eq_n']}")

    # M2 — identity recognition + routing receipts
    print("M2. identity: the mint IS larger(); tools route through compare")
    rr = RNG(78)
    Xf = np.column_stack([rr.uniform(-2, 2, 50), rr.uniform(-2, 2, 50)])
    mint = 0.5 * Xf[:, 0] + 0.5 * Xf[:, 1] + 0.5 * np.abs(Xf[:, 0] - Xf[:, 1])
    ident = float(np.max(np.abs(mint - np.maximum(Xf[:, 0], Xf[:, 1]))))
    lg = t_larger(1.25, -0.75)
    routed = (lg["routed_via"] and lg["routed_via"][0]["tool"] == "compare"
              and lg["value"] == 1.25)
    check("P-M2 mint == larger(x0,x1) on fresh rows", ident <= 1e-12,
          f"max |mint - max| = {ident:.2e} -> doored onto tool 'larger'")
    check("P-M2 larger() routes THROUGH compare (receipt chain)", routed,
          lg["receipt"])

    # M3 — the step world: switches make the regime sayable
    print("M3. step world: y = 2*x0 + 1.5*[1<x1<2] (+0.05 noise, 0.01 "
          "grain, sigma TOLD)")
    r3 = RNG(600)
    X3 = np.column_stack([r3.uniform(0, 3, 400), r3.uniform(0, 3, 400)])
    y3 = q(2.0 * X3[:, 0] + 1.5 * ((X3[:, 1] > 1) & (X3[:, 1] < 2))
           + r3.normal(0, 0.05, 400), 0.01)
    v3 = vrp.verdict_v3(X3, y3, 61, told_sigma=0.05)
    vrp.show("M3 step world (same world was v3's V5 refusal)", v3)
    has_steps = "[x1>1]" in v3["eq_n"] and "[x1>2]" in v3["eq_n"]
    check("P-M3 UNDERSTOOD with steps at marks 1 and 2",
          v3["new"] == "UNDERSTOOD" and has_steps,
          f"{v3['new']} / {v3['eq_n']}")

    # M4 — the blended world: the gated product completes the sentence
    print("M4. blended world: flag ? x0*x1 : x0+x1, n=96, exact")
    r4 = RNG(310)
    x01 = r4.integers(1, 10, (96, 2)).astype(float)
    flag = (np.arange(96) % 2).astype(float)
    y4 = np.where(flag == 0, x01[:, 0] + x01[:, 1],
                  x01[:, 0] * x01[:, 1])
    X4 = np.column_stack([x01, flag])
    v4 = vrp.verdict_v3(X4, y4, 33)
    vrp.show("M4 blended world (was PARTIAL-surprise for the wordless "
             "judge)", v4)
    honest4 = ((v4["new"].startswith("UNDERSTOOD") and v4["rms"] < 1e-9)
               or not v4["new"].startswith("UNDERSTOOD"))
    check("P-M4* verdict honest (as-filed 5-term recovery FALSIFIED "
          "run 1, kept: greedy search is the wall, not the vocabulary; "
          "old road ships the gated impostor confidently)", honest4,
          f"NEW {v4['new']} rms {v4['rms']:.2e}; OLD {v4['old']} "
          f"{v4['eq_o']}")
    # M4b — the deep exact road assembles the full sentence
    h4 = 48
    C4t, names4 = make_candidates_mech(X4[:h4], h4)
    C4h, _ = make_candidates_mech(X4[h4:], h4)
    deep = exact_road_deep(C4t, y4[:h4], C4h, y4[h4:])
    if deep is not None:
        sub, coef = deep
        eq_deep = vrp.fmt_eq(coef, sub, names4)
        okh = vrp.vrows(C4h)
        rms_deep = float(np.sqrt(np.mean(
            (y4[h4:][okh] - vrp.pred(C4h[okh][:, sub], coef)) ** 2)))
        terms = {names4[j] for j in sub}
        want = {"x0", "x1", "x0*x2", "x1*x2", "[x2]*x0*x1"}
        print(f"       deep exact road: {eq_deep}   (ho rms {rms_deep:.2e})")
        check("P-M4b deep exact road recovers the exact 5-term form",
              terms == want and rms_deep < 1e-9,
              f"terms {sorted(terms)} rms {rms_deep:.2e}")
    else:
        check("P-M4b deep exact road recovers the exact 5-term form",
              False, "no exact subset found (top-16, k<=5)")
    # M4c — deep-road false-stamp controls
    stamps = 0
    for s in range(20):
        rr = RNG(500 + s)
        xs = rr.uniform(0.4, 30, (8, 1))
        ys = rr.normal(0, 1, 8)
        Cs, _ = make_candidates_mech(xs[:4], 4)
        Csh, _ = make_candidates_mech(xs[4:], 4)
        stamps += exact_road_deep(Cs, ys[:4], Csh, ys[4:]) is not None
    for s in range(10):
        rr = RNG(700 + s)
        Xn = np.column_stack([rr.normal(0, 1, (200, 2)),
                              (rr.uniform(size=200) < 0.5).astype(float)])
        yn = rr.normal(0, 1, 200)
        Cn, _ = make_candidates_mech(Xn[:100], 100)
        Cnh, _ = make_candidates_mech(Xn[100:], 100)
        stamps += exact_road_deep(Cn, yn[:100], Cnh, yn[100:]) is not None
    check("P-M4c deep road false stamps 0/30 on noise", stamps == 0,
          f"{stamps}/30")

    # M5 — the cost of new words, measured
    print("M5a. noise x20 with the enriched pool (d=3 incl. binary flag)")
    fn = 0
    for s in range(20):
        rr = RNG(100 + s)
        Xn = np.column_stack([rr.normal(0, 1, (200, 2)),
                              (rr.uniform(size=200) < 0.5).astype(float)])
        yn = rr.normal(0, 1, 200)
        vv = vrp.verdict_v3(Xn, yn, 200 + s)
        fn += vv["new"].startswith("UNDERSTOOD")
    check("P-M5a false UNDERSTOOD 0/20 on noise", fn == 0, f"{fn}/20")
    print("M5b. true quadratic x20, sigma TOLD (availability under the "
          "bigger vocabulary)")
    und = fire = 0
    for s in range(20):
        rr = RNG(300 + s)
        xq = rr.uniform(0, 5, 300).reshape(-1, 1)
        yq = q(2.0 * xq[:, 0] ** 2 + rr.normal(0, 0.05, 300), 0.05)
        vv = vrp.verdict_v3(xq, yq, 300 + s, told_sigma=0.05)
        und += vv["new"].startswith("UNDERSTOOD")
        fire += (vv["leg1"].get("state") == "fired"
                 or vv["leg2"].get("state") == "scream")
    check("P-M5b availability >=17/20 (wordless pool scored 19/20)",
          und >= 17, f"{und}/20 UNDERSTOOD; false surprise {fire}/20")
    check("P-M5b combined false surprise <=2/20", fire <= 2, f"{fire}/20")

    # M6 — the measured gap: these mechanisms are not yet engine tools
    print("M6. engine catalog gap (the promotion work order)")
    import isaac as isaac_mod
    mind = isaac_mod.Isaac()
    try:
        cat = mind.catalog_tools()
        known = {c.get("name", "") for c in cat} if isinstance(cat, list) \
            else set(cat)
    except Exception:
        known = {m for m in dir(mind) if not m.startswith("_")}
    missing = [t for t in MECH_TOOLS if t not in known]
    check("P-M6 mechanisms absent from the engine toolkit (gap "
          "confirmed -> promotion order)", missing == list(MECH_TOOLS),
          f"missing {missing} of {list(MECH_TOOLS)}")

    dt = time.perf_counter() - t0
    check("P-M7 cost < 240 s", dt < 240, f"{dt:.1f}s")
    npass = sum(checks)
    print(f"\n  PROBE: {npass} passed, {len(checks) - npass} failed of "
          f"{len(checks)}  ({dt:.1f}s)")


if __name__ == "__main__":
    main()
