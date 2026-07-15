"""
ADDITIVE BOUNDARY PROBE -- attacking discover()'s greedy first admission.

THE CLAIM UNDER TEST (SESSION_HANDOFF_2026-07-13 s6, measured there):
discover() cannot say y = x0 + x1. On clean exact additive data it admits
the PRODUCT atom first (x0*x1 outscores each bare variable alone on
positive data), then chases the bent residual with trig atoms, claiming
UNDERSTOOD at r2~0.96. Greedy first-admission never reconsiders.

Three tasks, one file:
  (a) REPRO      clean y=x0+x1, positive uniform, N=2000: first admitted
                 term (engine's own round-1 choice via _Enumerator with
                 max_terms=1 on the same train half), final verdict, r2,
                 equation. Plus the mechanism number the claim rests on:
                 |corr| of x0*x1 vs each bare variable against y.
  (b) BOUNDARY   ten planted forms, fresh mind each, tabulated:
                 verdict / r2 / equation / symbolically-correct (term
                 structure from the mind's own findings model, matched to
                 truth up to constants). Sharp diagnostic: x0+x1 on
                 MIXED-SIGN U(-2,2) -- product-first was measured on
                 positive data; do mixed signs close the additive hole?
  (c) FIX DIRECTION (probe-side only, engine untouched): plain
                 least-squares on [x0,x1,1] fit on the SAME train half,
                 scored on the SAME held-out half -- does the linear pool
                 beat the admitted model? (BASELINE only, never fed back
                 to Isaac; numpy lstsq, no sklearn.)

PREDICTION (filed before the first run):
  P1 repro: first admitted term IS x0*x1; final verdict UNDERSTOOD with
     0.93 <= r2 <= 0.995 (not ~1.0); equation carries x0*x1 plus trig/pow
     chasers; NOT symbolically correct.
  P2 boundary: mixed-sign x0+x1 heals (|corr(x0*x1,y)| collapses when
     signs mix, so bare vars win round 1) -> symbolically correct.
     x0-x1 dodges the product (odd symmetry) but a near-linear
     sind(k*(x0-x1)) atom may complete >= 0.995 and win as an impostor.
     2*x0+3*x1 and x0+x1+noise fail like the clean sum. x0*x1, x0^2,
     sin(x0) recover exactly; x0*x1+0.5*x0^2 recovers or PARTIAL;
     x0/x1 uncertain (needs a beam-grown var*pow^-1 term).
  P3 control: shuffled-y world -> NO RULE (honest refusal), and the
     linear-pool baseline scores ~0 held-out there (duel sanity).
  P4 fix: on every additive failure the plain linear pool scores
     held-out r2 = 1.000 (clean) / ~0.996 (sigma=0.1), beating the
     admitted model.

Controls: shuffled-y null world (must be NO RULE); mixed-sign world is
the mechanism contrast; the linear duel is also run on the shuffled
control so the baseline itself is null-checked.

Deterministic: run as PYTHONHASHSEED=0 python3 additive_boundary_probe.py
(data rng seeded 7; isaac's module rng is seeded 0 at import and worlds
run in fixed order). N=2000 per world, 11 discover() calls -- sized to
finish well under 4 minutes.

House rule: failures below are REPORTED, never patched away.
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # repo-local isaac.py
from isaac import Isaac, _Enumerator, _atom_label

rng = np.random.default_rng(7)
N = 2000
HALF = N // 2


def corr(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    az = (a - a.mean()) / (a.std() + 1e-12)
    bz = (b - b.mean()) / (b.std() + 1e-12)
    return float(np.mean(az * bz))


def first_admitted(X, y):
    """The engine's OWN round-1 admission: _Enumerator capped at one term,
    fit on the same train half discover() uses. No probe-side reimplementation
    of the selection logic -- max_terms=1 just stops the loop after round 1."""
    enum = _Enumerator(max_terms=1)
    enum.fit(np.asarray(X[:HALF], float), np.asarray(y[:HALF], float))
    if not enum.terms:
        return "NONE-ADMITTED"
    return " | ".join("*".join(_atom_label(enum.specs[a]) for a in t)
                      for t in enum.terms)


# ── symbolic-correctness judge: term structure from the mind's own
#    findings model (specs+terms), matched to truth up to constants ──
def canon_terms(model):
    terms = []
    for t in model["terms"]:
        atoms = []
        for a in t:
            kind, i, j, k = model["specs"][a]
            atoms.append((str(kind), int(i),
                          None if j is None else int(j),
                          None if k is None else float(k)))
        terms.append(sorted(atoms, key=str))
    return terms


def a_lin(atom, i):
    kind, ii, j, k = atom
    return (kind == "var" and ii == i) or \
           (kind == "pow" and ii == i and k is not None and abs(k - 1.0) <= 0.02)


def a_pow(atom, i, p):
    kind, ii, j, k = atom
    return kind == "pow" and ii == i and k is not None \
        and abs(k - p) <= 0.02 * max(1.0, abs(p))


def a_sin(atom, i, kk):
    kind, ii, j, k = atom
    return kind == "sin" and ii == i and k is not None and abs(k - kk) <= 0.02 * kk


def t_lin(i):
    return lambda t: len(t) == 1 and a_lin(t[0], i)


def t_prod01():
    return lambda t: len(t) == 2 and (
        (a_lin(t[0], 0) and a_lin(t[1], 1)) or (a_lin(t[0], 1) and a_lin(t[1], 0)))


def t_sq0():
    return lambda t: (len(t) == 1 and a_pow(t[0], 0, 2.0)) or \
                     (len(t) == 2 and a_lin(t[0], 0) and a_lin(t[1], 0))


def t_div01():
    return lambda t: len(t) == 2 and (
        (a_lin(t[0], 0) and a_pow(t[1], 1, -1.0)) or
        (a_lin(t[1], 0) and a_pow(t[0], 1, -1.0)))


def t_sin0():
    return lambda t: len(t) == 1 and a_sin(t[0], 0, 1.0)


def match_terms(terms, matchers):
    """Exact bijection: every admitted term must be one planted truth term,
    every truth term must appear once. Extra terms = symbolically wrong."""
    if len(terms) != len(matchers):
        return False
    used = [False] * len(matchers)
    for t in terms:
        hit = next((m for m in range(len(matchers))
                    if not used[m] and matchers[m](t)), None)
        if hit is None:
            return False
        used[hit] = True
    return True


def linear_duel(X, y):
    """BASELINE (probe-side, never fed to Isaac): least squares on
    [x0, x1, 1], trained on discover()'s train half, scored held-out.
    Returns (heldout_r2, coefs)."""
    A = np.column_stack([X[:, :2], np.ones(len(y))])
    beta, *_ = np.linalg.lstsq(A[:HALF], y[:HALF], rcond=None)
    p = A[HALF:] @ beta
    yh = y[HALF:]
    ss = float(np.sum((yh - yh.mean()) ** 2))
    return 1 - float(np.sum((yh - p) ** 2)) / max(ss, 1e-12), beta


def run_world(tag, X, y, matchers, want_first=False):
    t0 = time.time()
    mind = Isaac()                       # fresh throwaway mind
    res = mind.discover(X, y)
    secs = time.time() - t0
    model = mind.findings[-1].get("model") if mind.findings else None
    sym = bool(model) and matchers is not None and \
        match_terms(canon_terms(model), matchers)
    n_terms = len(model["terms"]) if model else 0
    first = first_admitted(X, y) if want_first else ""
    lin, lin_beta = linear_duel(X, y)
    row = dict(tag=tag, verdict=res["verdict"], r2=res["r2"],
               r2_edge=res["r2_edge"], sym=sym, lin_r2=lin,
               lin_beta=lin_beta, n_terms=n_terms,
               first=first, eq=res["equation"], secs=secs)
    print(f"{tag:24s} {res['verdict']:10s} r2 {res['r2']:+8.5f} "
          f"edge {res['r2_edge']:+8.5f} sym={'YES' if sym else 'no ':3s} "
          f"terms={n_terms:2d} linLSQ {lin:+8.5f} [{secs:4.1f}s]")
    print(f"{'':24s} eq: {res['equation'][:150]}")
    if want_first:
        print(f"{'':24s} first admitted: {first}")
    return row


def main():
    t_all = time.time()
    U = lambda lo, hi: rng.uniform(lo, hi, N)   # noqa: E731

    # data pools (positive default matches the s6 setting; ranges printed)
    print("data ranges: positive U(0.1,2.1); mixed U(-2,2); "
          "denom U(1,3); sin-world U(-3,3); N=%d" % N)

    print("\n================ (a) REPRO: clean y = x0 + x1, positive ================")
    x0, x1 = U(0.1, 2.1), U(0.1, 2.1)
    X_add, y_add = np.column_stack([x0, x1]), x0 + x1
    print("mechanism |corr vs y|:  x0 %.4f   x1 %.4f   x0*x1 %.4f"
          % (abs(corr(x0, y_add)), abs(corr(x1, y_add)),
             abs(corr(x0 * x1, y_add))))
    m0, m1 = U(-2, 2), U(-2, 2)
    y_mix = m0 + m1
    print("mixed-sign contrast:    x0 %.4f   x1 %.4f   x0*x1 %.4f"
          % (abs(corr(m0, y_mix)), abs(corr(m1, y_mix)),
             abs(corr(m0 * m1, y_mix))))

    add_truth = [t_lin(0), t_lin(1)]
    rows = []
    rows.append(run_world("W1  y=x0+x1 (pos)", X_add, y_add, add_truth,
                          want_first=True))

    print("\n================ (b) BOUNDARY MAP ================")
    a, b = U(0.1, 2.1), U(0.1, 2.1)
    rows.append(run_world("W2  y=x0-x1 (pos)",
                          np.column_stack([a, b]), a - b, add_truth,
                          want_first=True))
    a, b = U(0.1, 2.1), U(0.1, 2.1)
    rows.append(run_world("W3  y=2x0+3x1 (pos)",
                          np.column_stack([a, b]), 2 * a + 3 * b, add_truth,
                          want_first=True))
    rows.append(run_world("W4  y=x0+x1 (MIXED)",
                          np.column_stack([m0, m1]), y_mix, add_truth,
                          want_first=True))
    a, b = U(0.1, 2.1), U(0.1, 2.1)
    rows.append(run_world("W5  y=x0*x1 (pos)",
                          np.column_stack([a, b]), a * b, [t_prod01()],
                          want_first=True))
    a, b = U(0.1, 2.1), U(0.1, 2.1)
    rows.append(run_world("W6  y=x0*x1+0.5*x0^2",
                          np.column_stack([a, b]), a * b + 0.5 * a * a,
                          [t_prod01(), t_sq0()]))
    a, b = U(-2, 2), U(-2, 2)
    rows.append(run_world("W7  y=x0^2 (mixed)",
                          np.column_stack([a, b]), a * a, [t_sq0()]))
    a, b = U(0.1, 2.1), U(1.0, 3.0)
    rows.append(run_world("W8  y=x0/x1 (den 1..3)",
                          np.column_stack([a, b]), a / b, [t_div01()]))
    a, b = U(-3, 3), U(-3, 3)
    rows.append(run_world("W9  y=sin(x0)",
                          np.column_stack([a, b]), np.sin(a), [t_sin0()]))
    a, b = U(0.1, 2.1), U(0.1, 2.1)
    noise = rng.normal(0, 0.1, N)
    rows.append(run_world("W10 y=x0+x1+N(0,0.1)",
                          np.column_stack([a, b]), a + b + noise, add_truth,
                          want_first=True))

    print("\n================ CONTROL: shuffled y (must refuse) ================")
    a, b = U(0.1, 2.1), U(0.1, 2.1)
    y_sh = rng.permutation(a + b)
    rows.append(run_world("C1  y=perm(x0+x1)",
                          np.column_stack([a, b]), y_sh, None))

    print("\n================ (c) FIX DIRECTION: the linear duel ================")
    print("plain LSQ on [x0,x1,1] (probe-side BASELINE) vs admitted model, "
          "same split, held-out r2 (accuracy) AND parsimony (n terms, coefs):")
    for r in rows:
        if r["tag"].startswith(("W1", "W2", "W3", "W4", "W10", "C1")):
            gap = r["lin_r2"] - r["r2"]
            who = "LINEAR-POOL WINS" if gap > 0.005 else \
                  ("accuracy tie" if abs(gap) <= 0.005 else "isaac wins")
            bstr = " ".join(f"{v:+.3f}" for v in r["lin_beta"])
            print(f"  {r['tag']:24s} linLSQ {r['lin_r2']:+8.5f}  "
                  f"isaac {r['r2']:+8.5f}  gap {gap:+8.5f}  -> {who}")
            print(f"  {'':24s} lin coefs [x0 x1 1] = [{bstr}]  "
                  f"({2} terms) vs isaac {r['n_terms']} terms, sym="
                  f"{'YES' if r['sym'] else 'no'}")

    print("\n================ VERDICT SHEET ================")
    def sheet(cap, verdict, ev):
        print(f"  {cap:34s} {verdict:15s} {ev}")
    w = {r["tag"].split()[0]: r for r in rows}

    r1 = w["W1"]
    sheet("s6a product admitted first (W1)",
          "REPRODUCED" if "x0*x1" in r1["first"] else "NOT-REPRODUCED",
          f"first admitted = {r1['first']!r}; mechanism corrs printed above")
    sheet("s6b accuracy hole r2~0.96 (W1)",
          "REPRODUCED" if r1["r2"] <= 0.99 else "NOT-REPRODUCED",
          f"measured r2={r1['r2']:.5f} (claimed ~0.96)")
    sheet("s6c wrong-form articulation (W1)",
          "REPRODUCED" if (r1["verdict"] == "UNDERSTOOD" and not r1["sym"])
          else "NOT-REPRODUCED",
          f"verdict={r1['verdict']} sym={r1['sym']} terms={r1['n_terms']} "
          f"(truth needs 2)")
    r3 = w["W3"]
    sheet("s6d 'never reconsiders' (W3)",
          "REFUTED" if ("x0*x1" in r3["first"] and r3["sym"])
          else ("REPRODUCED" if "x0*x1" in r3["first"] else "UNTESTED-HERE"),
          f"W3 first={r3['first']!r} yet final sym={r3['sym']} "
          f"terms={r3['n_terms']} (weight-prune can evict the product)")
    for k in ["W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8", "W9", "W10"]:
        r = w[k]
        if r["sym"] and r["verdict"] == "UNDERSTOOD" and r["r2"] >= 0.9:
            v = "WORKS"
        elif r["verdict"] == "UNDERSTOOD" and not r["sym"]:
            v = "BROKEN"          # fake understanding: wrong form said as law
        elif r["verdict"] in ("PARTIAL", "CAPTURED"):
            v = "PARTIAL"
        elif r["verdict"] == "NO RULE":
            v = "BROKEN"          # a sayable planted law refused
        else:
            v = "PARTIAL"
        sheet(f"say truth of {r['tag']}", v,
              f"verdict={r['verdict']} r2={r['r2']:.4f} sym={r['sym']}")
    rc = w["C1"]
    sheet("shuffled-y control", "HONEST-REFUSAL" if rc["verdict"] == "NO RULE"
          else "BROKEN", f"verdict={rc['verdict']} r2={rc['r2']:.4f} "
          f"linLSQ={rc['lin_r2']:.4f}")
    fails = [r for r in rows if r["tag"].split()[0] in
             ("W1", "W2", "W3", "W4", "W10") and not r["sym"]]
    if fails:
        won = [r for r in fails if r["lin_r2"] > r["r2"] + 0.005]
        tied = [r for r in fails if abs(r["lin_r2"] - r["r2"]) <= 0.005]
        slimmer = [r for r in fails if r["n_terms"] > 2]
        sheet("fix-direction duel (accuracy)",
              "CONFIRMED" if len(won) == len(fails) else
              ("ACCURACY-TIE" if len(won) == 0 and len(tied) == len(fails)
               else "PARTIAL"),
              f"linear pool beats admitted model on {len(won)}/{len(fails)} "
              f"additive failures; ties {len(tied)}/{len(fails)}")
        sheet("fix-direction duel (parsimony)",
              "CONFIRMED" if len(slimmer) == len(fails) else "PARTIAL",
              f"{len(slimmer)}/{len(fails)} additive failures carry >2 terms "
              f"vs the 2-term linear pool at equal accuracy")
    else:
        sheet("fix-direction linear duel", "MOOT",
              "no additive failures to duel")
    print(f"\ntotal wall time {time.time() - t_all:.1f}s")


if __name__ == "__main__":
    main()
