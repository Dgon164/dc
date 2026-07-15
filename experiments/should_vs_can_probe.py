"""
SHOULD vs CAN PROBE -- reference toy, NOT the engine.
Question: can the should/can distinction be TAUGHT, not programmed?

Design under test:
  THE ROUTER: a correction (said -> corrected) is evaluated on random
  assignments. VALUES EQUAL -> the tutor taught CONVENTION (a should):
  minted as a context-scoped NORM with rate-measured strength -- never a
  gate, always violable, deviation would carry a receipt. VALUES DIFFER
  -> the tutor made a TRUTH CLAIM: it faces the gates; a false one is
  refused and wrong-kept (norms cannot be poisoned by a lying tutor).
  GUARDS: a convention that fails at special points (x/x at 0) is minted
  WITH its discovered domain guard -- "technically allowed" has edges,
  found by probing, not programmed.
  CONTEXT: the same identity can carry OPPOSITE norms in different
  contexts (expand for integration, factor for root-finding) -- no
  conflict, because a should is indexed by the want it serves.

(Transcribed to dc/experiments 2026-07-15 from the 07-14 session's paste;
punctuation re-encoded to ASCII, code verbatim.)
"""
import numpy as np
rng = np.random.default_rng(3)

def size(e): return 1 + sum(size(c) for c in e[1:]) if isinstance(e, tuple) else 1
def show(e):
    if not isinstance(e, tuple): return str(e)
    return "(" + f" {e[0]} ".join(show(c) for c in e[1:]) + ")"

def evaluate(e, env):
    if isinstance(e, tuple):
        a = [evaluate(c, env) for c in e[1:]]
        op = e[0]
        if op == '+': return a[0] + a[1]
        if op == '-': return a[0] - a[1]
        if op == '*': return a[0] * a[1]
        if op == '/': return a[0] / a[1]
        if op == '^': return a[0] ** a[1]
        raise ValueError(op)
    return env[e] if isinstance(e, str) else float(e)

def evars(t):
    if isinstance(t, tuple): return set().union(*[evars(c) for c in t[1:]])
    return {t} if isinstance(t, str) else set()

def values_equal(a, b, n=30):
    """Probe on random nonzero assignments; also probe zeros to find guards.
    Returns (equal, guard) -- guard names variables whose zero breaks a side."""
    vs = sorted(evars(a) | evars(b))
    for _ in range(n):
        env = {v: float(rng.uniform(0.5, 4) * rng.choice([-1, 1])) for v in vs}
        try:
            if not np.isclose(evaluate(a, env), evaluate(b, env), rtol=1e-10):
                return False, None
        except ZeroDivisionError:
            return False, None
    guard = []
    for v in vs:                       # special-point probe: each var at 0
        env = {u: (0.0 if u == v else 1.7) for u in vs}
        try:
            ok = np.isclose(evaluate(a, env), evaluate(b, env), rtol=1e-10)
        except ZeroDivisionError:
            ok = False
        if not ok: guard.append(f"{v} != 0")
    return True, (", ".join(guard) if guard else None)

# ------------------------------------------------------------------ corpus
# corrections: (context/want, what he said, what the tutor corrected it to)
CORRECTIONS = [
    # convention drilled by repetition -> strong norm in 'simplify'
    ("simplify", ('*', 1, 'a'), 'a'),
    ("simplify", ('*', 1, 'b'), 'b'),
    ("simplify", ('*', 1, 'c'), 'c'),
    ("simplify", ('+', 'a', 0), 'a'),
    ("simplify", ('+', 'b', 0), 'b'),
    # convention with a domain edge: x/x -> 1, EXCEPT at 0
    ("simplify", ('/', 'a', 'a'), 1),
    ("simplify", ('/', 'b', 'b'), 1),
    # the SAME identity, OPPOSITE norms in different wants:
    ("integrate", ('^', ('+', 'a', 'b'), 2),
                  ('+', ('+', ('^', 'a', 2), ('*', 2, ('*', 'a', 'b'))), ('^', 'b', 2))),
    ("integrate", ('^', ('+', 'c', 'd'), 2),
                  ('+', ('+', ('^', 'c', 2), ('*', 2, ('*', 'c', 'd'))), ('^', 'd', 2))),
    ("root-find", ('+', ('+', ('^', 'a', 2), ('*', 2, ('*', 'a', 'b'))), ('^', 'b', 2)),
                  ('^', ('+', 'a', 'b'), 2)),
    ("root-find", ('+', ('+', ('^', 'c', 2), ('*', 2, ('*', 'c', 'd'))), ('^', 'd', 2)),
                  ('^', ('+', 'c', 'd'), 2)),
    # a lying tutor: value-changing "correction" -- must be refused
    ("simplify", ('+', 'a', 1), 'a'),
    # a one-off style quirk: seen once, norm strength must reflect that
    ("simplify", ('*', 'q', 2), ('*', 2, 'q')),
]

def route():
    norms, refused = {}, []
    for ctx, said, corr in CORRECTIONS:
        eq, guard = values_equal(said, corr)
        if not eq:
            refused.append((ctx, show(said), show(corr),
                            "VALUE CHANGED: not convention; truth claim -> gates -> refused"))
            continue
        key = (ctx, show(said), show(corr))
        # aggregate by shape family (toy: by op signature of 'said')
        fam = (ctx, said[0] if isinstance(said, tuple) else 'atom',
               tuple(type(c).__name__ if not isinstance(c, (int, float)) else 'const'
                     for c in (said[1:] if isinstance(said, tuple) else ())))
        rec = norms.setdefault(fam, {"count": 0, "guard": None, "examples": []})
        rec["count"] += 1
        rec["examples"].append((show(said), show(corr)))
        if guard and rec["guard"] is None: rec["guard"] = guard  # first evidenced guard
    return norms, refused

def run():
    norms, refused = route()
    ADOPT_MIN = 2
    print("ROUTED AS NORMS (value-preserving corrections = convention lane):")
    for (ctx, op, sig), rec in norms.items():
        strength = f"{rec['count']} sightings"
        status = "ADOPTED" if rec["count"] >= ADOPT_MIN else "noted, below adoption floor"
        g = f"  GUARD DISCOVERED: {rec['guard']}" if rec["guard"] else ""
        ex = rec["examples"][0]
        print(f"  [{ctx:<9}] {ex[0]} -> {ex[1]}   ({strength}: {status}){g}")
    print("\nROUTED AS TRUTH CLAIMS AND REFUSED (wrong-kept):")
    for ctx, s, c, why in refused:
        print(f"  [{ctx}] {s} -> {c}: {why}")

    # context application: same expression, two wants, two answers
    print("\nCONTEXT TEST -- same identity, opposite shoulds, no conflict:")
    print("  want=integrate : (a+b)^2 SHOULD become the expanded form (norm adopted, 2 sightings)")
    print("  want=root-find : expanded form SHOULD become (a+b)^2 (norm adopted, 2 sightings)")
    print("  both remain CAN in every context -- a norm never gates; deviation carries a receipt")

if __name__ == "__main__":
    run()
