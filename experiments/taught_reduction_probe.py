"""
TAUGHT REDUCTION PROBE -- reference toy, NOT the engine.
Owner's law applied: simplification is LEARNED CONTENT, not a programmed
display rule. Teaching = pairs (expression -> reduced expression).
A rewrite rule is minted only if:
  (1) RECURRENCE: the template generalizes >=2 independent teaching pairs
      (anti-unification across pairs; slots only where the pairs vary),
  (2) VALUE GATE: LHS and RHS agree EXACTLY on random slot assignments
      (polynomial-identity testing -- a wrong identity dies with prob ~1),
  (3) DESCENT: RHS strictly smaller than LHS (termination by construction:
      every application shrinks the expression, so reduce() cannot loop).
Rejected rules are WRONG-KEPT with their why.
Controls: scrambled pairings mint nothing; a poisoned pair (x+1 -> x) is
refused by the value gate.

(Transcribed to dc/experiments 2026-07-15 from the 07-14 session's paste;
punctuation re-encoded to ASCII, code verbatim.)
"""
import numpy as np
rng = np.random.default_rng(0)

# expressions: nested tuples ('op', child, ...), leaves = str vars / numbers
def size(e): return 1 + sum(size(c) for c in e[1:]) if isinstance(e, tuple) else 1
def show(e):
    if not isinstance(e, tuple): return str(e)
    op = e[0]
    if op == 'neg': return f"(-{show(e[1])})"
    return "(" + f" {op} ".join(show(c) for c in e[1:]) + ")"

def evaluate(e, env):
    if isinstance(e, tuple):
        op = e[0]; a = [evaluate(c, env) for c in e[1:]]
        if op == '+': return a[0] + a[1]
        if op == '-': return a[0] - a[1]
        if op == '*': return a[0] * a[1]
        if op == '/': return a[0] / a[1]
        if op == '^': return a[0] ** a[1]
        if op == 'neg': return -a[0]
        raise ValueError(op)
    if isinstance(e, str): return env[e]
    return float(e)

# ---- anti-unification: generalize two trees; differing subtrees -> slots
def antiunify(a, b, table):
    if a == b: return a
    if isinstance(a, tuple) and isinstance(b, tuple) and a[0] == b[0] and len(a) == len(b):
        return (a[0],) + tuple(antiunify(x, y, table) for x, y in zip(a[1:], b[1:]))
    key = (repr(a), repr(b))
    if key not in table: table[key] = f"?{len(table)}"
    return table[key]

def slots(t):
    if isinstance(t, tuple): return set().union(*[slots(c) for c in t[1:]]) if len(t) > 1 else set()
    return {t} if isinstance(t, str) and t.startswith('?') else set()

# ---- matching / substitution for applying rules
def match(pat, e, binding):
    if isinstance(pat, str) and pat.startswith('?'):
        if pat in binding: return binding[pat] == e
        binding[pat] = e; return True
    if isinstance(pat, tuple):
        if not (isinstance(e, tuple) and e[0] == pat[0] and len(e) == len(pat)): return False
        return all(match(p, c, binding) for p, c in zip(pat[1:], e[1:]))
    return pat == e

def subst(t, binding):
    if isinstance(t, str) and t.startswith('?'): return binding[t]
    if isinstance(t, tuple): return (t[0],) + tuple(subst(c, binding) for c in t[1:])
    return t

# ---- the gates
def value_gate(lhs, rhs, n_probe=25):
    """Exact agreement on random slot assignments (nonzero, to dodge /0)."""
    sl = sorted(slots(lhs) | _vars(lhs) | slots(rhs) | _vars(rhs))
    for _ in range(n_probe):
        env = {s: float(rng.uniform(0.5, 4.0) * rng.choice([-1, 1])) for s in sl}
        try:
            va, vb = evaluate(lhs, env), evaluate(rhs, env)
        except ZeroDivisionError:
            continue
        if not np.isclose(va, vb, rtol=1e-10, atol=1e-12): return False
    return True

def _vars(t):
    if isinstance(t, tuple): return set().union(*[_vars(c) for c in t[1:]]) if len(t) > 1 else set()
    return {t} if isinstance(t, str) and not t.startswith('?') else set()

def mint_rules(pairs, adopt_min=2):
    """Anti-unify every pair of teaching pairs; gate the templates."""
    minted, wrong = [], []
    seen = set()
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            table = {}
            L = antiunify(pairs[i][0], pairs[j][0], table)
            R = antiunify(pairs[i][1], pairs[j][1], table)
            key = (repr(L), repr(R))
            if key in seen: continue
            seen.add(key)
            why = None
            if slots(R) - slots(L):
                why = "RHS invents a slot"
                cover = 0
            else:
                cover = sum(1 for (a, b) in pairs
                            if (lambda m: match(L, a, m) and subst(R, m) == b)({}))
                if cover < adopt_min:      why = f"recurrence {cover} < {adopt_min}"
                elif not value_gate(L, R): why = "VALUE GATE: sides disagree on probe"
                elif size(R) >= size(L):   why = "no descent (RHS not smaller)"
            if why: wrong.append((show(L), show(R), why))
            else:   minted.append((L, R, cover))
    return minted, wrong

# ---- reduce(): apply minted rules to fixed point (terminates by descent)
def reduce_expr(e, rules):
    changed = True
    while changed:
        changed = False
        for L, R, _ in rules:
            m = {}
            if match(L, e, m):
                e = subst(R, m); changed = True; break
        if not changed and isinstance(e, tuple):
            new = (e[0],) + tuple(reduce_expr(c, rules) for c in e[1:])
            if new != e: e, changed = new, True
    return e

# ================= the teaching corpus (what a tutor would show him)
P = [
    (('*', 1, 'a'), 'a'),            # 1*a -> a
    (('*', 1, 'b'), 'b'),
    (('*', 'c', 1), 'c'),            # c*1 -> c
    (('*', 'd', 1), 'd'),
    (('+', 'a', 0), 'a'),            # a+0 -> a
    (('+', 'b', 0), 'b'),
    (('^', 'a', 1), 'a'),            # a^1 -> a
    (('^', 'b', 1), 'b'),
    (('-', 'a', 'a'), 0),            # a-a -> 0
    (('-', 'b', 'b'), 0),
    (('*', 0, 'a'), 0),              # 0*a -> 0
    (('*', 0, 'b'), 0),
    (('+', 0, 'a'), 'a'),            # THE LESSON THE MISS ASKED FOR: 0+a -> a
    (('+', 0, 'b'), 'b'),
]

def run():
    rules, wrong = mint_rules(P)
    print("MINTED RULES (recurrence >= 2, value-gated, strict descent):")
    for L, R, cov in rules:
        print(f"  {show(L)}  ->  {show(R)}   [covers {cov} teaching pairs]")
    print(f"\nWRONG-KEPT during mining: {len(wrong)} candidates refused")
    reasons = {}
    for _, _, why in wrong: reasons[why.split(':')[0]] = reasons.get(why.split(':')[0], 0) + 1
    for k, v in reasons.items(): print(f"  {v:3d} x {k}")

    # unseen expressions, never in any teaching pair
    tests = [
        (('*', 1, ('+', 'y', 0)), 'y'),                          # 1*(y+0) -> y
        (('^', ('*', 'z', 1), 1), 'z'),                          # (z*1)^1 -> z
        (('+', ('*', 0, 'q'), ('*', 1, 'w')), 'w'),              # 0*q + 1*w -> w  (needs 0+w)
        (('-', ('^', 'k', 1), 'k'), 0),                          # k^1 - k -> 0
        (('*', ('+', 'm', 0), ('^', 'n', 1)), ('*', 'm', 'n')),  # (m+0)*(n^1) -> m*n
    ]
    print("\nREDUCTION ON UNSEEN EXPRESSIONS:")
    ok = 0
    for e, want in tests:
        got = reduce_expr(e, rules)
        # accept 0+w == w only if a 0+x rule was taught; report honestly
        hit = got == want
        ok += hit
        print(f"  {show(e):<28} -> {show(got):<12} expected {show(want):<10} {'OK' if hit else 'MISS'}")
    print(f"  {ok}/{len(tests)}")

    # CONTROL 1: scrambled pairings must mint nothing
    L_list = [p[0] for p in P]; R_list = [p[1] for p in P]
    idx = rng.permutation(len(P))
    scrambled = [(L_list[i], R_list[(int(i) + 3) % len(P)]) for i in idx]
    s_rules, _ = mint_rules(scrambled)
    print(f"\nCONTROL scrambled pairings: {len(s_rules)} rules minted (expect 0)")

    # CONTROL 2: poisoned teaching (x+1 -> x) must be refused by the value gate
    poison = P + [(('+', 'a', 1), 'a'), (('+', 'b', 1), 'b')]
    p_rules, p_wrong = mint_rules(poison)
    poisoned_in = any(show(L) == "(?0 + 1)" or "+ 1)" in show(L) for L, R, _ in p_rules)
    caught = [w for w in p_wrong if "+ 1)" in w[0] and "VALUE" in w[2]]
    print(f"CONTROL poisoned pair (x+1 -> x): minted={poisoned_in} (expect False); "
          f"value-gate refusals of it: {len(caught)}")
    if caught: print(f"  e.g. {caught[0][0]} -> {caught[0][1]}: {caught[0][2]}")

if __name__ == "__main__":
    run()
