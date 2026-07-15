"""gate6_operator_mining.py — GATE 6: OPERATOR MINING (the handoff's
hardest open problem: "Learn a new Tier-2 rule from observed
(before, action, after) transitions: effects that survive noise
across contexts become the operator's intent. Full self-improvement
stands or falls here. Unproven; the reference files dodge it
deliberately.")

THE CLAIM UNDER TEST: dc already owns the hard parts. Operator
mining decomposes into organs Isaac has:

  EFFECT MINING       — the add/delete template that RECURS across
                        transitions (noise effects don't recur —
                        survive-noise as recurrence gating, and the
                        violators are wrong-kept with their why)
  PRECONDITION MINING — before-cells always present when the effect
                        held: constants stay literal, varying values
                        become SLOTS (the co-occurrence discipline);
                        jittering distractor cells fail the gate
  THE BINDING FUNCTION— the after-value as a FUNCTION of the
                        before-values: mind.discover() on transition
                        rows. Learning 'answer = x + y' from watched
                        episodes IS equation discovery — the engine's
                        oldest organ, pointed at transitions.

The minted Rule (the interpreter handoff's own class) then runs in
the interpreter and must pass GATE 2's binary test — unseen sums
from ONE rule nobody authored — plus controls: scrambled transitions
mint nothing; distractors never enter preconditions; corrupted
episodes are wrong-kept, not learned.

stdlib + numpy + dc isaac + the handoff interpreter.
"""
import os
import sys
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
HANDOFF = os.path.join(os.path.expanduser("~"), "Downloads",
                       "New folder (3)", "interpreter_handoff")
sys.path.insert(0, HANDOFF)
import isaac as mod                                      # noqa: E402
from isaac_interpreter import Var, Rule, Interpreter      # noqa: E402


# ═════════════════════════════════════════════════════════════════
# THE WORLD — someone performs arithmetic; Isaac only WATCHES
# (before, action, after). Distractor cells jitter; some episodes
# are corrupted (a wrong answer written down).
# ═════════════════════════════════════════════════════════════════
def episode(rng, op, corrupt=False):
    x, y = int(rng.integers(1, 10)), int(rng.integers(1, 10))
    sym = "+" if op == "add" else "*"
    before = {("num", 1, str(x)), ("sym", 2, sym), ("num", 3, str(y)),
              ("sym", 4, "="), ("want", "answer")}
    for k in range(int(rng.integers(0, 3))):          # jittering
        before.add(("noise", int(rng.integers(0, 9))))  # distractors
    v = x + y if op == "add" else x * y
    if corrupt:
        v = int(rng.integers(30, 90))                 # a wrong answer
    after = (before - {("want", "answer")}) | {("answer", str(v))}
    return frozenset(before), op, frozenset(after)


# ═════════════════════════════════════════════════════════════════
# THE MINER — effects, preconditions, and the binding function
# ═════════════════════════════════════════════════════════════════
def mine_operator(mind, transitions, name):
    """(before, after) pairs for ONE action -> a Rule, or None."""
    report = dict(name=name, wrong_kept=[], rule=None)
    # ── 1. EFFECT template: adds/deletes abstracted by (head, arity),
    #      values that vary become slots; the template must RECUR
    adds = [sorted(a - b) for b, _, a in transitions]
    dels = [sorted(b - a) for b, _, a in transitions]
    shape = Counter((tuple((c[0], len(c)) for c in ad),
                     tuple((c[0], len(c)) for c in dl))
                    for ad, dl in zip(adds, dels))
    (add_shape, del_shape), n_shape = shape.most_common(1)[0]
    if n_shape < max(3, len(transitions) // 2):
        return report                                  # nothing recurs
    keep = [i for i, (ad, dl) in enumerate(zip(adds, dels))
            if tuple((c[0], len(c)) for c in ad) == add_shape
            and tuple((c[0], len(c)) for c in dl) == del_shape]
    # ── 2. PRECONDITIONS: cells always present across kept episodes,
    #      grouped by (head, position-constants); varying value -> slot
    slots = {}

    def abstract(cells_per_ep):
        groups = {}
        for i in keep:
            for c in cells_per_ep[i]:
                groups.setdefault((c[0], len(c)), []).append((i, c))
        out = []
        for (head, ln), items in sorted(groups.items()):
            per_ep = Counter(i for i, _ in items)
            if any(per_ep.get(i, 0) != per_ep[keep[0]] for i in keep):
                continue                       # not always present:
            width = per_ep[keep[0]]            # distractors die here
            by_ep = {i: sorted(c for j, c in items if j == i)
                     for i in keep}
            for k in range(width):
                cols = []
                for pos in range(1, ln):
                    vals = {by_ep[i][k][pos] for i in keep}
                    if len(vals) == 1:
                        cols.append(vals.pop())        # constant: literal
                    else:
                        vname = f"v{len(slots)}"
                        slots[vname] = {i: by_ep[i][k][pos]
                                        for i in keep}
                        cols.append(Var(vname))        # varies: a SLOT
                out.append((head,) + tuple(cols))
        return out

    pre = abstract([b for b, _, _ in transitions])
    add_t = abstract([a - b for b, _, a in transitions])
    del_t = abstract([b - a for b, _, a in transitions])
    # ── 3. THE BINDING FUNCTION: slot in the EFFECT = f(slots in the
    #      precondition) — discover() on the transition rows; rows
    #      that violate the law are WRONG-KEPT and the law refit
    pre_vars = sorted(v for c in pre for v in
                      (a.name for a in c if isinstance(a, Var)))
    add_vars = [a for c in add_t for a in c if isinstance(a, Var)]
    if not add_vars:
        return report
    out_var = add_vars[0].name
    rows_i = list(keep)
    X = np.array([[float(slots[v][i]) for v in pre_vars]
                  for i in rows_i])
    yv = np.array([float(slots[out_var][i]) for i in rows_i])
    # SURVIVE-NOISE, before the referee: a cheap probe (linear +
    # product proposal) flags gross violators; they are WRONG-KEPT
    # with their why; discover() then referees the survivors (its
    # null-discipline rightly REFUSES a polluted table — measured)
    A = np.column_stack([X, (X[:, :1] * X[:, 1:2]
                             if X.shape[1] >= 2 else X[:, :1]),
                         np.ones(len(X))])
    inlier = np.ones(len(yv), bool)
    for _ in range(4):              # iterated trim: one survivor can
        coef, *_ = np.linalg.lstsq(A[inlier], yv[inlier], rcond=None)
        resid = np.abs(A @ coef - yv)    # drag the fit (measured);
        cut = max(3.0 * np.median(resid[inlier] + 1e-9), 1.0)
        nxt = resid < cut                # refit on survivors until
        if (nxt == inlier).all():        # the consensus stabilizes
            break
        inlier = nxt
    for i, ok_row in zip(rows_i, inlier):
        if not ok_row:
            report["wrong_kept"].append(
                dict(episode=i, why="violates the consensus effect"))
    X, yv = X[inlier], yv[inlier]
    law = f"{name}_effect"
    # SIMPLICITY DUEL (owner's law: simpler with the same accuracy
    # is better). ENGINE FINDING, measured tonight: discover() on
    # exact y=x0+x1 locks onto the product atom first (it outscores
    # each bare variable alone on positive data) and never says the
    # sum — so a plain linear candidate competes OUTRIGHT, and wins
    # only if EXACT.
    L = np.column_stack([X, np.ones(len(X))])
    lc, *_ = np.linalg.lstsq(L, yv, rcond=None)
    lc = np.round(lc, 6)
    if np.max(np.abs(L @ lc - yv)) < 1e-6:
        terms = [f"{c:+.2f}*x{j}" for j, c in enumerate(lc[:-1])
                 if abs(c) > 1e-9]
        if abs(lc[-1]) > 1e-9:
            terms.append(f"{lc[-1]:+.2f}")
        report["equation"] = " ".join(terms) + "   [linear-exact]"
        predict = lambda xs: xs @ lc[:-1] + lc[-1]
    else:
        mind.discover(X, yv, name=law)
        pred = mind.apply_law(law, X)
        if pred is None or np.max(np.abs(pred - yv)) > 1e-6:
            return report                      # no consistent intent
        report["equation"] = next(
            (f["equation"] for f in reversed(mind.findings)
             if f.get("name") == law and "equation" in f), None)
        predict = lambda xs: mind.apply_law(law, xs)

    def grounded(b, _predict=predict, _vars=pre_vars):
        xs = np.array([[float(b[v]) for v in _vars]])
        val = int(round(float(_predict(xs)[0])))
        out = []
        for c in add_t:
            out.append(tuple(str(val) if isinstance(a, Var) else a
                             for a in c))
        return out

    rule = Rule(f"mined-{name}", pre=pre,
                delete=[c for c in del_t], fn=grounded)
    report["rule"] = rule
    report["pre"] = pre
    return report


# ═════════════════════════════════════════════════════════════════
# THE RUN
# ═════════════════════════════════════════════════════════════════
def main():
    rng = np.random.default_rng(13)
    mind = mod.Isaac()
    print("=" * 64)
    print("GATE 6 — operator mining: watched transitions become rules")
    print("=" * 64)

    adds = [episode(rng, "add") for _ in range(40)]
    adds += [episode(rng, "add", corrupt=True) for _ in range(5)]
    muls = [episode(rng, "mul") for _ in range(40)]
    rng.shuffle(adds)

    r_add = mine_operator(mind, adds, "add")
    r_mul = mine_operator(mind, muls, "mul")
    for r in (r_add, r_mul):
        print(f"\n[mine:{r['name']}] equation: {r.get('equation')}")
        if r["rule"]:
            print(f"  preconditions: {r['pre']}")
            print(f"  corrupted episodes wrong-kept: "
                  f"{len(r['wrong_kept'])}/5 planted"
                  if r["name"] == "add" else
                  f"  corrupted episodes wrong-kept: "
                  f"{len(r['wrong_kept'])}")

    # ── GATE 2, run by the interpreter, on rules NOBODY authored ──
    print("\n[gate 2] the minted rules, in the interpreter, unseen:")
    minted = [r["rule"] for r in (r_add, r_mul) if r["rule"]]
    if len(minted) < 2:
        print("  REFUSED: an operator failed to mint — gate not run")
        return
    interp = Interpreter(minted, trace=False)
    ok = n = 0
    for x, y, sym, want in ((3, 5, "+", 8), (2, 4, "+", 6),
                            (7, 6, "+", 13), (9, 9, "+", 18),
                            (3, 5, "*", 15), (7, 6, "*", 42)):
        wm = {("num", 1, str(x)), ("sym", 2, sym), ("num", 3, str(y)),
              ("sym", 4, "="), ("want", "answer")}
        final, _ = interp.run(wm)
        ans = [f for f in final if f[0] == "answer"]
        got = ans[0][1] if ans else None
        n += 1
        ok += (got == str(want))
        print(f"  {x} {sym} {y} = -> {got}   "
              f"({'ok' if got == str(want) else 'WRONG, want ' + str(want)})")

    # ── controls ──────────────────────────────────────────────────
    scr = [(b, op, a) for (b, op, _), (_, _, a) in
           zip(adds, [adds[i] for i in
                      np.random.default_rng(5).permutation(len(adds))])]
    r_scr = mine_operator(mod.Isaac(), scr, "scrambled")
    print(f"\n[control] scrambled transitions mint a rule: "
          f"{'NO (honest)' if r_scr['rule'] is None else 'YES (bad)'}")
    noise_in_pre = any(
        c[0] == "noise" for c in (r_add.get("pre") or []))
    print(f"  distractor cells in preconditions: "
          f"{'YES (bad)' if noise_in_pre else 'none'}")

    print("\n" + "=" * 64)
    print("COMPLETE NUMBERS")
    print(f"  operators mined: "
          f"{sum(1 for r in (r_add, r_mul) if r['rule'])}/2; "
          f"equations: add='{r_add.get('equation')}', "
          f"mul='{r_mul.get('equation')}'")
    print(f"  gate-2 on unseen inputs: {ok}/{n}")
    print(f"  corrupted episodes wrong-kept: "
          f"{len(r_add['wrong_kept'])}/5; scrambled mints: "
          f"{'0' if r_scr['rule'] is None else '1'} (want 0); "
          f"distractors in pre: {noise_in_pre}")
    print("=" * 64)


if __name__ == "__main__":
    main()
