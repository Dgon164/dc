"""gate6_preconditions_probe.py — GATE 6 ATTACK: constant distractors +
multi-cell / conditional effects (07-13 next item 1). Action labels GIVEN
(each world is one operator's episodes — this attacks the mining
discipline itself, not the clustering, which gate6_unlabeled_probe.py
already attacked).

THE CLAIM UNDER TEST: the gate6 precondition/effect/binding discipline
(always-present + constant-multiplicity cells -> template; constant value
-> LITERAL, varying value -> SLOT; binding via fresh-mind discover(), the
probe-side linear duel as an explicitly labeled BASELINE) survives three
harder worlds:

  W1 CONSTANT DISTRACTOR — a semantically irrelevant cell ("bg","7")
     identical across ALL episodes. Co-occurrence is trivially satisfied
     (presence 1.0, multiplicity constant, value constant), so nothing in
     the discipline can exclude it. If it enters the preconditions as a
     required literal, a future state WITHOUT it can never fire — the trap
     is demonstrated concretely with fire counts via a probe-side matcher.
  W2 MULTI-CELL EFFECT — the operator writes TWO cells per episode:
     ("sum", a+b) and ("prod", a*b). Does effect mining recover a
     two-cell add-template with two output slots and TWO binding
     functions?
  W3 CONDITIONAL EFFECT — answer = a+b if flag==0 else a*b, flag a
     before-cell taking values {0,1}. Does the discipline mint ONE
     blended broken rule, or (via a piecewise fallback: partition on a
     low-cardinality precondition slot, re-mine so the split slot becomes
     a LITERAL) TWO rules with flag in their preconditions?

PREDICTIONS (filed before the first run):
  P1 (W1) ("bg","7") ENTERS the mined preconditions as a LITERAL —
     co-occurrence cannot exclude a constant — while the jittering noise
     cells (multiplicity 0-2) are excluded by the multiplicity gate. The
     contrast is the point: the gate works on jitter, is blind to
     constants.
  P2 (W1 trap) the minted rule fires 6/6 on unseen states WITH bg=7
     (answers exact), 0/6 on the SAME states with the bg cell removed,
     0/6 with bg=8 — a falsely required literal blocks all transfer.
  P3 (W1 binding) prod binding is engine-exact (discover() UNDERSTOOD,
     train maxerr < 1e-6), unseen 3/3.
  P4 (W2) the add-template has exactly TWO cells (heads sum, prod), one
     slot each; prod binds engine-exact; sum's raw discover() is NOT
     exact (the known additive blind spot) and the probe-side linear duel
     [BASELINE, never feeds Isaac] rescues it; unseen 3/3 + 3/3.
  P5 (W3 naive) the whole-cluster binding does NOT mint a single exact
     blended rule. Predicted mechanism: the robust trim gravitates to the
     add half (linear-expressible) and collapses below min_keep=29 ->
     binding REFUSED (the collapse guard is what stands between the
     discipline and a flag-blind half-rule). The untrimmed raw-discover
     diagnostic on the blended rows is also not exact.
  P6 (W3 piecewise) partitioning on the only low-cardinality slot (flag)
     re-mines flag as LITERAL "0"/"1" in each partition's preconditions;
     flag=1 (mul) binds engine-exact, flag=0 (add) needs the duel rescue;
     unseen 3/3 per branch; the flag=1 rule fires 0/6 on flag=0 states
     and 6/6 on flag=1 states (and vice versa) — genuinely conditional.
  P7 (controls) scrambled versions (befores paired with permuted afters)
     of all three worlds mint NOTHING — effect-shape recurrence falls
     below threshold. The flag-SHUFFLED W3 control (flags reassigned
     across episodes, so flag no longer predicts the branch) passes the
     shape gate but every piecewise partition is ~50/50 mixed -> trim
     collapse -> 0 mints: flag in the preconditions is EARNED, not free.

Deterministic: run as
  cd /home/user/dc/experiments && PYTHONHASHSEED=0 python3 gate6_preconditions_probe.py
Worlds sized at 48 episodes each to finish well under 4 min. stdlib +
numpy for the cognition; sklearn unused; fresh throwaway Isaac() minds
only; nothing persisted. The matcher/miner scaffolding is probe-side
(house gate6 pattern); all equation cognition is engine discover() plus
the explicitly labeled linear-duel BASELINE.
"""
import os
import sys
import time
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import isaac as mod                                       # noqa: E402

rng = np.random.default_rng(17)
TOL = 1e-6
PART_MIN = 4

UNSEEN_AB = [(7, 6), (9, 9), (2, 8)]
RESERVED = set(UNSEEN_AB) | {(3, 5), (4, 7), (8, 8)}
DEMO_AB = sorted(RESERVED)          # 6 reserved pairs for fire demos


# ═════════════════════════════════════════════════════════════════
# THE WORLDS — labels given; every episode in a world is one operator
# ═════════════════════════════════════════════════════════════════
def draw_ab():
    while True:
        a, b = int(rng.integers(1, 10)), int(rng.integers(1, 10))
        if (a, b) not in RESERVED:
            return a, b


def add_jitter(before):
    for _ in range(int(rng.integers(0, 3))):              # jittering
        before.add(("noise", int(rng.integers(0, 9))))    # distractors


def ep_w1():
    """answer = a*b; ("bg","7") constant across ALL episodes, never read."""
    a, b = draw_ab()
    before = {("num", 1, str(a)), ("num", 3, str(b)), ("sym", 2, "*"),
              ("want", "answer"), ("bg", "7")}
    add_jitter(before)
    after = (before - {("want", "answer")}) | {("answer", str(a * b))}
    return frozenset(before), frozenset(after)


def ep_w2():
    """operator writes TWO cells: sum AND prod."""
    a, b = draw_ab()
    before = {("num", 1, str(a)), ("num", 3, str(b)), ("sym", 2, "@"),
              ("want", "answer")}
    add_jitter(before)
    after = (before - {("want", "answer")}) | {("sum", str(a + b)),
                                               ("prod", str(a * b))}
    return frozenset(before), frozenset(after)


def ep_w3(f):
    """answer = a+b if flag==0 else a*b; flag is a before-cell."""
    a, b = draw_ab()
    before = {("num", 1, str(a)), ("num", 3, str(b)), ("flag", str(f)),
              ("want", "answer")}
    add_jitter(before)
    v = a + b if f == 0 else a * b
    after = (before - {("want", "answer")}) | {("answer", str(v))}
    return frozenset(before), frozenset(after)


def scrambled(trans, seed):
    perm = np.random.default_rng(seed).permutation(len(trans))
    return [(trans[i][0], trans[int(j)][1]) for i, j in enumerate(perm)]


def w3_flagshuffle(trans, seed=23):
    """Reassign flag values across episodes; afters keep the ORIGINAL
    branch's answer — flag no longer predicts anything."""
    flags = [next(c[1] for c in b if c[0] == "flag") for b, _ in trans]
    perm = np.random.default_rng(seed).permutation(len(flags))
    out = []
    for (b, a), j in zip(trans, perm):
        nf = ("flag", flags[int(j)])
        nb = frozenset({c for c in b if c[0] != "flag"} | {nf})
        na = frozenset({c for c in a if c[0] != "flag"} | {nf})
        out.append((nb, na))
    return out


# ═════════════════════════════════════════════════════════════════
# THE MINER — house gate6 discipline (templates, slots, trim, bind)
# ═════════════════════════════════════════════════════════════════
def sig_of(cells):
    return tuple((c[0], len(c)) for c in sorted(cells))


def shape_sig(b, a):
    return (sig_of(a - b), sig_of(b - a))


def mine_template(idxs, cells_of):
    """Always-present, constant-multiplicity (head, arity) groups become
    template cells: constant positions LITERAL, varying become SLOTS."""
    groups = {}
    for i in idxs:
        for c in cells_of(i):
            groups.setdefault((c[0], len(c)), []).append((i, c))
    out, slots, stats = [], {}, []
    for (head, ln), items in sorted(groups.items()):
        per_ep = Counter(i for i, _ in items)
        pres = sum(1 for i in idxs if per_ep.get(i, 0) > 0) / len(idxs)
        mults = sorted({per_ep.get(i, 0) for i in idxs})
        stats.append((head, ln, pres, mults))
        if len(mults) != 1:
            continue                    # jitter distractors die HERE
        width = mults[0]
        by_ep = {i: sorted(c for j, c in items if j == i) for i in idxs}
        for k in range(width):
            cols = []
            for pos in range(1, ln):
                vals = {by_ep[i][k][pos] for i in idxs}
                if len(vals) == 1:
                    cols.append(("LIT", vals.pop()))
                else:
                    nm = f"{head}{ln}k{k}p{pos}"
                    slots[nm] = {i: by_ep[i][k][pos] for i in idxs}
                    cols.append(("SLOT", nm))
            out.append((head, tuple(cols)))
    return out, slots, stats


def numeric_slot(vals):
    try:
        for v in vals.values():
            float(v)
        return True
    except (TypeError, ValueError):
        return False


def fmt_template(t):
    return " ".join(
        f"{h}(" + ",".join(str(v) if kind == "LIT" else "?" + str(v)
                           for kind, v in cols) + ")" for h, cols in t) \
        or "(empty)"


def robust_trim(X, yv):
    A = np.column_stack([X, (X[:, :1] * X[:, 1:2] if X.shape[1] >= 2
                             else X[:, :1]), np.ones(len(X))])
    inlier = np.ones(len(yv), bool)
    for _ in range(6):
        coef, *_ = np.linalg.lstsq(A[inlier], yv[inlier], rcond=None)
        resid = np.abs(A @ coef - yv)
        cut = max(3.0 * np.median(resid[inlier] + 1e-9), 1.0)
        nxt = resid < cut
        if (nxt == inlier).all():
            break
        inlier = nxt
    return inlier


def bind_one(tag, idxs, num_pre, pre_s, out_nm, out_vals):
    """Bind ONE output slot = f(numeric precondition slots).
    Engine path: fresh-mind discover(). Linear duel is an explicitly
    labeled BASELINE/rescue — its output never feeds Isaac."""
    rec = dict(tag=tag, out=out_nm, n=len(idxs), idxs=list(idxs),
               minted=False, reason=None, wrong_kept=[], num_pre=num_pre)
    if not num_pre:
        rec["reason"] = "no numeric precondition slots"
        return rec
    X = np.array([[float(pre_s[nm][i]) for nm in num_pre] for i in idxs])
    yv = np.array([float(out_vals[i]) for i in idxs])
    inlier = robust_trim(X, yv)
    kept = int(inlier.sum())
    rec["n_kept"] = kept
    rec["wrong_kept"] = [i for i, k in zip(idxs, inlier) if not k]
    min_keep = max(8, int(np.ceil(0.6 * len(idxs))))
    rec["min_keep"] = min_keep
    if kept < min_keep:
        rec["reason"] = (f"trim consensus collapsed ({kept} kept < "
                         f"{min_keep} required) — binding refused")
        return rec
    Xk, yk = X[inlier], yv[inlier]
    mind = mod.Isaac()                       # fresh throwaway mind
    order = rng.permutation(len(yk))
    law = f"bind_{tag}_{out_nm}"
    res = mind.discover(Xk[order], yk[order], name=law)
    rec["engine_verdict"], rec["engine_eq"] = res["verdict"], res["equation"]
    rec["engine_r2"] = res["r2"]
    pt = mind.apply_law(law, Xk)
    rec["engine_train_maxerr"] = (float(np.max(np.abs(pt - yk)))
                                  if pt is not None else float("inf"))
    rec["engine_exact"] = rec["engine_train_maxerr"] < TOL
    rec["mind"], rec["law"] = mind, law
    L = np.column_stack([Xk, np.ones(len(Xk))])
    lc, *_ = np.linalg.lstsq(L, yk, rcond=None)
    lc = np.round(lc, 8)
    rec["duel_train_maxerr"] = float(np.max(np.abs(L @ lc - yk)))
    rec["duel_exact"] = rec["duel_train_maxerr"] < TOL
    rec["duel_coefs"] = lc
    rec["minted"] = bool(rec["engine_exact"] or rec["duel_exact"])
    rec["mint_path"] = ("engine" if rec["engine_exact"] else
                        "linear-duel" if rec["duel_exact"] else None)
    return rec


def predict(rec, U):
    if rec["engine_exact"]:
        return rec["mind"].apply_law(rec["law"], U)
    lc = rec["duel_coefs"]
    return U @ lc[:-1] + lc[-1]


def print_bind(b, indent="  "):
    hd = f"{indent}[bind {b['tag']} -> {b['out']}] n={b['n']}"
    if b["reason"]:
        print(f"{hd}  REFUSED: {b['reason']}"
              + (f" (trim kept {b.get('n_kept')}/{b['n']})"
                 if "n_kept" in b else ""))
        return
    print(f"{hd}  trim kept {b['n_kept']}/{b['n']} "
          f"(min_keep {b['min_keep']}); wrong-kept {b['wrong_kept']}")
    print(f"{indent}  engine discover(): {b['engine_verdict']}  "
          f"eq: {b['engine_eq']}  r2={b['engine_r2']:.4f}  "
          f"train maxerr {b['engine_train_maxerr']:.4g}  "
          f"exact={b['engine_exact']}")
    print(f"{indent}  linear duel [BASELINE]: coefs {list(b['duel_coefs'])}"
          f"  train maxerr {b['duel_train_maxerr']:.4g}  "
          f"exact={b['duel_exact']}")
    print(f"{indent}  minted: {b['minted']} via {b['mint_path']}")


def mine_world(name, trans):
    """Labeled mining: shape recurrence gate -> templates -> bindings."""
    N = len(trans)
    sigs = [shape_sig(b, a) for b, a in trans]
    cnt = Counter(sigs)
    top_sig, sup = cnt.most_common(1)[0]
    thr = max(3, N // 2)
    rec = dict(name=name, n=N, top_support=sup, thr=thr,
               n_shapes=len(cnt), bindings=[], keep=None, pw=[])
    print(f"\n[{name}] {N} labeled transitions; effect-shape gate: "
          f"top support {sup}/{N} vs threshold {thr}; "
          f"distinct shapes {len(cnt)}")
    if sup < thr:
        rec["reason"] = "no recurring effect shape — nothing mined"
        print(f"  {rec['reason']}")
        return rec
    keep = [i for i, s in enumerate(sigs) if s == top_sig]
    rec["keep"] = keep
    pre_t, pre_s, stats = mine_template(keep, lambda i: trans[i][0])
    add_t, add_s, _ = mine_template(keep,
                                    lambda i: trans[i][1] - trans[i][0])
    del_t, _, _ = mine_template(keep, lambda i: trans[i][0] - trans[i][1])
    rec.update(pre_t=pre_t, add_t=add_t, del_t=del_t,
               pre_s=pre_s, add_s=add_s, stats=stats)
    print(f"  PRE : {fmt_template(pre_t)}")
    print(f"  ADD : {fmt_template(add_t)}   DEL: {fmt_template(del_t)}")
    for head, ln, pres, mults in stats:
        gate = "PASS" if len(mults) == 1 else "FAIL (distractor-like)"
        print(f"    pre gate {head}/{ln}: presence {pres:.2f} "
              f"multiplicity {mults} -> {gate}")
    num_pre = [nm for nm in sorted(pre_s) if numeric_slot(pre_s[nm])]
    outs = [nm for nm in sorted(add_s) if numeric_slot(add_s[nm])]
    rec["num_pre"], rec["outs"] = num_pre, outs
    print(f"  numeric pre slots (X columns, in order): {num_pre}; "
          f"output slots: {outs}")
    for out in outs:
        b = bind_one(name, keep, num_pre, pre_s, out, add_s[out])
        b["pre_t"] = pre_t
        rec["bindings"].append(b)
        print_bind(b)
    return rec


def piecewise(worldrec, trans, name):
    """Fallback when whole-cluster binding failed: partition on each
    low-cardinality (2-3 values) precondition slot, re-mine templates per
    partition (the split slot becomes a LITERAL there), re-bind."""
    keep, pre_s = worldrec["keep"], worldrec["pre_s"]
    cands = [nm for nm in sorted(pre_s)
             if 2 <= len({pre_s[nm][i] for i in keep}) <= 3]
    print(f"  piecewise fallback: low-cardinality slot candidates {cands}")
    results = []
    for nm in cands:
        parts = {}
        for i in keep:
            parts.setdefault(pre_s[nm][i], []).append(i)
        if any(len(p) < PART_MIN for p in parts.values()):
            print(f"    slot {nm}: some partition below "
                  f"PART_MIN={PART_MIN} — skipped")
            continue
        srec = dict(slot=nm, parts=[], all_minted=True)
        for val, pidx in sorted(parts.items()):
            p_pre_t, p_pre_s, _ = mine_template(
                pidx, lambda i: trans[i][0])
            p_add_t, p_add_s, _ = mine_template(
                pidx, lambda i: trans[i][1] - trans[i][0])
            p_num = [k for k in sorted(p_pre_s)
                     if numeric_slot(p_pre_s[k])]
            p_outs = [k for k in sorted(p_add_s)
                      if numeric_slot(p_add_s[k])]
            print(f"    partition {nm}={val}: n={len(pidx)}")
            print(f"      PRE: {fmt_template(p_pre_t)}")
            binds = []
            for out in p_outs:
                b = bind_one(f"{name}.{nm}={val}", pidx, p_num,
                             p_pre_s, out, p_add_s[out])
                b["pre_t"] = p_pre_t
                binds.append(b)
                print_bind(b, indent="      ")
                if not b["minted"]:
                    srec["all_minted"] = False
            if not p_outs:
                srec["all_minted"] = False
                print("      no numeric output slot — cannot bind")
            srec["parts"].append(dict(val=val, n=len(pidx),
                                      pre_t=p_pre_t, binds=binds))
        results.append(srec)
    return results


def full_pipeline(name, trans):
    rec = mine_world(name, trans)
    n_mint = sum(1 for b in rec["bindings"] if b["minted"])
    if rec.get("keep") and rec["bindings"] and \
            not all(b["minted"] for b in rec["bindings"]):
        print("  whole-cluster binding incomplete -> piecewise fallback")
        rec["pw"] = piecewise(rec, trans, name)
        n_mint += sum(1 for s in rec["pw"] for p in s["parts"]
                      for b in p["binds"] if b["minted"])
    rec["total_minted"] = n_mint
    return rec


# ═════════════════════════════════════════════════════════════════
# THE MATCHER (probe-side) — fires a minted rule against a state
# ═════════════════════════════════════════════════════════════════
def match(pre_t, state):
    """Bind template cells to state cells: LIT must equal, SLOT binds
    consistently, cells used injectively. Returns binding or None."""
    cells = sorted(state)

    def bt(k, binding, used):
        if k == len(pre_t):
            return binding
        head, cols = pre_t[k]
        for c in cells:
            if c in used or c[0] != head or len(c) != len(cols) + 1:
                continue
            nb, ok = dict(binding), True
            for (kind, v), actual in zip(cols, c[1:]):
                if kind == "LIT":
                    if actual != v:
                        ok = False
                        break
                else:
                    if v in nb and nb[v] != actual:
                        ok = False
                        break
                    nb[v] = actual
            if ok:
                r = bt(k + 1, nb, used | {c})
                if r is not None:
                    return r
        return None

    return bt(0, {}, frozenset())


def fire(bind_rec, state):
    """(fired, predicted output value or None)."""
    bnd = match(bind_rec["pre_t"], state)
    if bnd is None:
        return False, None
    U = np.array([[float(bnd[nm]) for nm in bind_rec["num_pre"]]])
    return True, float(predict(bind_rec, U)[0])


def fire_demo(label, bind_rec, states, wants):
    fired = 0
    exact = 0
    for st, w in zip(states, wants):
        f, v = fire(bind_rec, st)
        fired += f
        if f and w is not None and abs(v - w) < TOL:
            exact += 1
    print(f"    {label}: fired {fired}/{len(states)}, "
          f"exact answers {exact}/{len(states)}")
    return fired, exact


def grade_unseen(rec, U, want, label):
    got = predict(rec, U)
    errs = np.abs(got - np.asarray(want, float))
    n_ok = int((errs < TOL).sum())
    for u, g, w in zip(U, got, want):
        mark = "ok" if abs(g - w) < TOL else "WRONG"
        print(f"    UNSEEN {[int(x) for x in u]} -> {g:.6f} "
              f"(want {w}) {mark}")
    mx = float(np.max(errs))
    print(f"    {label}: unseen maxerr {mx:.4g} ({n_ok}/{len(want)} exact)")
    return mx, n_ok


# ═════════════════════════════════════════════════════════════════
# THE RUN
# ═════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    verdicts = {}
    print("=" * 70)
    print("GATE 6 PRECONDITIONS ATTACK — constant distractors, multi-cell")
    print("and conditional effects (labels given)")
    print("=" * 70)

    # ── W1: CONSTANT DISTRACTOR ────────────────────────────────────
    trans1 = [ep_w1() for _ in range(48)]
    rec1 = full_pipeline("W1_const_distractor", trans1)
    pre1 = rec1.get("pre_t", [])
    bg_cell = next(((h, cols) for h, cols in pre1 if h == "bg"), None)
    bg_lit = bool(bg_cell) and all(k == "LIT" for k, _ in bg_cell[1])
    noise_in_pre = any(h == "noise" for h, _ in pre1)
    print(f"\n[P1] constant distractor ('bg','7') in preconditions: "
          f"{bg_cell is not None} as "
          f"{'LITERAL' if bg_lit else 'slot/absent'}; "
          f"jitter noise cells in preconditions: {noise_in_pre} "
          f"(want: gate excludes jitter)")

    b1 = next((b for b in rec1["bindings"] if b["minted"]), None)
    fw = fo = fx = (0, 0)
    if b1 is not None:
        print("[P2] the trap, concretely (probe-side matcher):")

        def w1_state(a, b, bg):
            st = {("num", 1, str(a)), ("num", 3, str(b)),
                  ("sym", 2, "*"), ("want", "answer")}
            if bg is not None:
                st.add(("bg", bg))
            return frozenset(st)

        wants = [a * b for a, b in DEMO_AB]
        fw = fire_demo("unseen states WITH ('bg','7')",
                       b1, [w1_state(a, b, "7") for a, b in DEMO_AB],
                       wants)
        fo = fire_demo("SAME states, bg cell ABSENT",
                       b1, [w1_state(a, b, None) for a, b in DEMO_AB],
                       wants)
        fx = fire_demo("SAME states, bg present but =8",
                       b1, [w1_state(a, b, "8") for a, b in DEMO_AB],
                       wants)
        print("[P3] W1 binding on unseen inputs (engine path):")
        U = np.array([[float(a), float(b)] for a, b in UNSEEN_AB])
        mx1, ok1 = grade_unseen(b1, U, [a * b for a, b in UNSEEN_AB],
                                "W1 prod")
    trap_bites = (bg_lit and fw[0] == len(DEMO_AB) and fo[0] == 0
                  and fx[0] == 0)
    verdicts["W1-constant-distractor-exclusion"] = (
        ("BROKEN", f"('bg','7') minted as required LITERAL (presence "
         f"1.00, mult [1]); rule fires {fw[0]}/6 with bg=7 "
         f"({fw[1]}/6 exact) but {fo[0]}/6 without the cell and "
         f"{fx[0]}/6 with bg=8 — transfer fully blocked; jitter noise "
         f"correctly excluded ({noise_in_pre})")
        if trap_bites else
        ("PARTIAL", f"bg in pre: {bg_cell is not None} lit={bg_lit}; "
         f"fires with/without/wrong-val: {fw[0]}/{fo[0]}/{fx[0]} of 6; "
         f"noise in pre: {noise_in_pre}"))
    if b1 is not None:
        verdicts["W1-binding-product"] = (
            ("WORKS", f"engine {b1['engine_verdict']} eq {b1['engine_eq']}"
             f" train maxerr {b1['engine_train_maxerr']:.4g}; unseen "
             f"{ok1}/3 exact (maxerr {mx1:.4g}) via {b1['mint_path']}")
            if b1["engine_exact"] and ok1 == 3 else
            ("PARTIAL", f"mint via {b1['mint_path']}; unseen {ok1}/3 "
             f"(maxerr {mx1:.4g})"))
    else:
        verdicts["W1-binding-product"] = (
            "BROKEN", f"no binding minted: "
            f"{[b['reason'] for b in rec1['bindings']]}")

    # ── W2: MULTI-CELL EFFECT ──────────────────────────────────────
    trans2 = [ep_w2() for _ in range(48)]
    rec2 = full_pipeline("W2_multicell", trans2)
    add2 = rec2.get("add_t", [])
    heads2 = sorted(h for h, _ in add2)
    two_cell = (heads2 == ["prod", "sum"] and
                all(sum(1 for k, _ in cols if k == "SLOT") == 1
                    for _, cols in add2))
    print(f"\n[P4] add-template heads: {heads2} "
          f"(two-cell with one slot each: {two_cell}); "
          f"binding functions minted: "
          f"{sum(1 for b in rec2['bindings'] if b['minted'])}/2")
    verdicts["W2-multicell-effect-template"] = (
        ("WORKS", f"add-template = {fmt_template(add2)} — 2 cells, "
         f"2 output slots; del-template = "
         f"{fmt_template(rec2.get('del_t', []))}")
        if two_cell else ("BROKEN", f"add-template {fmt_template(add2)}"))
    U = np.array([[float(a), float(b)] for a, b in UNSEEN_AB])
    w2_ev, w2_ok, w2_paths = [], 0, []
    for b in rec2["bindings"]:
        if not b["minted"]:
            w2_ev.append(f"{b['out']}: NOT minted ({b['reason'] or 'inexact'})")
            continue
        head = "sum" if b["out"].startswith("sum") else "prod"
        want = ([a + b_ for a, b_ in UNSEEN_AB] if head == "sum"
                else [a * b_ for a, b_ in UNSEEN_AB])
        print(f"  W2 {head} binding on unseen inputs "
              f"(path {b['mint_path']}):")
        mx, ok = grade_unseen(b, U, want, f"W2 {head}")
        w2_ok += (ok == 3)
        w2_paths.append(f"{head}:{b['mint_path']}")
        w2_ev.append(f"{head}: engine {b['engine_verdict']} train_maxerr "
                     f"{b['engine_train_maxerr']:.4g} duel_exact "
                     f"{b['duel_exact']} -> {b['mint_path']}, unseen "
                     f"{ok}/3 (maxerr {mx:.4g})")
    n_engine = sum(1 for p in w2_paths if p.endswith("engine"))
    verdicts["W2-two-binding-functions"] = (
        ("WORKS" if n_engine == 2 else "PARTIAL" if w2_ok == 2 else
         "BROKEN"), "; ".join(w2_ev))

    # ── W3: CONDITIONAL EFFECT ─────────────────────────────────────
    plan3 = [0] * 24 + [1] * 24
    rng.shuffle(plan3)
    trans3 = [ep_w3(f) for f in plan3]
    rec3 = full_pipeline("W3_conditional", trans3)

    # naive whole-cluster path: what did it do with the blend?
    whole = rec3["bindings"][0] if rec3["bindings"] else None
    if whole is not None and whole.get("n_kept") is not None:
        surv = [i for i in whole["idxs"] if i not in set(whole["wrong_kept"])]
        comp = Counter("add" if plan3[i] == 0 else "mul" for i in surv)
        print(f"\n[P5] whole-cluster trim survivors by true branch: "
              f"{dict(comp)} (kept {whole.get('n_kept')}/{whole['n']}, "
              f"min_keep {whole.get('min_keep')})")
    # untrimmed raw-discover diagnostic on the blended rows
    keep3, pre_s3, num3 = rec3["keep"], rec3["pre_s"], rec3["num_pre"]
    Xall = np.array([[float(pre_s3[nm][i]) for nm in num3] for i in keep3])
    yall = np.array([float(rec3["add_s"][rec3["outs"][0]][i])
                     for i in keep3])
    mind = mod.Isaac()
    order = rng.permutation(len(yall))
    res = mind.discover(Xall[order], yall[order], name="w3_blend_diag")
    pd_ = mind.apply_law("w3_blend_diag", Xall)
    merr = (float(np.max(np.abs(pd_ - yall))) if pd_ is not None
            else float("inf"))
    print(f"[P5 diag] raw discover() on ALL {len(yall)} blended rows "
          f"(X columns {num3}): {res['verdict']}  eq: {res['equation']}  "
          f"r2={res['r2']:.4f}  train maxerr {merr:.4g}  "
          f"exact={merr < TOL}")
    blended_minted = bool(whole and whole["minted"])
    blend_ev = (f"whole binding: "
                f"{'MINTED (' + str(whole['mint_path']) + ')' if blended_minted else 'refused: ' + str(whole['reason'] if whole else 'none')}; "
                f"raw-discover diag: {res['verdict']} train_maxerr "
                f"{merr:.4g}")
    if blended_minted:
        # a single blended rule got minted — grade it on both branches
        Ub = np.array([[float(f), float(a), float(b)]
                       for f in (0, 1) for a, b in UNSEEN_AB])
        wantb = [a + b_ if f == 0 else a * b_
                 for f in (0, 1) for a, b_ in UNSEEN_AB]
        mxb, okb = grade_unseen(whole, Ub, wantb, "W3 blended rule")
        verdicts["W3-naive-blended-binding"] = (
            ("WORKS", f"single rule exact on BOTH branches: unseen "
             f"{okb}/6 (maxerr {mxb:.4g}); {blend_ev}")
            if okb == len(wantb) else
            ("BROKEN", f"one blended rule minted but wrong: unseen "
             f"{okb}/{len(wantb)} (maxerr {mxb:.4g}); {blend_ev}"))
    else:
        verdicts["W3-naive-blended-binding"] = (
            ("HONEST-REFUSAL", blend_ev) if merr >= TOL else
            ("PARTIAL", f"trim refused but raw discover was exact: "
             f"{blend_ev}"))

    # piecewise result: two rules with flag as a precondition LITERAL?
    pw_flag = next((s for s in rec3["pw"] if s["slot"].startswith("flag")),
                   None)
    pw_ok, pw_ev = False, "piecewise never produced a flag split"
    if pw_flag:
        lits, uns_ok, paths = [], 0, []
        rules = {}
        for p in pw_flag["parts"]:
            has_lit = any(h == "flag" and cols == (("LIT", p["val"]),)
                          for h, cols in p["pre_t"])
            lits.append(has_lit)
            b = next((b for b in p["binds"] if b["minted"]), None)
            rules[p["val"]] = b
            if b is None:
                continue
            want = ([a + b_ for a, b_ in UNSEEN_AB] if p["val"] == "0"
                    else [a * b_ for a, b_ in UNSEEN_AB])
            print(f"  W3 partition flag={p['val']} on unseen inputs "
                  f"(path {b['mint_path']}):")
            mx, ok = grade_unseen(b, U, want, f"W3 flag={p['val']}")
            uns_ok += (ok == 3)
            paths.append(f"flag={p['val']}:{b['mint_path']}")

        # cross-fire: each conditional rule must respect its flag literal
        def w3_state(a, b, f):
            return frozenset({("num", 1, str(a)), ("num", 3, str(b)),
                              ("flag", str(f)), ("want", "answer")})

        cross = {}
        if rules.get("0") and rules.get("1"):
            print("[P6] cross-fire (matcher): conditional rules vs states")
            s0 = [w3_state(a, b, 0) for a, b in DEMO_AB]
            s1 = [w3_state(a, b, 1) for a, b in DEMO_AB]
            wsum = [a + b for a, b in DEMO_AB]
            wmul = [a * b for a, b in DEMO_AB]
            cross["add_on_f0"] = fire_demo("add-rule on flag=0 states",
                                           rules["0"], s0, wsum)
            cross["add_on_f1"] = fire_demo("add-rule on flag=1 states",
                                           rules["0"], s1, wsum)
            cross["mul_on_f1"] = fire_demo("mul-rule on flag=1 states",
                                           rules["1"], s1, wmul)
            cross["mul_on_f0"] = fire_demo("mul-rule on flag=0 states",
                                           rules["1"], s0, wmul)
        cross_ok = (cross and cross["add_on_f0"] == (6, 6)
                    and cross["add_on_f1"][0] == 0
                    and cross["mul_on_f1"] == (6, 6)
                    and cross["mul_on_f0"][0] == 0)
        pw_ok = (all(lits) and len(lits) == 2 and uns_ok == 2
                 and pw_flag["all_minted"] and cross_ok)
        pw_ev = (f"flag LITERAL in both partition pres: {lits}; paths "
                 f"{paths}; unseen exact partitions {uns_ok}/2; "
                 f"cross-fire add:{cross.get('add_on_f0')}on-f0/"
                 f"{cross.get('add_on_f1')}on-f1, "
                 f"mul:{cross.get('mul_on_f1')}on-f1/"
                 f"{cross.get('mul_on_f0')}on-f0")
    verdicts["W3-piecewise-conditional-mining"] = (
        ("WORKS", pw_ev) if pw_ok else ("BROKEN", pw_ev))

    # ── CONTROLS ───────────────────────────────────────────────────
    print("\n" + "-" * 70)
    print("[P7 controls] scrambled worlds (befores + permuted afters)")
    ctl_ev = []
    total_scr = 0
    for nm, tr, seed in (("W1scr", trans1, 5), ("W2scr", trans2, 6),
                         ("W3scr", trans3, 7)):
        r = full_pipeline(nm, scrambled(tr, seed))
        total_scr += r["total_minted"]
        ctl_ev.append(f"{nm}: top shape {r['top_support']}/{r['n']} vs "
                      f"thr {r['thr']}, minted {r['total_minted']}")
        print(f"  {nm}: minted {r['total_minted']} (want 0)")
    verdicts["scrambled-controls"] = (
        ("WORKS", "; ".join(ctl_ev) + " — all 0 as demanded")
        if total_scr == 0 else
        ("BROKEN", "; ".join(ctl_ev) + f" — {total_scr} minted from garbage"))

    print("\n[P7b control] W3 with flags SHUFFLED across episodes "
          "(flag no longer predicts the branch)")
    rfs = full_pipeline("W3flagshuffle", w3_flagshuffle(trans3))
    fs_ev = (f"top shape {rfs['top_support']}/{rfs['n']} vs thr "
             f"{rfs['thr']}; whole+piecewise minted "
             f"{rfs['total_minted']} (want 0)")
    print(f"  W3flagshuffle: minted {rfs['total_minted']} (want 0)")
    verdicts["flagshuffle-piecewise-null"] = (
        ("WORKS", fs_ev + " — flag in preconditions is earned, not free")
        if rfs["total_minted"] == 0 else
        ("BROKEN", fs_ev))

    # ═══ VERDICT_SHEET ═════════════════════════════════════════════
    dt = time.time() - t0
    print("\n" + "=" * 70)
    print("VERDICT_SHEET (gate6 preconditions attack)")
    for cap, (v, ev) in verdicts.items():
        print(f"  {cap:36s} {v:15s} {ev}")
    print(f"  runtime: {dt:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
