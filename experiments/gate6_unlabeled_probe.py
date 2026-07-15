"""gate6_unlabeled_probe.py — GATE 6 ATTACK: operator mining with the
action labels REMOVED (07-15 open item 1; 07-13 caveat: "action labels
were GIVEN (no clustering)").

THE CLAIM UNDER TEST: gate6_operator_mining.py minted add/mul rules from
(before, action, after) episodes, but the transitions arrived PRE-GROUPED
by action. Here episodes are (before, after) ONLY — two hidden operators
(add: c=a+b; mul: c=a*b) interleaved, plus jittering distractor cells and
planted corrupted episodes. The miner must: (1) cluster transitions by
recurring effect template (add/delete shape recurrence); (2) per cluster
mine preconditions (constants literal, varying values SLOTS; a slot whose
values are non-numeric cannot feed a binding function and therefore
PARTITIONS the cluster — co-occurrence discipline); (3) mine the binding
via fresh-mind discover() on each partition's transition rows; (4) verify
minted bindings on UNSEEN inputs directly in numpy.

NOTE: gate6's isaac_interpreter import is UNAVAILABLE here — this probe
tests the MINING side only (no interpreter needed).

PREDICTION (filed before the first run):
  P1 effect-shape clustering yields ONE shape cluster covering all 126
     episodes — add and mul share the effect shape, so shape alone canNOT
     separate them (the label really is gone).
  P2 the categorical sym slot ('+'/'*') partitions that cluster into 2
     operator clusters with purity 1.00 vs hidden ground truth.
  P3 distractor (noise) cells never enter preconditions or partition keys
     (presence rate < 1.0 / varying multiplicity fails the gate).
  P4 the 6 planted corrupted episodes are wrong-kept by the binding trim,
     never learned.
  P5 mul partition: fresh-mind discover() minted, exact on unseen
     (7,6 -> 42).
  P6 add partition: raw discover() FAILS exactness (the known additive
     blind spot COUPLING with unlabeled mining — two known issues
     meeting); the probe-side linear duel (explicit BASELINE, never fed
     to Isaac) rescues it (9,9 -> 18 exact).
  P7 scrambled control (befores paired with permuted afters) mints
     NOTHING — top shape support falls below the recurrence threshold,
     and every downstream gate (trim-collapse guard, discover null
     discipline) refuses whatever leaks through.
  P8 the UNPARTITIONED mixed cluster cannot bind: raw discover on mixed
     add+mul rows is not exact, and the robust trim cannot keep >= 60%
     of a mixed cluster — the partition is load-bearing, not decorative.

Deterministic: run as
  cd /home/user/dc/experiments && PYTHONHASHSEED=0 python3 gate6_unlabeled_probe.py
World sized (60 add + 6 corrupt-add + 60 mul) to finish well under 4 min.
stdlib + numpy for the cognition; sklearn unused; fresh throwaway
Isaac() minds only; nothing persisted.
"""
import os
import sys
import time
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import isaac as mod                                       # noqa: E402

rng = np.random.default_rng(7)
PART_MIN = 4          # a partition below this support is not recurrence
TOL = 1e-6


# ═════════════════════════════════════════════════════════════════
# THE WORLD — same species as gate6_operator_mining.py, minus labels
# ═════════════════════════════════════════════════════════════════
def episode(op, corrupt=False):
    x, y = int(rng.integers(1, 10)), int(rng.integers(1, 10))
    sym = "+" if op == "add" else "*"
    before = {("num", 1, str(x)), ("sym", 2, sym), ("num", 3, str(y)),
              ("sym", 4, "="), ("want", "answer")}
    for _ in range(int(rng.integers(0, 3))):              # jittering
        before.add(("noise", int(rng.integers(0, 9))))    # distractors
    v = x + y if op == "add" else x * y
    if corrupt:
        v = int(rng.integers(30, 90))                     # wrong answer
    after = (before - {("want", "answer")}) | {("answer", str(v))}
    return frozenset(before), frozenset(after)


def gen_world():
    plan = [("add", False)] * 60 + [("add", True)] * 6 + [("mul", False)] * 60
    rng.shuffle(plan)
    trans = [episode(op, c) for op, c in plan]
    ops = [op for op, _ in plan]                # ground truth: GRADING ONLY
    corrupt_idx = {i for i, (_, c) in enumerate(plan) if c}
    return trans, ops, corrupt_idx


# ═════════════════════════════════════════════════════════════════
# THE MINER — unlabeled: shapes -> clusters -> slots -> partitions
# ═════════════════════════════════════════════════════════════════
def sig_of(cells):
    return tuple((c[0], len(c)) for c in sorted(cells))


def shape_sig(b, a):
    return (sig_of(a - b), sig_of(b - a))


def mine_template(idxs, cells_of):
    """Always-present, constant-multiplicity (head, arity) groups become
    template cells: constant positions stay LITERAL, varying become SLOTS.
    Returns (template, slots{name: {i: value}}, gate_stats)."""
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
            continue                    # distractors die HERE (measured)
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
                           for kind, v in cols) + ")" for h, cols in t)


def robust_trim(X, yv):
    """Iterated consensus trim (gate6's discipline): linear+product probe,
    gross violators wrong-kept, refit until stable."""
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


def bind_partition(tag, idxs, trans):
    """Mine one partition's templates and its binding function.
    Engine path: fresh-mind discover(). Probe-side linear duel is an
    explicitly labeled BASELINE/rescue — its output never feeds Isaac."""
    pre_t, pre_s, stats = mine_template(idxs, lambda i: trans[i][0])
    add_t, add_s, _ = mine_template(idxs,
                                    lambda i: trans[i][1] - trans[i][0])
    del_t, _, _ = mine_template(idxs, lambda i: trans[i][0] - trans[i][1])
    rec = dict(tag=tag, idxs=idxs, n=len(idxs), minted=False, reason=None,
               pre_t=pre_t, add_t=add_t, del_t=del_t, stats=stats,
               wrong_kept=[])
    num_pre = [nm for nm in sorted(pre_s) if numeric_slot(pre_s[nm])]
    cat_pre = [nm for nm in sorted(pre_s) if not numeric_slot(pre_s[nm])]
    outs = [nm for nm in sorted(add_s) if numeric_slot(add_s[nm])]
    rec["pre_slots"], rec["out_slots"] = num_pre, outs
    if cat_pre:
        rec["reason"] = f"categorical slot(s) remain: {cat_pre}"
        return rec
    if not outs:
        rec["reason"] = "no numeric output slot in the add-template"
        return rec
    if not num_pre:
        rec["reason"] = "no numeric precondition slots"
        return rec
    out = outs[0]
    X = np.array([[float(pre_s[nm][i]) for nm in num_pre] for i in idxs])
    yv = np.array([float(add_s[out][i]) for i in idxs])
    inlier = robust_trim(X, yv)
    kept = int(inlier.sum())
    rec["n_kept"] = kept
    rec["wrong_kept"] = [i for i, ok in zip(idxs, inlier) if not ok]
    min_keep = max(8, int(np.ceil(0.6 * len(idxs))))
    rec["min_keep"] = min_keep
    if kept < min_keep:
        rec["reason"] = (f"trim consensus collapsed ({kept} kept < "
                         f"{min_keep} required) — binding refused")
        return rec
    Xk, yk = X[inlier], yv[inlier]
    # ── engine path: fresh throwaway mind, discover() on the rows ──
    mind = mod.Isaac()
    order = rng.permutation(len(yk))
    law = f"bind_{tag}"
    res = mind.discover(Xk[order], yk[order], name=law)
    rec["engine_verdict"] = res["verdict"]
    rec["engine_eq"] = res["equation"]
    rec["engine_r2"] = res["r2"]
    pt = mind.apply_law(law, Xk)
    rec["engine_train_maxerr"] = (float(np.max(np.abs(pt - yk)))
                                  if pt is not None else float("inf"))
    rec["engine_exact"] = rec["engine_train_maxerr"] < TOL
    rec["mind"], rec["law"] = mind, law
    # ── probe-side LINEAR DUEL (BASELINE/rescue, never feeds Isaac) ──
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


def mine_unlabeled(trans, name):
    """The full unlabeled pipeline on (before, after) transitions."""
    N = len(trans)
    sigs = [shape_sig(b, a) for b, a in trans]
    cnt = Counter(sigs)
    thr = max(4, N // 5)
    print(f"\n[{name}] {N} transitions; shape recurrence threshold {thr}")
    for s, sup in cnt.most_common(3):
        print(f"  shape support {sup:3d}: adds={s[0]} dels={s[1]}")
    print(f"  distinct shapes: {len(cnt)}")
    report = dict(n=N, thr=thr, top_support=cnt.most_common(1)[0][1],
                  n_shapes=len(cnt), clusters=[], bindings=[])
    for sig, sup in cnt.most_common():
        if sup < thr:
            break
        idxs = [i for i, s in enumerate(sigs) if s == sig]
        pre_t, pre_s, stats = mine_template(idxs, lambda i: trans[i][0])
        add_t, add_s, _ = mine_template(idxs,
                                        lambda i: trans[i][1] - trans[i][0])
        cat = [nm for nm in sorted(pre_s) if not numeric_slot(pre_s[nm])]
        clu = dict(sig=sig, support=sup, idxs=idxs, stats=stats,
                   pre_s=pre_s, add_s=add_s, cat=cat, partitions={})
        report["clusters"].append(clu)
        print(f"  cluster ({sup} eps): precondition gate per (head,arity):")
        for head, ln, pres, mults in stats:
            gate = "PASS" if len(mults) == 1 else "FAIL (distractor-like)"
            print(f"    {head}/{ln}: presence {pres:.2f} multiplicity "
                  f"{mults} -> {gate}")
        print(f"    categorical (non-numeric) slots: {cat or 'none'}")
        if cat:
            parts = {}
            for i in idxs:
                parts.setdefault(
                    tuple(pre_s[nm][i] for nm in cat), []).append(i)
            clu["partitions"] = dict(sorted(parts.items()))
            for key, pidx in sorted(parts.items()):
                print(f"    partition {key}: {len(pidx)} episodes")
                if len(pidx) < PART_MIN:
                    print(f"      below PART_MIN={PART_MIN} — skipped")
                    continue
                report["bindings"].append(
                    bind_partition(f"{name}_{'_'.join(key)}", pidx, trans))
        else:
            report["bindings"].append(
                bind_partition(f"{name}_whole", idxs, trans))
    return report


# ═════════════════════════════════════════════════════════════════
# THE RUN
# ═════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    print("=" * 70)
    print("GATE 6 UNLABELED — mining operators with NO action labels")
    print("=" * 70)
    trans, ops, corrupt_idx = gen_world()
    n_add = sum(1 for o in ops if o == "add")
    print(f"world: {len(trans)} episodes ({n_add} add incl "
          f"{len(corrupt_idx)} corrupted, {len(ops) - n_add} mul), "
          f"interleaved, unlabeled")

    rep = mine_unlabeled(trans, "main")

    # ── grade the clustering vs hidden ground truth ────────────────
    verdicts = {}
    clu = rep["clusters"][0] if rep["clusters"] else None
    one_shape = (len(rep["clusters"]) == 1
                 and clu["support"] == len(trans))
    print(f"\n[P1] shape clusters passing threshold: {len(rep['clusters'])}"
          f"; top covers {clu['support'] if clu else 0}/{len(trans)}"
          f" -> add/mul share the effect shape: {one_shape}")
    purities, part_sizes = [], []
    if clu and clu["partitions"]:
        for key, pidx in clu["partitions"].items():
            comp = Counter(ops[i] for i in pidx)
            pur = max(comp.values()) / len(pidx)
            purities.append(pur)
            part_sizes.append(len(pidx))
            print(f"[P2] partition {key}: n={len(pidx)} composition "
                  f"{dict(comp)} purity {pur:.3f}")
    verdicts["effect-clustering-unlabeled"] = (
        ("WORKS", f"1 shape cluster {clu['support']}/{len(trans)}; "
         f"sym-partition purities {[round(p, 3) for p in purities]} "
         f"sizes {part_sizes}")
        if one_shape and purities and min(purities) == 1.0
        else ("BROKEN", f"clusters={len(rep['clusters'])} "
              f"purities={purities}"))

    # ── distractor discipline ──────────────────────────────────────
    noise_in_pre = any(
        any(h == "noise" for h, _ in b.get("pre_t", []))
        for b in rep["bindings"])
    noise_in_key = any(nm.startswith("noise") for nm in (clu["cat"] if clu
                                                         else []))
    print(f"[P3] distractor cells in preconditions: {noise_in_pre}; "
          f"in partition keys: {noise_in_key} (want False/False)")
    verdicts["distractor-exclusion"] = (
        ("WORKS", "noise head in preconditions: False; in partition "
         "keys: False") if not (noise_in_pre or noise_in_key)
        else ("BROKEN", f"pre={noise_in_pre} key={noise_in_key}"))

    # ── bindings: report each, grade vs truth, verify UNSEEN ───────
    unseen = {"add": [(9, 9, 18), (7, 6, 13), (2, 4, 6)],
              "mul": [(7, 6, 42), (3, 5, 15), (9, 9, 81)]}
    add_rec = mul_rec = None
    for b in rep["bindings"]:
        truth = Counter(ops[i] for i in b["idxs"]).most_common(1)[0][0]
        print(f"\n[bind {b['tag']}] n={b['n']} truth-majority={truth}")
        if b["reason"]:
            print(f"  refused: {b['reason']}")
            continue
        print(f"  preconditions: {fmt_template(b['pre_t'])}")
        print(f"  adds: {fmt_template(b['add_t'])}  "
              f"dels: {fmt_template(b['del_t'])}")
        print(f"  trim: kept {b['n_kept']}/{b['n']} "
              f"(min_keep {b['min_keep']}); wrong-kept {b['wrong_kept']}")
        print(f"  engine discover(): {b['engine_verdict']}  "
              f"eq: {b['engine_eq']}  r2={b['engine_r2']:.4f}  "
              f"train maxerr {b['engine_train_maxerr']:.4g}  "
              f"exact={b['engine_exact']}")
        print(f"  probe-side linear duel [BASELINE]: coefs "
              f"{list(b['duel_coefs'])} train maxerr "
              f"{b['duel_train_maxerr']:.4g} exact={b['duel_exact']}")
        print(f"  minted: {b['minted']} via {b['mint_path']}")
        if b["minted"]:
            U = np.array([[float(x), float(y)] for x, y, _ in unseen[truth]])
            want = np.array([float(w) for _, _, w in unseen[truth]])
            got = predict(b, U)
            errs = np.abs(got - want)
            for (x, y, w), g in zip(unseen[truth], got):
                print(f"  UNSEEN {x},{y} -> {g:.6f} (want {w}) "
                      f"{'ok' if abs(g - w) < TOL else 'WRONG'}")
            b["unseen_ok"] = bool(np.max(errs) < TOL)
            b["unseen_maxerr"] = float(np.max(errs))
        if truth == "add":
            add_rec = b
        else:
            mul_rec = b

    # [P5] mul binding, engine path
    if mul_rec is None or mul_rec.get("reason"):
        verdicts["binding-mul-fresh-discover"] = (
            "BROKEN", f"mul binding refused: "
            f"{mul_rec['reason'] if mul_rec else 'no mul partition'}")
    else:
        ok = mul_rec["engine_exact"] and mul_rec.get("unseen_ok", False)
        verdicts["binding-mul-fresh-discover"] = (
            "WORKS" if ok else
            ("PARTIAL" if mul_rec["minted"] else "BROKEN"),
            f"verdict={mul_rec['engine_verdict']} eq={mul_rec['engine_eq']} "
            f"train_maxerr={mul_rec['engine_train_maxerr']:.4g} "
            f"unseen_maxerr={mul_rec.get('unseen_maxerr', float('nan')):.4g}")

    # [P6] add binding — the MEASURED COUPLING of two known issues
    if add_rec is None or add_rec.get("reason"):
        verdicts["binding-add-coupling"] = (
            "BROKEN", f"add binding refused: "
            f"{add_rec['reason'] if add_rec else 'no add partition'}")
    else:
        eng, duel = add_rec["engine_exact"], add_rec["duel_exact"]
        uns = add_rec.get("unseen_ok", False)
        print(f"\n[P6 COUPLING] add cluster: raw discover exact={eng} "
              f"(verdict {add_rec['engine_verdict']}, eq "
              f"{add_rec['engine_eq']}, train maxerr "
              f"{add_rec['engine_train_maxerr']:.4g}); "
              f"probe-side linear duel exact={duel}, unseen ok={uns}")
        if eng and uns:
            verdicts["binding-add-coupling"] = (
                "WORKS", "additive blind spot did NOT bite: engine exact, "
                f"unseen_maxerr={add_rec['unseen_maxerr']:.4g}")
        elif duel and uns:
            verdicts["binding-add-coupling"] = (
                "PARTIAL", "MEASURED COUPLING: raw discover() verdict="
                f"{add_rec['engine_verdict']} train_maxerr="
                f"{add_rec['engine_train_maxerr']:.4g} (additive blind "
                "spot) x unlabeled mining; probe-side linear duel rescues "
                f"(unseen_maxerr={add_rec['unseen_maxerr']:.4g})")
        else:
            verdicts["binding-add-coupling"] = (
                "BROKEN", f"engine exact={eng} duel exact={duel} "
                f"unseen ok={uns}")

    # [P4] corruption wrong-kept
    trimmed = set()
    for b in rep["bindings"]:
        trimmed |= set(b.get("wrong_kept", []))
    hits = trimmed & corrupt_idx
    false_trims = trimmed - corrupt_idx
    print(f"\n[P4] corrupted episodes wrong-kept: {len(hits)}/"
          f"{len(corrupt_idx)} planted; false trims of clean episodes: "
          f"{len(false_trims)}")
    verdicts["corruption-wrong-keep"] = (
        ("WORKS", f"{len(hits)}/{len(corrupt_idx)} corrupted trimmed, "
         f"{len(false_trims)} false trims")
        if hits == corrupt_idx and not false_trims
        else ("PARTIAL" if hits == corrupt_idx else "BROKEN",
              f"hits {len(hits)}/{len(corrupt_idx)}, "
              f"false {len(false_trims)}"))

    # ── [P8] CONTROL: the UNPARTITIONED mixed cluster must not bind ─
    print("\n" + "-" * 70)
    print("[P8 control] binding the mixed cluster WITHOUT the partition")
    ok_mixed = None
    if clu:
        num_pre = [nm for nm in sorted(clu["pre_s"])
                   if numeric_slot(clu["pre_s"][nm])]
        outs = [nm for nm in sorted(clu["add_s"])
                if numeric_slot(clu["add_s"][nm])]
        idxs = clu["idxs"]
        Xm = np.array([[float(clu["pre_s"][nm][i]) for nm in num_pre]
                       for i in idxs])
        ym = np.array([float(clu["add_s"][outs[0]][i]) for i in idxs])
        mind = mod.Isaac()
        order = rng.permutation(len(ym))
        res = mind.discover(Xm[order], ym[order], name="bind_mixed")
        pm = mind.apply_law("bind_mixed", Xm)
        merr = (float(np.max(np.abs(pm - ym))) if pm is not None
                else float("inf"))
        print(f"  raw discover on mixed rows: {res['verdict']} eq "
              f"{res['equation']} r2={res['r2']:.4f} train maxerr "
              f"{merr:.4g} exact={merr < TOL}")
        inl = robust_trim(Xm, ym)
        comp = Counter(ops[i] for i, k in zip(idxs, inl) if k)
        mk = max(8, int(np.ceil(0.6 * len(idxs))))
        print(f"  robust trim on mixed rows: kept {int(inl.sum())}/"
              f"{len(idxs)} (min_keep {mk}) composition {dict(comp)} -> "
              f"{'binding REFUSED (collapse guard)' if inl.sum() < mk else 'kept enough — trim became a de-facto clusterer'}")
        ok_mixed = (merr >= TOL) and (inl.sum() < mk)
        verdicts["mixed-cluster-control"] = (
            ("WORKS", f"mixed discover train_maxerr={merr:.4g} (not "
             f"exact); trim kept {int(inl.sum())}/{len(idxs)} < {mk} -> "
             "refused; partition is load-bearing")
            if ok_mixed else
            ("PARTIAL", f"mixed exact={merr < TOL}, trim kept "
             f"{int(inl.sum())}/{len(idxs)} composition {dict(comp)}"))

    # ── [P7] CONTROL: scrambled transitions must mint NOTHING ──────
    print("\n" + "-" * 70)
    print("[P7 control] scrambled: befores paired with permuted afters")
    perm = np.random.default_rng(5).permutation(len(trans))
    scr = [(trans[i][0], trans[int(j)][1]) for i, j in enumerate(perm)]
    srep = mine_unlabeled(scr, "scrambled")
    s_mint = sum(1 for b in srep["bindings"] if b["minted"])
    s_refuse = [(b["tag"], b["reason"]) for b in srep["bindings"]
                if b["reason"]]
    print(f"  scrambled: top shape support {srep['top_support']} vs "
          f"threshold {srep['thr']}; clusters passing "
          f"{len(srep['clusters'])}; binding attempts "
          f"{len(srep['bindings'])}; refusals {s_refuse}; "
          f"MINTED {s_mint} (want 0)")
    verdicts["scrambled-control"] = (
        ("WORKS", f"top shape support {srep['top_support']}/{srep['n']} "
         f"vs thr {srep['thr']}; clusters {len(srep['clusters'])}; "
         f"minted {s_mint}/0 wanted")
        if s_mint == 0 else ("BROKEN", f"minted {s_mint} from garbage"))

    # ═══ VERDICT_SHEET ═════════════════════════════════════════════
    dt = time.time() - t0
    print("\n" + "=" * 70)
    print("VERDICT_SHEET (gate6 unlabeled attack)")
    for cap, (v, ev) in verdicts.items():
        print(f"  {cap:32s} {v:8s} {ev}")
    print(f"  runtime: {dt:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
