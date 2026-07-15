"""vocab_grounding.py — THE VOCABULARY EXPERIMENT (owner, 2026-07-13:
"we would need a full vocabulary. we need something that can teach him
all the tricks we got him to learn and convert them into symbolic
meaning of string.")

The claim under test: a word acquires SYMBOLIC MEANING by binding to
an executable concept Isaac already owns. The teacher walks the organ
inventory, generates grounded episodes (utterance + a situation his
organs can measure), and the word->concept binding is ADMITTED only
past a selection-corrected excess null — then proven by USE:

  hand-built = the concept inventory (organs he already has: the
               number line, residues, comparison, addition, shape
               perception) + the translation layer (numeric tokens
               are arguments; alphabetic tokens are candidate words)
               + the epistemics (excess over base rate, the
               scrambled-pairing null)
  LEARNED    = WHICH WORD MEANS WHICH CONCEPT. No mapping in code:
               swap the curriculum and the meanings follow the
               evidence (the liar control proves it).

Bindings live IN THE NET: word:<w> -means-> concept:<c> doors, every
measurement kept on the word node. Not meta side-channels for the
knowledge itself — the doors ARE the meaning.

The tests are USE, never recall (generation-not-duplication):
  1. unseen queries, numbers beyond the training range, yes AND no
  2. told rules phrased in TAUGHT words -> generated behavior from
     zero examples (the FizzBuzz road with a learned vocabulary —
     _RULE_CONDS was a hand-list; this is that list, taught)
  3. refusals are answers: ungrounded words, arity mismatch,
     ambiguity
  4. controls: scrambled pairing kills every binding; ubiquitous
     words bind to nothing (the excess law); a liar curriculum
     binds the lie (evidence decides, not the teacher's intent)

Complete numbers throughout. stdlib + numpy only.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import isaac as mod                                    # noqa: E402


# ═════════════════════════════════════════════════════════════════
# THE CONCEPT INVENTORY — his tricks, as executable checks.
# Pools: a concept only competes on episodes whose situation it can
# read (the translation layer's arity/type matching).
# ═════════════════════════════════════════════════════════════════
def _describe_kinds(desc):
    return [p["kind"] for p in (desc or {}).get("parts", [])]


CONCEPTS = [
    dict(name="greater",  pool="nums2", fn=lambda a: a[0] > a[1]),
    dict(name="less",     pool="nums2", fn=lambda a: a[0] < a[1]),
    dict(name="equal",    pool="nums2", fn=lambda a: a[0] == a[1]),
    dict(name="multiple", pool="nums2",
         fn=lambda a: a[1] >= 2 and a[0] % a[1] == 0),
    dict(name="even",     pool="nums1", fn=lambda a: a[0] % 2 == 0),
    dict(name="odd",      pool="nums1", fn=lambda a: a[0] % 2 == 1),
    dict(name="sum",      pool="nums3", fn=lambda a: a[0] + a[1] == a[2]),
    dict(name="product",  pool="nums3", fn=lambda a: a[0] * a[1] == a[2]),
    dict(name="round_shape",    pool="image",
         fn=lambda d: any(k in ("ring", "disc") for k in _describe_kinds(d))),
    dict(name="straight_shape", pool="image",
         fn=lambda d: bool(_describe_kinds(d))
         and all(k == "line" for k in _describe_kinds(d))),
]
BY_POOL = {}
for c in CONCEPTS:
    BY_POOL.setdefault(c["pool"], []).append(c)


def _is_num(tok):
    return tok.lstrip("-").isdigit()


# ═════════════════════════════════════════════════════════════════
# THE CURRICULUM — every utterance truthfully describes its
# situation. 'the' and 'is' appear everywhere (ubiquity controls);
# no word is ever labeled with its meaning.
# ═════════════════════════════════════════════════════════════════
def ring_img(rng, n=24):
    img = np.zeros((n, n))
    cx, cy = rng.integers(9, 15, 2)
    r = rng.integers(5, 9)
    yy, xx = np.mgrid[0:n, 0:n]
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    img[np.abs(d - r) <= 1.1] = 1.0
    return img


def line_img(rng, n=24):
    img = np.zeros((n, n))
    ang = rng.uniform(0, np.pi)
    cx, cy = rng.integers(9, 15, 2)
    for t in np.linspace(-9, 9, 160):
        x = int(round(cx + t * np.cos(ang)))
        y = int(round(cy + t * np.sin(ang)))
        if 0 <= x < n and 0 <= y < n:
            img[y, x] = 1.0
    return img


def _truths(pool, situation):
    return {c["name"]: bool(c["fn"](situation)) for c in BY_POOL[pool]}


def make_episodes(mind, rng, per_word=60, per_image_word=25, lo=0, hi=60,
                  swap_bigger_smaller=False):
    """The teacher. Situations first, then a true sentence about each.
    swap_bigger_smaller is the liar control: same situations, the two
    words exchanged — the teacher's labels follow the lie."""
    eps = []
    W_BIG, W_SMALL = (("smaller", "bigger") if swap_bigger_smaller
                      else ("bigger", "smaller"))

    def add(pool, tokens, situation):
        eps.append(dict(pool=pool, tokens=tokens,
                        truths=_truths(pool, situation)))

    for _ in range(per_word):                     # bigger / smaller / same
        a, b = sorted(rng.integers(lo, hi, 2))
        if a == b:
            b = a + int(rng.integers(1, 9))
        add("nums2", ["the", str(b), "is", W_BIG, str(a)], (b, a))
        a2, b2 = sorted(rng.integers(lo + 1, hi, 2))
        if a2 == b2:
            b2 = a2 + int(rng.integers(1, 9))
        add("nums2", ["the", str(a2), "is", W_SMALL, str(b2)], (a2, b2))
        s = int(rng.integers(lo, hi))
        add("nums2", ["the", str(s), "is", "same", str(s)], (s, s))
        k = int(rng.integers(2, 10))              # multiple: m varied so
        m = int(rng.choice([0, 1, 1, 2, 3, 4]))   # nothing rides along
        add("nums2", ["the", str(k * m), "is", "multiple", str(k)],
            (k * m, k))
    for _ in range(per_word):                     # even / odd
        e = int(rng.integers(lo, hi // 2)) * 2
        add("nums1", ["the", str(e), "is", "even"], (e,))
        o = int(rng.integers(lo, hi // 2)) * 2 + 1
        add("nums1", ["the", str(o), "is", "odd"], (o,))
    for _ in range(per_word):                     # plus / times
        a, b = (int(x) for x in rng.integers(lo, hi // 2, 2))
        add("nums3", ["the", str(a), "plus", str(b), "is", str(a + b)],
            (a, b, a + b))
        a2, b2 = (int(x) for x in rng.integers(lo, 9, 2))
        add("nums3", ["the", str(a2), "times", str(b2), "is",
                      str(a2 * b2)], (a2, b2, a2 * b2))
    for _ in range(per_image_word):               # round / straight
        d = mind.describe_shape(ring_img(rng))
        add("image", ["the", "shape", "is", "round"], d)
        d = mind.describe_shape(line_img(rng))
        add("image", ["the", "shape", "is", "straight"], d)
    return eps


# ═════════════════════════════════════════════════════════════════
# THE TEACHER'S GATE — a word binds to a concept only if its excess
# over the pool base rate beats a selection-corrected null (max
# excess over ALL concepts on every shuffled draw — the same
# correction the dialogue probe forced on speech_learn).
# ═════════════════════════════════════════════════════════════════
def teach_vocabulary(mind, episodes, n_perm=200, pct=99, min_evidence=8,
                     seed=3):
    rng = np.random.default_rng(seed)
    admitted = {}
    for pool, concepts in BY_POOL.items():
        pool_eps = [e for e in episodes if e["pool"] == pool]
        if not pool_eps:
            continue
        T = {c["name"]: np.array([e["truths"][c["name"]]
                                  for e in pool_eps]) for c in concepts}
        base = {n: t.mean() for n, t in T.items()}
        words = sorted({t for e in pool_eps for t in e["tokens"]
                        if not _is_num(t)})
        for w in words:
            has = np.array([w in e["tokens"] for e in pool_eps])
            nh = int(has.sum())
            if nh < min_evidence:
                continue
            excess = {n: T[n][has].mean() - base[n] for n in T}
            null_max = np.empty(n_perm)
            for i in range(n_perm):
                pick = rng.choice(len(pool_eps), size=nh, replace=False)
                null_max[i] = max(T[n][pick].mean() - base[n] for n in T)
            gate = float(np.percentile(null_max, pct))
            found = sorted(((n, x) for n, x in excess.items() if x > gate),
                           key=lambda kv: -kv[1])
            if not found:
                continue
            node = f"word:{w}"
            grounding = mind.net.get_meta(node, "grounding") or {}
            for n, x in found:
                mind.net.add_door(node, f"concept:{n}", tag="means")
                # counts, kept raw: "seen once" and "seen a hundred
                # times" must be different facts. No weights anywhere —
                # these are tallies and the null is the gate.
                grounding[n] = dict(
                    excess=round(float(x), 4),
                    true_with_word=int(T[n][has].sum()),
                    with_word=nh,
                    true_in_pool=int(T[n].sum()),
                    pool_size=len(pool_eps),
                    base_rate=round(float(base[n]), 4),
                    gate=round(gate, 4), pool=pool,
                    null="scrambled-pairing", n_perm=n_perm, pct=pct)
            mind.net.set_meta(node, "grounding", grounding)
            admitted[w] = found
    return admitted


# ═════════════════════════════════════════════════════════════════
# USE — answering is translation + execution through the doors.
# Rank + shortlist + abstention: the top-excess concept answers; a
# near-tied rival that DISAGREES on these arguments is a refusal.
# ═════════════════════════════════════════════════════════════════
def _resolve(mind, w):
    doors = mind.net.get_doors(f"word:{w}", tag="means")
    if not doors:
        return None
    g = mind.net.get_meta(f"word:{w}", "grounding") or {}
    ranked = sorted(((n, g.get(n, {}).get("excess", 0.0))
                     for n in (d.split("concept:", 1)[1] for d in doors)),
                    key=lambda kv: -kv[1])
    return ranked


def answer(mind, tokens, img=None):
    words = [t for t in tokens if not _is_num(t) and t != "?"]
    grounded = [(w, _resolve(mind, w)) for w in words]
    grounded = [(w, r) for w, r in grounded if r]
    if not grounded:
        return dict(verdict="refuse",
                    why=f"no grounded word in {words}")
    if len({w for w, _ in grounded}) > 1:
        return dict(verdict="refuse",
                    why=f"more than one grounded word: "
                        f"{sorted({w for w, _ in grounded})}")
    w, ranked = grounded[0]
    by_name = {c["name"]: c for c in CONCEPTS}
    top = by_name[ranked[0][0]]
    if top["pool"] == "image":
        if img is None:
            return dict(verdict="refuse", why="no image given")
        situation = mind.describe_shape(img)
    else:
        args = tuple(int(t) for t in tokens if _is_num(t))
        need = int(top["pool"][-1])
        if len(args) != need:
            return dict(verdict="refuse",
                        why=f"'{w}' means {top['name']} which wants "
                            f"{need} numbers, got {len(args)}")
        situation = args
    val = bool(top["fn"](situation))
    for name, x in ranked[1:]:
        rival = by_name[name]
        if rival["pool"] == top["pool"] and \
                ranked[0][1] - x < 0.05 and \
                bool(rival["fn"](situation)) != val:
            return dict(verdict="refuse",
                        why=f"'{w}' is ambiguous here: {top['name']} "
                            f"and {name} disagree")
    return dict(verdict="yes" if val else "no", word=w,
                concept=top["name"], shortlist=ranked)


# ═════════════════════════════════════════════════════════════════
# TOLD RULES THROUGH THE TAUGHT VOCABULARY — _RULE_CONDS, learned.
# A clause condition is words (+ number parameters); each word must
# resolve through its own doors or the rule REFUSES at tell time.
# ═════════════════════════════════════════════════════════════════
def rule_from_words(mind, clauses):
    by_name = {c["name"]: c for c in CONCEPTS}
    compiled = []
    for cond_words, say in clauses:
        preds = []
        i = 0
        while i < len(cond_words):
            w = cond_words[i]
            ranked = _resolve(mind, str(w))
            if not ranked:
                raise ValueError(f"REFUSED: no grounded meaning for "
                                 f"'{w}' — teach it first")
            con = by_name[ranked[0][0]]
            if con["pool"] == "nums1":
                preds.append((con["name"],
                              lambda n, f=con["fn"]: f((n,))))
                i += 1
            elif con["pool"] == "nums2" and i + 1 < len(cond_words) \
                    and _is_num(str(cond_words[i + 1])):
                k = int(cond_words[i + 1])
                preds.append((f"{con['name']}({k})",
                              lambda n, f=con["fn"], k=k: f((n, k))))
                i += 2
            else:
                raise ValueError(f"REFUSED: '{w}' means {con['name']} "
                                 f"which cannot condition a number "
                                 f"without a parameter")
        compiled.append((preds, say))
    return compiled


def respond_words(compiled, n):
    best_spec, best = -1, []
    for preds, say in compiled:
        if all(p(n) for _, p in preds):
            if len(preds) > best_spec:
                best_spec, best = len(preds), [say]
            elif len(preds) == best_spec:
                best.append(say)
    if not best:
        return None
    if len(set(best)) > 1:
        return None
    return str(n) if best[0] == "__number__" else best[0]


# ═════════════════════════════════════════════════════════════════
# THE RUN
# ═════════════════════════════════════════════════════════════════
def main():
    rng = np.random.default_rng(11)
    mind = mod.Isaac()
    print("=" * 64)
    print("VOCAB GROUNDING — words as doors onto his own organs")
    print("=" * 64)

    eps = make_episodes(mind, rng)
    admitted = teach_vocabulary(mind, eps)
    print(f"\n[teach] {len(eps)} episodes, no meanings labeled")
    intended = dict(bigger="greater", smaller="less", same="equal",
                    multiple="multiple", even="even", odd="odd",
                    plus="sum", times="product",
                    round="round_shape", straight="straight_shape")
    ok_bind = 0
    for w, want in intended.items():
        got = admitted.get(w, [])
        primary = got[0][0] if got else None
        mark = "OK " if primary == want else "MISS"
        if primary == want:
            ok_bind += 1
        rest = ", ".join(f"{n} ({x:+.2f})" for n, x in got[1:]) or "-"
        top = f"{primary} ({got[0][1]:+.2f})" if got else "NOTHING"
        print(f"  [{mark}] {w:9s} -> {top:22s} runners-up: {rest}")
    func_bound = [w for w in ("the", "is", "shape") if w in admitted]
    print(f"  bindings correct: {ok_bind}/{len(intended)}   "
          f"ubiquitous words bound: {len(func_bound)}/3 "
          f"{func_bound or ''}")

    # ── USE: unseen queries, numbers beyond the training range ────
    print("\n[use] unseen queries (numbers up to 300; trained on 0-60)")
    n_ok = n_all = 0
    by_name = {c["name"]: c for c in CONCEPTS}
    for _ in range(60):
        for w, want in (("bigger", "greater"), ("smaller", "less"),
                        ("same", "equal"), ("multiple", "multiple")):
            a, b = (int(x) for x in rng.integers(0, 300, 2))
            if w == "same" and rng.random() < 0.5:
                b = a
            if w == "multiple" and rng.random() < 0.5:
                b = int(rng.integers(2, 12)); a = b * int(rng.integers(0, 20))
            out = answer(mind, ["the", str(a), "is", w, str(b), "?"])
            truth = "yes" if by_name[want]["fn"]((a, b)) else "no"
            n_all += 1
            n_ok += (out["verdict"] == truth)
    for _ in range(40):
        for w, want in (("even", "even"), ("odd", "odd")):
            a = int(rng.integers(0, 300))
            out = answer(mind, ["the", str(a), "is", w, "?"])
            truth = "yes" if by_name[want]["fn"]((a,)) else "no"
            n_all += 1
            n_ok += (out["verdict"] == truth)
    for _ in range(20):
        for w, want in (("plus", "sum"), ("times", "product")):
            a, b = (int(x) for x in rng.integers(0, 40, 2))
            c = a + b if w == "plus" else a * b
            if rng.random() < 0.5:
                c += int(rng.integers(1, 5))
            out = answer(mind, ["the", str(a), w, str(b), "is",
                                str(c), "?"])
            truth = "yes" if by_name[want]["fn"]((a, b, c)) else "no"
            n_all += 1
            n_ok += (out["verdict"] == truth)
    img_ok = img_all = 0
    for _ in range(15):
        for w, maker, want in (("round", ring_img, "round_shape"),
                               ("straight", line_img, "straight_shape")):
            img = maker(rng)
            out = answer(mind, ["the", "shape", "is", w, "?"], img=img)
            truth = ("yes" if by_name[want]["fn"](mind.describe_shape(img))
                     else "no")
            img_all += 1
            img_ok += (out["verdict"] == truth)
    print(f"  number queries: {n_ok}/{n_all} correct "
          f"({100.0 * n_ok / n_all:.1f}%)")
    print(f"  image queries:  {img_ok}/{img_all} correct "
          f"({100.0 * img_ok / img_all:.1f}%)")

    # refusals are answers
    ref_ok = 0
    for w in ("prime", "heavier", "between", "narrow"):
        for _ in range(5):
            a, b = (int(x) for x in rng.integers(0, 100, 2))
            out = answer(mind, ["the", str(a), "is", w, str(b), "?"])
            ref_ok += (out["verdict"] == "refuse")
    arity_ok = 0
    for _ in range(10):
        a = int(rng.integers(0, 100))
        out = answer(mind, ["the", str(a), "is", "bigger", "?"])
        arity_ok += (out["verdict"] == "refuse")
    print(f"  ungrounded words refused: {ref_ok}/20   "
          f"arity mismatch refused: {arity_ok}/10")

    # ── GENERATION: told rules in taught words, zero examples ─────
    print("\n[rules] told in TAUGHT words -> generated, zero examples")
    ruleA = rule_from_words(mind, [
        (["even", "multiple", 3], "pingpong"),
        (["even"], "ping"),
        (["multiple", 3], "pong"),
        ([], "__number__")])
    genA = [respond_words(ruleA, i) for i in range(1, 121)]
    truthA = ["pingpong" if i % 6 == 0 else "ping" if i % 2 == 0
              else "pong" if i % 3 == 0 else str(i)
              for i in range(1, 121)]
    okA = sum(g == t for g, t in zip(genA, truthA))
    print(f"  even/multiple-3 rule: {okA}/120 "
          f"(first 12: {' '.join(genA[:12])})")
    ruleB = rule_from_words(mind, [(["bigger", 100], "big"),
                                   ([], "__number__")])
    genB = [respond_words(ruleB, i) for i in range(90, 111)]
    truthB = ["big" if i > 100 else str(i) for i in range(90, 111)]
    okB = sum(g == t for g, t in zip(genB, truthB))
    print(f"  bigger-100 rule:      {okB}/21")
    try:
        rule_from_words(mind, [(["prime"], "p"), ([], "__number__")])
        untaught = "FAILED TO REFUSE"
    except ValueError as e:
        untaught = f"refused ({e})"
    print(f"  untaught word in a rule: {untaught}")

    # ── CONTROLS ──────────────────────────────────────────────────
    print("\n[controls]")
    scr = [dict(e) for e in eps]
    for pool in BY_POOL:
        idx = [i for i, e in enumerate(scr) if e["pool"] == pool]
        perm = rng.permutation(idx)
        moved = {i: scr[j]["truths"] for i, j in zip(idx, perm)}
        for i in idx:
            scr[i] = dict(scr[i], truths=moved[i])
    scr_admit = teach_vocabulary(mod.Isaac(), scr)
    print(f"  scrambled pairing: {sum(len(v) for v in scr_admit.values())}"
          f" bindings admitted (want 0) {dict(scr_admit) or ''}")
    liar_mind = mod.Isaac()
    liar_eps = make_episodes(liar_mind, np.random.default_rng(12),
                             swap_bigger_smaller=True)
    liar = teach_vocabulary(liar_mind, liar_eps)
    lb = (liar.get("bigger") or [(None, 0)])[0][0]
    print(f"  liar curriculum: 'bigger' bound to {lb} "
          f"(evidence decides, not the teacher's intent)")

    # ── COMPLETE NUMBERS ──────────────────────────────────────────
    print("\n" + "=" * 64)
    print("COMPLETE NUMBERS")
    print(f"  bindings: {ok_bind}/{len(intended)} words to the intended "
          f"concept; ubiquitous words bound {len(func_bound)}/3")
    print(f"  use: numbers {n_ok}/{n_all} "
          f"({100.0 * n_ok / n_all:.1f}%), images {img_ok}/{img_all} "
          f"({100.0 * img_ok / img_all:.1f}%)")
    print(f"  refusals: ungrounded {ref_ok}/20, arity {arity_ok}/10, "
          f"untaught-rule-word {'yes' if 'refused' in untaught else 'NO'}")
    print(f"  generation: rule A {okA}/120, rule B {okB}/21 "
          f"(zero examples either)")
    print(f"  scrambled-pairing bindings: "
          f"{sum(len(v) for v in scr_admit.values())} (want 0)")
    print("=" * 64)


if __name__ == "__main__":
    main()
