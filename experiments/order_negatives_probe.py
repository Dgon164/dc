"""
ORDER NEGATIVES PROBE -- attack probe, FRESH EYES items 10+11 (2026-07-15).
Target organs: isaac.py:2394 compare_order, isaac.py:2485 order_matters
(both landed 2026-07-15). If the mind fails a world here, that is the
FINDING -- reported with numbers, never patched away.

THE CLAIM UNDER TEST
 (10) order_matters(op) badges order-sensitivity by MEASURING swaps:
      every matters=True op must carry a witness pair that RECOMPUTES;
      every matters=False op may claim freedom only as
      no-counterexample-in-N-probes. ATTACK: the probe domain is
      hardcoded uniform(0.5, 6.0) -- POSITIVE-ONLY and zero-free. An op
      that commutes on positives but not everywhere tests whether the
      stored boolean badge (matters=False) overreaches the measured
      domain, and whether the badge RECORDS the domain at all.
 (11) compare_order(a, b) on duplicate-heavy sequences: naming priority
      (swap > reversed > rotated > reordered; the record chose 'swap'
      for ab->ba although ba is ALSO reversed and rotated-by-1); every
      positional claim carries a mapping VERIFIED to turn a into b; and
      moved_positions is fixed-points-first minimal (the record admits
      this is NOT proven) -- counterexample searched by exhaustive
      enumeration over duplicate multisets.

PREDICTIONS (filed before the first run)
 P1 posmul trap (a*b if a>0 and b>0 else a-b) is badged matters=False
    by the positive-only domain although it fails to commute on ~75%
    of [-6,6]^2. The 'why' text will hedge (absence, not proof) but
    neither the result nor the node badge will record the probed
    domain or seed, so the absence claim is unqualified downstream.
 P2 zero trap (a+b unless an operand is 0, else a-b) is also badged
    matters=False; the counterexample f(0,1) != f(1,0) exists. Random
    sampling cannot hit the boundary even on a widened domain -- there
    is no deliberate special-point probe (0, negatives, equal pairs).
 P3 stock badges: '-','/','^' matters=True with witnesses that
    recompute exactly; '+','*','max','min' matters=False with absence
    wording; each badge lands on the op:<sym> node and in the ledger.
 P4 unknown op -> refused=True (honest refusal road works).
 P5 ab vs ba is simultaneously swap-shaped, reversed AND rotated-by-1;
    the organ names 'swap' (priority). abab vs baba (reversed and
    rotated, NOT swap-shaped) is named 'reversed'.
 P6 ~260 random shuffles over duplicate-heavy alphabets: 0 mapping
    verification failures, 0 kind/priority mis-claims.
 P7 minimality: 0 counterexamples in exhaustive search. (Filed after
    reading the code, before running: a mapping fixed point is only
    possible where a[i]==b[i], so |moved| = Hamming(a,b) is a hard
    floor and the greedy achieves it -- the record's admitted doubt
    should be dischargeable by enumeration.)
 P8 controls: a genuinely order-free custom op entered by the same
    road is badged matters=False, and a genuinely order-sensitive
    custom op entered by the same road is caught with a verified
    witness -- isolating the DOMAIN, not the entry road, as the hole.

ENTRY ROAD (read from the code first, per house rules): order_matters
resolves ops via the CLASS dict Isaac._ORDER_OPS -- the signature takes
no callable. An instance-level copy is invisible to it (measured in
A1). The only road for a new op is rebinding the class attribute to an
EXTENDED COPY in memory for the duration of this process; the original
dict object is never mutated and isaac.py is never touched.

Run: cd /home/user/dc/experiments && PYTHONHASHSEED=0 \
     OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 \
     order_negatives_probe.py
"""
import itertools
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import isaac as mod                                    # noqa: E402
import numpy as np                                     # noqa: E402

TOL = 1e-9


def differs(x, y):
    """The engine's own inequality test (isaac.py:2505), replicated."""
    return abs(x - y) > TOL * max(1.0, abs(x), abs(y))


# ── trap and control ops ─────────────────────────────────────────
def posmul(a, b):
    """Commutes on positives (a*b); order-sensitive elsewhere (a-b)."""
    return a * b if (a > 0 and b > 0) else a - b


def zerotrap(a, b):
    """Commutes everywhere EXCEPT when an operand is exactly 0."""
    return a + b if (a != 0 and b != 0) else a - b


def posadd(a, b):
    """CONTROL: genuinely order-free everywhere, same entry road."""
    return a + b


def revsub(a, b):
    """CONTROL: genuinely order-sensitive on positives, same road."""
    return a - b


# independent recompute table for stock ops (NOT read from the engine)
LOCAL_STOCK = {
    "+": lambda a, b: a + b, "-": lambda a, b: a - b,
    "*": lambda a, b: a * b, "/": lambda a, b: a / b,
    "^": lambda a, b: a ** b,
    "max": lambda a, b: max(a, b), "min": lambda a, b: min(a, b),
}


# ── compare_order verification helpers ───────────────────────────
def apply_mapping(a, mapping, n):
    out = [None] * n
    for i, j in enumerate(mapping):
        out[j] = a[i]
    return out


def swap_shape(a, b):
    moved = [i for i in range(len(a)) if a[i] != b[i]]
    return (len(moved) == 2 and a[moved[0]] == b[moved[1]]
            and a[moved[1]] == b[moved[0]])


def rotations(a, b):
    return [k for k in range(1, len(a)) if b == a[k:] + a[:k]]


def kind_claim_true(a, b, kind):
    """Is the named transform ACTUALLY a transform of a into b?"""
    if kind == "swap":
        return swap_shape(a, b)
    if kind == "reversed":
        return len(a) > 1 and b == a[::-1]
    if kind.startswith("rotated by "):
        k = int(kind.split()[-1])
        return b == a[k:] + a[:k]
    if kind == "reordered":       # fallback: no nicer name may apply
        return (not swap_shape(a, b) and b != a[::-1]
                and not rotations(a, b))
    return False


def priority_ok(a, b, kind):
    """No lower-priority name may be used when a higher one applies."""
    if kind == "reversed" and swap_shape(a, b):
        return False
    if kind.startswith("rotated by ") and (swap_shape(a, b)
                                           or b == a[::-1]):
        return False
    return True


def brute_min_moved(a, b):
    """Exhaustive: min #non-fixed-points over ALL position-permutations
    that verifiably turn a into b."""
    n, best = len(a), None
    for p in itertools.permutations(range(n)):
        placed = [None] * n
        for i, j in enumerate(p):
            placed[j] = a[i]
        if placed == b:
            m = sum(1 for i, j in enumerate(p) if i != j)
            best = m if best is None else min(best, m)
    return best


VERDICTS = []


def verdict(cap, v, evidence):
    VERDICTS.append((cap, v, evidence))
    print(f"  -> VERDICT [{v}] {cap}: {evidence}")


def main():
    print("=" * 72)
    print("ORDER NEGATIVES PROBE -- fresh eyes 10+11 "
          "(order_matters / compare_order)")
    print("predictions P1-P8 filed in the docstring BEFORE this run")
    print("=" * 72)

    # ── A1: the entry road itself (measured, not assumed) ────────
    print("\n[A1] entry road: is an instance-level _ORDER_OPS copy "
          "visible?")
    m = mod.Isaac()
    m._ORDER_OPS = dict(mod.Isaac._ORDER_OPS)
    m._ORDER_OPS["posmul"] = posmul
    r = m.order_matters("posmul")
    inst_refused = bool(r.get("refused"))
    print(f"  instance-copy road -> refused={inst_refused} "
          f"(order_matters reads the CLASS dict Isaac._ORDER_OPS; "
          f"there is no per-mind op road and no callable parameter)")

    ORIG = mod.Isaac._ORDER_OPS          # saved; restored in finally
    mod.Isaac._ORDER_OPS = dict(ORIG, posmul=posmul, zerotrap=zerotrap,
                                posadd=posadd, revsub=revsub)
    try:
        # ── A2: the positive-only trap ────────────────────────────
        print("\n[A2] TRAP posmul = a*b if both>0 else a-b "
              "(commutes on positives ONLY)")
        m = mod.Isaac()
        r = m.order_matters("posmul", n_probe=50, seed=0)
        print(f"  engine badge: matters={r['matters']} "
              f"probes={r['probes']}")
        print(f"  why: {r['why']}")
        rng = np.random.default_rng(3)
        n_mc, n_diff = 2000, 0
        for _ in range(n_mc):
            a = float(rng.uniform(-6, 6)); b = float(rng.uniform(-6, 6))
            if differs(posmul(a, b), posmul(b, a)):
                n_diff += 1
        frac = n_diff / n_mc
        print(f"  ground truth on [-6,6]^2: {n_diff}/{n_mc} = "
              f"{frac:.3f} of random pairs do NOT commute")
        print(f"  explicit counterexample: posmul(-1,2)={posmul(-1, 2)}"
              f"  posmul(2,-1)={posmul(2, -1)}")
        node_badge = m.net.get_meta("op:posmul", "order")
        print(f"  node badge op:posmul -> matters="
              f"{node_badge['matters']}")
        dom_keys = [k for k in r if k.lower() in
                    ("domain", "lo", "hi", "low", "high", "range",
                     "seed", "interval")]
        print(f"  result keys: {sorted(r)}  -- domain/seed recorded: "
              f"{dom_keys if dom_keys else 'NONE'}")
        hedged = ("absence, not proof" in r["why"]
                  and "no counterexample" in r["why"])
        mis_badge = (r["matters"] is False and frac > 0.5)
        if mis_badge and hedged and not dom_keys:
            verdict("order_matters domain coverage (item 10a)",
                    "PARTIAL",
                    f"MIS-BADGE: matters=False stored on node+ledger "
                    f"for an op that fails to commute on {frac:.0%} of "
                    f"[-6,6]^2; why-text hedges honestly but the "
                    f"probed domain [0.5,6.0) and seed are recorded "
                    f"NOWHERE, so the absence claim is unqualified "
                    f"downstream")
        elif mis_badge and not hedged:
            verdict("order_matters domain coverage (item 10a)",
                    "BROKEN",
                    f"matters=False claimed without hedge for an op "
                    f"non-commuting on {frac:.0%} of [-6,6]^2")
        else:
            verdict("order_matters domain coverage (item 10a)",
                    "WORKS", f"trap caught: matters={r['matters']}")

        # ── A3: the zero trap ─────────────────────────────────────
        print("\n[A3] TRAP zerotrap = a+b unless an operand is 0, "
              "else a-b")
        m = mod.Isaac()
        r = m.order_matters("zerotrap", n_probe=50, seed=0)
        print(f"  engine badge: matters={r['matters']} "
              f"probes={r['probes']}")
        print(f"  counterexample at the boundary: zerotrap(0,1)="
              f"{zerotrap(0, 1)}  zerotrap(1,0)={zerotrap(1, 0)}")
        rng = np.random.default_rng(4)
        hits = sum(1 for _ in range(100000)
                   if float(rng.uniform(-6, 6)) == 0.0)
        print(f"  widened-domain simulation: exact zeros drawn in "
              f"100000 uniform(-6,6) samples: {hits} -- the boundary "
              f"is unreachable by random draws; only a DELIBERATE "
              f"special-point probe (0, negatives, equal pairs) "
              f"could catch this, and there is none")
        zero_misbadge = (r["matters"] is False)
        print(f"  zero-trap mis-badge: {zero_misbadge}")

        # ── A4: replicate the probed domain exactly ───────────────
        print("\n[A4] the probed domain, replicated draw-for-draw "
              "(default_rng(0), interleaved a,b)")
        r0 = np.random.default_rng(0)
        draws = [float(r0.uniform(0.5, 6.0)) for _ in range(100)]
        print(f"  100 draws (= 50 probe pairs): min={min(draws):.4f} "
              f"max={max(draws):.4f} #<=0: "
              f"{sum(1 for d in draws if d <= 0)} #==0: "
              f"{sum(1 for d in draws if d == 0.0)}")

        # ── A5: stock badges -- witness recompute + wording ───────
        print("\n[A5] stock badges (one fresh mind, defaults "
              "n_probe=50 seed=0)")
        ms = mod.Isaac()
        stock_ok, minus_w = True, None
        for op in ["+", "-", "*", "/", "^", "max", "min"]:
            r = ms.order_matters(op)
            node = ms.net.get_meta(f"op:{op}", "order")
            badge_lands = (node is not None
                           and node["matters"] == r["matters"])
            if r["matters"]:
                w = r["witness"]
                loc_ab = LOCAL_STOCK[op](w["a"], w["b"])
                loc_ba = LOCAL_STOCK[op](w["b"], w["a"])
                recomputes = (abs(loc_ab - w["ab"]) < 1e-12
                              and abs(loc_ba - w["ba"]) < 1e-12
                              and differs(w["ab"], w["ba"]))
                print(f"  '{op}': matters=True  witness a={w['a']:.4f} "
                      f"b={w['b']:.4f} ab={w['ab']:.4f} ba={w['ba']:.4f}"
                      f"  recomputes={recomputes} node={badge_lands}")
                stock_ok &= recomputes and badge_lands
                if op == "-":
                    minus_w = w
            else:
                wording = ("no counterexample in 50 probes" in r["why"]
                           and "absence, not proof" in r["why"])
                print(f"  '{op}': matters=False  absence-wording="
                      f"{wording} node={badge_lands}")
                stock_ok &= wording and badge_lands
        n_ledger = len([e for e in ms.findings
                        if e.get("kind") == "order"])
        print(f"  ledger entries kind='order': {n_ledger}/7")
        stock_ok &= (n_ledger == 7)
        if minus_w is not None:
            faithful = (minus_w["a"] == draws[0]
                        and minus_w["b"] == draws[1])
            print(f"  cross-check: '-' witness pair == first "
                  f"replicated domain pair: {faithful} "
                  f"(proves A4 replication is exact)")
            stock_ok &= faithful
        verdict("stock badges: '-' verified witness, '+' "
                "absence-only claim (item 10b)",
                "WORKS" if stock_ok else "BROKEN",
                f"witnesses recompute, wording + node badge + "
                f"{n_ledger}/7 ledger entries as claimed"
                if stock_ok else "see FAIL lines above")

        # ── A6: unknown op -> honest refusal ──────────────────────
        print("\n[A6] unknown op")
        r = ms.order_matters("nosuchop")
        print(f"  refused={r.get('refused')}  why={r.get('why')}")
        verdict("order_matters unknown-op road", "HONEST-REFUSAL"
                if r.get("refused") else "BROKEN",
                f"refused={r.get('refused')}, names the known ops")

        # ── A7: zeros against the stock ops directly ──────────────
        print("\n[A7] stock ops AT ZERO (evaluated outside the "
              "engine; its domain can never draw 0)")
        for op in ["+", "-", "*", "^", "max", "min"]:
            f = LOCAL_STOCK[op]
            print(f"  '{op}'(0,1)={f(0.0, 1.0):g}  "
                  f"'{op}'(1,0)={f(1.0, 0.0):g}  "
                  f"commutes_at_zero={not differs(f(0.0, 1.0), f(1.0, 0.0))}")
        try:
            LOCAL_STOCK["/"](1.0, 0.0)
        except ZeroDivisionError:
            print("  '/'(0,1)=0  '/'(1,0)=ZeroDivisionError -- an "
                  "order asymmetry that manifests as an EXCEPTION; "
                  "the engine's probe loop `continue`s on "
                  "ZeroDivisionError (isaac.py:2503-2504), so such a "
                  "pair would be discarded as no-evidence rather than "
                  "kept as a witness (dead code today: 0 is outside "
                  "the domain)")
        print("  note: no stock order-FREE badge ('+','*','max','min') "
              "is contradicted at zero -- the zero hole is real but "
              "the stock ops happen not to fall into it")

        # ── A8: controls -- same road, known truths ───────────────
        print("\n[A8] CONTROLS on the same entry road")
        m = mod.Isaac()
        rc1 = m.order_matters("posadd", n_probe=50, seed=0)
        rc2 = m.order_matters("revsub", n_probe=50, seed=0)
        c2w = rc2.get("witness")
        c2_verified = (c2w is not None
                       and abs(revsub(c2w["a"], c2w["b"]) - c2w["ab"])
                       < 1e-12
                       and differs(c2w["ab"], c2w["ba"]))
        print(f"  posadd (truly order-free):    matters="
              f"{rc1['matters']} (expect False)")
        print(f"  revsub (truly order-bound):   matters="
              f"{rc2['matters']} witness-verifies={c2_verified} "
              f"(expect True, True)")
        controls_ok = (rc1["matters"] is False
                       and rc2["matters"] is True and c2_verified)
        print(f"  controls isolate the hole to the DOMAIN, not the "
              f"road: {controls_ok}")

    finally:
        mod.Isaac._ORDER_OPS = ORIG      # restore; file never touched

    # ══ B: compare_order ═════════════════════════════════════════
    mB = mod.Isaac()

    # ── B1: naming priority on ambiguous transforms ───────────────
    print("\n[B1] naming priority (swap > reversed > rotated > "
          "reordered)")
    cases = [("ab", "ba", "swap"), ("abab", "baba", "reversed"),
             ("aaab", "aaba", "swap"), ("abcd", "cdab", "rotated by 2"),
             ("abcd", "dcba", "reversed")]
    b1_ok = True
    for sa, sb, expect in cases:
        a, b = list(sa), list(sb)
        r = mB.compare_order(a, b)
        kind = r["arrangement"]["kind"]
        sw, rev, rots = (swap_shape(a, b), b == a[::-1],
                         rotations(a, b))
        ok = kind == expect
        b1_ok &= ok
        print(f"  {sa}->{sb}: swap-shaped={sw} reversed={rev} "
              f"rotations={rots}  named='{kind}' expected="
              f"'{expect}' [{'PASS' if ok else 'FAIL'}]")
    verdict("compare_order naming priority incl. ab->ba triple "
            "ambiguity (item 11a)",
            "WORKS" if b1_ok else "BROKEN",
            "ab->ba is simultaneously swap+reversed+rotated-by-1 and "
            "'swap' is named; abab->baba (not swap-shaped) gets "
            "'reversed' over 'rotated by 1'" if b1_ok
            else "priority violated, see FAIL lines")

    # ── B2: random shuffles, duplicate-heavy alphabets ────────────
    print("\n[B2] random shuffles: mapping VERIFIED on every "
          "positional claim?")
    bases = ["aab", "abab", "aaab", "aabb", "aaabb", "aabbcc",
             "aaabbb", "aabc", "abcabc", "aaaab", "ab",
             "abcd", "abcdef"]              # last two: no-dup controls
    rng = np.random.default_rng(7)
    n_sh = 0
    f_map = f_bij = f_moved = f_kind = f_prio = f_same = 0
    for base in bases:
        a = list(base)
        for _ in range(20):
            b = list(a)
            rng.shuffle(b)
            n_sh += 1
            r = mB.compare_order(a, b)
            if a == b:
                if r["verdict"] != "same":
                    f_same += 1
                continue
            if r["verdict"] != "reordered":
                f_same += 1
                continue
            arr = r["arrangement"]
            mp = arr["mapping"]
            if sorted(mp) != list(range(len(a))):
                f_bij += 1
            if apply_mapping(a, mp, len(a)) != b:
                f_map += 1
            if arr["moved_positions"] != [i for i in range(len(a))
                                          if a[i] != b[i]]:
                f_moved += 1
            if not kind_claim_true(a, b, arr["kind"]):
                f_kind += 1
            if not priority_ok(a, b, arr["kind"]):
                f_prio += 1
    print(f"  {n_sh} shuffles over {len(bases)} alphabets "
          f"(11 duplicate-heavy + 2 distinct controls)")
    print(f"  mapping-verification failures: {f_map}/{n_sh}")
    print(f"  non-bijective mappings:        {f_bij}/{n_sh}")
    print(f"  moved_positions != Hamming set: {f_moved}/{n_sh}")
    print(f"  kind claim untrue:             {f_kind}/{n_sh}")
    print(f"  priority violations:           {f_prio}/{n_sh}")
    print(f"  same/reordered verdict wrong:  {f_same}/{n_sh}")
    b2_ok = (f_map + f_bij + f_moved + f_kind + f_prio + f_same) == 0
    verdict("compare_order mapping verification on duplicate-heavy "
            "shuffles (item 11b)",
            "WORKS" if b2_ok else "BROKEN",
            f"{n_sh} shuffles, 0 failures on all six invariants"
            if b2_ok else f"failures: map={f_map} bij={f_bij} "
            f"moved={f_moved} kind={f_kind} prio={f_prio} "
            f"verdict={f_same} of {n_sh}")

    # ── B3: minimality of moved_positions -- exhaustive search ────
    print("\n[B3] fixed-points-first minimality: exhaustive "
          "counterexample search on duplicate multisets")
    mbases = ["aab", "aaab", "aabb", "aabc", "aaabb", "aabbc",
              "aaabbb", "aabbcc"]
    n_pairs = n_cex = n_under = 0
    for base in mbases:
        a = list(base)
        perms = sorted(set(itertools.permutations(a)))
        cex_here = 0
        for bt in perms:
            b = list(bt)
            if b == a:
                continue
            n_pairs += 1
            r = mB.compare_order(a, b)
            reported = len(r["arrangement"]["moved_positions"])
            best = brute_min_moved(a, b)
            if reported > best:
                n_cex += 1
                cex_here += 1
                print(f"    COUNTEREXAMPLE {base}->{''.join(b)}: "
                      f"reported {reported} moved, brute min {best}")
            elif reported < best:
                n_under += 1
                print(f"    IMPOSSIBLE CLAIM {base}->{''.join(b)}: "
                      f"reported {reported} < achievable min {best}")
        print(f"  base '{base}': {len(perms) - 1} rearrangements, "
              f"{cex_here} counterexamples")
    print(f"  TOTAL: {n_pairs} pairs exhaustively brute-forced, "
          f"{n_cex} minimality counterexamples, {n_under} "
          f"impossible claims")
    verdict("compare_order moved_positions minimality on duplicates "
            "(item 11c; record admits unproven)",
            "WORKS" if (n_cex + n_under) == 0 else "BROKEN",
            f"0 counterexamples in {n_pairs} exhaustive pairs -- the "
            f"doubt discharges: a mapping fixed point is only "
            f"possible where a[i]==b[i], so |moved|=Hamming(a,b) is "
            f"a floor and the greedy meets it" if (n_cex + n_under)
            == 0 else f"{n_cex} counterexamples, {n_under} "
            f"impossible claims in {n_pairs} pairs")

    # ── B4: null control -- no positional claim without identity ──
    print("\n[B4] NULL: content-differing pairs must get NO "
          "positional mapping claim")
    rng = np.random.default_rng(9)
    n_null, n_viol = 0, 0
    for base in ["aab", "abab", "aaab", "aabbcc"]:
        for _ in range(10):
            a = list(base)
            b = list(a)
            rng.shuffle(b)
            b[int(rng.integers(0, len(b)))] = "z"   # break the multiset
            n_null += 1
            r = mB.compare_order(a, b)
            if ("arrangement" in r
                    or r["verdict"] not in ("content differs",
                                            "content and order differ")
                    or r.get("shared_order") not in ("kept",
                                                     "reordered")):
                n_viol += 1
    print(f"  {n_null} broken-multiset pairs: positional-claim "
          f"violations {n_viol}/{n_null}")
    verdict("compare_order null control (claims only where "
            "verifiable)",
            "WORKS" if n_viol == 0 else "BROKEN",
            f"{n_viol}/{n_null} violations")

    # entry-road observation as its own line item
    verdict("custom op entry road into order_matters", "PARTIAL",
            f"no per-mind road exists: instance-copy refused="
            f"{inst_refused}; signature takes no callable; the only "
            f"road is rebinding the CLASS dict (shared across all "
            f"minds in a process)")

    print("\n" + "=" * 72)
    print("PREDICTION SCORECARD")
    print(f"  P1 posmul mis-badge:   {'CONFIRMED' if mis_badge else 'REFUTED'}")
    print(f"  P2 zerotrap mis-badge: {'CONFIRMED' if zero_misbadge else 'REFUTED'}")
    print(f"  P3 stock badges:       {'CONFIRMED' if stock_ok else 'REFUTED'}")
    print(f"  P5 priority:           {'CONFIRMED' if b1_ok else 'REFUTED'}")
    print(f"  P6 shuffles clean:     {'CONFIRMED' if b2_ok else 'REFUTED'}")
    print(f"  P7 minimality holds:   "
          f"{'CONFIRMED' if (n_cex + n_under) == 0 else 'REFUTED'}")
    print(f"  P8 controls clean:     "
          f"{'CONFIRMED' if controls_ok else 'REFUTED'}")
    print("=" * 72)
    print("VERDICTS")
    for cap, v, ev in VERDICTS:
        print(f"  [{v}] {cap}")
    print("=" * 72)


if __name__ == "__main__":
    main()
