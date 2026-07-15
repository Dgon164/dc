"""complete_ideas.py — COMPLETE IDEAS (owner, 2026-07-13: "what we
need is structural flow charts... the thing we also worked with is
complete ideas. abc=d. action, actor, object, = idea. she threw
ball, he stood up, the dog sat 'and' [dog] took a nap.")

A sentence is not a string: it is parts filling the slots of an
admitted pattern, assembling into ONE idea-whole (abc=d — the
composition organ pointed at language, the flow-chart cycle finding
the pattern). The division of labor, per the usage law:

  hand-built = the translation (sentences -> word atoms; clause
               boundaries at the join word; the terminator) + the
               epistemics (template admission past a scrambled-
               sentence null; refusals)
  SELF-ASSEMBLED = the word TYPES (the crystallizer's own door road —
               no matmul engine room, no hand-listed classes) and
               the TEMPLATES (type-patterns admitted only because
               the real corpus repeats them and its shuffles cannot)
  THE IDEAS  = per sentence: the words fill an admitted template's
               slots -> an idea:<k> node in the net with a slot-
               tagged door per word. abc=d, literally.

THE CARRY (the owner's "[dog]"): "the dog sat and took a nap" — the
second clause alone satisfies NO admitted template. It is the SUFFIX
of one, missing its leading slots; the only type-consistent filler
in scope is the first clause's own leading segment. Completion is
what the template demands — there is no coreference rule anywhere.
If no completion satisfies a template: refusal, with the why.

Tests are USE (generation-not-duplication): decompose UNSEEN
sentences, retrieve every idea about a word, refuse salads and
unknown words; a scrambled corpus must admit nothing.

Complete numbers throughout. stdlib + numpy only.
"""
import os
import sys
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import isaac as mod                                    # noqa: E402
from isaac_store.structural import analyze_level       # noqa: E402


# ═════════════════════════════════════════════════════════════════
# THE WORLD'S SPEECH — the corpus generator IS the ground truth.
# The generator's word lists are the WORLD (who can act, what can be
# acted on), never shown to Isaac; he sees only sentences.
# ═════════════════════════════════════════════════════════════════
ACTORS = ["she", "he", "dog", "cat", "boy", "girl"]
TRANS = ["threw", "ate", "took", "held", "grabbed"]
INTRANS = ["sat", "ran", "slept", "smiled"]
OBJS = ["ball", "apple", "stick", "cup", "bone", "nap"]
DETS = ["the", "a"]


def gen_sentence(rng):
    """One sentence + its truth: the list of COMPLETE ideas, each a
    tuple of the full clause tokens (carried slots filled in)."""
    a = ACTORS[rng.integers(0, len(ACTORS))]
    d1 = DETS[rng.integers(0, 2)]
    d2 = DETS[rng.integers(0, 2)]
    form = rng.integers(0, 6)
    if form == 0:                                # she threw ball .
        v, o = TRANS[rng.integers(0, 5)], OBJS[rng.integers(0, 6)]
        toks = [a, v, o, "."]
        ideas = [(a, v, o)]
    elif form == 1:                              # the dog took the bone .
        v, o = TRANS[rng.integers(0, 5)], OBJS[rng.integers(0, 6)]
        toks = [d1, a, v, d2, o, "."]
        ideas = [(d1, a, v, d2, o)]
    elif form == 2:                              # he ran .
        v = INTRANS[rng.integers(0, 4)]
        toks = [a, v, "."]
        ideas = [(a, v)]
    elif form == 3:                              # the cat slept .
        v = INTRANS[rng.integers(0, 4)]
        toks = [d1, a, v, "."]
        ideas = [(d1, a, v)]
    elif form == 4:          # THE CARRY: the dog sat and took a nap .
        v1 = INTRANS[rng.integers(0, 4)]
        v2, o = TRANS[rng.integers(0, 5)], OBJS[rng.integers(0, 6)]
        toks = [d1, a, v1, "and", v2, d2, o, "."]
        ideas = [(d1, a, v1), (d1, a, v2, d2, o)]
    else:                    # no carry: she threw ball and he slept .
        v1, o = TRANS[rng.integers(0, 5)], OBJS[rng.integers(0, 6)]
        a2 = ACTORS[rng.integers(0, len(ACTORS))]
        v2 = INTRANS[rng.integers(0, 4)]
        toks = [a, v1, o, "and", a2, v2, "."]
        ideas = [(a, v1, o), (a2, v2)]
    return toks, ideas


# ═════════════════════════════════════════════════════════════════
# TRANSLATION — sentences to clause atom-sequences (hand-built,
# wordless: a join atom and a terminator atom, nothing else known).
# ═════════════════════════════════════════════════════════════════
JOIN, END = "and", "."


def clauses_of(tokens):
    out, cur = [], []
    for t in tokens:
        if t == END:
            break
        if t == JOIN:
            out.append(tuple(cur))
            cur = []
        else:
            cur.append(t)
    if cur:
        out.append(tuple(cur))
    return out


# ═════════════════════════════════════════════════════════════════
# TEACHING — types self-assemble (crystallizer), templates earn
# admission: a clause type-pattern is knowledge only if the real
# corpus repeats it more than its own scrambles can (selection-
# corrected: the null draw scores its BEST template).
# ═════════════════════════════════════════════════════════════════
def learn_grammar(mind, corpus, n_null=40, pct=95, seed=9):
    rng = np.random.default_rng(seed)
    for toks, _ in corpus:
        mind.observe(list(toks))
    mind.consolidate()
    type_of = {}
    for n in mind.net.all_nodes():
        if str(n).startswith("type:"):
            for w in mind.net.get_predecessors(n, tag="type"):
                type_of[w] = n
    known = {w for toks, _ in corpus for w in toks}

    def template(clause):
        return tuple(type_of.get(w, w) for w in clause)

    real = Counter()
    initial = Counter()   # a COMPLETE idea can stand alone; a pattern
    n_initial = n_later = 0   # that NEVER opens a sentence is a
    for toks, _ in corpus:    # FRAGMENT (certified never — negatives
        for ci, c in enumerate(clauses_of(toks)):  # are first-class)
            t = template(c)
            real[t] += 1
            if ci == 0:
                initial[t] += 1
                n_initial += 1
            else:
                n_later += 1
    null_best = np.empty(n_null)
    for i in range(n_null):
        cnt = Counter()
        for toks, _ in corpus:
            s = [t for t in toks if t != END]
            s = [s[j] for j in rng.permutation(len(s))] + [END]
            cnt.update(template(c) for c in clauses_of(s))
        null_best[i] = max(cnt.values()) if cnt else 0
    gate = float(np.percentile(null_best, pct))
    p_init = n_initial / max(n_initial + n_later, 1)
    admitted, fragments = {}, {}
    for t, c in real.items():
        if c <= gate:
            continue
        # expected sentence-initial occurrences if position were
        # chance; zero seen when >= 3 expected = certified fragment
        if initial[t] == 0 and c * p_init >= 3.0:
            fragments[t] = c
        else:
            admitted[t] = c
    mind.net.set_meta("grammar:templates", "admitted", dict(
        templates={" ".join(map(str, t)): c for t, c in admitted.items()},
        fragments={" ".join(map(str, t)): c
                   for t, c in fragments.items()},
        gate=gate, p_initial=round(p_init, 3),
        null="scrambled-sentence", n_null=n_null, pct=pct,
        seen_patterns=len(real)))
    return dict(type_of=type_of, known=known, admitted=admitted,
                fragments=fragments, gate=gate, seen=len(real))


# ═════════════════════════════════════════════════════════════════
# THE IDEAS — parse fills admitted templates; the carry is slot
# completion (a suffix template borrows its missing lead from the
# previous clause ONLY if the types agree and the whole then
# satisfies an admitted template). Refusals are answers.
# ═════════════════════════════════════════════════════════════════
def parse(g, tokens):
    unknown = sorted({w for w in tokens if w not in g["known"]})
    if unknown:
        return dict(ideas=None, why=f"unknown words: {unknown}")

    def template(clause):
        return tuple(g["type_of"].get(w, w) for w in clause)

    ideas, prev = [], None
    for c in clauses_of(tokens):
        if template(c) in g["admitted"]:
            ideas.append(tuple(c))
            prev = tuple(c)
            continue
        # a fragment (or unknown pattern) is not an idea: it must
        # COMPLETE into an admitted template, borrowing its missing
        # lead from the previous clause, types agreeing throughout
        done = False
        if prev is not None:
            tpl = template(c)
            for full in sorted(g["admitted"], key=len, reverse=True):
                k = len(full) - len(tpl)
                if k <= 0 or full[k:] != tpl or k > len(prev):
                    continue
                borrow = prev[:k]
                if template(borrow) == full[:k]:
                    ideas.append(tuple(borrow) + tuple(c))
                    prev = ideas[-1]
                    done = True
                    break
        if not done:
            return dict(ideas=None,
                        why=f"no admitted pattern fits clause {c} "
                            f"(and no completion satisfies one)")
    return dict(ideas=ideas, why=None)


def mint_ideas(mind, g, corpus):
    """abc=d into the net: one idea node per parsed clause, a slot-
    tagged door per word. The reverse index makes 'every idea about
    w' one lookup."""
    n = 0
    for toks, _ in corpus:
        out = parse(g, toks)
        if out["ideas"] is None:
            continue
        for idea in out["ideas"]:
            node = f"idea:{n}"
            n += 1
            for i, w in enumerate(idea):
                mind.net.add_door(node, w, tag=f"s{i}")
            mind.net.set_meta(node, "clause", list(idea))
    return n


def ideas_about(mind, w):
    return {tuple(mind.net.get_meta(p, "clause"))
            for p in mind.net.get_predecessors(w)
            if str(p).startswith("idea:")}


# ═════════════════════════════════════════════════════════════════
# THE RUN
# ═════════════════════════════════════════════════════════════════
def main():
    rng = np.random.default_rng(21)
    print("=" * 64)
    print("COMPLETE IDEAS — abc=d through the flow-chart organs")
    print("=" * 64)
    corpus = [gen_sentence(rng) for _ in range(260)]
    mind = mod.Isaac()
    g = learn_grammar(mind, corpus)

    types = sorted({t for t in g["type_of"].values()})
    print(f"\n[types] self-assembled by the crystallizer: {len(types)}")
    for t in types:
        ms = sorted(w for w, tt in g["type_of"].items() if tt == t)
        print(f"  {t} = {ms}")
    print(f"\n[templates] {g['seen']} clause patterns seen, "
          f"{len(g['admitted'])} complete + {len(g['fragments'])} "
          f"fragments, past the scrambled null (gate {g['gate']:.0f}):")

    def short(t):
        return " ".join(x.split(":")[1].split("[")[0] if ":" in x else x
                        for x in t)
    for t, c in sorted(g["admitted"].items(), key=lambda kv: -kv[1]):
        print(f"  x{c:<4d} [{short(t)}]")
    for t, c in sorted(g["fragments"].items(), key=lambda kv: -kv[1]):
        print(f"  x{c:<4d} [{short(t)}]  FRAGMENT — never opens a "
              f"sentence; demands completion")

    # the flow chart at type level (the structural organ's own read)
    stream = []
    for toks, _ in corpus[:120]:
        stream += [g["type_of"].get(w, w) for w in toks]
    lv = analyze_level(stream)
    print(f"\n[flow chart] the structural organ on the type stream: "
          f"{lv['pattern']}")
    for line in lv["chart"][:6]:
        print(f"  {line}")

    n_ideas = mint_ideas(mind, g, corpus)
    print(f"\n[ideas] {n_ideas} idea-wholes minted into the net "
          f"(slot-tagged doors)")

    # ── USE 1: decompose UNSEEN sentences ─────────────────────────
    rng2 = np.random.default_rng(77)
    seen = {tuple(t) for t, _ in corpus}
    tests = []
    while len(tests) < 150:
        t, ideas = gen_sentence(rng2)
        if tuple(t) not in seen:
            tests.append((t, ideas))
    ok = carry_ok = carry_all = 0
    for toks, truth in tests:
        out = parse(g, toks)
        good = (out["ideas"] is not None and
                [tuple(i) for i in out["ideas"]] == [tuple(i)
                                                     for i in truth])
        ok += good
        if JOIN in toks and len(truth) == 2 and truth[0][:2] == truth[1][:2]:
            carry_all += 1
            carry_ok += good
    print(f"\n[use] unseen sentences fully decomposed: {ok}/150 "
          f"({100.0 * ok / 150:.1f}%)")
    print(f"  of those, carried-actor conjunctions: "
          f"{carry_ok}/{carry_all} (the '[dog]' completion)")

    # ── USE 2: every idea about a word, from the net ──────────────
    truth_about = {}
    for toks, ideas in corpus:
        if parse(g, toks)["ideas"] is None:
            continue
        for i in ideas:
            for w in i:
                truth_about.setdefault(w, set()).add(tuple(i))
    q_ok = q_all = 0
    for w in ACTORS + OBJS:
        got = ideas_about(mind, w)
        want = truth_about.get(w, set())
        q_all += 1
        q_ok += (got == want)
    print(f"  'every idea about w' retrievals exact: {q_ok}/{q_all}")

    # ── refusals are answers ──────────────────────────────────────
    salad_ref = salad_n = 0
    for _ in range(30):
        toks, _ = gen_sentence(rng2)
        body = [t for t in toks if t != END]
        body = [body[j] for j in rng2.permutation(len(body))]
        # a scramble that lands back on admitted grammar IS a
        # sentence — only truly pattern-breaking scrambles count
        cl = clauses_of(body + [END])
        tpl = [tuple(g["type_of"].get(w, w) for w in c) for c in cl]
        if all(t in g["admitted"] for t in tpl):
            continue
        salad_n += 1
        out = parse(g, body + [END])
        salad_ref += (out["ideas"] is None)
    unk_ref = 0
    for _ in range(10):
        toks, _ = gen_sentence(rng2)
        toks = list(toks)
        toks[int(rng2.integers(0, len(toks) - 1))] = "zebra"
        out = parse(g, toks)
        unk_ref += (out["ideas"] is None and "zebra" in str(out["why"]))
    print(f"  word-salad refused: {salad_ref}/{salad_n} "
          f"(pattern-breaking scrambles only)   "
          f"unknown word refused by name: {unk_ref}/10")

    # ── control: a scrambled corpus must teach nothing ────────────
    scr = []
    for toks, ideas in corpus:
        body = [t for t in toks if t != END]
        body = [body[j] for j in rng.permutation(len(body))]
        scr.append((body + [END], ideas))
    g_scr = learn_grammar(mod.Isaac(), scr)
    print(f"\n[control] scrambled corpus: {len(g_scr['admitted'])} "
          f"templates admitted (want 0; patterns seen: "
          f"{g_scr['seen']})")

    print("\n" + "=" * 64)
    print("COMPLETE NUMBERS")
    print(f"  types self-assembled: {len(types)}; templates: "
          f"{len(g['admitted'])} complete + {len(g['fragments'])} "
          f"fragments of {g['seen']} seen (gate {g['gate']:.0f})")
    print(f"  unseen decomposition {ok}/150 ({100.0 * ok / 150:.1f}%); "
          f"carry {carry_ok}/{carry_all}; retrieval {q_ok}/{q_all}")
    print(f"  refusals: salad {salad_ref}/{salad_n}, unknown "
          f"{unk_ref}/10; scrambled-corpus templates "
          f"{len(g_scr['admitted'])}")
    print("=" * 64)


if __name__ == "__main__":
    main()
