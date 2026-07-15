"""parts_of_speech.py — GROUPING AND IDENTIFICATION (owner,
2026-07-13: "strip that out. lets walk him through grouping and
identification. start with single simple things. he ran. who is the
subject and what is the verb. make sure he understands that. then
build out sentences and multiples... start with teaching him parts
of speech. have lists of words and say choose all the nouns etc")

THE PARSER IS STRIPPED. Nothing structured is ever handed to Isaac:
his input is token lists, full stop. The environment's generator
knows the truth about its own sentences — but that truth lives in
the GRADER only, never in what he receives (the previous world
handed him pre-shaped (actor, verb, object) tuples — my ontology,
wrong-kept with this why).

The walk, in the owner's order:
  1. HE READS. Plain sentences. His crystallizer mints word groups;
     sentence patterns admit past a scrambled null. All self-made.
  2. PARTS OF SPEECH ARE TAUGHT AS NAMES FOR HIS GROUPS. Lessons
     are just labeled examples ("dog is a noun"). A label binds to
     whichever of HIS OWN groups holds the examples, past a
     scrambled-pairing null. "Choose all the nouns" is then answered
     from GROUP MEMBERSHIP — including words never labeled in any
     lesson. Generation, not duplication: recall proves the group,
     not the memory.
  3. IDENTIFICATION. Lessons pair a sentence with a question and an
     answer ("he ran" / subject -> "he"). The label binds to a SLOT
     of his admitted patterns, past a shuffled-answer null. Tested
     on unseen sentences; a question with no bound slot in that
     pattern refuses honestly ("he ran" has no object).

Everything counted, gated, kept; refusals are answers; scrambled
controls must admit nothing. stdlib + numpy only.
"""
import os
import sys
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import isaac as mod                                    # noqa: E402


# ═════════════════════════════════════════════════════════════════
# THE ENVIRONMENT — it speaks in plain sentences and it grades.
# Its word lists are ITS knowledge; Isaac never sees them.
# ═════════════════════════════════════════════════════════════════
NOUNS = ["dog", "cat", "boy", "girl", "apple", "ball", "bone",
         "cup", "stick", "fox", "cow"]
PRONOUNS = ["he", "she", "it", "they", "we"]
V_INTR = ["ran", "sat", "slept", "smiled"]
V_TRAN = ["took", "held", "grabbed", "threw", "ate"]
DETS = ["the", "a"]


def env_sentence(rng):
    """A sentence (tokens) + the GRADER's truth (never shown)."""
    form = rng.integers(0, 4)
    n1 = NOUNS[rng.integers(0, len(NOUNS))]
    n2 = NOUNS[rng.integers(0, len(NOUNS))]
    p = PRONOUNS[rng.integers(0, len(PRONOUNS))]
    vi = V_INTR[rng.integers(0, len(V_INTR))]
    vt = V_TRAN[rng.integers(0, len(V_TRAN))]
    d1, d2 = DETS[rng.integers(0, 2)], DETS[rng.integers(0, 2)]
    if form == 0:                                   # he ran .
        return [p, vi, "."], dict(subject=p, verb=vi, object=None)
    if form == 1:                                   # the dog sat .
        return ([d1, n1, vi, "."],
                dict(subject=n1, verb=vi, object=None))
    if form == 2:                                   # she took the cup .
        return ([p, vt, d2, n2, "."],
                dict(subject=p, verb=vt, object=n2))
    return ([d1, n1, vt, d2, n2, "."],              # the dog took a bone .
            dict(subject=n1, verb=vt, object=n2))


def pos_truth(w):
    if w in NOUNS:
        return "noun"
    if w in PRONOUNS:
        return "pronoun"
    if w in V_INTR + V_TRAN:
        return "verb"
    return None


# ═════════════════════════════════════════════════════════════════
# 1. HE READS — groups and patterns, self-made
# ═════════════════════════════════════════════════════════════════
def read_world(mind, corpus, n_null=40, pct=95, seed=5):
    rng = np.random.default_rng(seed)
    for toks in corpus:
        mind.observe(list(toks))
    mind.consolidate()
    type_of = {}
    for nd in mind.net.all_nodes():
        if str(nd).startswith("type:"):
            for w in mind.net.get_predecessors(nd, tag="type"):
                type_of[w] = nd

    g = dict(type_of=type_of, known=set(type_of))
    g["units"] = subdivide(g, corpus)
    # patterns are built over his FINEST units (most-specific-wins,
    # applied to reading): '.' separates from the verbs, 'something'
    # from the determiners — sharper patterns AND an honest null
    # (coarse types double-counted identity permutations, inflating
    # the gate past what real patterns could clear; measured)
    finest = {}
    for u, ws in g["units"].items():
        for w in ws:
            cur = finest.get(w)
            if cur is None or len(g["units"][u]) < len(g["units"][cur]):
                finest[w] = u
    g["type_of"] = finest

    def template(toks):
        return tuple(finest.get(w, w) for w in toks)

    real = Counter(template(t) for t in corpus)
    null_best = np.empty(n_null)
    for i in range(n_null):
        cnt = Counter()
        for toks in corpus:
            s = [toks[j] for j in rng.permutation(len(toks))]
            cnt[template(s)] += 1
        null_best[i] = max(cnt.values()) if cnt else 0
    gate = float(np.percentile(null_best, pct))
    admitted = {t: c for t, c in real.items() if c > gate}
    mind.net.set_meta("grammar:patterns", "admitted", dict(
        patterns={" ".join(map(str, t)): c for t, c in admitted.items()},
        gate=gate, seen=len(real)))
    g.update(admitted=admitted, gate=gate, seen=len(real))
    return g


def subdivide(g, corpus):
    """HIERARCHY: subgroups are real things under parents (the
    sheet's law #7). Within each of his groups, members cluster by
    their dominant neighbor GROUPS (left, right — START/END are
    contexts too). Pronouns split from nouns because a determiner
    never precedes them; '.' splits from verbs because nothing ever
    follows it. Parents are kept; nothing is smushed."""
    left, right = {}, {}
    for toks in corpus:
        for i, w in enumerate(toks):
            lft = (str(g["type_of"].get(toks[i - 1], toks[i - 1]))
                   if i > 0 else "START")
            rgt = (str(g["type_of"].get(toks[i + 1], toks[i + 1]))
                   if i + 1 < len(toks) else "END")
            left.setdefault(w, Counter())[lft] += 1
            right.setdefault(w, Counter())[rgt] += 1
    members = {}
    for w, t in g["type_of"].items():
        members.setdefault(str(t), set()).add(w)
    units = {}
    for t, ws in members.items():
        units[t] = set(ws)                      # the parent, kept
        clusters = {}
        for w in ws:
            sig = (left[w].most_common(1)[0][0],
                   right[w].most_common(1)[0][0])
            clusters.setdefault(sig, set()).add(w)
        if len(clusters) > 1:
            for i, (sig, sub) in enumerate(sorted(
                    clusters.items(), key=lambda kv: str(kv[0]))):
                units[f"{t}/sub{i}"] = sub
    return units


# ═════════════════════════════════════════════════════════════════
# 2. PARTS OF SPEECH — labels bind to HIS groups, past a null
# ═════════════════════════════════════════════════════════════════
def teach_labels(mind, g, lessons, n_perm=200, pct=99, seed=7):
    """lessons: [(label, example_word), ...] — sparse, never
    exhaustive. A label binds to whichever of HIS units (groups and
    their subgroups) holds its examples beyond what scrambled
    labeling explains; the MOST SPECIFIC unit wins (highest excess —
    rarer base rate, same law that ranked equal above multiple)."""
    rng = np.random.default_rng(seed)
    units = g["units"]
    n_typed = max(len(g["type_of"]), 1)
    labels = sorted({L for L, _ in lessons})
    ex_words = [w for _, w in lessons]
    bound = {}
    for L in labels:
        ex = [w for lab, w in lessons if lab == L]
        excess = {u: sum(1 for w in ex if w in ws) / len(ex)
                  - len(ws) / n_typed for u, ws in units.items()}
        null_max = np.empty(n_perm)
        for i in range(n_perm):
            pick = [ex_words[j] for j in
                    rng.choice(len(ex_words), size=len(ex),
                               replace=False)]
            null_max[i] = max(
                sum(1 for w in pick if w in ws) / len(ex)
                - len(ws) / n_typed for ws in units.values())
        gate = float(np.percentile(null_max, pct))
        hits = sorted(((u, x) for u, x in excess.items() if x > gate),
                      key=lambda kv: -kv[1])
        if hits:
            top = hits[0][0]
            mind.net.set_meta(f"pos:{L}", "unit", top)
            mind.net.set_meta(f"pos:{L}", "members",
                              sorted(units[top]))
            mind.net.set_meta(f"pos:{L}", "evidence", dict(
                shortlist=[(u, round(x, 3)) for u, x in hits[:4]],
                gate=round(gate, 4), examples=len(ex),
                null="scrambled-labeling"))
            bound[L] = top
    return bound


def choose_all(mind, g, label, words):
    """'Choose all the nouns' — answered from group membership,
    never from lesson memory."""
    ms = mind.net.get_meta(f"pos:{label}", "members")
    if ms is None:
        return dict(chosen=None, why=f"'{label}' names nothing "
                                     f"he has been taught")
    ms = set(ms)
    chosen = [w for w in words if w in ms]
    unknown = [w for w in words if w not in g["known"]]
    return dict(chosen=chosen, unknown=unknown, why=None)


# ═════════════════════════════════════════════════════════════════
# 3. IDENTIFICATION — subject/verb/object bind to pattern slots
# ═════════════════════════════════════════════════════════════════
def teach_roles(mind, g, lessons, n_perm=200, pct=99, seed=9):
    """lessons: [(tokens, role, answer_word), ...]. A role binds to
    the slot of each admitted pattern whose filler IS the answer,
    past a shuffled-answers null."""
    rng = np.random.default_rng(seed)

    def template(toks):
        return tuple(g["type_of"].get(w, w) for w in toks)

    by_pat = {}
    for toks, role, ans in lessons:
        tpl = template(toks)
        if tpl in g["admitted"]:
            by_pat.setdefault(tpl, []).append((list(toks), role, ans))
    bound = {}
    for tpl, ls in by_pat.items():
        roles = sorted({r for _, r, _ in ls})
        answers = [a for _, _, a in ls]
        for role in roles:
            mine = [(t, a) for t, r, a in ls if r == role]
            hit = {k: sum(1 for t, a in mine if t[k] == a) / len(mine)
                   for k in range(len(tpl))}
            null_max = np.empty(n_perm)
            for i in range(n_perm):
                pick = [answers[j] for j in
                        rng.choice(len(answers), size=len(mine),
                                   replace=False)]
                null_max[i] = max(
                    sum(1 for (t, _), a in zip(mine, pick)
                        if t[k] == a) / len(mine)
                    for k in range(len(tpl)))
            gate = float(np.percentile(null_max, pct))
            best_k = max(hit, key=lambda k: hit[k])
            if hit[best_k] > gate:
                key = " ".join(map(str, tpl))
                mind.net.set_meta(f"role:{role}", key, dict(
                    slot=int(best_k), agree=round(hit[best_k], 3),
                    gate=round(gate, 3), lessons=len(mine)))
                bound.setdefault(role, {})[key] = int(best_k)
    return bound


def identify(mind, g, tokens, role):
    unknown = [w for w in tokens if w not in g["known"]]
    if unknown:
        return dict(answer=None, why=f"unknown words: {unknown}")
    tpl = tuple(g["type_of"].get(w, w) for w in tokens)
    if tpl not in g["admitted"]:
        return dict(answer=None, why="no admitted pattern fits")
    rec = mind.net.get_meta(f"role:{role}",
                            " ".join(map(str, tpl)))
    if rec is None:
        return dict(answer=None,
                    why=f"this pattern has no {role} he was "
                        f"taught — refusing beats guessing")
    return dict(answer=tokens[rec["slot"]], why=None)


# ═════════════════════════════════════════════════════════════════
# THE RUN
# ═════════════════════════════════════════════════════════════════
def main():
    rng = np.random.default_rng(13)
    mind = mod.Isaac()
    print("=" * 64)
    print("PARTS OF SPEECH — grouping and identification, his groups")
    print("=" * 64)

    # 1. he reads
    corpus = [env_sentence(rng)[0] for _ in range(900)]
    g = read_world(mind, corpus)
    print(f"\n[reads] 900 sentences -> {len(g['units'])} units "
          f"(groups + their subgroups), "
          f"{len(g['admitted'])}/{g['seen']} patterns admitted "
          f"(gate {g['gate']:.0f})")
    for u, ws in sorted(g["units"].items()):
        print(f"  {u.split('[')[0] + ('/' + u.split('/')[1] if '/' in u else ''):22s}"
              f" {sorted(ws)}")

    # 2. parts of speech: sparse labeled examples, never exhaustive
    lessons = [("noun", w) for w in ("dog", "apple", "cup", "girl")]
    lessons += [("verb", w) for w in ("ran", "took", "held", "smiled")]
    lessons += [("pronoun", w) for w in ("he", "she", "it")]
    bound = teach_labels(mind, g, lessons)
    print(f"\n[labels] taught {len(lessons)} examples; bindings:")
    for L, u in sorted(bound.items()):
        ms = mind.net.get_meta(f"pos:{L}", "members")
        print(f"  {L:8s} -> his unit "
              f"{u.split('[')[0] + ('/' + u.split('/')[1] if '/' in u else '')}"
              f" = {ms}")
    lab_ok = lab_n = 0
    rng2 = np.random.default_rng(29)
    for label, truth_set in (("noun", set(NOUNS)),
                             ("verb", set(V_INTR + V_TRAN)),
                             ("pronoun", set(PRONOUNS))):
        for _ in range(10):
            pool = NOUNS + PRONOUNS + V_INTR + V_TRAN + DETS
            words = [pool[i] for i in
                     rng2.choice(len(pool), size=10, replace=False)]
            out = choose_all(mind, g, label, words)
            want = sorted(w for w in words if w in truth_set)
            lab_n += 1
            lab_ok += (sorted(out["chosen"] or []) == want)
    print(f"  'choose all the X' on random 10-word lists "
          f"(most words never labeled): {lab_ok}/{lab_n}")
    out = choose_all(mind, g, "adverb", ["dog", "ran"])
    print(f"  untaught label refuses: "
          f"{'yes' if out['chosen'] is None else 'NO'} "
          f"({out['why']})")

    # 3. identification: he ran -> subject he, verb ran
    rlessons = []
    rng3 = np.random.default_rng(31)
    while len(rlessons) < 90:
        toks, truth = env_sentence(rng3)
        for role in ("subject", "verb", "object"):
            if truth[role] is not None:
                rlessons.append((toks, role, truth[role]))
    rbound = teach_roles(mind, g, rlessons)
    print(f"\n[roles] taught from {len(rlessons)} q/a lessons; "
          f"bindings per pattern:")
    for role, pats in sorted(rbound.items()):
        print(f"  {role:8s} -> slot "
              f"{sorted(set(pats.values()))} across "
              f"{len(pats)} pattern(s)")
    id_ok = id_n = ref_ok = ref_n = 0
    rng4 = np.random.default_rng(37)
    for _ in range(60):
        toks, truth = env_sentence(rng4)
        for role in ("subject", "verb", "object"):
            out = identify(mind, g, toks, role)
            if truth[role] is None:
                ref_n += 1
                ref_ok += (out["answer"] is None)
            else:
                id_n += 1
                id_ok += (out["answer"] == truth[role])
    print(f"  unseen sentences: {id_ok}/{id_n} roles identified")
    print(f"  no-such-role refusals (object of 'he ran'): "
          f"{ref_ok}/{ref_n}")
    out = identify(mind, g, ["zebra", "ran"], "subject")
    print(f"  unknown word refuses by name: "
          f"{'yes' if out['answer'] is None else 'NO'} ({out['why']})")

    # controls: scrambled lessons must bind nothing
    mind_c = mod.Isaac()
    g_c = read_world(mind_c, corpus)
    rngc = np.random.default_rng(41)
    ex = [w for _, w in lessons]
    scr = [(L, ex[i]) for (L, _), i in
           zip(lessons, rngc.permutation(len(ex)))]
    b_scr = teach_labels(mind_c, g_c, scr)
    ans = [a for _, _, a in rlessons]
    scr_r = [(t, r, ans[i]) for (t, r, _), i in
             zip(rlessons, rngc.permutation(len(ans)))]
    rb_scr = teach_roles(mind_c, g_c, scr_r)
    print(f"\n[controls] scrambled label lessons bind: "
          f"{len(b_scr)} labels (want 0); "
          f"scrambled role lessons bind: "
          f"{sum(len(v) for v in rb_scr.values())} pattern-slots "
          f"(want 0)")

    print("\n" + "=" * 64)
    print("COMPLETE NUMBERS")
    print(f"  units {len(g['units'])}; patterns {len(g['admitted'])}")
    print(f"  choose-all lists exact: {lab_ok}/{lab_n}; "
          f"untaught label refused: yes")
    print(f"  role identification {id_ok}/{id_n}; "
          f"honest no-object refusals {ref_ok}/{ref_n}")
    print(f"  scrambled controls: "
          f"{len(b_scr) + sum(len(v) for v in rb_scr.values())} "
          f"bindings (want 0)")
    print("=" * 64)


if __name__ == "__main__":
    main()
