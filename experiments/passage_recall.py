"""passage_recall.py — SHORT PASSAGES: "WHAT DO YOU RECALL?"
(owner, 2026-07-13: "i was thinking more like short paragraph
things. what do you recall from this passage")

THE RETENTION LAW OF THIS RUNG: he retains his READINGS, never the
raw text. Each sentence he can read becomes a structured fact —
subject phrase, verb, object phrase — through his own patterns
(prior rungs). Recall is REGENERATION from that structure. A
sentence he cannot read is retained as a NAMED GAP ("one sentence I
could not read"), never guessed at, never invented. The fragment
road would have echoed strings; this retains understanding.

Recall questions add RETRIEVAL: "who took a bone ?" must first find
WHICH retained fact is about bone-taking (content overlap on his own
readings), then answer through the question-shape bindings learned
by correction (open_questions rung). Two facts match -> refuse,
naming both. Nothing matches -> refuse. Rank + veto + abstention.

Complete numbers. stdlib + numpy only.
"""
import os
import sys
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import isaac as mod                                      # noqa: E402
from parts_of_speech import (env_sentence, read_world,    # noqa: E402
                             teach_roles, identify)
from open_questions import (env_dialogue, respond,        # noqa: E402
                            learn_answer, _qkey)


# ═════════════════════════════════════════════════════════════════
# THE ENVIRONMENT — speaks a short passage, then probes it.
# ═════════════════════════════════════════════════════════════════
def env_passage(rng, n_facts=4, with_unreadable=False,
                with_ambiguous=False):
    sents, truths, subjects = [], [], set()
    while len(sents) < n_facts:
        toks, truth = env_sentence(rng)
        if truth["subject"] in subjects:
            continue
        subjects.add(truth["subject"])
        sents.append(toks)
        truths.append(truth)
    if with_ambiguous:
        # a second fact with the SAME predicate, different subject
        if not any(t and t["object"] for t in truths):
            return env_passage(rng, n_facts, with_unreadable,
                               with_ambiguous)
        bi = next(i for i, t in enumerate(truths)
                  if t and t["object"] is not None)
        base_toks, base = sents[bi], truths[bi]
        while True:
            toks, truth = env_sentence(rng)
            if truth["subject"] not in subjects:
                break
        sl = 1 if toks[0] in ("he", "she", "it", "they", "we") else 2
        vpos = base_toks.index(base["verb"])
        twin = toks[:sl] + base_toks[vpos:]
        subjects.add(truth["subject"])
        sents.append(twin)
        truths.append(dict(subject=truth["subject"],
                           verb=base["verb"], object=base["object"]))
    if with_unreadable:
        sents.append(["the", "zebra", "grumbled", "."])
        truths.append(None)                    # HE can't read this;
    return sents, truths                       # the grader knows why


def _subj_prefix(t):
    """The asker speaks grammatically: pronouns take no article."""
    if t["subject"] in ("he", "she", "it", "they", "we"):
        return [t["subject"]]
    return ["the", t["subject"]]


def env_question(rng, truths):
    """A recall question about one fact — a fair asker asks
    ANSWERABLE questions (a subject-question about a predicate two
    facts share is genuinely ambiguous; the deliberate-ambiguity
    test covers those separately)."""
    cands = [t for t in truths if t is not None]
    while True:
        t = cands[rng.integers(0, len(cands))]
        kinds = (["subject", "verb"] +
                 (["object"] if t["object"] else []))
        kind = kinds[rng.integers(0, len(kinds))]
        if kind == "subject":
            same_pred = [c for c in cands if c["verb"] == t["verb"]
                         and c["object"] == t["object"]]
            if len(same_pred) > 1:
                continue                    # ask something answerable
            q = (["who", t["verb"]] +
                 (["the", t["object"]] if t["object"] else []) + ["?"])
            return q, t["subject"]
        if kind == "verb":
            return _subj_prefix(t) + ["did", "what", "?"], t["verb"]
        return (_subj_prefix(t) + [t["verb"], "the", "what", "?"],
                t["object"])       # echo keeps the article — the
                                   # shape his corrections trained


# ═════════════════════════════════════════════════════════════════
# ISAAC'S SIDE — retain readings; regenerate; retrieve; answer.
# ═════════════════════════════════════════════════════════════════
def read_fact(mind, g, toks):
    """His reading of one sentence, or None (a named gap)."""
    subj = identify(mind, g, toks, "subject")
    verb = identify(mind, g, toks, "verb")
    if subj["answer"] is None or verb["answer"] is None:
        return None
    k = toks.index(subj["answer"])
    v = toks.index(verb["answer"])
    obj = identify(mind, g, toks, "object")
    return dict(subj_phrase=toks[:k + 1], verb=verb["answer"],
                obj_phrase=toks[v + 1:-1],
                object=obj["answer"])


def hear_passage(mind, g, sents):
    """Retention: readings only. The raw text is not kept."""
    retained, gaps = [], 0
    for toks in sents:
        r = read_fact(mind, g, toks)
        if r is None:
            gaps += 1
        else:
            retained.append(r)
    return dict(facts=retained, gaps=gaps)


def recall(memory):
    """'What do you recall?' — regenerate every retained fact from
    structure; name the gaps plainly."""
    sents = [r["subj_phrase"] + [r["verb"]] + r["obj_phrase"] + ["."]
             for r in memory["facts"]]
    note = (f"and {memory['gaps']} sentence(s) i could not read"
            if memory["gaps"] else None)
    return sents, note


def bank_content_roles(mind, g, statement, question):
    """During question training he grounds MORE than the answer:
    each content word of the question fills some role of the
    statement per HIS reading — position -> role votes per shape.
    (Probe lesson: bag-of-words retrieval matched 'dog' in the
    wrong role; role-aware retrieval is learned, not assumed.)"""
    r = read_fact(mind, g, statement)
    if r is None:
        return
    key = _qkey(g, question)
    subj_n = r["subj_phrase"][-1]
    obj_n = r["obj_phrase"][-1] if r["obj_phrase"] else None
    store = mind.net.get_meta("qa:content", key) or {}
    for p, w in enumerate(question):
        role = ("verb" if w == r["verb"] else
                "subject" if w == subj_n else
                "object" if w == obj_n else None)
        if role:
            d = store.get(str(p)) or {}
            d[role] = d.get(role, 0) + 1
            store[str(p)] = d
    mind.net.set_meta("qa:content", key, store)


def _filler(fact, role):
    if role == "verb":
        return fact["verb"]
    if role == "subject":
        return fact["subj_phrase"][-1]
    return fact["obj_phrase"][-1] if fact["obj_phrase"] else None


def retrieve(mind, g, memory, question):
    """Which retained fact is this question about? Role-aware when
    the shape's content roles are learned (a question word must sit
    in the RIGHT role of the fact); overlap as fallback; ties
    refuse, naming the rivals."""
    key = _qkey(g, question)
    content = mind.net.get_meta("qa:content", key) or {}
    learned = {int(p): max(d.items(), key=lambda kv: kv[1])
               for p, d in content.items()}
    learned = {p: role for p, (role, c) in learned.items() if c >= 3}
    hits = []
    if learned:
        for i, fact in enumerate(memory["facts"]):
            if all(_filler(fact, role) == question[p]
                   for p, role in learned.items()):
                hits.append(i)
    else:
        qwords = [w for w in question
                  if w not in ("who", "what", "did", "?")]
        scored = []
        for r in memory["facts"]:
            c = set(r["subj_phrase"]) | {r["verb"]} | \
                set(r["obj_phrase"])
            scored.append(sum(1 for w in qwords if w in c))
        if scored and max(scored) > 0:
            best = max(scored)
            hits = [i for i, s in enumerate(scored) if s == best]
    if not hits:
        return None, "nothing i retain matches that question"
    if len(hits) > 1:
        rivals = " / ".join(
            " ".join(memory["facts"][i]["subj_phrase"]) for i in hits)
        return None, f"ambiguous — it could be about: {rivals}"
    return memory["facts"][hits[0]], None


def answer_from_passage(mind, g, memory, question):
    fact, why = retrieve(mind, g, memory, question)
    if fact is None:
        return "...", why
    toks = fact["subj_phrase"] + [fact["verb"]] + \
        fact["obj_phrase"] + ["."]
    return respond(mind, g, toks, question)


# ═════════════════════════════════════════════════════════════════
# THE RUN
# ═════════════════════════════════════════════════════════════════
def main():
    rng = np.random.default_rng(13)
    mind = mod.Isaac()
    print("=" * 64)
    print("PASSAGE RECALL — retained readings, not echoed strings")
    print("=" * 64)

    corpus = [env_sentence(rng)[0] for _ in range(900)]
    g = read_world(mind, corpus)
    rlessons = []
    while len(rlessons) < 90:
        toks, truth = env_sentence(rng)
        for role in ("subject", "verb", "object"):
            if truth[role] is not None:
                rlessons.append((toks, role, truth[role]))
    teach_roles(mind, g, rlessons)
    rng2 = np.random.default_rng(21)          # question-shape lessons
    for _ in range(60):                        # (open_questions rung)
        st, q, truth_ans = env_dialogue(rng2)
        said, _ = respond(mind, g, st, q)
        if said != truth_ans:
            learn_answer(mind, g, st, q, said, truth_ans)
        bank_content_roles(mind, g, st, q)    # ground the question's
        # other content words in his reading of the statement
    print("\n[foundation] groups, roles, question-shapes: prior rungs")

    # ── recall: what do you recall from this passage? ─────────────
    rng3 = np.random.default_rng(43)
    fact_ok = fact_n = invented = 0
    for _ in range(20):
        sents, truths = env_passage(rng3, n_facts=4)
        memory = hear_passage(mind, g, sents)
        got, note = recall(memory)
        fact_n += len(sents)
        fact_ok += sum(1 for s in sents if s in got)
        invented += sum(1 for s in got if s not in sents)
    print(f"\n[recall] 20 passages x 4 facts: {fact_ok}/{fact_n} "
          f"facts recalled exactly, {invented} invented (want 0)")
    sents, truths = env_passage(rng3, n_facts=4, with_unreadable=True)
    memory = hear_passage(mind, g, sents)
    got, note = recall(memory)
    print(f"  passage with an unreadable sentence: recalled "
          f"{len(got)}/4 readable facts + gap named: '{note}'")
    ex = " | ".join(" ".join(s) for s in got[:2])
    print(f"  (sample recall: {ex} ...)")

    # ── recall questions with retrieval ───────────────────────────
    rng4 = np.random.default_rng(77)
    q_ok = q_n = 0
    for _ in range(40):
        sents, truths = env_passage(rng4, n_facts=5)
        memory = hear_passage(mind, g, sents)
        q, truth_ans = env_question(rng4, truths)
        said, _ = answer_from_passage(mind, g, memory, q)
        q_n += 1
        q_ok += (said == truth_ans)
    amb_ok = amb_n = 0
    for _ in range(10):
        sents, truths = env_passage(rng4, n_facts=3,
                                    with_ambiguous=True)
        memory = hear_passage(mind, g, sents)
        t = next(x for x in truths if x and x["object"])
        q = ["who", t["verb"], "the", t["object"], "?"]
        said, why = answer_from_passage(mind, g, memory, q)
        amb_n += 1
        amb_ok += (said == "..." and "ambiguous" in str(why))
    np_ok = np_n = 0
    for _ in range(10):
        sents, truths = env_passage(rng4, n_facts=3)
        memory = hear_passage(mind, g, sents)
        said, why = answer_from_passage(
            mind, g, memory, ["who", "grumbled", "?"])
        np_n += 1
        np_ok += (said == "..." and "nothing" in str(why))
    print(f"\n[questions] across 5-fact passages: {q_ok}/{q_n} "
          f"answered right")
    print(f"  two facts match the question -> refused naming "
          f"rivals: {amb_ok}/{amb_n}")
    print(f"  question about nothing retained -> refused: "
          f"{np_ok}/{np_n}")

    print("\n" + "=" * 64)
    print("COMPLETE NUMBERS")
    print(f"  recall {fact_ok}/{fact_n} exact, {invented} invented; "
          f"gaps named honestly")
    print(f"  questions {q_ok}/{q_n}; ambiguity refusals "
          f"{amb_ok}/{amb_n}; nothing-retained refusals "
          f"{np_ok}/{np_n}")
    print("=" * 64)


if __name__ == "__main__":
    main()
