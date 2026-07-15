"""logical_gaps.py — LOGICAL STRUCTURE PROCESSING (owner,
2026-07-13: "how to turn understanding into sentences, how to
identify gaps in the information, and how to ask for information
regarding that gap")

The three capabilities, on the existing rungs:

  GAPS ARE STRUCTURAL. His patterns define what a complete fact
    looks like. The world sometimes speaks incompletely — "someone
    took the cup" — and the gap-markers ('someone', 'something')
    are LEARNED, never listed: when he answers a question with one
    and the correction is A QUESTION instead of an answer, that
    word earns gap-marker evidence. Recurrence adopts it.
  ASKING IS ANSWERING, INVERTED. His question shapes were learned
    by correction as (shape -> which role answers). To ask about a
    missing role, he emits the shape whose answer-role IS that
    role, its content positions filled from the fact he holds. The
    ask-template is extracted from the corrections themselves
    (literal words + role slots), adopted by recurrence.
  UNDERSTANDING BECOMES SENTENCES. When the oracle answers his
    question, he integrates the filler into the retained fact —
    the hedged version SUPERSEDED, kept, worn — answers the
    original question, and can STATE the completed fact as a
    sentence of his own. The loop closes: hear incompletely, know
    what is missing, ask exactly that, absorb the answer, say the
    whole thing.

Controls: unhedged passages flag no gaps and ask nothing; before
adoption he answers 'someone' honestly (it is all he has); the
marker needs recurrence. Complete numbers. stdlib + numpy only.
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
                             teach_roles)
from open_questions import (env_dialogue, respond,        # noqa: E402
                            learn_answer)
from passage_recall import (bank_content_roles, read_fact,  # noqa: E402
                            hear_passage, answer_from_passage,
                            _subj_prefix)

ADOPT_MIN = 3
PRON = ("he", "she", "it", "they", "we")


# ═════════════════════════════════════════════════════════════════
# THE ENVIRONMENT — sometimes it only knows part of the story.
# ═════════════════════════════════════════════════════════════════
def env_hedged(rng, obj_bias=0.5):
    """A fact the informant states incompletely + the full truth."""
    while True:
        toks, truth = env_sentence(rng)
        gap = "object" if rng.random() < obj_bias else "subject"
        if gap == "object" and truth["object"] is None:
            continue
        break
    subj_len = 1 if toks[0] in PRON else 2
    if gap == "subject":
        hedged = ["someone"] + toks[subj_len:]
    else:
        vpos = toks.index(truth["verb"])
        hedged = toks[:vpos + 1] + ["something", "."]
    return hedged, toks, truth, gap


def env_corpus(rng, n=2000):
    """His reading includes the world's incomplete talk — enough of
    it that the hedged patterns clear the gate. Measured twice:
    object-hedged patterns are their OWN patterns (nothing else puts
    a bare word after a transitive verb) and at 18%/1000 and
    25%/1600 they sat under the gate; the world must actually talk
    this way often enough to be learnable."""
    out = []
    for _ in range(n):
        if rng.random() < 0.30:
            hedged, _, _, _ = env_hedged(rng, obj_bias=0.65)
            out.append(hedged)
        else:
            out.append(env_sentence(rng)[0])
    return out


def env_gap_question(truth, gap):
    """The user asks about exactly what the informant left out."""
    if gap == "subject":
        q = (["who", truth["verb"]] +
             (["the", truth["object"]] if truth["object"] else []) +
             ["?"])
        return q, truth["subject"]
    return (_subj_prefix(truth) + [truth["verb"], "the", "what", "?"],
            truth["object"])


# ═════════════════════════════════════════════════════════════════
# ISAAC'S SIDE — markers learned, asks generated, facts completed.
# ═════════════════════════════════════════════════════════════════
def _is_question(toks):
    return bool(toks) and toks[-1] == "?"


def learn_from_ask_correction(mind, g, fact, role, said, correction):
    """He answered with a filler and was corrected with a QUESTION:
    the filler earns gap-marker evidence, and the correction is
    mined into an ASK-TEMPLATE for that role (literal words + role
    slots), adopted by recurrence."""
    if not _is_question(correction):
        return "not-a-question"
    mk = mind.net.get_meta("gap:markers", "counts") or {}
    mk[said] = mk.get(said, 0) + 1
    mind.net.set_meta("gap:markers", "counts", mk)
    # PHRASE-LEVEL slots: 'the dog' and 'he' both fill ONE subject
    # slot, so pronoun and det-noun corrections merge into one
    # template instead of splitting the vote (probe lesson)
    spans = dict(subject=list(fact["subj_phrase"]),
                 verb=[fact["verb"]],
                 object=list(fact["obj_phrase"] or []))
    tpl, i = [], 0
    while i < len(correction):
        placed = False
        for r, span in spans.items():
            if span and correction[i:i + len(span)] == span:
                tpl.append(("SLOT", r))
                i += len(span)
                placed = True
                break
        if not placed:
            tpl.append(("LIT", correction[i]))
            i += 1
    shape = f"{role}:{'t' if fact['obj_phrase'] else 'i'}"
    votes = mind.net.get_meta("gap:ask", shape) or {}
    key = " ".join(f"{a}:{b}" for a, b in tpl)
    votes[key] = votes.get(key, 0) + 1
    mind.net.set_meta("gap:ask", shape, votes)
    top, c = max(votes.items(), key=lambda kv: kv[1])
    if c >= ADOPT_MIN:
        mind.net.set_meta("gap:ask_rule", shape,
                          [tuple(x.split(":", 1)) for x in top.split()])
    wrongs = mind.net.get_meta("gap:wrong", "record") or []
    wrongs.append(dict(said=said, correction=list(correction)))
    mind.net.set_meta("gap:wrong", "record", wrongs)
    return f"marker:{said}+ask-template:{role}"


def markers(mind):
    mk = mind.net.get_meta("gap:markers", "counts") or {}
    return {w for w, c in mk.items() if c >= ADOPT_MIN}


def gaps_of(mind, memory):
    """A gap = a retained fact whose slot holds an adopted marker."""
    out = []
    for i, f in enumerate(memory["facts"]):
        for role, filler in (("subject", f["subj_phrase"][-1]),
                             ("object", f["obj_phrase"][-1]
                              if f["obj_phrase"] else None)):
            if filler in markers(mind):
                out.append((i, role))
    return out


def ask_about(mind, fact, role):
    """Generate the question that fills the gap — his learned shape
    for that role, content from the fact he holds."""
    shape = f"{role}:{'t' if fact['obj_phrase'] else 'i'}"
    rule = mind.net.get_meta("gap:ask_rule", shape)
    if rule is None:
        return None
    spans = dict(subject=list(fact["subj_phrase"]),
                 verb=[fact["verb"]],
                 object=list(fact["obj_phrase"] or []))
    out = []
    for kind, val in rule:
        if kind == "LIT":
            out.append(val)
        else:
            if not spans.get(val):
                return None
            out += spans[val]
    return out


def integrate(mind, memory, i, role, answer_phrase):
    """The oracle answered: fill the gap; the hedged fact is
    SUPERSEDED, kept, worn."""
    old = dict(memory["facts"][i])
    chain = mind.net.get_meta("gap:superseded", "record") or []
    chain.append(old)
    mind.net.set_meta("gap:superseded", "record", chain)
    f = memory["facts"][i]
    if role == "subject":
        f["subj_phrase"] = list(answer_phrase)
    else:
        f["obj_phrase"] = list(answer_phrase)
        f["object"] = answer_phrase[-1]
    return f


def state_fact(fact):
    return (fact["subj_phrase"] + [fact["verb"]] +
            fact["obj_phrase"] + ["."])


# ═════════════════════════════════════════════════════════════════
# THE RUN
# ═════════════════════════════════════════════════════════════════
def main():
    rng = np.random.default_rng(13)
    mind = mod.Isaac()
    print("=" * 64)
    print("LOGICAL GAPS — know it, miss it, ask it, say it whole")
    print("=" * 64)

    corpus = env_corpus(rng)
    g = read_world(mind, corpus)
    rlessons = []
    rngl = np.random.default_rng(17)
    while len(rlessons) < 260:
        if rngl.random() < 0.25:
            hedged, _, truth, gap = env_hedged(rngl)
            t = dict(truth)
            if gap == "subject":
                t["subject"] = "someone"
            else:
                t["object"] = "something"
            toks = hedged
        else:
            toks, t = env_sentence(rngl)
        for role in ("subject", "verb", "object"):
            if t[role] is not None:
                rlessons.append((toks, role, t[role]))
    teach_roles(mind, g, rlessons)
    rng2 = np.random.default_rng(21)
    for _ in range(60):
        st, q, ta = env_dialogue(rng2)
        said, _ = respond(mind, g, st, q)
        if said != ta:
            learn_answer(mind, g, st, q, said, ta)
        bank_content_roles(mind, g, st, q)
    print(f"\n[foundation] reading (incl. the world's hedged talk), "
          f"roles, question shapes: prior rungs")

    # ── learning: answering a gap with its marker gets corrected
    #    with the QUESTION he should have asked ────────────────────
    rng3 = np.random.default_rng(31)
    honest_before = lessons = 0
    for k in range(30):
        hedged, full, truth, gap = env_hedged(rng3)
        memory = hear_passage(mind, g, [hedged])
        q, true_ans = env_gap_question(truth, gap)
        said, _ = answer_from_passage(mind, g, memory, q)
        if said in ("someone", "something"):
            honest_before += (len(markers(mind)) < 2)
            fact = memory["facts"][0]
            learn_from_ask_correction(mind, g, fact, gap, said,
                                      list(q))
            lessons += 1
    print(f"\n[learn] 30 hedged dialogues: {lessons} ask-corrections "
          f"banked ({honest_before} while markers were still "
          f"unadopted — he answered with the marker, all he had)")
    print(f"  markers adopted: {sorted(markers(mind))}")
    for shape in ("subject:t", "subject:i", "object:t"):
        r = mind.net.get_meta("gap:ask_rule", shape)
        print(f"  ask-template [{shape}]: "
              f"{' '.join((v if k == 'LIT' else v.upper()) for k, v in r) if r else 'NOT ADOPTED'}")

    # ── the full loop on unseen passages ──────────────────────────
    rng4 = np.random.default_rng(43)
    flag_ok = ask_ok = ans_ok = say_ok = n = 0
    for _ in range(30):
        n += 1
        hedged, full, truth, gap = env_hedged(rng4)
        fillers = []
        while len(fillers) < 3:      # a fair asker: distractors must
            d, dt = env_sentence(rng4)   # not make the gap-question
            if dt["verb"] == truth["verb"] and (
                    (gap == "subject" and
                     dt["object"] == truth["object"]) or
                    (gap == "object" and
                     dt["subject"] == truth["subject"])):
                continue             # genuinely ambiguous
            fillers.append(d)
        memory = hear_passage(mind, g, fillers + [hedged])
        gs = gaps_of(mind, memory)
        flag_ok += (len(gs) == 1 and gs[0][1] == gap)
        if not gs:
            continue
        i, role = gs[0]
        ask = ask_about(mind, memory["facts"][i], role)
        want_q, _ = env_gap_question(truth, gap)
        # grade the GENERATED question against what fills the gap
        # (dets may differ — the oracle accepts either article)
        norm = [w if w not in ("a",) else "the" for w in (ask or [])]
        ask_ok += (norm == [w if w != "a" else "the" for w in want_q])
        if ask is None:
            continue
        # the oracle answers with the missing phrase
        if gap == "subject":
            sl = 1 if full[0] in PRON else 2
            phrase = full[:sl]
        else:
            vpos = full.index(truth["verb"])
            phrase = full[vpos + 1:-1]
        integrate(mind, memory, i, role, phrase)
        q, true_ans = env_gap_question(truth, gap)
        said, _ = answer_from_passage(mind, g, memory, q)
        ans_ok += (said == true_ans)
        say_ok += (state_fact(memory["facts"][i]) == full)
    print(f"\n[loop] 30 unseen hedged passages (3 distractors each):")
    print(f"  gap flagged (right fact, right role): {flag_ok}/{n}")
    print(f"  question GENERATED correctly: {ask_ok}/{n}")
    print(f"  after the oracle's answer: user's question answered "
          f"{ans_ok}/{n}; fact stated whole {say_ok}/{n}")
    sup = mind.net.get_meta("gap:superseded", "record") or []
    print(f"  hedged versions superseded and kept: {len(sup)}")

    # ── controls ──────────────────────────────────────────────────
    rng5 = np.random.default_rng(77)
    false_flags = asked = 0
    for _ in range(20):
        sents = [env_sentence(rng5)[0] for _ in range(4)]
        memory = hear_passage(mind, g, sents)
        gs = gaps_of(mind, memory)
        false_flags += len(gs)
        for i, role in gs:
            if ask_about(mind, memory["facts"][i], role):
                asked += 1
    print(f"\n[control] 20 complete passages: {false_flags} false "
          f"gaps flagged, {asked} needless questions (want 0/0)")

    print("\n" + "=" * 64)
    print("COMPLETE NUMBERS")
    print(f"  markers learned by correction: {sorted(markers(mind))}")
    print(f"  gaps flagged {flag_ok}/{n}; questions generated "
          f"{ask_ok}/{n}; answered-after-asking {ans_ok}/{n}; "
          f"stated whole {say_ok}/{n}")
    print(f"  false gaps on complete passages: {false_flags} "
          f"(want 0); superseded hedged facts kept: {len(sup)}")
    print("=" * 64)


if __name__ == "__main__":
    main()
