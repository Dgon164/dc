"""open_questions.py — OPEN QUESTIONS, ANSWERED BY CORRECTION
(owner, 2026-07-13: "now we need to start testing open ended
questions and correcting. just a say something, and then he responds
and then a correction.")

The loop, exactly as ordered: the environment SAYS something, asks
an open question about it, Isaac RESPONDS, a CORRECTION follows.
Nothing else teaches him.

What makes this honest rather than mimicry: Isaac already OWNS the
machinery that answers questions — his role bindings identify the
subject/verb/object of any sentence matching his patterns (built in
parts_of_speech.py, all self-made + corrected). What he does NOT
know is what question words MEAN. That is what corrections teach:

  hear:  "the dog took a bone ."  then  "who took a bone ?"
  he says: "..." (no binding yet — silence, never a guess)
  correction: "dog"
  he grounds the correction against HIS OWN reading: "dog" IS what
  his organs call the subject of the stored statement -> this
  question-shape votes for SUBJECT. Recurrence adopts the binding
  (counts kept; ambiguous matches — a word filling two roles —
  teach nothing, honestly).

After that, "who ... ?" is ANSWERED from his own identification on
sentences and questions never seen. A question-shape never corrected
stays silence; a question about a role the statement lacks ("the cat
took what ?" after "the cat ran .") REFUSES from his own grammar —
the statement has no object slot. Scrambled corrections must bind
nothing.

Complete numbers throughout. stdlib + numpy only.
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

ADOPT_MIN = 3      # a question-shape binds past this recurrence
ADOPT_SHARE = 0.6  # and only if the role dominates its votes


# ═════════════════════════════════════════════════════════════════
# THE ENVIRONMENT — says a statement, asks an open question, grades.
# Question construction uses ITS knowledge; Isaac sees tokens only.
# ═════════════════════════════════════════════════════════════════
def env_dialogue(rng, mismatch=False):
    toks, truth = env_sentence(rng)
    subj_len = 1 if toks[0] in ("he", "she", "it", "they", "we") else 2
    has_obj = truth["object"] is not None
    if mismatch:
        # an object question after an intransitive statement — the
        # right answer is a REFUSAL, and it must come from HIS OWN
        # grammar (the statement has no object slot)
        while has_obj:
            toks, truth = env_sentence(rng)
            has_obj = truth["object"] is not None
        q = toks[:-1] + ["what", "?"]
        return toks, q, None
    kind = rng.integers(0, 3 if has_obj else 2)
    if kind == 0:                    # who took a bone ? -> subject
        q = ["who"] + toks[subj_len:-1] + ["?"]
        return toks, q, truth["subject"]
    if kind == 1:                    # the dog did what ? -> verb
        q = toks[:subj_len] + ["did", "what", "?"]
        return toks, q, truth["verb"]
    q = toks[:-2] + ["what", "?"]    # the dog took what ? -> object
    return toks, q, truth["object"]


# ═════════════════════════════════════════════════════════════════
# ISAAC'S SIDE — respond from bindings; learn from corrections by
# grounding them against his own role reading of the statement.
# ═════════════════════════════════════════════════════════════════
def _qkey(g, question):
    return " ".join(str(g["type_of"].get(w, w)) for w in question)


def respond(mind, g, statement, question):
    """His answer, or '...' — silence is honest, guessing is not."""
    rec = mind.net.get_meta("qa:binding", _qkey(g, question))
    if rec is None or rec.get("role") is None:
        return "...", "no-binding"
    out = identify(mind, g, statement, rec["role"])
    if out["answer"] is None:
        return "...", f"refusal: {out['why']}"
    return out["answer"], rec["role"]


def learn_answer(mind, g, statement, question, said, corrected):
    """Ground the correction in HIS reading: which role of the
    statement is the corrected word? Exactly one match = one vote
    for this question-shape; zero or two = teaches nothing."""
    key = _qkey(g, question)
    matches = [r for r in ("subject", "verb", "object")
               if identify(mind, g, statement, r)["answer"] == corrected]
    votes = mind.net.get_meta("qa:votes", key) or {}
    lesson = "ambiguous-or-unmatched"
    if len(matches) == 1:
        votes[matches[0]] = votes.get(matches[0], 0) + 1
        mind.net.set_meta("qa:votes", key, votes)
        lesson = f"vote:{matches[0]}"
        total = sum(votes.values())
        top, c = max(votes.items(), key=lambda kv: kv[1])
        if c >= ADOPT_MIN and c / total > ADOPT_SHARE:
            old = mind.net.get_meta("qa:binding", key)
            if old is not None and old.get("role") != top:
                chain = mind.net.get_meta("qa:superseded", key) or []
                chain.append(old)
                mind.net.set_meta("qa:superseded", key, chain)
            mind.net.set_meta("qa:binding", key,
                              dict(role=top, votes=dict(votes)))
    wrongs = mind.net.get_meta("qa:wrong", "record") or []
    wrongs.append(dict(statement=list(statement),
                       question=list(question), said=said,
                       correction=corrected))
    mind.net.set_meta("qa:wrong", "record", wrongs)
    return lesson


# ═════════════════════════════════════════════════════════════════
# THE RUN
# ═════════════════════════════════════════════════════════════════
def main():
    rng = np.random.default_rng(13)
    mind = mod.Isaac()
    print("=" * 64)
    print("OPEN QUESTIONS — say something, he responds, correct him")
    print("=" * 64)

    # his foundation: reading + role identification (prior rungs)
    corpus = [env_sentence(rng)[0] for _ in range(900)]
    g = read_world(mind, corpus)
    rlessons = []
    while len(rlessons) < 90:
        toks, truth = env_sentence(rng)
        for role in ("subject", "verb", "object"):
            if truth[role] is not None:
                rlessons.append((toks, role, truth[role]))
    teach_roles(mind, g, rlessons)
    print(f"\n[foundation] {len(g['units'])} units, "
          f"{len(g['admitted'])} patterns, roles bound (prior rung)")

    # ── the correction loop, live ─────────────────────────────────
    rng2 = np.random.default_rng(21)
    log, silence_then, answered = [], 0, 0
    for turn in range(60):
        st, q, truth_ans = env_dialogue(rng2)
        said, how = respond(mind, g, st, q)
        if said == "...":
            silence_then += 1
        elif said == truth_ans:
            answered += 1
            rec = mind.net.get_meta("qa:binding", _qkey(g, q))
            votes = rec["votes"]                # a pass confirms
            votes[rec["role"]] = votes.get(rec["role"], 0) + 1
            mind.net.set_meta("qa:binding", _qkey(g, q),
                              dict(role=rec["role"], votes=votes))
            continue
        lesson = learn_answer(mind, g, st, q, said, truth_ans)
        if len(log) < 6:
            log.append(f"    heard '{' '.join(st)}' + "
                       f"'{' '.join(q)}' | said '{said}' | "
                       f"corrected '{truth_ans}' [{lesson}]")
    print(f"\n[learn] 60 dialogue turns: {answered} answered right, "
          f"{silence_then} honest silences, rest corrected:")
    for line in log:
        print(line)
    bindings = {k: v["role"] for k, v in
                (mind.net.meta.get("qa:binding") or {}).items()}
    print(f"  question-shapes bound: {len(bindings)} "
          f"-> roles {sorted(set(bindings.values()))}")

    # ── unseen dialogues, no corrections ──────────────────────────
    rng3 = np.random.default_rng(43)
    ok = n = 0
    for _ in range(60):
        st, q, truth_ans = env_dialogue(rng3)
        said, _ = respond(mind, g, st, q)
        n += 1
        ok += (said == truth_ans)
    ref_ok = ref_n = 0
    roads = Counter()
    for _ in range(20):
        st, q, _ = env_dialogue(rng3, mismatch=True)
        said, how = respond(mind, g, st, q)
        ref_n += 1
        ref_ok += (said == "...")     # any honest non-answer; the
        roads[how.split(":")[0]] += 1  # road taken is reported
    print(f"\n[use] unseen dialogues answered: {ok}/{n} "
          f"({100.0 * ok / n:.1f}%)")
    print(f"  object-question after intransitive statement, honest "
          f"non-answers: {ref_ok}/{ref_n} (roads: {dict(roads)})")
    st = ["the", "dog", "ran", "."]
    said, how = respond(mind, g, st,
                        ["where", "did", "the", "dog", "ran", "?"])
    print(f"  never-corrected question shape stays silent: "
          f"{'yes' if said == '...' else 'NO'} ({how})")

    # ── control: scrambled corrections bind nothing ───────────────
    mind_c = mod.Isaac()
    g_c = read_world(mind_c, corpus)
    teach_roles(mind_c, g_c, rlessons)
    rng4 = np.random.default_rng(77)
    pool = []
    for _ in range(60):
        pool.append(env_dialogue(rng4))
    answers = [a for _, _, a in pool]
    perm = np.random.default_rng(5).permutation(len(answers))
    for (st, q, _), i in zip(pool, perm):
        learn_answer(mind_c, g_c, st, q, "...", answers[i])
    b_scr = {k: v for k, v in
             (mind_c.net.meta.get("qa:binding") or {}).items()}
    print(f"\n[control] scrambled corrections: {len(b_scr)} "
          f"bindings (want 0)")

    print("\n" + "=" * 64)
    print("COMPLETE NUMBERS")
    print(f"  learning block: {answered} right + {silence_then} "
          f"silences + {60 - answered - silence_then} corrected of 60")
    print(f"  unseen: {ok}/{n} answered; refusals {ref_ok}/{ref_n}; "
          f"untaught shape silent: {'yes' if said == '...' else 'NO'}")
    print(f"  scrambled-correction bindings: {len(b_scr)} (want 0)")
    print("=" * 64)


if __name__ == "__main__":
    main()
