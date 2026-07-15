"""sentence_compression.py — UNDERSTANDING AND RETENTION: MULTI-
SENTENCE COMPRESSION (owner, 2026-07-13: "taking several sentences
and turning it into one.")

RETENTION: statements arrive as separate turns and he holds them.
UNDERSTANDING: what the sentences SHARE is detected by his own role
reading (parts_of_speech rung) — "the dog took a bone" and "the dog
ran" share a subject PHRASE; "the dog sat" and "the cat ran" share
only a determiner, and omitting that would be a string-matcher's
mistake, not an understander's. The share relation is his judgment.
THE FORM IS LEARNED: what to omit, where the join word goes, what
happens to the terminators — all extracted from corrections by
aligning them against his own readings (blocks: SHARED once, then
the first remainder, a JOIN word he has never seen before, the
second remainder, one terminator). Recurrence adopts an arrangement
per share-relation; scrambled corrections adopt nothing.

THE FAITHFULNESS NULL, POINTED AT COMPRESSION: before speaking, he
verifies his one sentence still contains everything both sentences
said (shared content once, both remainders whole, in order). A
compression that loses content is SILENCE, never spoken.

COMPOSITION: three sentences compress by folding the learned pair
rule — compress(compress(s1, s2), s3) — wholes as parts, upward.

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

ADOPT_MIN = 3


# ═════════════════════════════════════════════════════════════════
# THE ENVIRONMENT — speaks statement pairs and grades compressions.
# ═════════════════════════════════════════════════════════════════
def env_pair(rng, shared=None):
    """Two statements + the compressed truth (grader-side)."""
    if shared is None:
        shared = bool(rng.integers(0, 2))
    s1, t1 = env_sentence(rng)
    if shared:
        subj_len = 1 if s1[0] in ("he", "she", "it", "they",
                                  "we") else 2
        while True:
            s2, t2 = env_sentence(rng)
            if t2["verb"] != t1["verb"]:
                break
        sl2 = 1 if s2[0] in ("he", "she", "it", "they", "we") else 2
        s2 = s1[:subj_len] + s2[sl2:]           # same subject phrase
        truth = (s1[:-1] + ["and"] + s2[subj_len:-1] + ["."])
    else:
        while True:
            s2, t2 = env_sentence(rng)
            if s2[:2] != s1[:2]:                # different subjects
                break
        truth = s1[:-1] + ["and"] + s2[:-1] + ["."]
    return s1, s2, truth


# ═════════════════════════════════════════════════════════════════
# ISAAC'S SIDE
# ═════════════════════════════════════════════════════════════════
def subject_phrase(mind, g, toks):
    """His subject reading, as a PHRASE: the tokens up to and
    including the bound subject slot (understanding, not string
    matching)."""
    out = identify(mind, g, toks, "subject")
    if out["answer"] is None:
        return None
    k = toks.index(out["answer"])
    return toks[:k + 1]


def share_relation(mind, g, s1, s2):
    p1, p2 = subject_phrase(mind, g, s1), subject_phrase(mind, g, s2)
    if p1 is None or p2 is None:
        return None, None
    return ("same-subject" if p1 == p2 else "different-subject"), p1


def _strip(toks):
    return toks[:-1] if toks and toks[-1] == "." else list(toks)


def learn_compress(mind, g, s1, s2, corrected):
    """Align the correction against HIS readings; extract the block
    arrangement + the join word; vote; adopt by recurrence."""
    rel, shared = share_relation(mind, g, s1, s2)
    if rel is None:
        return "no-reading"
    a = _strip(s1)[len(shared):] if rel == "same-subject" else _strip(s1)
    b = _strip(s2)[len(shared):] if rel == "same-subject" else _strip(s2)
    head = shared if rel == "same-subject" else []
    blocks, join_words, i = [], [], 0
    parts = ([("S", list(head))] if head else []) + \
        [("A", list(a)), ("B", list(b))]
    remaining = {k: list(v) for k, v in parts}
    order = []
    for w in corrected:
        placed = False
        for k, v in remaining.items():
            if v and v[0] == w:
                v.pop(0)
                if not order or order[-1] != k:
                    order.append(k)
                placed = True
                break
        if not placed:
            if w == "." and all(not v for v in remaining.values()):
                order.append("T")
            else:
                join_words.append(w)
                order.append("J")
    if any(v for v in remaining.values()):
        wrongs = mind.net.get_meta("compress:wrong", "record") or []
        wrongs.append(dict(s1=s1, s2=s2, corrected=list(corrected),
                           why="content unaccounted"))
        mind.net.set_meta("compress:wrong", "record", wrongs)
        return "unfactored"
    votes = mind.net.get_meta("compress:votes", rel) or {}
    key = " ".join(order)
    votes[key] = votes.get(key, 0) + 1
    mind.net.set_meta("compress:votes", rel, votes)
    for j in join_words:
        jw = mind.net.get_meta("compress:join", "words") or {}
        jw[j] = jw.get(j, 0) + 1
        mind.net.set_meta("compress:join", "words", jw)
    top, c = max(votes.items(), key=lambda kv: kv[1])
    if c >= ADOPT_MIN:
        mind.net.set_meta("compress:rule", rel,
                          dict(blocks=top.split(), votes=dict(votes)))
    return f"vote:{rel}:{key}"


def compress(mind, g, sents):
    """Fold the learned pair rule across retained sentences. He
    RETAINS THE STRUCTURE he builds (head + body) — he never
    re-parses his own utterance (probe lesson: a merged sentence
    matches no simple pattern, and it shouldn't need to). The
    faithfulness null runs at every fold; silence otherwise."""
    if len(sents) < 2:
        return None, "need at least two retained statements"
    jw = mind.net.get_meta("compress:join", "words") or {}
    if not jw:
        return None, "no join word learned"
    join = max(jw.items(), key=lambda kv: kv[1])[0]
    head = subject_phrase(mind, g, sents[0])
    if head is None:
        return None, "cannot read a statement (no pattern)"
    body = _strip(sents[0])[len(head):]
    for nxt in sents[1:]:
        p = subject_phrase(mind, g, nxt)
        if p is None:
            return None, "cannot read a statement (no pattern)"
        rel = ("same-subject" if head and p == head
               else "different-subject")
        rule = mind.net.get_meta("compress:rule", rel)
        if rule is None:
            return None, f"no learned form for {rel}"
        if rel == "same-subject":
            hd, a, b = head, body, _strip(nxt)[len(head):]
        else:
            hd, a, b = [], (head or []) + body, _strip(nxt)
        out = []
        for blk in rule["blocks"]:
            if blk == "S":
                out += hd
            elif blk == "A":
                out += a
            elif blk == "B":
                out += b
            elif blk == "J":
                out.append(join)
            elif blk == "T":
                out.append(".")
        # the faithfulness null: nothing lost, shared said once
        want = Counter(hd) + Counter(a) + Counter(b)
        got = Counter(t for t in out if t != ".") - Counter(
            {join: sum(1 for t in out if t == join)
             - a.count(join) - b.count(join)})
        if want != got:
            return None, "faithfulness check failed — silence"
        head = hd if hd else []
        body = out[len(hd):-1] if hd else out[:-1]
    return (head or []) + body + ["."], None


# ═════════════════════════════════════════════════════════════════
# THE RUN
# ═════════════════════════════════════════════════════════════════
def main():
    rng = np.random.default_rng(13)
    mind = mod.Isaac()
    print("=" * 64)
    print("MULTI-SENTENCE COMPRESSION — retained, understood, learned")
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
    print(f"\n[foundation] roles bound; compression untaught: "
          f"{compress(mind, g, [['the', 'dog', 'sat', '.'], ['the', 'dog', 'ran', '.']])}")

    # ── the correction loop ───────────────────────────────────────
    rng2 = np.random.default_rng(21)
    log, corrected_n, spoke_right = [], 0, 0
    for turn in range(40):
        s1, s2, truth = env_pair(rng2)
        said, why = compress(mind, g, [s1, s2])
        if said == truth:
            spoke_right += 1
            continue
        corrected_n += 1
        lesson = learn_compress(mind, g, s1, s2, truth)
        if len(log) < 5:
            log.append(f"    '{' '.join(s1)}' + '{' '.join(s2)}'")
            log.append(f"      he: "
                       f"{('silence (' + str(why) + ')') if said is None else ' '.join(said)}"
                       f" | corrected: '{' '.join(truth)}' [{lesson}]")
    print(f"\n[learn] 40 pair turns: {spoke_right} already right, "
          f"{corrected_n} corrected:")
    for line in log:
        print(line)
    jw = mind.net.get_meta("compress:join", "words") or {}
    print(f"  join word learned: {jw}")
    for rel in ("same-subject", "different-subject"):
        r = mind.net.get_meta("compress:rule", rel)
        print(f"  {rel}: blocks {r['blocks'] if r else 'NOT ADOPTED'}")

    # ── unseen pairs, no corrections ──────────────────────────────
    rng3 = np.random.default_rng(43)
    ok = n = 0
    for _ in range(60):
        s1, s2, truth = env_pair(rng3)
        said, _ = compress(mind, g, [s1, s2])
        n += 1
        ok += (said == truth)
    print(f"\n[use] unseen pairs compressed exactly: {ok}/{n} "
          f"({100.0 * ok / n:.1f}%)")

    # ── three sentences: the learned rule FOLDS ───────────────────
    ok3 = n3 = 0
    for _ in range(20):
        s1, s2, t12 = env_pair(rng3, shared=True)
        while True:
            s3, tr3 = env_sentence(rng3)
            sl = 1 if s3[0] in ("he", "she", "it", "they", "we") else 2
            s3 = s1[:2 if s1[0] in ("the", "a") else 1] + s3[sl:]
            if s3 != s1 and s3 != s2:
                break
        subj_len = 2 if s1[0] in ("the", "a") else 1
        truth = t12[:-1] + ["and"] + s3[subj_len:-1] + ["."]
        said, _ = compress(mind, g, [s1, s2, s3])
        n3 += 1
        ok3 += (said == truth)
    print(f"  THREE retained sentences, rule folded pairwise: "
          f"{ok3}/{n3}")

    # ── controls ──────────────────────────────────────────────────
    mind_c = mod.Isaac()
    g_c = read_world(mind_c, corpus)
    teach_roles(mind_c, g_c, rlessons)
    rng4 = np.random.default_rng(77)
    pool = [env_pair(rng4) for _ in range(40)]
    tr = [t for _, _, t in pool]
    perm = np.random.default_rng(5).permutation(len(tr))
    unf = 0
    for (s1, s2, _), i in zip(pool, perm):
        if learn_compress(mind_c, g_c, s1, s2, tr[i]) == "unfactored":
            unf += 1
    rules = sum(1 for rel in ("same-subject", "different-subject")
                if mind_c.net.get_meta("compress:rule", rel))
    said_c, why_c = compress(mind_c, g_c,
                             [["the", "dog", "sat", "."],
                              ["the", "dog", "ran", "."]])
    print(f"\n[control] scrambled corrections: {unf}/40 unfactored "
          f"(content unaccounted), rules adopted: {rules}")
    print(f"  scrambled-taught mind compresses: "
          f"{'SILENCE (' + str(why_c) + ')' if said_c is None else 'SPOKE (bad)'}")

    print("\n" + "=" * 64)
    print("COMPLETE NUMBERS")
    print(f"  learning: {corrected_n} corrections in 40 turns "
          f"({spoke_right} right during learning)")
    print(f"  unseen pairs: {ok}/{n}; three-sentence fold: {ok3}/{n3}")
    print(f"  scrambled control: {unf}/40 unfactored, "
          f"{rules} rules adopted (want 0)")
    print("=" * 64)


if __name__ == "__main__":
    main()
