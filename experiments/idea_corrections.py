"""idea_corrections.py — TEACHING WHOLE IDEAS BY CORRECTION (owner,
2026-07-13: "how do we teach him these kind of things. whole ideas.
he can get better with the vocab and stuff as he learns. if he says
something and it keeps getting corrected to the other orientation,
then it should work.")

The correction loop — the owner's residual for language — pointed at
IDEA STRUCTURE instead of strings. Isaac OWNS the idea before he
speaks: the event, who did what to what, his slots, his words. The
correction is compared against HIS OWN idea by CONTENT, and factors
into lessons at three levels:

  ORIENTATION — same words, different order: the permutation is a
    VOTE. Orientation is decided by recurrence (a sliding window of
    recent votes), exactly the owner's sentence: corrections that
    KEEP coming in the other orientation flip him; one noisy echo
    from a sloppy tutor cannot (measured live: mistral echoed the
    wrong order back twice in 30 events). Passes vote too — being
    right confirms. Dethroned orientations are KEPT with their vote
    totals (supersession, losers worn).
  WORD — one of his slot words replaced: told knowledge, usable on
    the next utterance; the lesson history kept.
  DECORATION — extra words in the correction that match no slot
    ('the', 'a'): counted by (arity, the slot they precede). Words
    whose counts recur are adopted INTO his speech — function words
    are learned from correction evidence, never listed (measured
    live: the natural tutor inserts articles; refusing to learn them
    made 0/6 corrections factorable in probe v1 — wrong-kept, this
    version is the cure).

Every wrong utterance is KEPT with its correction (wisdom). The
tutor here is scripted (batteries must be deterministic); the live
road is live_idea_probe.py (mistral 7B over ollama).

Complete numbers throughout. stdlib + numpy only.
"""
import os
import sys
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import isaac as mod                                    # noqa: E402

ACTORS = ["she", "he", "dog", "cat", "boy", "girl"]
TRANS = ["threw", "ate", "took", "held", "grabbed"]
INTRANS = ["sat", "ran", "slept", "smiled"]
OBJS = ["ball", "apple", "stick", "cup", "bone", "nap"]

WINDOW = 8        # recurrence window for orientation votes
DECO_MIN = 3      # a decoration word is adopted past this count


def gen_event(rng):
    """An event is HIS: slot-tagged, language-free. (actor, action[,
    object]) — the idea exists before any words are arranged."""
    if rng.random() < 0.5:
        return (ACTORS[rng.integers(0, 6)], TRANS[rng.integers(0, 5)],
                OBJS[rng.integers(0, 6)])
    return (ACTORS[rng.integers(0, 6)], INTRANS[rng.integers(0, 4)])


# ═════════════════════════════════════════════════════════════════
# ORIENTATION — recurrence-voted, supersession kept
# ═════════════════════════════════════════════════════════════════
def _node(arity):
    return f"speak:orientation:{arity}"


def current_orientation(mind, n, rng=None):
    node = _node(n)
    recent = mind.net.get_meta(node, "recent") or []
    if recent:
        best = Counter(tuple(p) for p in recent).most_common(1)[0][0]
        return tuple(best)
    explore = mind.net.get_meta(node, "explore")
    if explore is None and rng is not None:
        explore = tuple(rng.permutation(n).tolist())  # no belief yet:
        mind.net.set_meta(node, "explore", explore)   # explore, but
    return tuple(explore) if explore else None        # consistently


def vote_orientation(mind, n, perm):
    node = _node(n)
    before = current_orientation(mind, n)
    recent = mind.net.get_meta(node, "recent") or []
    recent = (recent + [list(perm)])[-WINDOW:]
    mind.net.set_meta(node, "recent", recent)
    total = mind.net.get_meta(node, "total") or {}
    key = ",".join(map(str, perm))
    total[key] = total.get(key, 0) + 1
    mind.net.set_meta(node, "total", total)
    after = current_orientation(mind, n)
    if before is not None and after != before:        # dethroned:
        chain = mind.net.get_meta(node, "superseded") or []
        chain.append(dict(orientation=list(before),   # kept, worn
                          votes=total.get(",".join(map(str, before)), 0)))
        mind.net.set_meta(node, "superseded", chain)


# ═════════════════════════════════════════════════════════════════
# SPEAKING — his idea, his learned orientation, his learned words,
# his learned decoration. Nothing hand-listed.
# ═════════════════════════════════════════════════════════════════
def speak(mind, event, rng):
    n = len(event)
    perm = current_orientation(mind, n, rng)
    words = mind.net.get_meta("speak:words", "map") or {}
    deco = mind.net.get_meta("speak:decoration", "counts") or {}
    out = []
    for i in perm:
        d = deco.get(f"{n}:{i}") or {}
        if d:
            w, c = max(d.items(), key=lambda kv: kv[1])
            if c >= DECO_MIN:
                out.append(w)
        out.append(words.get(event[i], event[i]))
    return out


# ═════════════════════════════════════════════════════════════════
# WORD ELECTIONS — rivals with receipts (owner law, 2026-07-15:
# "if he knows the old vs new, and the new keeps failing he is
# supposed to revert to the old.")
# WRONG-KEPT, measured on his live account: speak:words was
# WRITE-ONLY — corrections were aligned BACKWARD through the map, so
# a correction restoring the original word never had a missing slot
# and the only door into the map never opened. The revert signal for
# 'ate' arrived 25 times and was filed as decoration/orientation 25
# times. The repair: the identity and the substitution both stay on
# the ballot; corrections VOTE (using the original word where the
# map says otherwise = an identity vote); every USE banks its
# outcome (a thinking receipt at the smallest scale); the seat faces
# RE-ELECTION in the recent window; reversion is supersession
# running backward — the dethroned word kept wearing its numbers.
# ═════════════════════════════════════════════════════════════════
REVERT_VOTES = 3   # identity votes in the window that dethrone (told)
REVERT_USES = 4    # seat uses in the window before its record counts
INSIST = 3         # seat votes that mean the tutor actively insists


def _word_vote(mind, concept, word):
    votes = mind.net.get_meta("speak:words", "votes") or {}
    votes[concept] = ((votes.get(concept) or []) + [word])[-WINDOW:]
    mind.net.set_meta("speak:words", "votes", votes)


def _reelect(mind, concept):
    """One seat faces its record. REVERT when the tutor keeps using
    the original (told road), or when the seat keeps failing and the
    tutor is not actively insisting on it (outcome road — recent
    telling outranks a stale record, the poisoning-cure precedent)."""
    words = mind.net.get_meta("speak:words", "map") or {}
    s = words.get(concept)
    if s is None:
        return False
    votes = (mind.net.get_meta("speak:words", "votes")
             or {}).get(concept, [])
    vw, vs = votes.count(concept), votes.count(s)
    rec = ((mind.net.get_meta("speak:words", "record") or {})
           .get(concept, {}).get(s, {}))
    uses = rec.get("recent", [])
    told_road = vw >= REVERT_VOTES and vw > vs
    insisting = vs >= INSIST and vs > vw
    outcome_road = (len(uses) >= REVERT_USES
                    and sum(uses) / len(uses) < 0.5 and not insisting)
    if not (told_road or outcome_road):
        return False
    why = (f"told road: the tutor used '{concept}' {vw}x vs "
           f"'{s}' {vs}x in the recent window" if told_road else
           f"outcome road: '{s}' passed {sum(uses)}/{len(uses)} "
           f"recent uses and the tutor is not insisting "
           f"({vs} recent votes for it)")
    del words[concept]
    mind.net.set_meta("speak:words", "map", words)
    chain = mind.net.get_meta("speak:words", "superseded") or []
    chain.append(dict(concept=concept, dethroned=s, why=why,
                      votes=dict(identity=vw, seat=vs),
                      record=dict(uses=len(uses), passed=sum(uses))))
    mind.net.set_meta("speak:words", "superseded", chain)
    return True


def bank_word_outcomes(mind, event, said, passed):
    """A thinking receipt at the smallest scale: the judged outcome
    lands on the candidate he ACTUALLY USED, per mapped concept —
    then the seat faces re-election. Called on every judged event,
    pass and fail alike (a seat holds by winning, not by default)."""
    words = mind.net.get_meta("speak:words", "map") or {}
    record = mind.net.get_meta("speak:words", "record") or {}
    touched = []
    for w in event:
        s = words.get(w)
        if s is None:
            continue
        used = s if s in said else (w if w in said else None)
        if used is None:
            continue
        r = record.setdefault(w, {}).setdefault(
            used, dict(recent=[], used=0, passed=0))
        r["recent"] = (r["recent"] + [1 if passed else 0])[-WINDOW:]
        r["used"] += 1
        r["passed"] += 1 if passed else 0
        touched.append(w)
    if touched:
        mind.net.set_meta("speak:words", "record", record)
        for w in set(touched):
            _reelect(mind, w)


def remine_word_elections(mind, transcripts_dir):
    """Wisdom paying rent: replay the KEPT life through the election
    law. Every judged event banks its outcome; every kept correction
    votes. A seat poisoned before the law existed falls on evidence
    already paid for; nothing in the history is altered."""
    import json
    reverted, n_votes, n_out = [], 0, 0
    files = sorted(f for f in os.listdir(transcripts_dir)
                   if f.endswith(".json"))
    for f in files:
        try:
            tr = json.load(open(os.path.join(transcripts_dir, f),
                                encoding="utf-8"))
        except Exception:
            continue
        if not (isinstance(tr, list) and tr
                and isinstance(tr[0], dict) and "event" in tr[0]):
            continue
        for e in tr:
            v = e.get("verdict")
            if v not in ("pass", "fail"):
                continue
            before = dict(mind.net.get_meta("speak:words", "map") or {})
            bank_word_outcomes(mind, e["event"], e["said"], v == "pass")
            n_out += 1
            if v == "fail" and e.get("say"):
                words = mind.net.get_meta("speak:words", "map") or {}
                voted = []
                for w in e["event"]:
                    s = words.get(w)
                    if s is None:
                        continue
                    if w in e["say"]:
                        _word_vote(mind, w, w)
                        voted.append(w)
                    elif s in e["say"]:
                        _word_vote(mind, w, s)
                        voted.append(w)
                n_votes += len(voted)
                for w in set(voted):
                    _reelect(mind, w)
            after = mind.net.get_meta("speak:words", "map") or {}
            reverted += [w for w in before if w not in after]
    return dict(reverted=sorted(set(reverted)), votes_replayed=n_votes,
                outcomes_replayed=n_out)


# ═════════════════════════════════════════════════════════════════
# THE FRONTIER — he sees what keeps losing (owner, 2026-07-15: "he
# just needs to see that hey, this keeps losing in these conditions,
# group them and find why and what makes these failures different
# than the successes.") The commonality miner — his own engine organ
# — pointed at his own recent life after every session: fat records
# from the judged events, failures differential-mined against
# successes, the named conditions stored on HIS mind
# (commonality:frontier:speech — active + superseded chain, so the
# frontier's own history is kept as he clears it). The window is
# RECENT sessions only: the frontier is the PRESENT — an all-time
# mine would let a cured era dominate forever (the E3 lesson).
# ═════════════════════════════════════════════════════════════════
DECO_WORDS = {"the", "a", "an"}
FRONTIER_SESSIONS = 10     # the recent-life window, in sessions


def _order_kind(event, said):
    """His word order vs the event's, form-tolerant (nap/napkin)."""
    pos = []
    for w in event:
        hit = next((i for i, s in enumerate(said)
                    if s == w or (len(w) >= 3 and s.startswith(w))
                    or (len(s) >= 3 and w.startswith(s))), None)
        if hit is None:
            return "word-missing"
        pos.append(hit)
    return "event-order" if pos == sorted(pos) else "reordered"


def frontier_records(events):
    """Judged drill events -> fat records for the miner (the context
    a failure receipt must carry — can't mine what wasn't kept)."""
    recs, failed = [], []
    for e in events:
        v = e.get("verdict")
        if v not in ("pass", "fail"):
            continue
        ev, said = e["event"], e["said"]
        recs.append(dict(
            arity=str(len(ev)), actor=ev[0], action=ev[1],
            obj=(ev[2] if len(ev) > 2 else "none"),
            deco=("yes" if any(w in DECO_WORDS for w in said)
                  else "no"),
            extra=str(min(3, max(-1, len(said) - len(ev)))),
            order=_order_kind(ev, said)))
        failed.append(v == "fail")
    return recs, failed


def mine_frontier(mind, transcripts_dir, last_n=FRONTIER_SESSIONS,
                  name="speech"):
    """After a session: mine the recent window of his own life. The
    named conditions ARE the frontier — a gap with evidence, and the
    next curriculum. Returns the miner's result (honest refusals
    included); stores at commonality:frontier:<name> in his net."""
    import json
    files = sorted(f for f in os.listdir(transcripts_dir)
                   if f.endswith(".json"))
    events = []
    for f in files[-last_n:]:
        try:
            tr = json.load(open(os.path.join(transcripts_dir, f),
                                encoding="utf-8"))
        except Exception:
            continue
        if (isinstance(tr, list) and tr and isinstance(tr[0], dict)
                and "event" in tr[0]):
            events += tr
    recs, failed = frontier_records(events)
    return mind.mine_commonality(recs, failed, name=f"frontier:{name}")


# ═════════════════════════════════════════════════════════════════
# THE FACTORING — content alignment against his own idea
# ═════════════════════════════════════════════════════════════════
def learn_from_correction(mind, event, said, say):
    """Returns the lesson kinds banked ('orientation'/'word'/
    'decoration', joined) or 'unfactored' (kept raw, teaches nothing
    yet — honesty over guessing). Since 2026-07-15 the correction also
    VOTES on every located slot that carries a taught seat — using the
    original word where the map says otherwise used to be INVISIBLE;
    now it is an identity vote and the seat faces re-election."""
    n = len(event)
    lesson = []
    words = mind.net.get_meta("speak:words", "map") or {}
    back = {v: k for k, v in words.items()}
    norm = [back.get(w, w) for w in say]
    pos = {}                       # slot index -> position in say
    for i, w in enumerate(event):
        ps = [j for j, x in enumerate(norm) if x == w]
        if len(ps) == 1:
            pos[i] = ps[0]
    missing = [i for i in range(n) if i not in pos]
    extra = [j for j in range(len(norm)) if j not in pos.values()]
    deco = mind.net.get_meta("speak:decoration", "counts") or {}
    known_deco = {w for d in deco.values() for w in d}
    if len(missing) == 1:
        # one absent slot: the unmatched non-decoration word is its
        # replacement — a WORD lesson, told, used at once
        cands = [j for j in extra if norm[j] not in known_deco]
        if len(cands) == 1:
            j, i = cands[0], missing[0]
            hist = mind.net.get_meta("speak:words", "lessons") or []
            hist.append(dict(meant=event[i],
                             he_said=words.get(event[i], event[i]),
                             tutor=say[j]))
            words[event[i]] = say[j]
            mind.net.set_meta("speak:words", "map", words)
            mind.net.set_meta("speak:words", "lessons", hist)
            pos[i] = j
            extra.remove(j)
            missing = []
            lesson.append("word")
    if not missing:
        # every slot located: the say's slot order is an ORIENTATION
        # vote; leftover words are DECORATION evidence, counted by
        # the slot they immediately precede
        perm = tuple(sorted(pos, key=lambda i: pos[i]))
        vote_orientation(mind, n, perm)
        lesson.append("orientation")
        for j in sorted(extra):
            nxt = [i for i in pos if pos[i] > j]
            if not nxt:
                continue
            slot = min(nxt, key=lambda i: pos[i])
            key = f"{n}:{slot}"
            d = deco.get(key) or {}
            d[say[j]] = d.get(say[j], 0) + 1
            deco[key] = d
            lesson.append("decoration")
        mind.net.set_meta("speak:decoration", "counts", deco)
    # the correction VOTES on every located slot with a taught seat
    # (the visibility fix): the tutor using the original word is an
    # identity vote; using the seat's word confirms the seat
    words = mind.net.get_meta("speak:words", "map") or {}
    voted = []
    for i, j in pos.items():
        w, actual = event[i], say[j]
        s = words.get(w)
        if s is None:
            continue
        if actual == w:
            _word_vote(mind, w, w)
            voted.append(w)
        elif actual == s:
            _word_vote(mind, w, s)
            voted.append(w)
    for w in set(voted):
        _reelect(mind, w)
    if not lesson:
        lesson = ["unfactored"]
    wrongs = mind.net.get_meta("speak:wrong", "record") or []
    wrongs.append(dict(event=list(event), said=list(said),
                       correction=list(say)))
    mind.net.set_meta("speak:wrong", "record", wrongs)
    return "+".join(sorted(set(lesson)))


# ═════════════════════════════════════════════════════════════════
# THE SCRIPTED TUTOR — deterministic battery
# ═════════════════════════════════════════════════════════════════
def tutor(event, utterance, convention="svo", lexicon=None):
    canon = list(event) if convention == "svo" else list(event)[::-1]
    if lexicon:
        canon = [lexicon.get(w, w) for w in canon]
    if utterance == canon:
        return dict(verdict="pass", say=canon)
    return dict(verdict="fail", say=canon)


def run_block(mind, rng, n_events, convention="svo", lexicon=None,
              log=None):
    corrections = 0
    for _ in range(n_events):
        ev = gen_event(rng)
        said = speak(mind, ev, rng)
        j = tutor(ev, said, convention, lexicon)
        if j["verdict"] == "fail":
            corrections += 1
            kind = learn_from_correction(mind, ev, said, j["say"])
            if log is not None and len(log) < 8:
                log.append(f"    event {ev} | he said "
                           f"'{' '.join(said)}' | FAIL -> "
                           f"'{' '.join(j['say'])}' [{kind}]")
        else:                          # a pass CONFIRMS: it votes
            vote_orientation(mind, len(ev),
                             current_orientation(mind, len(ev)))
    return corrections


# ═════════════════════════════════════════════════════════════════
# THE RUN
# ═════════════════════════════════════════════════════════════════
def main():
    rng = np.random.default_rng(31)
    mind = mod.Isaac()
    print("=" * 64)
    print("WHOLE IDEAS BY CORRECTION — recurrence-voted orientation")
    print("=" * 64)

    log = []
    c1 = run_block(mind, rng, 40, log=log)
    print(f"\n[learn] 40 events, {c1} corrections received:")
    for line in log:
        print(line)
    print(f"  orientation: arity-3 {current_orientation(mind, 3)}, "
          f"arity-2 {current_orientation(mind, 2)}")

    rng2 = np.random.default_rng(77)
    ok = 0
    for _ in range(60):
        ev = gen_event(rng2)
        ok += (speak(mind, ev, rng2) == list(ev))
    print(f"\n[use] unseen events spoken in convention: {ok}/60")

    # the owner's sentence, literally: corrections KEEP coming in the
    # other orientation -> he flips; one echo could not have done it
    c3 = run_block(mind, rng, 30, convention="ovs")
    rng3 = np.random.default_rng(99)
    ok3 = 0
    for _ in range(60):
        ev = gen_event(rng3)
        ok3 += (speak(mind, ev, rng3) == list(ev)[::-1])
    chain3 = mind.net.get_meta(_node(3), "superseded") or []
    print(f"\n[flip] convention reversed: {c3} corrections to "
          f"re-stabilize; unseen in NEW convention: {ok3}/60")
    print(f"  superseded (arity 3): {chain3}")

    # noise: a single wrong-echo correction cannot dethrone
    before = current_orientation(mind, 3)
    ev = ("she", "threw", "ball")
    said = speak(mind, ev, rng)
    learn_from_correction(mind, ev, said, list(said))  # echo = 1 vote
    stable = current_orientation(mind, 3) == before
    print(f"\n[noise] one echoed-wrong correction dethrones nothing: "
          f"{'HELD' if stable else 'FLIPPED (bad)'}")

    # a word lesson through the same loop
    mind4 = mod.Isaac()
    rng4 = np.random.default_rng(55)
    lex = {"threw": "tossed"}
    run_block(mind4, rng4, 120, lexicon=lex)
    ok4 = n4 = 0
    for _ in range(60):
        ev = gen_event(rng4)
        if "threw" not in ev:
            continue
        n4 += 1
        ok4 += (speak(mind4, ev, rng4) == [lex.get(w, w) for w in ev])
    lessons = mind4.net.get_meta("speak:words", "lessons") or []
    print(f"\n[vocab] tutor's dialect says 'tossed': "
          f"{len(lessons)} word lesson(s); unseen 'threw'-events in "
          f"dialect: {ok4}/{n4}")

    # decoration through the same loop: a det-speaking tutor
    mind5 = mod.Isaac()
    rng5 = np.random.default_rng(66)

    def det_tutor(ev, utt, convention="svo", lexicon=None):
        canon = []
        for i, w in enumerate(ev):
            if i != 1:                      # dets before non-verbs
                canon.append("the")
            canon.append(w)
        if utt == canon:
            return dict(verdict="pass", say=canon)
        return dict(verdict="fail", say=canon)

    global tutor
    plain, tutor = tutor, det_tutor
    c5 = run_block(mind5, rng5, 40)
    tutor = plain
    ok5 = 0
    rng6 = np.random.default_rng(88)
    for _ in range(40):
        ev = gen_event(rng6)
        want = []
        for i, w in enumerate(ev):
            if i != 1:
                want.append("the")
            want.append(w)
        ok5 += (speak(mind5, ev, rng6) == want)
    deco = mind5.net.get_meta("speak:decoration", "counts") or {}
    print(f"\n[decoration] article-speaking tutor: {c5} corrections; "
          f"learned decoration {deco}")
    print(f"  unseen events spoken WITH the article: {ok5}/40")

    wrongs = mind.net.get_meta("speak:wrong", "record") or []
    print(f"\n[wisdom] wrongs kept with corrections: {len(wrongs)}")

    print("\n" + "=" * 64)
    print("COMPLETE NUMBERS")
    print(f"  from nothing: {c1} corrections/40 events; unseen {ok}/60")
    print(f"  flip by recurrence: {c3} corrections; unseen (new) "
          f"{ok3}/60; noise echo held: {stable}")
    print(f"  word lesson: {ok4}/{n4} unseen; decoration: {ok5}/40 "
          f"unseen; wrongs kept {len(wrongs)}")
    print("=" * 64)


if __name__ == "__main__":
    main()
