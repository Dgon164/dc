"""concept_groups.py — CONCEPT GROUPS, AND HE MAKES THEM (owner,
2026-07-13: "what it needs is concept groups and it needs to make
them.")

The 600-event block measured three failures with one root: behavior
was keyed to the wrong thing. Articles keyed to SLOT POSITION said
"the he"; tense keyed to nothing leaked 'ate -> eats' through the
word channel; 'nap -> napkin' overwrote his concept instead of
competing beside it. The root fix is GROUPS — made by Isaac, from
his own correction corpus, by counting behavior past gates:

  DECO-BEHAVIOR GROUPS — for every slot word: how often does the
    tutor's correction put a decoration word immediately before it?
    Words where decoration is expected >= 2 sightings and observed
    ZERO join the bare group (certified never — negatives first-
    class: he/she). Words where it dominates join the decorated
    group (dog/boy/apple). Group nodes minted in the net with member
    doors and every count kept.
  FORM GROUPS — when a correction's unmatched slot word pairs with
    an unmatched say-word sharing a stem (common prefix >= 3), that
    is a SURFACE FORM of his concept, not a new concept: it joins
    the concept's form group with a count. The concept keeps its
    identity — 'napkin' can win nap's SURFACE by evidence but can
    never overwrite the concept. Suppletion (sat/sitting, prefix 1)
    stays honestly unfactored: the frontier.
  TYPE TRANSFER — the crystallizer's distributional types (minted
    from reading the tutor's own corrected sentences) door onto the
    SAME groups: a word with no correction history inherits its
    type-mates' group. All methods, one concept — 'fox' says
    "the fox" on first sight.

Speaking conditions on the groups: decoration by word (group
fallback), surface form by form-group evidence, 'is' learned as
decoration-before-the-ing-form. Nothing hand-listed anywhere in
Isaac's side; the scripted tutor is the ENVIRONMENT (the world's
English speaker) and owns its own irregulars.

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
from idea_corrections import (gen_event, _node,           # noqa: E402
                              current_orientation, vote_orientation,
                              ACTORS, TRANS, INTRANS, OBJS)

PREFIX_MIN = 3     # stem proposal gate (suppletion stays frontier)
FORM_MIN = 2       # a surface form is adopted past this recurrence
DECO_MIN = 3       # word-level decoration adoption
NEVER_EXP = 2.0    # certified-never needs this much expectation


# ═════════════════════════════════════════════════════════════════
# GROUP MAKING — his corpus is his kept wrongs; he counts, gates,
# and mints. No member is ever listed by hand.
# ═════════════════════════════════════════════════════════════════
def _slot_words(mind):
    """Every word that has ever filled one of his event slots (from
    the kept wrongs) PLUS every banked surface form — a form is his
    word too (probe lesson: rebuilding groups from raw wrongs
    without the forms silently discarded 'is'-before-'sleeping')."""
    out = set()
    for w in (mind.net.get_meta("speak:wrong", "record") or []):
        out |= set(w["event"])
    for d in (mind.net.get_meta("groups:forms", "map") or {}).values():
        out |= set(d["counts"] if isinstance(d, dict) and "counts" in d
                   else d)
    return out


def make_groups(mind):
    wrongs = mind.net.get_meta("speak:wrong", "record") or []
    his = _slot_words(mind)
    # decoration behavior per word, measured over all corrections
    seen, decod, deco_words = Counter(), Counter(), {}
    for rec in wrongs:
        say = rec["correction"]
        if len(say) > 2 * len(rec["event"]) + 2:
            continue                      # commentary hygiene
        for j, w in enumerate(say):
            if w not in his:
                continue
            seen[w] += 1
            if j > 0 and say[j - 1] not in his:
                decod[w] += 1
                deco_words.setdefault(w, Counter())[say[j - 1]] += 1
    base = (sum(decod.values()) / sum(seen.values())) if seen else 0.0
    bare, dressed = [], []
    for w, k in seen.items():
        if decod[w] == 0 and k * base >= NEVER_EXP:
            bare.append(w)                      # certified never
        elif k >= 2 and decod[w] / k > 0.5:
            dressed.append(w)
    for name, members in (("deco_no", bare), ("deco_yes", dressed)):
        for w in members:
            mind.net.add_door(f"group:{name}", w, tag="member")
    mind.net.set_meta("group:deco", "measured", dict(
        base=round(base, 3),
        per_word={w: dict(seen=seen[w], decorated=decod[w],
                          words=dict(deco_words.get(w, {})))
                  for w in seen}))
    mind.net.set_meta("group:deco", "members",
                      dict(deco_no=sorted(bare),
                           deco_yes=sorted(dressed)))
    return dict(deco_no=sorted(bare), deco_yes=sorted(dressed))


def type_transfer(mind, groups):
    """The crystallizer's types (his reading = the tutor's corrected
    sentences, observed and consolidated) door onto the SAME groups:
    an unwitnessed word inherits the group of its type-mates when
    the mates vote unanimously (>= 3 of them)."""
    for rec in (mind.net.get_meta("speak:wrong", "record") or []):
        mind.observe(list(rec["correction"]) + ["."])
    mind.consolidate()
    type_of = {}
    for n in mind.net.all_nodes():
        if str(n).startswith("type:"):
            for w in mind.net.get_predecessors(n, tag="type"):
                type_of[w] = n
    transferred = {}
    for w, t in type_of.items():
        if any(w in groups[g] for g in groups):
            continue
        mates = [m for m, tt in type_of.items() if tt == t and m != w]
        per = (mind.net.get_meta("group:deco", "measured") or
               {}).get("per_word", {})
        own = (per.get(w) or {}).get("decorated", 0)
        for g in ("deco_no", "deco_yes"):
            votes = [m for m in mates if m in groups[g]]
            own_g = own if g == "deco_yes" else 0
            # EVIDENCE FUSION: unanimous mates >= 2, or one unanimous
            # mate PLUS the word's own sighting — two thin roads that
            # agree beat either alone (all methods, one concept)
            if (len(votes) + min(own_g, 1)) >= 2 and votes and all(
                    m not in groups["deco_no" if g == "deco_yes"
                                    else "deco_yes"] for m in mates):
                mind.net.add_door(f"group:{g}", w, tag="member")
                mind.net.set_meta(f"group:{g}", f"transfer:{w}",
                                  dict(road="type", type=str(t),
                                       mates=len(votes)))
                transferred[w] = g
    return transferred


# ═════════════════════════════════════════════════════════════════
# FORM GROUPS — surface forms of one concept, evidence-counted
# ═════════════════════════════════════════════════════════════════
def _prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def bank_form(mind, concept, form):
    """Totals are the measurement (kept forever); the RECENT window
    is the usage — same design as orientation votes (block-3 lesson:
    49 stale 'smiling' votes held the present hostage)."""
    forms = mind.net.get_meta("groups:forms", "map") or {}
    d = forms.get(concept)
    if not isinstance(d, dict) or "counts" not in d:
        d = dict(counts=dict(d) if d else {}, recent=[])
    d["counts"][form] = d["counts"].get(form, 0) + 1
    d["recent"] = (d["recent"] + [form])[-8:]
    forms[concept] = d
    mind.net.set_meta("groups:forms", "map", forms)


def remine_forms(mind):
    """Rebuild ALL form evidence from the kept wrongs through the
    CURRENT gates — the wisdom law paying rent: wrongs are kept
    precisely so an organ can be rebuilt when its gate improves.
    (Built after the poisoned block: junk banked by the old gates
    washes out; nothing in the history is deleted.)"""
    mind.net.set_meta("groups:forms", "map", {})
    for rec in (mind.net.get_meta("speak:wrong", "record") or []):
        ev, say = rec["event"], rec["correction"]
        if len(say) > 2 * len(ev) + 2:
            continue                      # commentary hygiene
        used = set()
        missing = []
        for w in ev:
            ps = [j for j, x in enumerate(say)
                  if x == w and j not in used]
            if ps:
                used.add(ps[0])
                bank_form(mind, w, w)     # identity votes
            else:
                missing.append(w)
        for w in missing:
            best, bj = 0, None
            for j, x in enumerate(say):
                if j in used:
                    continue
                p = _prefix(w, x)
                if p > best:
                    best, bj = p, j
            if best >= PREFIX_MIN:
                bank_form(mind, w, say[bj])
                used.add(bj)
    return mind.net.get_meta("groups:forms", "map")


def surface(mind, concept):
    """The identity form COMPETES, and the RECENT window decides —
    a form speaks only if it beats the concept's own recent direct
    confirmations. Totals remain as kept measurement."""
    d = (mind.net.get_meta("groups:forms", "map") or {}).get(concept)
    if not d:
        return concept
    if isinstance(d, dict) and "counts" in d:
        counts, recent = d["counts"], d["recent"]
    else:
        counts, recent = dict(d), []
    pool = Counter(recent) if recent else Counter(counts)
    own = pool.pop(concept, 0)
    if not pool:
        return concept
    form, c = pool.most_common(1)[0]
    return form if c >= FORM_MIN and c > own else concept


# ═════════════════════════════════════════════════════════════════
# SPEAKING AND FACTORING, GROUP-CONDITIONED
# ═════════════════════════════════════════════════════════════════
def _deco_for(mind, word, concept):
    """Decoration must DOMINATE the word's sightings, not merely
    recur (poisoning lesson: chatty-tutor 'be' hit a raw count of 3
    against ~50 undecorated sightings and entered his speech)."""
    per = (mind.net.get_meta("group:deco", "measured") or {}).get(
        "per_word", {})
    rec = per.get(word) or per.get(concept)
    if rec:
        d = Counter(rec.get("words", {}))
        seen = max(rec.get("seen", 0), 1)
        if d:
            w, c = d.most_common(1)[0]
            if c >= DECO_MIN and c / seen > 0.5:
                return w
        if seen >= DECO_MIN:
            return None        # well-measured word: its rate said no
        # thinly-measured word falls through to its group's habit
    yes = mind.net.get_doors("group:deco_yes", tag="member")
    no = mind.net.get_doors("group:deco_no", tag="member")
    if (word in yes or concept in yes) and \
            not (word in no or concept in no):
        agg = Counter()
        for r in per.values():
            agg.update(r.get("words", {}))
        if agg:
            return agg.most_common(1)[0][0]     # the group's habit
    return None


def speak_g(mind, event, rng):
    perm = current_orientation(mind, len(event), rng)
    out = []
    for i in perm:
        form = surface(mind, event[i])
        d = _deco_for(mind, form, event[i])
        if d:
            out.append(d)
        out.append(form)
    return out


def factor_g(mind, event, said, say):
    """Content alignment through form groups; stem proposals for new
    forms; decoration counted per word; orientation voted. Returns
    lesson kinds or 'unfactored'."""
    n = len(event)
    lesson = []
    if len(say) > 2 * n + 2:      # commentary hygiene: a correction
        wrongs = mind.net.get_meta("speak:wrong", "record") or []
        wrongs.append(dict(event=list(event), said=list(said),
                           correction=list(say)))   # twice the event
        mind.net.set_meta("speak:wrong", "record", wrongs)  # teaches
        return "unfactored"                              # nothing
    forms = mind.net.get_meta("groups:forms", "map") or {}
    back = {}
    for concept, d in forms.items():
        fs = d["counts"] if isinstance(d, dict) and "counts" in d \
            else d
        for f in fs:
            back[f] = concept
    norm = [back.get(w, w) for w in say]
    pos = {}
    for i, w in enumerate(event):
        ps = [j for j, x in enumerate(norm) if x == w]
        if len(ps) == 1:
            pos[i] = ps[0]
            bank_form(mind, w, say[ps[0]])   # a direct use of HIS
            # word is a vote FOR it; a form sighting is a vote for
            # the form — identity competes (the poisoning cure)
    missing = [i for i in range(n) if i not in pos]
    extra = [j for j in range(len(norm)) if j not in pos.values()]
    # stem proposals pair missing slots with unmatched say-words
    for i in list(missing):
        best, bj = 0, None
        for j in extra:
            p = _prefix(event[i], norm[j])
            if p > best:
                best, bj = p, j
        if best >= PREFIX_MIN:
            bank_form(mind, event[i], say[bj])
            pos[i] = bj
            missing.remove(i)
            extra.remove(bj)
            lesson.append("form")
    # a SINGLE leftover pairing is evidence by itself (ate->eats,
    # prefix 2; sat->sitting once 'is' is established decoration):
    # recurrence still gates adoption — a proposal is not a belief
    if len(missing) == 1:
        agg = Counter()
        for rec in ((mind.net.get_meta("group:deco", "measured") or
                     {}).get("per_word", {})).values():
            agg.update(rec.get("words", {}))
        estab = {w for w, c in agg.items() if c >= DECO_MIN}
        cands = [j for j in extra if norm[j] not in estab]
        if len(cands) == 1:
            bank_form(mind, event[missing[0]], say[cands[0]])
            pos[missing[0]] = cands[0]
            extra.remove(cands[0])
            missing = []
            lesson.append("form")
    if not missing:
        perm = tuple(sorted(pos, key=lambda i: pos[i]))
        vote_orientation(mind, n, perm)
        lesson.append("orientation")
        # decoration evidence, keyed by the word it precedes
        per = (mind.net.get_meta("group:deco", "measured") or
               dict(per_word={}, base=0.0))
        for j in sorted(extra):
            nxt = [i for i in pos if pos[i] > j]
            if not nxt:
                continue
            slot = min(nxt, key=lambda i: pos[i])
            w = say[pos[slot]]
            rec = per["per_word"].setdefault(
                w, dict(seen=0, decorated=0, words={}))
            rec["words"][say[j]] = rec["words"].get(say[j], 0) + 1
            rec["decorated"] += 1
            lesson.append("decoration")
        for i in pos:
            w = say[pos[i]]
            rec = per["per_word"].setdefault(
                w, dict(seen=0, decorated=0, words={}))
            rec["seen"] += 1
        mind.net.set_meta("group:deco", "measured", per)
    if not lesson:
        lesson = ["unfactored"]
    wrongs = mind.net.get_meta("speak:wrong", "record") or []
    wrongs.append(dict(event=list(event), said=list(said),
                       correction=list(say)))
    mind.net.set_meta("speak:wrong", "record", wrongs)
    return "+".join(sorted(set(lesson)))


# ═════════════════════════════════════════════════════════════════
# THE SCRIPTED TUTOR — the environment's English: articles for
# nouns, none for pronouns, progressive for intransitives. The
# tutor owns its irregulars (it is the world, not Isaac).
# ═════════════════════════════════════════════════════════════════
PRONOUNS = {"she", "he"}                     # the TUTOR knows this
ING = dict(sat="sitting", ran="running", slept="sleeping",
           smiled="smiling")                 # and this


def english(event):
    out = []
    a = event[0]
    if a not in PRONOUNS:
        out.append("the")
    out.append(a)
    v = event[1]
    if len(event) == 2:
        out += ["is", ING[v]]
    else:
        out.append(v)
        out += ["the", event[2]]
    return out


def script_tutor(event, utterance):
    canon = english(event)
    if utterance == canon:
        return dict(verdict="pass", say=canon)
    return dict(verdict="fail", say=canon)


# ═════════════════════════════════════════════════════════════════
# THE BATTERY
# ═════════════════════════════════════════════════════════════════
def main():
    rng = np.random.default_rng(71)
    mind = mod.Isaac()
    print("=" * 64)
    print("CONCEPT GROUPS — made by him, from his corrections")
    print("=" * 64)

    # ── a tutoring block with the scripted environment ────────────
    stats = Counter()
    for k in range(120):
        ev = gen_event(rng)
        said = speak_g(mind, ev, rng)
        j = script_tutor(ev, said)
        if j["verdict"] == "pass":
            stats["pass"] += 1
            vote_orientation(mind, len(ev),
                             current_orientation(mind, len(ev)))
        else:
            stats[factor_g(mind, ev, said, j["say"])] += 1
        if k in (39, 79):
            groups = make_groups(mind)      # he re-makes his groups
    groups = make_groups(mind)
    print(f"\n[block] 120 events: {dict(stats)}")
    print(f"  groups he made: bare={groups['deco_no']}")
    print(f"                  decorated={groups['deco_yes']}")
    forms = mind.net.get_meta("groups:forms", "map") or {}
    print(f"  form groups: {forms}")

    # ── checks, complete numbers ──────────────────────────────────
    rng2 = np.random.default_rng(88)
    ok = n = supl_fail = other_fail = 0
    for _ in range(60):
        ev = gen_event(rng2)
        got = speak_g(mind, ev, rng2)
        want = english(ev)
        n += 1
        if got == want:
            ok += 1
        elif len(ev) == 2 and _prefix(ev[1], ING[ev[1]]) < PREFIX_MIN:
            supl_fail += 1     # ran/sitting-class: the prefix road's
        else:                  # honest limit, not a learning failure
            other_fail += 1
    print(f"\n[use] unseen events spoken as the environment speaks: "
          f"{ok}/{n}")
    print(f"  misses: {supl_fail} suppletion (ran/sat — the prefix "
          f"road's measured limit), {other_fail} other")

    pron_ok = pron_n = 0
    rng3 = np.random.default_rng(99)
    for _ in range(40):
        ev = gen_event(rng3)
        if ev[0] not in PRONOUNS:
            continue
        pron_n += 1
        pron_ok += (speak_g(mind, ev, rng3)[0] == ev[0])
    print(f"  pronoun events start BARE (no 'the he'): "
          f"{pron_ok}/{pron_n}")

    # ── type transfer: words he has never been corrected on ──────
    new_events = [("fox", "grabbed", "ball"), ("cow", "ate", "apple")]
    for w in ("fox", "cow"):        # he READS about them only — in
        for _ in range(18):         # the same frames the world talks
            ev = gen_event(rng)
            ev = ((w,) + ev[1:] if rng.random() < 0.5 or len(ev) == 2
                  else ev[:2] + (w,))
            mind.observe(english(ev) + ["."])
    transferred = type_transfer(mind, groups)
    first_try = 0
    for ev in new_events:
        got = speak_g(mind, ev, np.random.default_rng(1))
        first_try += (got == english(ev))
    print(f"\n[transfer] never-corrected words grouped by type: "
          f"{transferred}")
    print(f"  first-try correct speech for fox/cow events: "
          f"{first_try}/2")
    # both roads, one group: a word whose reading evidence is too
    # thin gets there by FUSION — its own correction sighting plus
    # one agreeing type-mate, or two sightings alone. Two thin roads
    # that agree beat either alone.
    second_try, needed = 0, {}
    for ev in new_events:
        rr = np.random.default_rng(2)
        for c in range(3):
            if speak_g(mind, ev, rr) == english(ev):
                second_try += 1
                needed[ev[0]] = c
                break
            said = speak_g(mind, ev, rr)
            factor_g(mind, ev, said, script_tutor(ev, said)["say"])
            groups = make_groups(mind)
            type_transfer(mind, groups)
    print(f"  with fusion (own sightings + type-mates): "
          f"{second_try}/2 correct, corrections needed: {needed}")

    # ── concept identity survives a surface takeover ──────────────
    forms = mind.net.get_meta("groups:forms", "map") or {}
    his = set()
    for rec in (mind.net.get_meta("speak:wrong", "record") or []):
        his |= set(rec["event"])
    ident = all(c in his for c in forms)
    print(f"\n[identity] every form group hangs off HIS concept "
          f"word (no overwrite): "
          f"{'yes' if ident else 'NO — ' + str(sorted(set(forms) - his))}")

    print("\n" + "=" * 64)
    print("COMPLETE NUMBERS")
    print(f"  block: {stats['pass']}/120 passes, "
          f"{sum(v for k, v in stats.items() if k != 'pass')} "
          f"corrections, unfactored {stats.get('unfactored', 0)}")
    print(f"  groups: bare {len(groups['deco_no'])}, decorated "
          f"{len(groups['deco_yes'])}, form groups {len(forms)}")
    print(f"  unseen speech {ok}/{n} ({supl_fail} suppletion + "
          f"{other_fail} other misses); pronouns bare "
          f"{pron_ok}/{pron_n}")
    print(f"  type transfer first-try: {first_try}/2")
    print("=" * 64)


if __name__ == "__main__":
    main()
