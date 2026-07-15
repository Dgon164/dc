"""idea_practice_run.py — LET HIM COOK (owner, 2026-07-13: "just let
him do some tutoring, repeat ad nauseum.")

Isaac's account continues its life on the IDEA road: block after
block of live tutoring with the local 7B. His world supplies the
event; he speaks from the idea (learned orientation + learned words
+ learned decoration); the tutor corrects; the factoring banks what
each correction carries. Everything appends to HIS account —
transcripts, journal, mind — nothing is cleaned or reset.

usage: python idea_practice_run.py [sessions] [events_per_session]
       (defaults 20 x 30)

Per session the journal gets: passes, corrections, factorable,
lesson kinds, broken-format count. The frontier (unfactored
corrections — the tutor's rephrasings the factoring honestly
refuses to guess at) accumulates in the mind as speak:wrong records:
that pile IS the next lesson plan.
"""
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import isaac as mod                                      # noqa: E402
from isaac_store.sparring import OllamaPartner            # noqa: E402
from idea_corrections import (gen_event, speak,           # noqa: E402
                              learn_from_correction,
                              current_orientation, vote_orientation,
                              bank_word_outcomes, mine_frontier)
from live_idea_probe import SYSTEM, parse_judgment, toks  # noqa: E402

HOME = os.path.join(os.path.expanduser("~"), "Documents", "isaac_home")
MODEL = "mistral:latest"


def sessions_so_far(journal):
    if not os.path.exists(journal):
        return 0
    return sum(1 for _ in open(journal, encoding="utf-8"))


def one_session(mind, rng, n_events, speak_fn=None, factor_fn=None):
    speak_fn = speak_fn or speak
    factor_fn = factor_fn or learn_from_correction
    partner = OllamaPartner(MODEL, max_calls=n_events + 5,
                            temperature=0.2)
    tr, stats = [], dict(passes=0, corrections=0, factorable=0,
                         broken=0, lessons={})
    for _ in range(n_events):
        ev = gen_event(rng)
        said = speak_fn(mind, ev, rng)
        user = f"FACT: {' '.join(ev)}\nISAAC SAID: {' '.join(said)}"
        try:
            reply = partner.say([dict(role="system", content=SYSTEM),
                                 dict(role="user", content=user)])
        except Exception as e:
            stats["broken"] += 1
            tr.append(dict(event=list(ev), said=said,
                           verdict="error", why=type(e).__name__))
            continue
        j = parse_judgment(reply)
        if j is None:
            stats["broken"] += 1
            tr.append(dict(event=list(ev), said=said,
                           verdict="unparseable", raw=(reply or "")[:80]))
            continue
        if j["verdict"] == "pass":
            stats["passes"] += 1
            vote_orientation(mind, len(ev),
                             current_orientation(mind, len(ev)))
            bank_word_outcomes(mind, ev, said, True)   # the receipt:
            tr.append(dict(event=list(ev), said=said, verdict="pass"))
        else:                                          # every judged
            stats["corrections"] += 1                  # use lands on
            say = toks(j["say"])                       # the candidate
            bank_word_outcomes(mind, ev, said, False)  # he leaned on
            kind = factor_fn(mind, ev, said, say)
            if kind != "unfactored":
                stats["factorable"] += 1
            stats["lessons"][kind] = stats["lessons"].get(kind, 0) + 1
            tr.append(dict(event=list(ev), said=said, verdict="fail",
                           say=say, lesson=kind))
    return tr, stats


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    groups_mode = "--groups" in sys.argv
    n_sessions = int(args[0]) if len(args) > 0 else 20
    n_events = int(args[1]) if len(args) > 1 else 30
    mind_path = os.path.join(HOME, "mind.isaac")   # superseded road: kept,
    packs = os.path.join(HOME, "packs")            # frozen; packs are live
    journal = os.path.join(HOME, "practice_journal.jsonl")
    tdir = os.path.join(HOME, "transcripts")
    os.makedirs(tdir, exist_ok=True)
    # storage 2026-07-15: the account lives on the pack road (versioned,
    # supersedes-never-overwrites, partial-loadable); the pickle stays on
    # disk wearing its date. Falls back only if no pack exists yet.
    try:
        mind = mod.Isaac.load_pack(packs)
        print(f"  [account] loaded from packs ({packs})")
    except (FileNotFoundError, OSError):
        mind = (mod.Isaac.load(mind_path) if os.path.exists(mind_path)
                else mod.Isaac())
        print("  [account] no packs yet — loaded pickle; packs from here on")
    if not OllamaPartner(MODEL).reachable():
        print(f"REFUSED: {MODEL} not reachable — start ollama first.")
        return
    speak_fn = factor_fn = None
    if groups_mode:
        from concept_groups import (speak_g, factor_g, make_groups,
                                    remine_forms)
        speak_fn, factor_fn = speak_g, factor_g
        g = make_groups(mind)   # mine his existing kept wrongs first
        forms = remine_forms(mind)   # rebuild form evidence through
        print(f"  [groups from his life so far] "  # the current gates
              f"bare={g['deco_no']}")
        print(f"                            dressed={g['deco_yes']}")
        contested = {k: v["counts"] for k, v in forms.items()
                     if len(v["counts"]) > 1}
        print(f"  [forms re-mined from his wrongs] {contested}")
    rng = np.random.default_rng(int(time.time()) % 100000)
    print(f"IDEA PRACTICE — {n_sessions} sessions x {n_events} events, "
          f"{MODEL}, account {HOME}"
          + (" [CONCEPT GROUPS]" if groups_mode else ""))
    curve = []
    for s in range(n_sessions):
        tr, st = one_session(mind, rng, n_events,
                             speak_fn=speak_fn, factor_fn=factor_fn)
        if groups_mode:
            from concept_groups import make_groups
            make_groups(mind)   # he re-makes his groups each session
        n0 = sessions_so_far(journal)
        with open(os.path.join(tdir, f"session_{n0:05d}.json"), "w",
                  encoding="utf-8") as f:
            json.dump(tr, f, indent=1)
        entry = dict(session=n0, kind="idea_drill", events=n_events,
                     **st, ts=time.time())
        with open(journal, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        # the frontier habit (owner, 2026-07-15): after every session
        # he mines his own recent life — what keeps losing, named,
        # stored in HIS mind (supersedes as he clears it)
        fr = mine_frontier(mind, tdir)
        if fr["named"]:
            print("  [frontier] keeps losing when: " + ", ".join(
                f"{k}={v} (+{e:.2f})" for k, v, e in fr["named"][:5]))
        else:
            print("  [frontier] nothing named past the null "
                  "this window")
        # the tools query call in the main loop (owner, 2026-07-15):
        # he asks his own tools node what applies to his named
        # frontier; nothing applicable -> he just skips to next
        want = ("keeps losing when " + " ".join(
            f"{k} {v}" for k, v, _ in fr["named"])
            if fr["named"] else "speech practice drill session")
        tq = mind.query_tools(want)
        if tq["matches"]:
            print("  [tools] applicable: " + "; ".join(
                f"{t['name']} — {t['does'][:48]}"
                for t in tq["matches"][:3]))
        else:
            print(f"  [tools] {tq['why']}")
        mind.save_pack(packs)
        curve.append(st)
        print(f"  session {n0}: pass {st['passes']}/{n_events}  "
              f"corrections {st['corrections']} "
              f"(factorable {st['factorable']})  broken {st['broken']}"
              f"  lessons {st['lessons']}")

    print("\n" + "=" * 64)
    print("COMPLETE NUMBERS — the block")
    total_ev = n_sessions * n_events
    print(f"  pass curve: " + " ".join(
        f"{st['passes']}/{n_events}" for st in curve))
    print(f"  totals: {sum(st['passes'] for st in curve)}/{total_ev} "
          f"passes, {sum(st['corrections'] for st in curve)} "
          f"corrections ({sum(st['factorable'] for st in curve)} "
          f"factorable), {sum(st['broken'] for st in curve)} broken")
    print(f"  orientation: arity-3 {current_orientation(mind, 3)}, "
          f"arity-2 {current_orientation(mind, 2)}")
    print(f"  words map: {mind.net.get_meta('speak:words', 'map')}")
    print(f"  decoration: "
          f"{mind.net.get_meta('speak:decoration', 'counts')}")
    wrongs = mind.net.get_meta("speak:wrong", "record") or []
    unf = [w for w in wrongs]
    print(f"  wrongs kept (all time): {len(wrongs)}")
    print("  frontier samples (last 5 corrections kept raw):")
    for w in wrongs[-5:]:
        print(f"    {w['event']} | said {w['said']} | "
              f"corrected {w['correction']}")
    print("=" * 64)


if __name__ == "__main__":
    main()
