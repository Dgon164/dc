"""word_reelection_battery.py — WORD-MAP RE-ELECTION (owner law,
2026-07-15: "if he knows the old vs new, and the new keeps failing he
is supposed to revert to the old"). Battery FIRST, per discipline.

The defect under repair (measured on his live account): speak:words
was WRITE-ONLY — a taught substitution ('ate'->'eats', poisoning-era)
could never be contradicted, because corrections were aligned BACKWARD
through the map (the revert signal arrived 25 times, filed as
decoration/orientation 25 times, word lessons 0). The repair:
  RIVALS WITH RECEIPTS — the identity and the substitution both stay
    on the ballot; every USE gets its outcome banked (thinking
    receipts at the smallest scale); corrections VOTE (using the
    original word where the map says otherwise = an identity vote —
    the visibility fix)
  RE-ELECTION by recent window (the decoration/forms election law):
    REVERT when the tutor keeps using the original (told road), or
    when the seat keeps failing and the tutor is not actively
    insisting on it (outcome road)
  REVERSION = SUPERSESSION RUNNING BACKWARD — the dethroned word is
    kept wearing its numbers; re-teaching re-adopts (the chain grows)
  REMINE — his kept life replays through the new law (wisdom paying
    rent): the poisoned seat falls on evidence already paid for.

Checks W1-W8. Run: PYTHONHASHSEED=0 python word_reelection_battery.py
"""
import json
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import isaac as mod                                    # noqa: E402
from idea_corrections import (speak, learn_from_correction,  # noqa: E402
                              bank_word_outcomes,
                              remine_word_elections)

rng = np.random.default_rng(5)
CHECKS = []


def check(name, ok, detail=""):
    CHECKS.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" +
          (f"   {detail}" if detail else ""))


def poison(mind):
    """Teach ate->eats by the legitimate word-lesson road: 'ate' is
    missing from the correction, 'eats' is the one unmatched word."""
    ev = ("cat", "ate", "apple")
    learn_from_correction(mind, ev, ["cat", "ate", "apple"],
                          ["cat", "eats", "apple"])
    return (mind.net.get_meta("speak:words", "map") or {}).get("ate")


def seat(mind, w):
    return (mind.net.get_meta("speak:words", "map") or {}).get(w)


def main():
    print("=" * 70)
    print("WORD RE-ELECTION BATTERY — rivals with receipts")
    print("=" * 70)

    # ── W1: the seat keeps failing -> outcome-road revert ──
    m = mod.Isaac()
    check("W1a the poison takes the seat (word lesson still works)",
          poison(m) == "eats")
    for _ in range(5):
        said = speak(m, ("cat", "ate", "apple"), rng)
        bank_word_outcomes(m, ("cat", "ate", "apple"), said, False)
    check("W1b five failing uses, no tutor insistence -> REVERTED",
          seat(m, "ate") is None)
    said = speak(m, ("cat", "ate", "apple"), rng)
    check("W1c he speaks the original again", "ate" in said
          and "eats" not in said, " ".join(said))

    # ── W2: the tutor keeps using the original -> told-road revert ──
    m = mod.Isaac()
    poison(m)
    for _ in range(3):
        # correction USES 'ate' where the map says 'eats' — previously
        # invisible; now an identity vote
        learn_from_correction(m, ("cat", "ate", "apple"),
                              ["cat", "eats", "apple"],
                              ["the", "cat", "ate", "apple"])
    check("W2 three identity votes from corrections -> REVERTED",
          seat(m, "ate") is None)

    # ── W3: a GOOD substitution holds its seat ──
    m = mod.Isaac()
    ev = ("girl", "threw", "ball")
    learn_from_correction(m, ev, ["girl", "threw", "ball"],
                          ["girl", "tossed", "ball"])
    for _ in range(5):
        said = speak(m, ev, rng)
        bank_word_outcomes(m, ev, said, True)      # it keeps WINNING
    check("W3a winning seat holds under passing receipts",
          seat(m, "threw") == "tossed")
    # one stray identity echo cannot dethrone a winning seat
    learn_from_correction(m, ev, ["girl", "tossed", "ball"],
                          ["girl", "threw", "ball"])
    check("W3b one echo cannot dethrone (noise resistance)",
          seat(m, "threw") == "tossed")

    # ── W4: failing seat BUT the tutor actively insists -> holds ──
    m = mod.Isaac()
    poison(m)
    for _ in range(3):                       # tutor keeps saying eats
        learn_from_correction(m, ("cat", "ate", "apple"),
                              ["cat", "ate", "apple"],
                              ["cat", "eats", "apple"])
    for _ in range(4):
        said = speak(m, ("cat", "ate", "apple"), rng)
        bank_word_outcomes(m, ("cat", "ate", "apple"), said, False)
    check("W4 failing seat with active tutor insistence holds "
          "(recent telling outranks the record)",
          seat(m, "ate") == "eats")

    # ── W5: reversion is SUPERSESSION — kept, and re-adoptable ──
    m = mod.Isaac()
    poison(m)
    for _ in range(3):
        learn_from_correction(m, ("cat", "ate", "apple"),
                              ["cat", "eats", "apple"],
                              ["the", "cat", "ate", "apple"])
    chain = m.net.get_meta("speak:words", "superseded") or []
    ok = (len(chain) == 1 and chain[0]["concept"] == "ate"
          and chain[0]["dethroned"] == "eats" and "why" in chain[0])
    check("W5a dethroned word kept wearing its numbers", ok,
          chain[0]["why"][:58] if chain else "no chain")
    # re-adoption must WIN the window — one echo of re-teaching cannot
    # instantly overturn three recent identity votes (the same
    # noise-resistance law, running in the other direction)
    poison(m)
    one_shot = seat(m, "ate")                # reverts right back
    for _ in range(2):
        poison(m)                            # recurrence re-elects
    check("W5b one re-teach cannot flip; recurrence re-adopts",
          one_shot is None and seat(m, "ate") == "eats",
          f"after 1 re-teach: {one_shot}; after 3: {seat(m, 'ate')}")

    # ── W6: the election state survives the pack road ──
    a = json.dumps([m.net.get_meta("speak:words", k)
                    for k in ("map", "votes", "record", "superseded")],
                   sort_keys=True, default=str)
    with tempfile.TemporaryDirectory() as td:
        m.save_pack(td)
        m2 = mod.Isaac.load_pack(td)
    b = json.dumps([m2.net.get_meta("speak:words", k)
                    for k in ("map", "votes", "record", "superseded")],
                   sort_keys=True, default=str)
    check("W6 votes, records, chain survive packs", a == b)

    # ── W7: REMINE — a poisoned history heals the mind that kept it ──
    m = mod.Isaac()
    poison(m)
    with tempfile.TemporaryDirectory() as td:
        tdir = os.path.join(td, "transcripts")
        os.makedirs(tdir)
        events = []
        for _ in range(12):                  # his shape: says eats,
            events.append(dict(               # fails; tutor says ate
                event=["cat", "ate", "apple"],
                said=["the", "cat", "eats", "apple"],
                verdict="fail",
                say=["the", "cat", "ate", "the", "apple"],
                lesson="decoration+orientation"))
        with open(os.path.join(tdir, "session_00001.json"), "w",
                  encoding="utf-8") as f:
            json.dump(events, f)
        rep = remine_word_elections(m, tdir)
    ok = (seat(m, "ate") is None and "ate" in rep["reverted"])
    check("W7 remine from kept history reverts the poisoned seat", ok,
          f"reverted {rep['reverted']}, votes replayed "
          f"{rep['votes_replayed']}, outcomes {rep['outcomes_replayed']}")

    # ── W8: remine on a clean mind changes nothing ──
    m = mod.Isaac()
    with tempfile.TemporaryDirectory() as td:
        tdir = os.path.join(td, "transcripts")
        os.makedirs(tdir)
        with open(os.path.join(tdir, "session_00001.json"), "w",
                  encoding="utf-8") as f:
            json.dump([dict(event=["boy", "ran"], said=["the", "boy",
                       "ran"], verdict="pass")], f)
        rep = remine_word_elections(m, tdir)
    check("W8 clean mind: remine reverts nothing",
          rep["reverted"] == [])

    n_ok = sum(CHECKS)
    print("=" * 70)
    print(f"WORD RE-ELECTION BATTERY: {n_ok}/{len(CHECKS)}")
    print("=" * 70)
    sys.exit(0 if n_ok == len(CHECKS) else 1)


if __name__ == "__main__":
    main()
