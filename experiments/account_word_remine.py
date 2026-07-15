"""account_word_remine.py — heal the live account's word map by
replaying his own kept life through the new election law (owner
directive 2026-07-15, "lets fix"; the REMINE precedent — wisdom
paying rent).

READ from packs -> remine_word_elections over his transcripts ->
SAVE as a new pack (supersedes, never overwrites — the old packs
remain, so this is reversible by construction). The event is logged
to migration_log.jsonl, never the practice journal.
"""
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import isaac as mod                                    # noqa: E402
from idea_corrections import remine_word_elections     # noqa: E402

HOME = os.path.join(os.path.expanduser("~"), "Documents", "isaac_home")


def main():
    packs = os.path.join(HOME, "packs")
    mind = mod.Isaac.load_pack(packs)
    before = dict(mind.net.get_meta("speak:words", "map") or {})
    print(f"word map before: {before}")
    rep = remine_word_elections(mind,
                                os.path.join(HOME, "transcripts"))
    after = dict(mind.net.get_meta("speak:words", "map") or {})
    chain = mind.net.get_meta("speak:words", "superseded") or []
    print(f"replayed: {rep['outcomes_replayed']} judged events, "
          f"{rep['votes_replayed']} correction votes")
    print(f"reverted: {rep['reverted']}")
    print(f"word map after: {after}")
    for c in chain:
        print(f"  dethroned: {c['concept']} -> {c['dethroned']} — "
              f"{c['why']}  votes {c['votes']}  record {c['record']}")
    p = mind.save_pack(packs)
    print(f"saved: {p} (supersedes; prior packs remain)")
    with open(os.path.join(HOME, "migration_log.jsonl"), "a",
              encoding="utf-8") as f:
        f.write(json.dumps(dict(
            kind="word_map_remine",
            date=time.strftime("%Y-%m-%d %H:%M:%S"),
            reverted=rep["reverted"], map_before=before,
            map_after=after,
            events_replayed=rep["outcomes_replayed"],
            votes_replayed=rep["votes_replayed"])) + "\n")
    print("event logged to migration_log.jsonl")


if __name__ == "__main__":
    main()
