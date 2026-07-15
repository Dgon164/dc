"""account_pack_migration.py — move Isaac's LIVE account onto the pack
road (the plain-file storage built and battery-proven 2026-07-11),
with the pickle kept frozen as the superseded road.

Owner law applied (2026-07-15): everything is a tool in the chain —
the pickle is not deleted, it wears its date; packs supersede it and
can themselves be superseded later.

What this script does (READ + ADD only; never modifies existing files):
  1. load the pickle mind (isaac_home/mind.isaac) — READ ONLY
  2. save it as pack v1 in isaac_home/packs/ (a new directory — an ADD)
  3. load pack v1 back; verify EVERYTHING matches the pickle mind,
     compared in the codec's canonical plain form (numpy-safe)
  4. save pack v2 FROM the pack-loaded mind, reload, verify again
     (proves the new road round-trips on itself, not just once)
  5. spot-check the account's life content by name: orientation votes,
     word map, decoration counts, wrongs kept, journal-visible state
  6. append the migration event to isaac_home/migration_log.jsonl
     (a NEW append-only file — NEVER the practice journal, which the
     runner counts lines of to number sessions)

REFUSES (exits nonzero, changes nothing further) on any mismatch.
"""
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import isaac as mod                                    # noqa: E402
from isaac_store import codec                          # noqa: E402

HOME = os.path.join(os.path.expanduser("~"), "Documents", "isaac_home")
PICKLE = os.path.join(HOME, "mind.isaac")
PACKS = os.path.join(HOME, "packs")
MIGLOG = os.path.join(HOME, "migration_log.jsonl")


def canon(mind):
    """Canonical plain form of a whole mind, via the store's own bridge
    + codec (numpy-safe, deterministic order)."""
    from isaac_store import bridge
    codec.register_module(sys.modules[mod.Isaac.__module__], as_name="isaac")
    recs = bridge.extract(mind)
    return {k: json.dumps(codec.enc(v), sort_keys=True)
            for k, v in recs.items()}


def diff(a, b):
    """Names of records that differ between two canonical forms."""
    keys = set(a) | set(b)
    return sorted(k for k in keys if a.get(k) != b.get(k))


def spot(mind):
    """The account's life content, by name (what sessions 25-94 built)."""
    g = mind.net.get_meta
    wrongs = g("speak:wrong", "record") or []
    return dict(
        orientation3=g("speak:orientation:3", "total"),
        orientation2=g("speak:orientation:2", "total"),
        words=g("speak:words", "map"),
        decoration=g("speak:decoration", "counts"),
        wrongs_kept=len(wrongs),
        nodes=len(set(mind.net.doors) | set(mind.net.meta)),
        observations=len(mind.observed_sequences),
        findings=len(mind.findings),
    )


def main():
    print("=" * 70)
    print("ACCOUNT -> PACK MIGRATION (read + add only; pickle untouched)")
    print("=" * 70)
    if not os.path.exists(PICKLE):
        print(f"REFUSED: {PICKLE} not found"); sys.exit(1)

    # 1. the pickle mind (the account as it lives today)
    m0 = mod.Isaac.load(PICKLE)
    s0 = spot(m0)
    print(f"[1] pickle mind loaded: {s0['nodes']} nodes, "
          f"{s0['observations']} observations, "
          f"{s0['wrongs_kept']} wrongs kept")

    # 2. pack v1
    p1 = m0.save_pack(PACKS)
    print(f"[2] pack v1 written: {p1} "
          f"({os.path.getsize(p1)} bytes)")

    # 3. verify v1 == pickle, record by record
    m1 = mod.Isaac.load_pack(PACKS)
    c0, c1 = canon(m0), canon(m1)
    d1 = diff(c0, c1)
    print(f"[3] pickle vs pack-v1: {len(c0)} records vs {len(c1)}; "
          f"differing: {len(d1)}")
    if d1:
        print("    REFUSED — first differing records:")
        for k in d1[:10]:
            print(f"      {k}")
        sys.exit(1)

    # 4. round-trip stability: v2 from the pack-loaded mind
    p2 = m1.save_pack(PACKS)
    m2 = mod.Isaac.load_pack(PACKS)
    d2 = diff(c1, canon(m2))
    print(f"[4] pack-v1 mind -> pack v2 -> reload: differing: {len(d2)}")
    if d2:
        print("    REFUSED — the new road does not round-trip on itself:")
        for k in d2[:10]:
            print(f"      {k}")
        sys.exit(1)

    # 5. spot-check his life content by name, both roads
    s1 = spot(m1)
    same = all(json.dumps(codec.enc(s0[k]), sort_keys=True) ==
               json.dumps(codec.enc(s1[k]), sort_keys=True) for k in s0)
    print(f"[5] life-content spot check (orientation votes, word map, "
          f"decoration, wrongs, counts): {'MATCH' if same else 'MISMATCH'}")
    for k in s0:
        print(f"      {k}: {s0[k] if not isinstance(s0[k], dict) else str(s0[k])[:64]}")
    if not same:
        print("    REFUSED"); sys.exit(1)

    # 6. the migration event — its own log, NEVER the practice journal
    with open(MIGLOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(
            kind="storage_migration", date=time.strftime("%Y-%m-%d %H:%M:%S"),
            from_road="pickle (mind.isaac, kept frozen — superseded, not deleted)",
            to_road=f"packs ({PACKS}), v1+v2 verified identical",
            records=len(c0), wrongs_kept=s0["wrongs_kept"],
            nodes=s0["nodes"], verified="record-level + round-trip + life-content",
        )) + "\n")
    print(f"[6] migration event appended to {MIGLOG}")
    print("=" * 70)
    print("MIGRATION VERIFIED — the account can now live on the pack road.")
    print("mind.isaac remains on disk, frozen at session 94, superseded.")
    print("=" * 70)


if __name__ == "__main__":
    main()
