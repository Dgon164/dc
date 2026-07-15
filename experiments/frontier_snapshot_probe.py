"""frontier_snapshot_probe.py — READ-ONLY investigation of Isaac's
live account (owner ask, 2026-07-15: "a snapshot of consistent points
of failure... do we know specifically how he is using which doors?").

Runs HIS OWN commonality miner over HIS OWN life transcripts: every
drill event becomes a fat record (arity, actor, action, object,
decoration used, word-order, extra words, era), failed = the tutor's
verdict. The miner names what the failures share that the passes lack
— the frontier pile self-assembling from receipts, exactly as the
07-14 record predicted (§8).

Also: the door-usage inventory — what his mind can currently SAY
about what he uses (observation counts, votes, rates) and what it
cannot (no per-decision usage receipts — the gap this probe motivates).

READ-ONLY on isaac_home. Nothing written anywhere.
"""
import json
import os
import sys
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import isaac as mod                                    # noqa: E402

HOME = os.path.join(os.path.expanduser("~"), "Documents", "isaac_home")
TDIR = os.path.join(HOME, "transcripts")
DECO = {"the", "a", "an"}


def order_kind(event, said):
    """His word order vs the event's: the orientation he actually used."""
    pos = []
    for w in event:
        # the form he used may differ (nap->napkin); prefix-match
        hit = next((i for i, s in enumerate(said)
                    if s == w or (len(w) >= 3 and s.startswith(w))
                    or (len(s) >= 3 and w.startswith(s))), None)
        if hit is None:
            return "word-missing"
        pos.append(hit)
    return "event-order" if pos == sorted(pos) else "reordered"


def main():
    records, failed = [], []
    n_sessions, n_events, n_broken = 0, 0, 0
    tutor_echo_fails = 0
    sessions = sorted(f for f in os.listdir(TDIR) if f.endswith(".json"))
    idx = {f: i for i, f in enumerate(sessions)}
    for f in sessions:
        try:
            tr = json.load(open(os.path.join(TDIR, f), encoding="utf-8"))
        except Exception:
            continue
        if not (isinstance(tr, list) and tr
                and isinstance(tr[0], dict) and "event" in tr[0]):
            continue                      # pre-idea-era transcript shape
        n_sessions += 1
        era = f"E{min(3, idx[f] * 4 // len(sessions)) + 1}"
        for e in tr:
            v = e.get("verdict")
            if v not in ("pass", "fail"):
                n_broken += 1
                continue
            n_events += 1
            ev, said = e["event"], e["said"]
            rec = dict(
                arity=str(len(ev)),
                actor=ev[0],
                action=ev[1],
                obj=(ev[2] if len(ev) > 2 else "none"),
                deco=("yes" if any(w in DECO for w in said) else "no"),
                extra=str(min(3, max(-1, len(said) - len(ev)))),
                order=order_kind(ev, said),
                era=era,
            )
            records.append(rec)
            failed.append(v == "fail")
            if v == "fail" and e.get("say") == said:
                tutor_echo_fails += 1

    n_fail = sum(failed)
    print("=" * 70)
    print("FRONTIER SNAPSHOT — his life, mined by his own organ")
    print("=" * 70)
    print(f"  {n_sessions} idea-drill sessions, {n_events} judged events "
          f"({n_fail} fails / {n_events - n_fail} passes; "
          f"{n_broken} broken-format excluded)")
    print(f"  TUTOR ECHO FAILS: {tutor_echo_fails} failures where the "
          f"correction was VERBATIM what he said\n  (pure tutor "
          f"self-inconsistency, measured: "
          f"{tutor_echo_fails}/{n_fail} = "
          f"{100 * tutor_echo_fails / max(n_fail, 1):.0f}% of all fails)")

    mind = mod.Isaac()                        # fresh mind; his account
    r = mind.mine_commonality(records, failed)  # is never touched
    print(f"\n  THE MINER'S VERDICT (null threshold "
          f"{r['threshold']:.3f}, top excess {r['top']:.3f}):")
    if not r["named"]:
        print("    nothing named — failures carry no shared signature "
              "past the null")
    for k, v, e in r["named"]:
        print(f"    named: {k}={v}   excess {e:+.3f}")

    # complete numbers behind the verdict: per-feature fail rates
    print("\n  FAIL-RATE TABLES (complete numbers behind the verdict):")
    for feat in ("action", "arity", "deco", "order", "era", "extra"):
        pools = {}
        for rec, fl in zip(records, failed):
            pools.setdefault(rec[feat], [0, 0])
            pools[rec[feat]][fl] += 1
        rows = sorted(pools.items(),
                      key=lambda kv: -(kv[1][1] / max(sum(kv[1]), 1)))
        cells = [f"{v}={f_}/{p + f_}" for v, (p, f_) in rows[:6]]
        print(f"    {feat:<7} " + "  ".join(cells))

    # ── the door-usage inventory: what his mind can currently say ──
    print("\n" + "=" * 70)
    print("DOOR-USAGE INVENTORY — what is knowable today (read-only)")
    print("=" * 70)
    m = mod.Isaac.load_pack(os.path.join(HOME, "packs"))
    nodes = set(m.net.doors) | set(m.net.meta)
    print(f"  his net: {len(nodes)} nodes, "
          f"{sum(len(v) for v in m.net.doors.values())} doors")
    top = Counter(dict(m.net.door_counts)).most_common(8)
    print("  top doors by OBSERVATION count (how often SEEN at intake"
          " — the only per-door number that exists):")
    for (a, b), c in top:
        print(f"    {a} -> {b}: seen {c}x")
    o3 = m.net.get_meta("speak:orientation:3", "total") or {}
    o2 = m.net.get_meta("speak:orientation:2", "total") or {}
    print(f"  orientation votes (consulted at every speak): "
          f"arity-3 {o3}, arity-2 {o2}")
    deco = m.net.get_meta("speak:decoration", "counts") or {}
    print(f"  decoration evidence (consulted at every speak): {deco}")
    words = m.net.get_meta("speak:words", "map") or {}
    print(f"  word map (consulted at every speak): {words}")
    wrongs = m.net.get_meta("speak:wrong", "record") or []
    print(f"  wrongs kept: {len(wrongs)} — each holds event/said/"
          f"correction, NOT the evidence state he consulted")
    print("\n  WHAT DOES NOT EXIST (the gap): no door or meta carries a"
          "\n  USE count or a per-decision receipt — observation counts"
          "\n  say what he has SEEN, never what he LEANED ON to answer.")


if __name__ == "__main__":
    main()
