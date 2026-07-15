"""frontier_battery.py — THE FRONTIER HABIT (owner, 2026-07-15: "he
just needs to see that hey, this keeps losing in these conditions,
group them and find why"). Battery for mine_frontier: his own miner
over his own recent life, every session, stored in his own mind.

Checks N1-N4. Run: PYTHONHASHSEED=0 python frontier_battery.py
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
from idea_corrections import mine_frontier              # noqa: E402

rng = np.random.default_rng(9)
CHECKS = []


def check(name, ok, detail=""):
    CHECKS.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" +
          (f"   {detail}" if detail else ""))


def session(events):
    return events


def write_sessions(td, sessions, start=0):
    tdir = os.path.join(td, "transcripts")
    os.makedirs(tdir, exist_ok=True)
    for k, ev in enumerate(sessions):
        with open(os.path.join(tdir, f"session_{start + k:05d}.json"),
                  "w", encoding="utf-8") as f:
            json.dump(ev, f)
    return tdir


def make_events(n, fail_when_smiled=0.9, base=0.1):
    out = []
    for _ in range(n):
        act = str(rng.choice(["smiled", "ran", "sat", "threw"]))
        ev = ["boy", act] if act != "threw" else ["boy", act, "ball"]
        p = fail_when_smiled if act == "smiled" else base
        v = "fail" if rng.random() < p else "pass"
        e = dict(event=ev, said=["the"] + ev, verdict=v)
        if v == "fail":
            e["say"] = ev
        out.append(e)
    return out


def main():
    print("=" * 70)
    print("FRONTIER BATTERY — the see-what-keeps-losing habit")
    print("=" * 70)

    # ── N1: a planted losing condition gets NAMED and STORED ──
    m = mod.Isaac()
    with tempfile.TemporaryDirectory() as td:
        tdir = write_sessions(td, [make_events(30) for _ in range(10)])
        r = mine_frontier(m, tdir)
    named = {(k, v) for k, v, _ in r["named"]}
    stored = m.net.get_meta("commonality:frontier:speech", "active")
    ok = (("action", "smiled") in named and stored is not None
          and stored["named"] == r["named"])
    check("N1 'keeps losing when action=smiled' named + stored in his "
          "net", ok, f"named {sorted(named)}")

    # ── N2: random failures -> honest nothing ──
    m2 = mod.Isaac()
    with tempfile.TemporaryDirectory() as td:
        tdir = write_sessions(td, [make_events(30, fail_when_smiled=0.15,
                                               base=0.15)
                                   for _ in range(10)])
        r = mine_frontier(m2, tdir)
    check("N2 no shared condition -> nothing named", r["named"] == [],
          f"top {r['top']:.3f} vs thr {r['threshold']:.3f}")

    # ── N3: the window is the PRESENT — a cured era washes out ──
    with tempfile.TemporaryDirectory() as td:
        old = [make_events(30) for _ in range(10)]            # smiled era
        new = [make_events(30, fail_when_smiled=0.1,          # cured;
                           base=0.1) for _ in range(10)]      # now clean
        tdir = write_sessions(td, old + new)
        r = mine_frontier(m, tdir, last_n=10)                 # recent only
    chain = m.net.get_meta("commonality:frontier:speech",
                           "superseded") or []
    ok = (r["named"] == [] and len(chain) == 1
          and any(k == "action" and v == "smiled"
                  for k, v, _ in chain[0]["record"]["named"]))
    check("N3 cured era washes out; old frontier kept superseded", ok,
          f"chain {len(chain)}")

    # ── N4: the frontier survives the pack road ──
    a = json.dumps(m.net.get_meta("commonality:frontier:speech",
                                  "superseded"), sort_keys=True,
                   default=str)
    with tempfile.TemporaryDirectory() as td:
        m.save_pack(td)
        m3 = mod.Isaac.load_pack(td)
    b = json.dumps(m3.net.get_meta("commonality:frontier:speech",
                                   "superseded"), sort_keys=True,
                   default=str)
    check("N4 frontier history survives packs", a == b and a != "null")

    n_ok = sum(CHECKS)
    print("=" * 70)
    print(f"FRONTIER BATTERY: {n_ok}/{len(CHECKS)}")
    print("=" * 70)
    sys.exit(0 if n_ok == len(CHECKS) else 1)


if __name__ == "__main__":
    main()
