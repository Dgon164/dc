"""containers_probe.py — CONTAINERS: world-scoped truth (owner,
2026-07-15 night: "certain details or facts change depending on the
world or situation you put it in. fiction world harry potter has
different rules than fiction world hail mary").

RESEARCH, NOT THE ENGINE. The probe builds the container mechanism
as a THIN SHIM over primitives the engine already owns — door TAGS
(add_door(tag=), get_door_tags) and the fact index — to prove the
design before any promotion:

  A CONTAINER IS A TAG saying which facts are IN FORCE. The word node
  stays ONE node (identity = relations — brooms in Harry Potter and
  brooms in the kitchen are the same concept wearing different doors);
  the ACTIVE container is routing state (the execution cycle's "which
  world is open"); asking filters doors by the container tag.
  INHERITANCE IS ONE-WAY: a fiction world overlays base (untold facts
  fall back to base WITH a receipt — HP water still boils at 100),
  but base NEVER inherits from fiction (the anti-poisoning wall: the
  practice corpus contains fiction, and the live account must not).

PREDICTIONS (filed before the first run; wrong ones kept)
  C1 PER-CONTAINER ANSWERS: the same question answers differently by
     world — brooms are-for: base=sweeping, HP=flying; a three-way
     conflicting fact (stars do: shine/dim/sing) answers correctly
     under each of base / hail-mary / harry-potter.
  C2 CROSS-CONTAINER CONFLICT IS NOT CONTRADICTION: teaching all
     three star-facts raises ZERO tension flags (different worlds
     legitimately differ).
  C3 IN-CONTAINER CONTRADICTION STILL FLAGS: teaching HP both
     "brooms are-for flying" and "brooms are-for sinking" records a
     tension IN harry-potter (both kept — wrongs kept — but flagged).
  C4 LEAKAGE 0: over the full fact battery, asking under BASE never
     returns a fiction-only fact (0 leaks in every probe), and asking
     under one fiction never returns the OTHER fiction's facts.
  C5 INHERITANCE WITH RECEIPT: HP has no boiling-point fact of its
     own -> asking under HP returns base's 100C with an explicit
     "inherited from base" receipt; base asking a fiction-only fact
     (portraits do talk) returns NOTHING.
  C6 THE POISONING GUARD: base answers for every base fact are
     byte-identical before and after ingesting all fiction (0 drift).
  C7 THE KEYSTONE CHECK: one node carries both worlds — the doors of
     'brooms' include sweeping AND flying on the SAME node,
     distinguished only by their container tags.
  C8 CONTAINER-LESS FACTS land in base with a receipt saying so.

Run: cd experiments && PYTHONHASHSEED=0 python containers_probe.py
"""
import os
import sys
import time

import numpy as np  # noqa: F401  (engine import path convention)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import isaac as isaac_mod  # noqa: E402

Isaac = isaac_mod.Isaac


class ContainerShim:
    """The candidate design, engine-primitives only. Promotion would
    move these three verbs into the engine (active-container state on
    the mind, container= kwarg on teach, container routing on ask,
    provenance stamping at intake)."""

    def __init__(self, mind):
        self.m = mind
        self.active = "base"
        self.tensions = []                 # in-container flags, kept

    def open_world(self, container):
        """Routing state: which world is open (do op, take result
        door — the container is the door's address space)."""
        self.active = container

    def _tags(self, s, o):
        return self.m.net.get_door_tags(str(s), str(o))

    def _in(self, s, o, c):
        return f"in:{c}" in self._tags(s, o)

    def teach_in(self, s, r, o, container=None):
        c = container or self.active
        prior = self.ask_in(s, r, container=c, inherit=False)
        self.m.teach(s, r, o)
        self.m.net.add_door(str(s), str(o), tag=f"in:{c}")
        if prior["answers"] and str(o) not in prior["answers"]:
            self.tensions.append(dict(container=c, subject=s,
                                      relation=r,
                                      held=list(prior["answers"]),
                                      new=str(o)))

    def ask_in(self, s, r, container=None, inherit=True):
        c = container or self.active
        objs = [str(o) for o in self.m.fact_index.get((s, r), [])]
        seen = list(dict.fromkeys(objs))
        infic = [o for o in seen if self._in(s, o, c)]
        if infic:
            return dict(answers=infic, container=c,
                        why=f"{len(infic)} fact(s) in force in '{c}'")
        if c == "base":
            # container-less facts belong to base, with a receipt
            untagged = [o for o in seen
                        if not any(t.startswith("in:")
                                   for t in self._tags(s, o))]
            if untagged:
                return dict(answers=untagged, container="base",
                            why="container-less fact(s): landed in "
                                "base (receipt)")
            return dict(answers=[], container="base",
                        why="nothing in force in base")
        if inherit:
            base = self.ask_in(s, r, container="base")
            if base["answers"]:
                return dict(answers=base["answers"], container=c,
                            why=f"inherited from base (no '{c}' fact "
                                f"overrides) — {base['why']}")
        return dict(answers=[], container=c,
                    why=f"nothing in force in '{c}'")


def main():
    t0 = time.perf_counter()
    checks = []

    def check(name, ok, detail):
        checks.append(ok)
        print(f"  [{'ok ' if ok else 'FAIL'}] {name} — {detail}")

    mind = Isaac()
    w = ContainerShim(mind)

    # the worlds
    w.teach_in("brooms", "are-for", "sweeping", "base")
    w.teach_in("water", "boils-at", "100C", "base")
    w.teach_in("stars", "do", "shine", "base")
    w.teach_in("owls", "are", "birds", "base")

    w.open_world("harry-potter")
    w.teach_in("brooms", "are-for", "flying")
    w.teach_in("owls", "deliver", "mail")
    w.teach_in("portraits", "do", "talk")
    w.teach_in("stars", "do", "sing")          # HP astronomy, why not

    w.open_world("hail-mary")
    w.teach_in("stars", "do", "dim")
    w.teach_in("astrophage", "eats", "starlight")

    # C1 — per-container answers
    a_base = w.ask_in("brooms", "are-for", "base")["answers"]
    a_hp = w.ask_in("brooms", "are-for", "harry-potter")["answers"]
    s_b = w.ask_in("stars", "do", "base")["answers"]
    s_hm = w.ask_in("stars", "do", "hail-mary")["answers"]
    s_hp = w.ask_in("stars", "do", "harry-potter")["answers"]
    check("C1 same question, per-world answers",
          a_base == ["sweeping"] and a_hp == ["flying"]
          and s_b == ["shine"] and s_hm == ["dim"] and s_hp == ["sing"],
          f"brooms: base={a_base} hp={a_hp}; stars: {s_b}/{s_hm}/{s_hp}")

    # C2 — cross-container conflict raised no tension
    check("C2 cross-world difference is NOT contradiction",
          len(w.tensions) == 0, f"{len(w.tensions)} tensions flagged")

    # C3 — in-container contradiction flags (both kept)
    w.teach_in("brooms", "are-for", "sinking", "harry-potter")
    both = w.ask_in("brooms", "are-for", "harry-potter")["answers"]
    check("C3 in-world contradiction flagged, both kept",
          len(w.tensions) == 1
          and w.tensions[0]["container"] == "harry-potter"
          and set(both) == {"flying", "sinking"},
          f"tensions {len(w.tensions)}; held {both}")

    # C4 — leakage battery: every question under every world
    QA = [("brooms", "are-for"), ("water", "boils-at"), ("stars", "do"),
          ("owls", "deliver"), ("owls", "are"), ("portraits", "do"),
          ("astrophage", "eats")]
    fiction_only = {("owls", "deliver"): "harry-potter",
                    ("portraits", "do"): "harry-potter",
                    ("astrophage", "eats"): "hail-mary"}
    leaks = 0
    for (s, r) in QA:
        rb = w.ask_in(s, r, "base")
        if (s, r) in fiction_only and rb["answers"]:
            leaks += 1                       # fiction reached base
    hp_from_hm = w.ask_in("astrophage", "eats", "harry-potter")
    hm_from_hp = w.ask_in("portraits", "do", "hail-mary")
    cross = int(bool(hp_from_hm["answers"])) \
        + int(bool(hm_from_hp["answers"]))
    check("C4 leakage 0 (fiction->base and fiction->fiction)",
          leaks == 0 and cross == 0,
          f"fiction->base leaks {leaks}/3; cross-fiction {cross}/2")

    # C5 — inheritance with receipt, one-way only
    hp_water = w.ask_in("water", "boils-at", "harry-potter")
    base_portraits = w.ask_in("portraits", "do", "base")
    check("C5 fiction inherits base WITH receipt; base never inherits",
          hp_water["answers"] == ["100C"]
          and "inherited from base" in hp_water["why"]
          and base_portraits["answers"] == [],
          f"hp water={hp_water['answers']} ({hp_water['why'][:40]}); "
          f"base portraits={base_portraits['answers']}")

    # C6 — the poisoning guard: base answers identical pre/post fiction
    fresh = Isaac()
    w2 = ContainerShim(fresh)
    for (s, r, o) in [("brooms", "are-for", "sweeping"),
                      ("water", "boils-at", "100C"),
                      ("stars", "do", "shine"),
                      ("owls", "are", "birds")]:
        w2.teach_in(s, r, o, "base")
    before = {(s, r): w2.ask_in(s, r, "base")["answers"]
              for (s, r) in QA}
    w2.open_world("harry-potter")
    for (s, r, o) in [("brooms", "are-for", "flying"),
                      ("owls", "deliver", "mail"),
                      ("portraits", "do", "talk"),
                      ("stars", "do", "sing")]:
        w2.teach_in(s, r, o)
    w2.open_world("hail-mary")
    w2.teach_in("stars", "do", "dim")
    w2.teach_in("astrophage", "eats", "starlight")
    after = {(s, r): w2.ask_in(s, r, "base")["answers"]
             for (s, r) in QA}
    check("C6 base answers byte-identical after fiction ingestion",
          before == after, f"drift on {sum(before[k] != after[k] for k in before)}/"
                           f"{len(before)} questions")

    # C7 — the keystone: one node, both worlds' doors
    tags_fly = mind.net.get_door_tags("brooms", "flying")
    tags_sweep = mind.net.get_door_tags("brooms", "sweeping")
    check("C7 one concept node carries both worlds (tags disambiguate)",
          "in:harry-potter" in tags_fly and "in:base" in tags_sweep,
          f"flying tags {sorted(tags_fly)}; sweeping tags "
          f"{sorted(tags_sweep)}")

    # C8 — container-less fact lands in base with a receipt
    mind.teach("cats", "chase", "mice")      # raw engine teach, no shim
    r8 = w.ask_in("cats", "chase", "base")
    check("C8 container-less fact -> base, receipt says so",
          r8["answers"] == ["mice"] and "container-less" in r8["why"],
          f"{r8['answers']} ({r8['why']})")

    dt = time.perf_counter() - t0
    npass = sum(checks)
    print(f"\n  PROBE: {npass} passed, {len(checks) - npass} failed of "
          f"{len(checks)}  ({dt:.1f}s)")


if __name__ == "__main__":
    main()
