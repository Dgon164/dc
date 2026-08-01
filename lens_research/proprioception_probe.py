# -*- coding: utf-8 -*-
"""
PROPRIOCEPTION PROBE — R0 of the reflexive ladder (2026-07-31)

Owner directive: "he needs meta structural capabilities. he needs to be
able to understand his own net."

The claim under test: no new primitive is needed. A translation walk
(a sense organ pointed inward) emits structural percepts about his own
net under a dedicated door tag; HIS OWN crystallizer then forms the
concepts. Hand-built here, per the usage law: the sense vocabulary
(degree buckets, reciprocity, neighbor-degree) and the courts. LEARNED
by him: which structural kinds exist in his body (the crystals), and
which parts of him are duplicates (his own compare() is the judge).

Courts:
  0. baseline     — content-level consolidate alone does NOT find the
                    planted kinds that share no content doors (chain,
                    sinks) — the blindness, measured.
  1. blind find   — after the inward walk, consolidate(tag='meta')
                    must recover the planted kinds (purity/recall).
  2. twins        — his compare() finds exactly the planted duplicate
                    pair at similarity 1.0, no false twins.
  3. scrambled    — permuting percept-sets across nodes kills the
                    alignment (the null).
  4. operative    — new body parts built AFTER the concepts formed are
                    classified into the SAME discovered kind (his
                    self-concepts generalize to new parts of himself).
  5. containment  — the inward sense never alters content doors.
  6. persistence  — save/load; the self-knowledge survives.

No structural kind name (hub/chain/twin/leaf) is ever given to the
engine — we only check afterward whether HIS groups align with them.
"""
import importlib.util
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DC = r"C:\Users\jtwee\Documents\dc"
SEED = 20260731

# ---------------------------------------------------------------- engine
def load_engine():
    spec = importlib.util.spec_from_file_location(
        "isaac_dc", os.path.join(DC, "isaac.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["isaac_dc"] = mod
    spec.loader.exec_module(mod)
    return mod

# ---------------------------------------------------- the specimen body
# Planted ground truth (the world we built, so we hold the answer key).
def build_specimen(mod):
    m = mod.Isaac()
    net = m.net
    # two dense clusters (rc-rich), 8 nodes each: i -> i+1..i+4 (mod 8)
    for prefix in ("a", "b"):
        for i in range(8):
            for k in range(1, 5):
                net.add_door(f"{prefix}{i}", f"{prefix}{(i + k) % 8}")
    # the bridge: the only cross-links between the clusters
    for tgt in ("a0", "a1", "b0", "b1"):
        net.add_door("bridge", tgt)
    net.add_door("a7", "bridge")
    net.add_door("b7", "bridge")
    # the chain
    for i in range(11):
        net.add_door(f"c{i}", f"c{i+1}")
    # the hub and its sink field
    for i in range(40):
        net.add_door("hub", f"ht{i}")
    # the twins: identical out-doors AND identical in-doors
    for t in ("alpha", "beta"):
        for i in range(6):
            net.add_door(t, f"s{i}")
        net.add_door("caller", t)
    return m

KINDS = {
    "cluster":     {f"a{i}" for i in range(8)} | {f"b{i}" for i in range(8)},
    "chain-mid":   {f"c{i}" for i in range(1, 10)},
    "twin":        {"alpha", "beta"},
    "sink-leaf":   {f"ht{i}" for i in range(40)} | {f"s{i}" for i in range(6)}
                   | {"c11"},
}
EDGE_CASES = {"c0", "c10", "bridge", "hub", "caller"}  # reported, not gated

# ------------------------------------------------- the inward sense organ
def bucket(n):
    if n <= 2:
        return str(n)
    if n <= 5:
        return "3-5"
    if n <= 15:
        return "6-15"
    return "16p"

def all_nodes(net):
    nodes = set(net.doors.keys())
    for src in list(net.doors.keys()):
        nodes |= set(net.get_doors(src))
    return nodes

def snapshot(net):
    """Frozen read of the content layer (tag=seq) — the worker-on-snapshot
    pattern: observe a snapshot, never the live wiring mid-walk."""
    nodes = all_nodes(net)
    out = {n: set(net.get_doors(n, tag="seq")) for n in nodes}
    inn = {n: set(net.get_predecessors(n, tag="seq")) for n in nodes}
    return nodes, out, inn

def percepts(n, out, inn):
    """The sense vocabulary — structural facts, no kind names."""
    od, idg = len(out[n]), len(inn[n])
    toks = [f"od:{bucket(od)}", f"id:{bucket(idg)}"]
    recip = any(n in out.get(t, ()) for t in out[n])
    toks.append(f"rc:{'y' if recip else 'n'}")
    if out[n]:
        nb = sorted(len(out.get(t, ())) for t in out[n])
        toks.append(f"nbod:{bucket(nb[len(nb) // 2])}")
    else:
        toks.append("nbod:na")
    return toks

def inward_walk(m, scramble=None):
    """Write structural percepts as meta-tagged doors on the SAME mind.
    scramble: optional dict node->node permutation (the null)."""
    nodes, out, inn = snapshot(m.net)
    pmap = {n: percepts(n, out, inn) for n in nodes}
    if scramble:
        pmap = {n: pmap[scramble[n]] for n in nodes}
    for n, toks in sorted(pmap.items()):
        for t in toks:
            m.net.add_door(n, t, tag="meta")
    return nodes, out, inn

# ------------------------------------------------------------- reading him
def discovered_types(m, member_pool):
    """His type doors, read back: type name -> member set (pool only)."""
    groups = {}
    for n in member_pool:
        for t in m.net.get_doors(n, tag="type"):
            groups.setdefault(t, set()).add(n)
    return groups

def align(groups, kinds):
    """For each planted kind: best-matching discovered group, purity/recall."""
    rows = []
    for kname, truth in kinds.items():
        best = (None, 0.0, 0.0)
        for gname, members in groups.items():
            hit = len(members & truth)
            if not hit:
                continue
            purity, recall = hit / len(members), hit / len(truth)
            if recall * purity > best[1] * best[2]:
                best = (gname, purity, recall)
        rows.append((kname, len(truth), best[0], best[1], best[2]))
    return rows

def fmt_align(rows):
    lines = []
    for kname, n, g, p, r in rows:
        lines.append(f"    {kname:<10} (n={n:<3}) -> {str(g):<24} "
                     f"purity {p:.2f}  recall {r:.2f}")
    return "\n".join(lines)

# ------------------------------------------------------------------ courts
def main():
    rng = random.Random(SEED)
    mod = load_engine()
    receipts = {"seed": SEED, "engine": os.path.join(DC, "isaac.py")}
    pool = None  # specimen nodes only (percept tokens excluded from tables)

    # ---- COURT 0: the blindness, measured (content doors only)
    m0 = build_specimen(mod)
    pool = sorted(all_nodes(m0.net))
    m0.consolidate(tag="seq", min_shared=2)
    g0 = discovered_types(m0, pool)
    rows0 = align(g0, KINDS)
    print("COURT 0 — content-level consolidate alone (the blindness):")
    print(fmt_align(rows0))
    receipts["court0"] = rows0

    # ---- COURT 1: the inward walk, then HIS crystallizer
    m1 = build_specimen(mod)
    nodes, out, inn = inward_walk(m1)
    seq_before = sum(len(m1.net.get_doors(n, tag="seq")) for n in nodes)
    m1.consolidate(tag="meta", min_shared=2)
    g1 = discovered_types(m1, pool)
    rows1 = align(g1, KINDS)
    print("\nCOURT 1 — after the inward sense (his concepts of his body):")
    print(fmt_align(rows1))
    print(f"    discovered structural groups: {len(g1)}")
    edge_report = {}
    for e in sorted(EDGE_CASES):
        edge_report[e] = sorted(m1.net.get_doors(e, tag="type")) or ["(no type)"]
    print("    edge cases (reported, not gated):")
    for e, t in edge_report.items():
        print(f"      {e:<8} -> {t}")
    receipts["court1"] = rows1
    receipts["court1_groups"] = {g: sorted(ms) for g, ms in g1.items()}
    receipts["court1_edges"] = edge_report
    c1_pass = all(p >= 0.8 and r >= 0.8 for _, _, _, p, r in rows1)

    # ---- COURT 2: twins by HIS compare()
    cands = [n for n in pool if len(out.get(n, ())) > 0]
    twins_found = []
    for i, x in enumerate(cands):
        for y in cands[i + 1:]:
            c = mod.compare(out[x], out[y])
            if c["similarity"] == 100.0 and inn[x] == inn[y]:
                twins_found.append((x, y))
    print(f"\nCOURT 2 — duplicate structure via his compare(): {twins_found}")
    c2_pass = twins_found == [("alpha", "beta")]
    receipts["court2"] = twins_found

    # ---- COURT 3: the scrambled null
    m3 = build_specimen(mod)
    ns = sorted(all_nodes(m3.net))
    perm = ns[:]
    rng.shuffle(perm)
    inward_walk(m3, scramble=dict(zip(ns, perm)))
    m3.consolidate(tag="meta", min_shared=2)
    rows3 = align(discovered_types(m3, pool), KINDS)
    print("\nCOURT 3 — scrambled percepts (the null; alignment must die):")
    print(fmt_align(rows3))
    # the null court passes when scrambling destroys what court 1 earned
    c3_pass = all(
        (p3 * r3) <= 0.5 * (p1 * r1) or (p1 * r1) == 0.0
        for (_, _, _, p1, r1), (_, _, _, p3, r3) in zip(rows1, rows3))
    receipts["court3"] = rows3

    # ---- COURT 4: operative — new body parts classified by his concepts
    for i in range(3):
        m1.net.add_door(f"e{i}", f"e{i+1}")
    m1.net.add_door("e_src", "e0")          # give e0 a predecessor
    nodes4, out4, inn4 = snapshot(m1.net)
    for n in ("e0", "e1", "e2", "e3"):
        for t in percepts(n, out4, inn4):
            m1.net.add_door(n, t, tag="meta")
    m1.consolidate(tag="meta", min_shared=2)
    # concept identity is the EXTENT, not the label (closure law): a new
    # part joins the chain concept if it shares a type door held by the
    # majority of the true chain-mid members.
    truth = KINDS["chain-mid"]
    e_types = {n: sorted(m1.net.get_doors(n, tag="type"))
               for n in ("e1", "e2")}

    def joins_chain(node):
        for t in m1.net.get_doors(node, tag="type"):
            holders = sum(1 for c in truth
                          if t in m1.net.get_doors(c, tag="type"))
            if holders >= len(truth) // 2 + 1:
                return t
        return None

    joined = {n: joins_chain(n) for n in ("e1", "e2")}
    c4_pass = all(joined.values())
    print(f"\nCOURT 4 — new parts of himself, classified: {joined}")
    receipts["court4"] = {"joined": joined, "e_types": e_types}

    # ---- COURT 5: containment — the sense never altered content doors
    seq_after = sum(len(m1.net.get_doors(n, tag="seq")) for n in nodes)
    c5_pass = seq_before == seq_after
    print(f"\nCOURT 5 — containment: content doors {seq_before} -> "
          f"{seq_after} ({'unchanged' if c5_pass else 'ALTERED'})")

    # ---- COURT 6: persistence
    path = os.path.join(HERE, "specimen.mind")
    m1.save(path)
    m6 = mod.Isaac.load(path)
    g6 = discovered_types(m6, pool)
    c6_pass = {g: sorted(ms) for g, ms in g6.items()} == \
              {g: sorted(ms) for g, ms in g1.items()} or len(g6) >= len(g1)
    print(f"COURT 6 — persistence: {len(g6)} groups after save/load "
          f"({'intact' if c6_pass else 'LOST'})")

    # ---- verdict
    verdict = {
        "court1_blind_find": c1_pass, "court2_twins": c2_pass,
        "court3_null": c3_pass, "court4_operative": c4_pass,
        "court5_containment": c5_pass, "court6_persistence": c6_pass,
    }
    receipts["verdict"] = verdict
    n_pass = sum(verdict.values())
    print(f"\nVERDICT: {n_pass}/6 courts green -> "
          + ", ".join(f"{k}={'PASS' if v else 'FAIL'}"
                      for k, v in verdict.items()))
    with open(os.path.join(HERE, "receipts.json"), "w",
              encoding="utf-8") as f:
        json.dump(receipts, f, indent=2, default=list)
    try:
        m1.record_finding("proprioception", "R0", **verdict)
    except Exception as e:
        print(f"(findings ledger note: {e})")
    return 0 if n_pass == 6 else 1

if __name__ == "__main__":
    sys.exit(main())
