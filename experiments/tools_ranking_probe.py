"""
TOOLS RANKING PROBE -- attack on the tools-node QUERY organ.

THE CLAIM UNDER TEST (fresh-eyes item 7): query_tools is bag-of-words
overlap -- the ranking can mislead (T3's top-3 included
complete_symmetry for an instrument query); catalog descriptions are
the FIRST DOCSTRING SENTENCE, which can mislead out of context.
Surface: isaac.py:2284-2367 (register_tool / catalog_tools /
query_tools / run_tool).

WHAT IS MEASURED
  A. Catalog census on a FRESH mind (the record says 107 engine
     organs auto-harvest) + description audit: length distribution,
     the shortest descriptions verbatim, and SELF-RETRIEVAL (does a
     tool's own catalog description rank the tool first?).
  B. A 17-query relevance set with known-correct answers drawn from
     the docs (PAPERCLIP_MANUAL.md, docs/ISAAC_MANUAL.md, the organs'
     own docstrings). Top-1 / top-3 hit rates vs a shuffled-ranking
     null; worst misleads listed with scores.
  C. Score calibration attacks: a gibberish query (must skip -- the
     honest-refusal control) and a pure stop-word query (semantics-
     free; bag-of-words should leak nonzero scores).
  D. SAFETY (fresh-eyes item 3): run_tool executes on the LIVE mind;
     the fork-first sandbox is designed but NOT BUILT. Demonstrated
     ONCE -- one fresh throwaway mind, ONE mutating tool via
     run_tool, state change shown with numbers. NO catalog sweep.

PREDICTION (filed before the first run):
  P1. catalog_tools() harvests ~107 organs (the record's number;
      more only if organs joined since the record).
  P2. Relevance top-1 well above the shuffled null but far from
      clean -- predicted top-1 in [0.4, 0.7], top-3 in [0.6, 0.85].
  P3. Specific predicted misses, from reading the catalog texts:
      'what keeps failing' -> mine_frontier shares ZERO content
      words with its description ("After a session: mine the recent
      window of his own life.") -- the documented session-loop query
      cannot retrieve its documented tool. 'what comes next' ->
      lookup fails because first-sentence truncation DROPS the
      docstring's own "What comes next?" sentence. 'solve for the
      unknown' suffers no-stemming (unknown vs unknowns).
  P4. Shortest descriptions are misleading out of context --
      discover's catalog entry is literally "THE CHALLENGE LOOP."
      (no 'equation', no 'data', no 'rule').
  P5. Gibberish query -> honest skip (matches=[]); stop-word query
      -> nonzero scores (overlap has no IDF / calibration).
  P6. run_tool('teach', ...) on a fresh mind mutates the live net
      (doors added, fact indexed) with no fork and no undo.

Run: cd experiments && PYTHONHASHSEED=0 OPENBLAS_NUM_THREADS=1 \
     OMP_NUM_THREADS=1 python3 tools_ranking_probe.py
stdlib + numpy only. Fresh throwaway minds. Nothing patched:
every wrong found here is REPORTED, not repaired.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import isaac as mod                                    # noqa: E402

rng = np.random.default_rng(7)
N_NULL = 2000                     # shuffled-ranking draws per query
RECORD_COUNT = 107                # docs/ISAAC_MANUAL.md:1425 "107 on
                                  # the current engine"

# ── the relevance set ────────────────────────────────────────────
# Known-correct answers drawn from the docs. Provenance per query:
#   [M]  PAPERCLIP_MANUAL.md Part 2A (feeding/querying/solving)
#   [IM] docs/ISAAC_MANUAL.md:1438-1441 (the account seeding:
#        "what keeps failing" -> mine_frontier)
#   [D]  the organ's own docstring in isaac.py
#   [T3] tools_battery.py T3 verbatim (the fresh-eyes cited query)
# A query scores a hit if the ranked tool is in its acceptable set.
RELSET = [
    ("what keeps failing",                             # [IM]:1440
     {"mine_frontier", "mine_commonality"}),
    ("read the instrument resolution of a column",     # [D] read_floor
     {"read_floor", "tell_resolution"}),
    ("how precise is the measuring instrument -- the resolution "
     "floor of these values",                          # [T3]
     {"read_floor", "tell_resolution"}),
    ("do these two sequences differ in order",         # [D] compare_order
     {"compare_order"}),
    ("does operand order matter",                      # [D] order_matters
     {"order_matters"}),
    ("what do the failures share",                     # [D] mine_commonality
     {"mine_commonality", "mine_frontier"}),
    ("discover an equation from data",                 # [M] 2C law lens
     {"discover"}),
    ("fill in unknown points using an assumed axis of symmetry",
     {"complete_symmetry"}),                           # [D]
    ("save the whole mind to disk",                    # [D] save_pack;
     # 'save' (isaac.py:1959 "Snapshot the WHOLE mind ... one file")
     # is equally correct -- accepted after a first-run ground-truth
     # audit (probe bug, not an engine pardon).
     {"save_pack", "save"}),
    ("teach him a fact",                               # [M] teach
     {"teach"}),
    ("generate a sentence about a concept",            # [M] generate
     {"generate"}),
    ("what comes next after this context",             # [M] lookup
     {"lookup", "predict_next"}),
    ("is this specific claim true",                    # [M] check
     {"check"}),
    ("is a cat a kind of animal",                      # [M] is_a_check
     {"is_a_check"}),
    ("solve for the unknown",                          # [M] solve
     {"solve"}),
    ("multiply two numbers he has never seen",         # [M] compute
     {"compute"}),
    ("let him sleep and crystallize while idle",       # [M] consolidate
     {"consolidate", "crystallize"}),
]

STOPWORD_QUERY = "the a of in is to for with and"
GIBBERISH_QUERY = "zqvirp glomtak sfezzle wubbin"


def fresh_cataloged_mind():
    """A fresh throwaway mind with the engine catalog harvested, plus
    the ONE documented drill-layer tool this probe's relevance set
    names: mine_frontier (docs/ISAAC_MANUAL.md:1438 -- the account
    pack carries 107 engine + 7 drill-layer tools incl. mine_frontier;
    a fresh mind gets only the 107, so the account's registration is
    replicated here with the function's real first docstring sentence,
    experiments/idea_corrections.py:312)."""
    m = mod.Isaac()
    n = m.catalog_tools()
    m.register_tool(
        "mine_frontier",
        "After a session: mine the recent window of his own life.",
        how="mine_frontier(mind, transcripts_dir)", tags=("drill",))
    return m, n


def full_ranking(m, query, n_tools):
    r = m.query_tools(query, top=n_tools)
    return r.get("matches", []), r.get("why", "")


def rank_of_best(matches, acceptable):
    for i, x in enumerate(matches):
        if x["name"] in acceptable:
            return i + 1, x["score"]
    return None, 0


def main():
    print("=" * 72)
    print("TOOLS RANKING PROBE -- query_tools / catalog descriptions "
          "(fresh-eyes item 7) + run_tool live-mind hazard (item 3)")
    print("=" * 72)

    # ── PART A: catalog census + description audit ──────────────
    m, n_harvest = fresh_cataloged_mind()
    doors = sorted(m.net.get_doors("tools", tag="tool"))
    names = [d.split("tool:", 1)[1] for d in doors]
    n_tools = len(names)
    print(f"\nPART A -- catalog census (fresh mind)")
    print(f"  catalog_tools() returned n = {n_harvest}")
    print(f"  tool doors under 'tools'  = {n_tools} "
          f"(engine {n_tools - 1} + 1 drill-layer registered here)")
    print(f"  record says {RECORD_COUNT} engine organs -> "
          f"{'MATCH' if n_harvest == RECORD_COUNT else 'MISMATCH'} "
          f"(harvest {n_harvest} vs record {RECORD_COUNT})")

    recs = {nm: m.net.get_meta(f"tool:{nm}", "record") for nm in names}
    lens = np.array([len(recs[nm]["does"].split()) for nm in names])
    print(f"  description length (words): min {lens.min()}  "
          f"median {np.median(lens):.0f}  max {lens.max()}")
    shortest = sorted(names, key=lambda nm: len(recs[nm]["does"].split()))
    print("  5 shortest descriptions (first-docstring-sentence "
          "truncation, verbatim):")
    for nm in shortest[:5]:
        print(f"    {nm:24s} -> {recs[nm]['does']!r}")

    # self-retrieval: query with a tool's own catalog description --
    # if descriptions were distinctive, the owner should rank first.
    # chance (analytic control): 1/N = {:.4f}
    self_top1 = 0
    self_fail = []
    for nm in names:
        matches, _ = full_ranking(m, recs[nm]["does"], n_tools)
        if matches and matches[0]["name"] == nm:
            self_top1 += 1
        else:
            got = matches[0]["name"] if matches else "(skip)"
            r, _ = rank_of_best(matches, {nm})
            self_fail.append((nm, got, r))
    print(f"  self-retrieval (query = own description): "
          f"{self_top1}/{n_tools} top-1 = {self_top1 / n_tools:.2f} "
          f"(chance 1/N = {1 / n_tools:.4f})")
    print(f"  self-retrieval failures: {len(self_fail)}; worst 6:")
    for nm, got, r in sorted(self_fail, key=lambda t: (t[2] is None,
                                                       -(t[2] or 0)))[:6]:
        print(f"    {nm:24s} lost to {got:24s} "
              f"(own rank {r if r else 'unranked'})")

    # ── PART B: the 17-query relevance set ──────────────────────
    print(f"\nPART B -- relevance set ({len(RELSET)} queries, "
          f"doc-grounded answers)")
    top1_hits = top3_hits = 0
    tie_top1 = 0
    misleads = []
    for q, acc in RELSET:
        matches, _ = full_ranking(m, q, n_tools)
        top3 = [x["name"] for x in matches[:3]]
        h1 = bool(top3) and top3[0] in acc
        h3 = any(t in acc for t in top3)
        top1_hits += h1
        top3_hits += h3
        tied = (len(matches) >= 2
                and matches[0]["score"] == matches[1]["score"])
        tie_top1 += tied
        r_acc, s_acc = rank_of_best(matches, acc)
        mark = "TOP1" if h1 else ("top3" if h3 else "MISS")
        print(f"  [{mark}] {q!r}")
        print(f"         top-3: "
              + ", ".join(f"{x['name']}({x['score']})"
                          for x in matches[:3])
              + (f"   [top-1 tie, alphabetical break]" if tied else ""))
        if not h1:
            best = sorted(acc)[0] if r_acc is None else None
            print(f"         correct {sorted(acc)} at rank "
                  f"{r_acc if r_acc else 'UNRANKED (zero overlap)'}"
                  f" score {s_acc}")
            misleads.append((q, top3[:1], sorted(acc), r_acc,
                             matches[0]["score"] if matches else 0,
                             s_acc))

    top1_rate = top1_hits / len(RELSET)
    top3_rate = top3_hits / len(RELSET)

    # shuffled-ranking null: same acceptable sets, random order of
    # the same catalog. Every positive claim ("the ranking works")
    # must beat this.
    null1 = np.zeros(N_NULL)
    null3 = np.zeros(N_NULL)
    name_arr = np.array(names)
    acc_sets = [acc & set(names) for _, acc in RELSET]
    for i in range(N_NULL):
        h1 = h3 = 0
        for acc in acc_sets:
            idx = rng.permutation(n_tools)[:3]
            picks = name_arr[idx]
            h1 += picks[0] in acc
            h3 += any(p in acc for p in picks)
        null1[i] = h1 / len(RELSET)
        null3[i] = h3 / len(RELSET)
    n99_1 = float(np.quantile(null1, 0.99))
    n99_3 = float(np.quantile(null3, 0.99))
    print(f"\n  TOP-1 hit rate: {top1_hits}/{len(RELSET)} = "
          f"{top1_rate:.2f}   (shuffled null mean "
          f"{null1.mean():.3f}, 99th pct {n99_1:.3f})")
    print(f"  TOP-3 hit rate: {top3_hits}/{len(RELSET)} = "
          f"{top3_rate:.2f}   (shuffled null mean "
          f"{null3.mean():.3f}, 99th pct {n99_3:.3f})")
    print(f"  top-1 decided by alphabetical tie-break: "
          f"{tie_top1}/{len(RELSET)} queries")
    print(f"  WORST MISLEADS (top-1 wrong):")
    for q, got, acc, r_acc, s_got, s_acc in misleads:
        print(f"    {q!r}: got {got[0] if got else '(skip)'} "
              f"(score {s_got}); correct {acc} at rank "
              f"{r_acc if r_acc else 'UNRANKED'} (score {s_acc})")

    # the fresh-eyes T3 citation, checked exactly
    t3_matches, _ = full_ranking(
        m, RELSET[2][0], n_tools)
    t3_top3 = [x["name"] for x in t3_matches[:3]]
    print(f"  T3 replication -- instrument query top-3: {t3_top3}; "
          f"complete_symmetry in top-3: "
          f"{'complete_symmetry' in t3_top3}")

    # ── PART C: calibration attacks ─────────────────────────────
    print(f"\nPART C -- score calibration")
    r = m.query_tools(GIBBERISH_QUERY)
    gib_skip = r["matches"] == [] and "skip" in r["why"]
    print(f"  gibberish query {GIBBERISH_QUERY!r}: "
          f"matches={len(r['matches'])}, why={r['why']!r} "
          f"-> {'HONEST SKIP' if gib_skip else 'FAILED TO SKIP'}")
    r = m.query_tools(STOPWORD_QUERY)
    sw = [(x["name"], x["score"]) for x in r["matches"][:3]]
    sw_top = sw[0][1] if sw else 0
    print(f"  stop-word query {STOPWORD_QUERY!r} (zero semantics): "
          f"top-3 {sw}")
    print(f"    -> a semantics-free query scores {sw_top} "
          f"(no IDF / no calibration: function words count as "
          f"evidence of applicability)")

    # ── PART D: run_tool executes on the LIVE mind (ONE demo) ───
    print(f"\nPART D -- run_tool live-mind hazard (fresh-eyes item 3; "
          f"ONE mutating tool, NO sweep)")
    m2 = mod.Isaac()
    m2.catalog_tools()
    doors_before = sum(len(v) for v in m2.net.doors.values())
    cat_before = "cat" in m2.net.doors
    facts_before = len(m2.fact_index.get(("cat", "is_a"), []))
    out = m2.run_tool("teach", "cat", "is_a", "mammal")
    doors_after = sum(len(v) for v in m2.net.doors.values())
    cat_after = "cat" in m2.net.doors
    facts_after = len(m2.fact_index.get(("cat", "is_a"), []))
    used = m2.net.get_meta("tool:teach", "record")["used"]
    print(f"  run_tool('teach','cat','is_a','mammal') -> {out!r}")
    print(f"  net doors: {doors_before} -> {doors_after} "
          f"(+{doors_after - doors_before})")
    print(f"  'cat' node present: {cat_before} -> {cat_after}")
    print(f"  fact_index ('cat','is_a'): {facts_before} -> "
          f"{facts_after} entries")
    print(f"  tool:teach used counter: {used}")
    mutated = (doors_after > doors_before and cat_after
               and facts_after > facts_before)
    print(f"  => the tool ran on the LIVE mind: mutation "
          f"{'CONFIRMED' if mutated else 'not detected'} -- no fork, "
          f"no undo. The design's own 'trial and error: run them "
          f"all' mode would apply this to every mutating organ in "
          f"the {n_tools}-tool catalog. Fork-first sandbox: designed, "
          f"NOT BUILT. (Reported, not patched.)")

    # ── VERDICTS ────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("VERDICTS")
    print("=" * 72)
    v_cat = "WORKS" if n_harvest >= RECORD_COUNT else "PARTIAL"
    print(f"  catalog auto-harvest: {v_cat} "
          f"({n_harvest} organs vs record {RECORD_COUNT})")
    beats_null = top1_rate > n99_1 and top3_rate > n99_3
    v_rank = ("WORKS" if beats_null and top1_rate >= 0.9
              else "PARTIAL" if beats_null else "BROKEN")
    print(f"  query_tools relevance ranking: {v_rank} "
          f"(top-1 {top1_rate:.2f} / top-3 {top3_rate:.2f} vs null99 "
          f"{n99_1:.3f} / {n99_3:.3f}; {len(misleads)} misleads; "
          f"{tie_top1} alphabetical tie-breaks at top-1)")
    v_desc = ("WORKS" if self_top1 / n_tools >= 0.95 and lens.min() >= 4
              else "PARTIAL" if self_top1 / n_tools >= 0.7
              else "BROKEN")
    print(f"  catalog descriptions: {v_desc} "
          f"(self-retrieval {self_top1}/{n_tools}; shortest "
          f"description {lens.min()} words: "
          f"{recs[shortest[0]]['does']!r})")
    print(f"  empty-query honesty: "
          f"{'HONEST-REFUSAL' if gib_skip else 'BROKEN'} "
          f"(gibberish skips) BUT stop-word query scores {sw_top} "
          f"(calibration hole)")
    print(f"  run_tool sandboxing: BROKEN (hazard demonstrated: live "
          f"mutation +{doors_after - doors_before} doors through the "
          f"tool node; fork-first sandbox not built)")
    print("=" * 72)


if __name__ == "__main__":
    main()
