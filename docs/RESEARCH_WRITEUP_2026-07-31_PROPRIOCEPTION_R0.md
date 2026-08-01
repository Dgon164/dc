# PROPRIOCEPTION R0 — he understood his own net
*(research writeup, 2026-07-31; probe: `lens_research/proprioception_probe.py`; 6/6 courts green, first full run after two probe-side bug fixes)*

## The owner's directive (verbatim)

> "he needs meta structural capabilities. he needs to be able to
> understand his own net. thats what we are missing."

And the standing law minted earlier the same day, which this probe obeys:

> "well there is our first indicator, when something is done with code
> instead of derived from the structure..."

## The claim tested

No new primitive is needed for self-understanding. A **sense organ pointed
inward** — a translation walk that reads a frozen snapshot of the net and
writes structural percepts (degree buckets, reciprocity, neighbor-degree)
as doors under a dedicated `meta` tag — is enough for HIS OWN crystallizer
to form concepts about his own body. Hand-built: the sense vocabulary and
the courts (translation + epistemics, legal per the usage law). Learned by
him: which structural kinds exist in his net, and which parts of him are
duplicates. No kind name (hub/chain/twin/leaf) is ever given to the engine;
alignment with the planted anatomy is checked only afterward.

The specimen body: a scratch mind with planted ground truth — two dense
clusters (16 nodes), a bridge, a 12-node chain, a hub with 40 sinks,
identical twins (`alpha`/`beta`, same out-doors AND in-doors), and leaf
fields. 135 content doors. Engine: `dc/isaac.py` untouched, driven by its
own public methods (`add_door`, `consolidate`, `compare`, `save/load`).

## The courts, complete numbers

**Court 0 — the blindness, measured.** Content-level `consolidate` alone:
chain-mid recall **0.00** (invisible — chain nodes share no successors),
sink-leaf recall 0.13, cluster recall 0.25. Only the twins were found
(they share actual content doors).

**Court 1 — after the inward sense.** `consolidate(tag='meta')` on the
same body formed **9 structural groups**:

| planted kind | n | his best group | purity | recall |
|---|---|---|---|---|
| cluster | 16 | type:a0_a1[17] | 0.94 | 1.00 |
| chain-mid | 9 | type:c0_c1[11] | 0.82 | 1.00 |
| twin | 2 | type:alpha_beta[2] | 1.00 | 1.00 |
| sink-leaf | 47 | type:c11_ht0[47] | 1.00 | 1.00 |

The kinds that were invisible to content crystallization became fully
visible to structural crystallization. **The delta between court 0 and
court 1 is the capability.**

**Court 2 — duplicate structure by his own primitive.** `compare()` over
out-door sets (plus in-door equality) found exactly `[('alpha','beta')]` —
no false twins among all candidate pairs. (Probe bug fixed en route:
`compare()` reports similarity as a percentage; identical = 100.0.)

**Court 3 — the scrambled null.** Permuting percept-sets across nodes
collapsed the alignment (cluster purity 0.94 → 0.19; chain 0.82 → 0.15;
twin 1.00 → 0.04). The structure he found is in the body, not in the
harness.

**Court 4 — operative: his self-concepts classify new parts of himself.**
Three new chain nodes (`e0..e3`) built AFTER the concepts formed, sensed,
re-consolidated: `e1` and `e2` both joined the chain concept
(`type:c0_c1[14]`, shared with a majority of the true chain members).
Concept identity judged by EXTENT, not label, per the closure law —
the re-mint gives a new type node name; the membership is what persists.

**Court 5 — containment.** Content doors 135 → 135. The inward sense
wrote only `meta`-tagged percept doors; the body it observed is unaltered.
(The OSI discipline: the meta layer is its own layer.)

**Court 6 — persistence.** Save/load: 15 type groups intact.

## What the edge cases said (reported, not gated)

- **The bridge got two type doors** — the cluster concept AND its own
  small group. A node genuinely of two kinds carried both memberships.
  Overlap-at-membership works; overlap-at-crystal-formation remains the
  closure doc's standing lattice gap. The bridge is the test case waiting.
- **The hub** landed in coarse piles rather than as a named outlier — a
  singleton cannot crystallize as a type of one. Hubs are sensed (its
  `od:16p` percept door exists, sole holder) but not yet CONCEPTED.
  Structural outliers need the outlier road, not the type road.
- **c10** (chain node whose successor is the chain end) sits in multiple
  groups — the bucket boundary is visible. Sense granularity is rung-1
  given-atoms, honestly.

## Honest limits

1. The sense vocabulary (degree buckets, reciprocity, neighbor-degree) is
   GIVEN, not grown — same rung-1 status as the signifier skeleton. The
   atoms-grown road (prediction-collapse boundaries, earned recurrence)
   applies here too.
2. Re-consolidation mints new type nodes rather than extending old ones —
   supersession by re-mint. Membership overlap carries identity; a
   supersession door between old and new type nodes would make the lineage
   explicit.
3. One structural altitude only. Laws about his own structure
   (`discover()` over structural magnitudes), the journal fed back as
   self-hearing, and the demotion of code self-judgments (hub ejection,
   `_flow_ok`) into HIS concepts are the named next rungs.

## Why this matters for the lane

This is the reflexive rung of the lens ladder running on the product
engine. The same move that let him see his own chain will let him see
"columns that behave alike," "laws that duplicate each other," and "flows
that keep losing" — the self-discovery the lens charter charters, entered
through the door the owner named: understand your own net first.

*Companion pieces: the token ruling and informational chemistry dialogue
(2026-07-31, filed in session memory, charter doc pending owner go);
DESIGN_LENS_LOOP_2026-07-30.md; the signifier corpus + courts (built,
run pending).*
