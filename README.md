# dc — the datacenter

**The LIVE repo.** Research happens in the sibling repo
(**isaac-paperclip**); dc is the live mirror of its blessed state —
everything Isaac needs to run in production, duplicated here by
promotion (owner model, 2026-07-12: "research and live").

> **EXPERIMENTAL STATUS (2026-07-15, owner note):** the 2026-07-13..15
> arc -- the language ladder (`experiments/` idea files), Gate 6
> operator mining, the instrument floor, the commonality miner, the
> audit face (`askisaac --audit`), word elections, the frontier habit,
> and the tools node -- is **UNFINISHED AND EXPERIMENTAL**. Engine
> organs passed their batteries and the full promotion gauntlet
> (engine 11/11 + 33 batteries), but broader TESTING IS THE NEXT STEP:
> adversarial probes, the data-ops bench harness (B1-B4), the verdict
> rewrite gates, and the interpreter-segment attack list are all
> still ahead. Treat these features as candidates, not products.

## The promotion rule

**Never hand-edit promoted files in dc.** Edit in research, then:

```
python promote.py
```

The script copies the research tree in, runs EVERY gate (engine suite
11/11 + all four store batteries) from dc against dc's own copies, and
refuses the promotion if anything fails (git restores the last good
live state). On success it stamps `PROVENANCE.md` with the research
commit, timestamp, and gate results — so what is live is always
traceable to what was proven. dc-only files (this README, promote.py,
PROVENANCE.md, .gitignore) are never overwritten.

Plain files and stdlib only. **No databases** (owner constraint). No pickle
for new saves. No hand-rolled crypto, ever.

## The roles (owner nomenclature, 2026-07-11)

| Role | What it is |
|---|---|
| **Prime** | Isaac himself — the full net, everything |
| **Orchestrator** | Prime's sole writer: queued edits → admission gate → journal → immutable packs |
| **Workers** | Do their own processing on cached snapshots; a leaf worker is a whole all-in-one Isaac with its security net |
| **Scrubbers** | Strip identifiable information on every intake path to Prime |
| **Client side** | Customer/user side: overlays, security nets, user-held data — where ALL personalization lives |

**The Identity Law:** Isaac understands the world; he does not identify
people or businesses. No datasets, no user info, no usage patterns flow to
Prime — telemetry included. Full design intent: `isaac-paperclip/docs/DESIGN_INTENT.md`.

## What's here

`isaac_store/` — the infrastructure, one module per organ:

- `codec.py` — complete + loud plain-data codec (cycles, aliasing, numpy;
  anything unencodable fails with the exact path, never a silent drop)
- `journal.py` — append-only JSONL event log; crash-tolerant tail
- `pack.py` — immutable indexed pack snapshots; partial loading is a
  property of the format (open the index, seek one node: measured 0.04 ms
  vs 5.3 s full load at 100k nodes)
- `orchestrator.py` — the sole writer; recovery invariant: latest pack +
  journal tail replays to exactly the live state
- `cache.py` — `LiveCache`, the Worker read side: 30-minute fixed recheck
  (owner-settled), delta invalidation, one pinned view per job
- `leaf.py` — `LeafIsaac`, the security-net worker: layered local-wins
  reads, graft with provenance, ephemeral / user-held modes, structurally
  no upstream channel
- `server.py` — `PrimeServer`, the orchestrator over HTTP (localhost-only
  by default; bearer token before any wider bind; refusals are answers;
  access log never records query strings or content)
- `bridge.py` — live Isaac instance ⇄ store records (per-node splitting)

## Running the batteries

Requires the engine: a sibling checkout of `isaac-paperclip` (or set
`ISAAC_ENGINE` to its `isaac.py`). Then:

```
cd isaac_store
python test_store.py     # 29 — codec/journal/pack/orchestrator/round-trip/migration
python test_cache.py     # 13 — recheck, pinning, staleness-cannot-corrupt
python test_leaf.py      # 13 — the quarantine proof (planted-secret byte-scan)
python test_server.py    # 17 — the doorway, auth, refusal contract
python scale_test.py     # measured size/speed numbers
```

On Windows set `PYTHONUTF8=1`. All claims in this repo are battery-backed:
settle design disputes by battery, not argument.

## Founding document

`docs/INFRASTRUCTURE_ARC.md` — the full plan: phases, batteries, the
orchestrator architecture, customer subnets, security nets, and the
build-state ledger.
