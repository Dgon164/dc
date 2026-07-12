# THE INFRASTRUCTURE ARC — 2026-07-11

Owner declared the next major shift: **persistence, structure processing,
batch processing, API calls, system/operational management, and partial
net loading for responses and processing.** This is the plan, checked
against DESIGN_INTENT.md before anything is written. Every phase ends in
a battery, not an argument.

---

## What exists today (measured 2026-07-11)

| Area | Current state |
|---|---|
| Persistence | `save()/load()/fork()` = whole-object pickle (isaac.py:1505–1529). No versioning, no schema, full deserialize always. Findings ledger = findings.jsonl (append-only, the good pattern). |
| Structure | One 7,252-line monolith (paperclip isaac.py). Net = 5 dicts + `preds` reverse index. Incremental compile exists (`_ensure_compiled`, dirty-tracking). **DIVERGENCE ALARM: isaac_project/code/isaac.py is 4,739 lines — 2,500 behind paperclip. The three-divergent-copies failure is recurring.** |
| Batch | `discover()` fully numpy; `observe()` one token-sequence at a time; `observe_matrix()` loops rows. Known knees: preds scan (fixed), block-perm null cost, composite width (capped). |
| API | stdlib HTTP server ui/isaac_ui.py, 11 routes on :8674 (status/teach/observe/ask/discover/apply/findings/save/load/new/fork). Older Flask app.py on :5050 = a divergent duplicate. Newer organs (recognize_glyph, observe_spatial, interaction_map, interrogate, information_ledger) NOT exposed. |
| Ops | findings.jsonl + RESULTS_LEDGER.md + 11/11 inline suite. No config, no structured logging, ~49 try/except total, print-based debug. |
| Partial loading | NONE. Load = everything in RAM. (Lazy *compilation* exists; lazy *loading* does not.) |

## Design-intent check (done before writing this plan)

- **Which organ already does this?** The findings **ledger** is the
  persistence pattern (append-only, provenance, rehydratable). The
  **reverse index** is the partial-loading pattern (a query's tokens
  point at exactly the doors it needs). The **stdlib UI server** is the
  API seed. Nothing here is a new parallel system — each area is an
  existing organ turned toward storage/serving.
- **Keep every measurement (#4):** the persistence schema must be a
  complete mirror of the Net — meta measurements, spreads, normals,
  door_counts, the wrong-section, supersession chains, adequacy
  dossiers. Pickle currently keeps everything *by accident*; the new
  format must keep everything *by design*. Battery: save→load→re-run
  the full suite → identical behavior.
- **Wrong kept (#6):** append-only journal makes supersession the
  natural write; migration NEVER deletes old minds; the Flask app is
  retired as superseded-with-provenance, not erased.
- **Measured, not hand-told (#3):** partial-load hop depth, cache
  sizes, batch chunk sizes — all chosen from measured curves, none
  guessed. No optimizing before a measured profile (the law).
- **Benchmark law (owner, 2026-07-11):** performance work is graded
  against the owner's ~3-second single-core trainer under a hard
  compute budget — his engine is contender #1, not an internal
  before/after.

---

## The orchestrator architecture (owner, 2026-07-11)

Owner's design: one **orchestrator**, workers do their own processing,
all updates/finished processes route through the orchestrator; live
pulls (reads) as needed; **all edits queued**.

```
                    ┌──────────── ORCHESTRATOR ────────────┐
                    │ owns the mind — SOLE WRITER           │
                    │ edit queue → ADMISSION GATE → journal │
                    │ → snapshot compaction → version stamp │
                    └────┬─────────────▲───────────────▲────┘
             read snapshots      queued proposed    results /
             ("live pulls")          edits          finished jobs
                    │                 │                 │
        ┌───────────┴──┐   ┌──────────┴───┐   ┌─────────┴────┐
        │ worker:      │   │ worker:      │   │ API front-end│
        │ discover /   │   │ perception / │   │ (:8674) —    │
        │ refine jobs  │   │ recognize /  │   │ reads live,  │
        │              │   │ consolidate  │   │ writes queue │
        └──────────────┘   └──────────────┘   └──────────────┘
```

Why queued edits is the RIGHT call, not a compromise: in Isaac every
state change is already a *proposal that must pass admission* (null,
rivals, supersession). The edit queue is that gate made operational —
workers expand freely, the orchestrator compresses by test. One
writer also means the journal is a total order (crash-safe, replayable,
no torn state), and IMMUTABLE snapshot files give every worker
consistent live pulls *while* the orchestrator writes: a reader opens
the latest published snapshot (+ tails the journal if it wants to be
current) and can never see torn state — zero locking, ever.

**Storage constraint (owner, 2026-07-11): NO traditional database
systems** — owner has had major scaling problems with SQLite and is
averse to any DB. Plain files only, stdlib only.

Rules of the design:
- **Sole writer.** Nothing touches the Net or journal except the
  orchestrator applying queued edits in order.
- **Edits are events with provenance** — which worker, which evidence,
  which snapshot version it read. A worker's in-flight proposals are
  UNCONFIRMED until admitted (it may be behind the queue; that's fine —
  admission is the arbiter, not the worker).
- **Rejected edits are wrong-kept** with their why — queue rejections
  are ledger material like everything else.
- **Conflicts resolve by existing organs** (reuse checklist #1):
  commutative door-count increments merge trivially; contradictory
  claims are routed into RIVALS, never overwritten; duplicates
  supersede with provenance.
- **The orchestrator stays THIN** — queue + admission + journal +
  compaction + job registry. Gatekeeper, not thinker: all discovery,
  perception, and consolidation compute happens in workers on
  snapshots. Hand-built stays only the epistemics.

Phase impact: the orchestrator's write path IS Phase 1 (queue →
admission → journal → snapshot); it becomes its own module in the
Phase 2 split; workers are how Phases 3–5 consume it (batch jobs and
API requests are just workers/front-ends on the same queue).

---

## Nomenclature (owner, 2026-07-11 — the official role names)
- **PRIME** — Isaac himself: the full net, everything. (Earlier
  sections say "core"; Prime is the name.)
- **ORCHESTRATOR** — Prime's sole writer.
- **WORKERS** — processing on snapshots; a leaf worker is a whole
  all-in-one Isaac with its security net.
- **SCRUBBERS** — the strip-identity workers, a first-class role on
  EVERY intake path to Prime, not only the customer share path.
- **CLIENT SIDE** — customer/user side: overlays, security nets,
  user-held data, and ALL personalization.

## The Identity Law (owner, same message)
Isaac understands the world; he does not identify people or
businesses. Prime holds concepts and laws, never identity dossiers.
Public-facing information is usable for processing; user-submitted
information is reserved client-side and scrubbed before anything
crosses. **No datasets, no user info, no usage patterns flow to
Prime** — including telemetry: usage analytics stay client-side, and
Prime's ops ledger describes Prime's own operation, never user
behavior. Enforcement layers: structural tier quarantine + scrubbers
on every Prime intake + identity-shaped-content refusal at Prime's
admission gate (a certified-never at the gate itself).

## Round two of owner design (2026-07-11, same day)

### Customer subnets — the privacy quarantine (containerization)
Owner: customer deployments hold **local subnets that never feed back
to Isaac** — customer-specific, not used for education or anything
else; identifiable information lives there and only there.

Design — isolation is STRUCTURAL, not filtered:
- A customer container = engine + **core net as a read-only layer**
  (a pinned core snapshot pack) + the customer's own **overlay**: its
  own journal, its own packs, its own orchestrator. Reads see
  core+overlay merged (overlay wins on conflict, supersession-style);
  ALL writes land in the overlay.
- **No path exists** from a customer overlay to the core queue — the
  core orchestrator is unreachable from customer containers. This is
  not a policy check that could have bugs; the files and endpoints
  simply aren't there. PII never enters education because education
  pipelines never possess the overlay files.
- Sync is **one-way, core → customer**: an overlay tracks a pinned
  upstream snapshot version and upgrades deliberately (like a private
  git branch rebasing on upstream; it can never push).
- Reuse check: this is the existing **fork() organ made persistent** —
  a fork that lives in its own container and tracks upstream.
- **The share path (owner):** customer-controlled, opt-in. With share
  ON, an overlay discovery goes to the core orchestrator for
  evaluation → routed to a dedicated **scrubber worker** that strips
  all identifiable information and keeps only the general abstraction
  (equation/relation/door structure + measured stats — never rows,
  names, or literal customer values) → the scrubbed abstraction enters
  the normal edit queue and faces normal admission (nulls, rivals).
  Mechanics: shares land in a QUARANTINE INTAKE, never the Net; raw
  submissions are not retained core-side — kept = abstraction + scrub
  report (categories/counts stripped, not content). If the scrubber
  cannot cleanly separate abstraction from identity it REFUSES the
  share back to the customer — refusal is an answer, no partial
  admission. Share OFF (default): no path exists at all.
- Scrubber battery: planted identifiable details in synthetic shares →
  strip rate must be 100/100 with abstraction utility preserved
  (complete numbers on both sides); refusal triggers verified on
  ambiguous plants.
- Container layering literally mirrors the design: read-only base
  image layer (core pack) + writable volume (overlay). One image,
  many customers, no shared writable state.

### Worker caching + the recheck interval
Owner: workers live-load anything from the net, cache it locally, and
operate on the cache. **Recheck interval SETTLED by owner: FIXED at
30 minutes** — the workers are the ones producing the updates, not
the net, so they're never surprised by their own edits and the net
between contributions moves slowly. (Assistant's three-layer
heartbeat/delta/backstop rival superseded, kept below for the record;
revisit only if a measured staleness cost bites.)
**"The longer Isaac runs the less anything should be updated."**

- Immutable packs make caching trivially safe: a cached record is
  valid-as-of snapshot vN *by construction* — there is no way for it
  to be silently wrong, only behind.
- Recheck is a **delta, not a reload**: each worker keeps a journal
  watermark (last event sequence seen); recheck = pull the journal
  tail past the watermark and invalidate only touched nodes. The
  owner's law is exactly what append-only gives us: as Isaac matures,
  the tail between rechecks shrinks toward empty, so rechecks get
  cheaper for free.
- **Measured, not hand-told:** 30 min is the starting default, then
  the interval is set from measured churn. Per-node churn (count
  deltas, supersessions per unit time) is a measurement KEPT ON THE
  NODE (checklist #4) — a stability score. Old heavily-evidenced nodes
  cache effectively forever; young nodes recheck sooner. The cache
  policy reads the node's own evidence; nothing is guessed.
- Battery: staleness experiment — measure answer-delta between cached
  and current at 1×/10×/100× the interval, at different mind ages;
  the curve picks the interval, complete numbers.

### Security nets — the third tier (owner, 2026-07-11, later message)
User-side specific nets, one level below customer subnets: data that
Isaac processes, learns from, and answers from — but that is pullable
ONLY from the worker holding it. Never stored upstream, never shared.

The architecture becomes ONE mechanism at three scales:
```
core net            (shared knowledge; the education pipeline)
  └─ customer overlay   (per-customer container; own orchestrator;
                          never pushes up except the opt-in scrubbed
                          share path)
        └─ worker security net  (per-user/per-session; the worker IS
                                 a classic all-in-one Isaac at the
                                 leaf; NO upstream channel at all)
```
- Reads layer one-way: local ∪ customer ∪ core, local wins; a shared
  tier can never reach down into a local net.
- Writes: the worker runs its own mini-orchestrator over its own net
  (the "second orchestrator"); its queue simply goes nowhere else.
  Not even the scrubbed share path — security nets are total.
- Partial loading makes the leaf cheap: the local Isaac pulls only
  the shared nodes a question touches (0.04 ms each, measured) and
  doors its private evidence onto them locally.
- Two persistence modes: EPHEMERAL (RAM-only; evaporates at session
  end) or USER-HELD (persisted only on the user's side, encrypted,
  key user-side). Which is default = product decision, owner's call.
- **Logging is a leak channel:** ops events from a security-net
  worker carry counts and timing only, never content. Same for error
  reports — a stack trace with user data in it is a breach.
- Battery: prove no byte of security-net content appears in any
  upstream file (journal, packs, ops ledger, logs) after a full
  process-learn-answer cycle; prove layered reads resolve local-wins;
  prove session-end evaporation in ephemeral mode.

### Working nets — small enough, capable enough
Owner: the full net will be big (how big is unknown); working nets
must stay small but capable.

- Two mechanisms, both already designed: **demand paging** (index-only
  load + seek, Phase 3) bounds RAM regardless of net size; **working-
  set extraction** bounds the whole deployed artifact — a distilled
  subnet = the task's closure × measured usage/evidence thresholds.
  Deployment is the compress stroke of expand/compress.
- The information_ledger reports what was kept vs left behind:
  "this N-MB working net keeps X% of full-net capability on task Y" —
  complete numbers, both sides of the cut stated.
- The full net never shrinks — extraction produces a deployable child
  (wrong-kept applies: what was left out is listed, not lost).
- Battery: working-net size vs capability curve on MNIST/HIGGS/glyphs;
  report the knee.

---

---

## The phases (ordered by dependency, each with its battery)

### Phase 1 — Persistence v2 (FIRST, because it unblocks everything)
Pickle couples every saved mind to the exact class layout, so the
module split (Phase 2) would break every existing .isaac file. Fix the
format first.
- **Schema = the Net as plain data.** nodes, doors, tags, counts, meta
  (every kept measurement), findings, composites, lockouts, wrong-
  section, supersession chains — nothing dropped.
- **Two layers, both reusing the ledger organ:**
  1. **Journal** — append-only JSONL of events (door admitted, count
     incremented, node superseded, finding recorded). The findings
     ledger generalized to ALL state changes. Crash-safe, diffable,
     human-readable.
  2. **Snapshot** — periodic compaction into one IMMUTABLE indexed
     pack file (`mind.v<N>.isaacpack` — NO database; owner constraint):
     layout = header | node→(offset,length) index | packed node
     records (a node's record carries its doors, tags, counts, meta,
     wrong-section, supersession links). The loader reads header +
     index only (tiny), then seeks or mmaps individual records on
     demand — this IS Phase 3's partial loading, no engine underneath.
     Published by write-new-then-atomic-rename; a new snapshot
     SUPERSEDES the old (kept per retention policy, never overwritten
     in place — the wisdom laws as a file layout). Recovery invariant:
     latest snapshot + journal tail replays to current state.
- **Versioning:** every save stamps engine version + schema version;
  `load()` keeps reading old pickles (one-way migration, original kept).
- **Battery:** round-trip → 11/11 suite identical; old-mind migration
  → identical; journal replay == snapshot (the accounting: two roads,
  one state).
- **BUILD STATE 2026-07-11 (second pass): WIRED INTO THE ENGINE,
  battery 29/29, engine suite 11/11.** One home now:
  `isaac-paperclip/isaac_store/` (isaac_project copy removed).
  Engine gained `save_pack()`/`Isaac.load_pack()` beside the old
  pickle pair; manual updated same step (both copies).
  **Scale measured** (synthetic, `scale_test.py`): 100k nodes/500k
  doors → pack 141.5 MB vs pickle 27.1 MB (5.2×), save 10.1 s vs
  0.5 s, full load 5.3 s vs 0.4 s, **single-node partial fetch
  0.04 ms — 133,000× faster than full load** (index open 28 ms).
  Verdict: partial loading vindicated; pack size/save-time is the
  measured limit, next lever = per-record compression (NOT yet — no
  optimizing before the limit bites). Known v1 limits (wrong-kept):
  cross-record aliasing not preserved; orchestrator queue in-process
  only.

### Phase 2 — Structure processing (the package split)
- `isaac/` package: substrate (Net/Scratch/World), discovery
  (_Enumerator + discover/confirm/refine), perception, recognition,
  consolidation, interaction, persistence, serve. A thin top-level
  `isaac.py` façade keeps every experiment script importing unchanged.
- **Kills the divergence problem structurally:** isaac_project imports
  the paperclip package instead of carrying its own copy. One body.
- **Battery:** 11/11 + spatial 13/13 + a sample of RESULTS_LEDGER
  scripts reproduce their recorded numbers, before and after the split.

### Phase 3 — Partial net loading (the reverse index turned toward disk)
- Recall pages in only what a query touches: observed tokens/features →
  their doors (pack-index lookup: name → offset, then seek/mmap) →
  the concept nodes they reach →
  those nodes' nevers, variants, normals, meta → bounded closure.
  Everything else stays on disk. Responses need the shortlist's
  neighborhoods, not the whole mind.
- **Measured, not hand-told:** closure depth and cache size from
  measured latency/accuracy curves. Hard rule: partial-load answers
  must equal full-load answers, or the delta is reported as a complete
  number — never silently approximate.
- **Why it matters:** this is the Raspberry-Pi line of the mission — a
  mind bigger than RAM on cheap hardware.
- **Battery:** recognition/apply_law identical partial vs full; peak
  RAM and latency reported at 1×/10×/100× current mind size.

### Phase 4 — Batch processing
- `observe_batch()` — vectorized door counting over arrays of
  sequences, one compile per batch (the dirty-tracking machinery
  already supports this). Batch recognize / batch apply_law.
- **Profile first** (no optimizing before measured limits): measure
  where time actually goes at MNIST/HIGGS scale, attack only the top
  knee.
- **Battery:** graded vs the owner's ~3 s single-core trainer under a
  hard compute budget (pin its exact number in the head-to-head);
  complete numbers, wall-clock + accuracy together.

### Phase 5 — API
- **One canonical server** — grow ui/isaac_ui.py (stdlib, zero
  dependencies, Pi-friendly). Flask app retired as superseded.
- Expose ALL organs: recognize_glyph, observe_spatial, interaction_map,
  interrogate/graduate, information_ledger, law_confidence, batch
  endpoints. Stable JSON contract; **refusals are answers** — an
  abstention returns a structured verdict with its diagnosis, never an
  error code.
- Open question for the owner: "API calls" inbound (programs drive
  Isaac — this phase) vs outbound (Isaac calls an LLM as the
  gated-English mouth, the Omega tier). Inbound is the prerequisite
  either way; outbound is a later door on the same nodes.

### Phase 6 — System / operational management
- **Ops ledger** — the same ledger organ, universally applied: every
  operation may append an ops event (timing, counts, sizes, errors)
  to ops.jsonl. "Ledgers are not a findings tool."
- **Runtime wrong-kept:** errors filed with their why, superseded when
  fixed — the wisdom law applied to the system itself.
- **Defaults table with provenance:** existing hardcoded constants
  (max_terms=16, n_perm=40, pct=95, max_context=7…) become a visible
  table, each entry stamped with how it was measured or flagged
  hand-told (debt to be measured). No new hand-told numbers.
- **Every save stamps** engine version + suite result.

### Phase 7 — Customer subnets & containerization
- Overlay minds: core pack read-only + customer journal/packs +
  per-container orchestrator; merged reads, overlay-only writes;
  one-way core→customer sync with pinned, deliberate upgrades.
- Container image: engine + core pack as read-only layer; overlay on
  a writable volume. PII exists only inside the customer volume.
- Working-set extraction tooling: build a small-but-capable core pack
  per deployment tier (Pi / laptop / server), each shipped with its
  information_ledger statement of kept-vs-left-behind capability.
- **Battery:** prove the quarantine — no byte of overlay content
  reachable from core-side tooling; upgrade a pinned core under a
  live overlay and show customer answers stay consistent; capability
  curves per working-net tier.

---

## Immediate hygiene (before Phase 1)
1. isaac_project repo is dirty since 07-04 (ISAAC_MANUAL.md modified,
   3 untracked dirs) — commit or file it.
2. isaac_project/code/isaac.py (4,739 lines) vs paperclip (7,252):
   declare paperclip authoritative NOW, mark the project copy
   superseded (kept, wearing its defeat).
3. Project-root DESIGN_INTENT.md copy was stale (missing the usage law
   + relative law) — resynced 2026-07-11.
