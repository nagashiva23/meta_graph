# Project Memory — MatGraphRAG

A running log of what has been built, changed, and decided in this repository.
Updated **after every PR**, so anyone (including a future contributor with zero
context) can read this file top-to-bottom and understand how the project got to
its current state without digging through git history.

**Convention:** newest entry first. Each entry records what changed, why it
changed, and anything it deliberately did *not* do.

---

## Current status at a glance

| Phase | Description | Status |
|---|---|---|
| **Phase 1** | Knowledge graph construction from Materials Project data | ✅ Complete |
| **Phase 1.5** | Robustness/scalability hardening before retrieval work | 🔨 In progress |
| Phase 2 | GraphRAG retrieval + LLM-generated, evidence-backed recommendations | ⬜ Not started |
| Phase 3 | CGCNN crystal-structure embeddings, structural similarity, element substitution | ⬜ Not started |

**Phase 1 deliverable as built and verified:** 1,504 nodes (794 Materials,
261 FormulaPatterns, 143 ChemicalSystems, 131 OxidationStates, 106 SpaceGroups,
46 Elements, 11 PropertyClasses, 7 CrystalSystems, 5 ApplicationDomains) and
14,681 edges, built from cached Materials Project data with **zero LLM calls**.

---

## PR log

### PR #1 — Tier 1 robustness hardening (branch `hardening/tier1-robustness`)

**Status:** in progress

**Why:** Phase 2's retrieval layer will be built directly on top of the graph
produced by `enrich.py`. Before adding an LLM on top, the graph-construction
path needed to stop being able to fail in silent or all-or-nothing ways. Four
concrete failure modes were identified in a code review of the Phase 1
codebase (see "Known gaps" below for the ones deliberately deferred).

**Changes:** documented per-commit as they land — see the commits on this
branch. Summary will be filled in when the PR is opened.

---

## Pre-history — Phase 1 initial build (22 commits, direct to `main`)

No pull requests were used during the initial Phase 1 build; all 22 commits
landed directly on `main`. They are grouped below by what they accomplished,
in chronological order, so this file still tells the whole story.

### 1. Scaffolding and planning
- `e0e730e` chore: scaffold Python project (uv, deps, env template)
- `c3e9d3b` docs: Phase 1 knowledge graph construction plan — `docs/PHASE1_PLAN.md`.
  **Note:** this document is now historical. It describes the original
  10-material seed-list design, which was superseded by the 8-cluster fetch
  (see commit `27589b2`). `README.md` is the authoritative current document.
- `cdeb685` feat: seed material set and classification config

### 2. Core pipeline
- `287a12e` feat: Materials Project fetch client — `mp_client.py`, REST client
  with retry/backoff plus a disk cache (`data/raw/<material_id>.json` +
  `_manifest.json`) so reruns are free and offline.
- `f44513d` feat: Cognee DataPoint graph schema — `schema.py`, the node types
  and their `identity_fields` (which is what makes repeated runs deduplicate
  onto the same nodes instead of creating copies).
- `54a103f` feat: deterministic enrichment (classification, rules, similarity) —
  `enrich.py`. **No LLM calls anywhere in this path**, by design.
- `defcf35` feat: Cognee pipeline wiring, build entrypoint, and verification —
  `pipeline.py`, `verify.py`, `scripts/build_graph.py`.

### 3. A design decision that was tried and reverted
- `e0f79f1` feat: add StructureType node and bucket density/formation_energy as nodes
- `256ee2a` **Revert** "feat: add StructureType node and bucket density/..."

  **Why it was reverted:** making every property (including raw floats like
  density) its own graph node was implemented and verified working, then
  deliberately backed out. Near-unique floats produce singleton value-nodes
  that connect to exactly one material each — they add no groupability. This
  established the project's core design rule: *continuous values stay as
  attributes; only things used for grouping become nodes.*

### 4. Scale-up: 10 materials → ~800
- `e7c4766` feat: full periodic-table element data via pymatgen — generates
  `config/elements.json` (103 elements) once, so the fetch path has no
  pymatgen dependency at request time.
- `27589b2` feat: chemistry-driven cluster fetching (10 → ~800 materials) —
  replaced the hand-listed seed materials with 8 chemistry-motivated clusters,
  chosen so materials genuinely share elements and structure (a densely
  interconnected graph) rather than forming isolated islands.
- `487858d` chore: ignore `.claude/` harness state directory

### 5. Fixes found while scaling up
- `38315df` fix: top-K nearest-neighbour similarity instead of flat threshold.
  A single global cosine cutoff (0.7) produced tens of thousands of
  "similar enough" pairs at a few hundred materials — noise, not signal.
  Switched to each material's top-5 nearest neighbours, which keeps edge
  density meaningful regardless of dataset size.
- `c6e8b14` fix: raise recursion limit for provenance stamping at scale.
  Cognee's provenance walk is recursive with no depth cap; once materials are
  cross-linked by hundreds of `similar_to` edges it exceeds Python's default
  1,000-frame limit. Fixed with `sys.setrecursionlimit(20000)`.
- `5a38cdc` fix: scale-appropriate verification output — `verify.py` capped
  its per-section sample printing instead of dumping hundreds of rows.

### 6. Widening the data
- `8a73a1e` feat: widen fetch to elastic/dielectric/oxidation-state/structure fields.
  All of these came from the **same** `/materials/summary/` endpoint already
  being called — widening `_fields=` added ~20 attributes and two new node
  types at **zero extra HTTP requests**.
- `7bb942f` feat: add OxidationState/FormulaPattern nodes and elastic/dielectric attrs
- `bae638e` feat: populate new nodes/attrs and redesign `similar_to` to 10 dimensions.
  The similarity vector grew from 4 to 10 fields, with every inclusion and
  exclusion justified explicitly in a comment block above `SIMILARITY_FIELDS`:
  intensive/size-normalized fields only, universal-scale fields only, and no
  redundant/derived fields.
- `d67a8e8` feat: verify new node types and elastic/dielectric coverage

### 7. Documentation
- `3d61f1d` docs: update README for the cluster-based dataset
- `fb6bb4c` docs: comprehensive Phase 1 project report
- `ac9aeb8` docs: update report for OxidationState/FormulaPattern and 10-field similarity

---

## Notable engineering lessons from Phase 1

Kept here because they are the kind of thing that is easy to re-learn the hard
way:

- **A 50-material spot-check badly overestimated data coverage.** An early
  sample suggested ~84% elastic and ~70% dielectric coverage; measured against
  all 794 materials the real numbers are **18.0%** and **19.9%**. The sample
  was drawn disproportionately from well-characterized battery-cathode
  compounds, which are fetched first. *Lesson: never extrapolate coverage from
  an early slice of a clustered fetch order.*
- **The Materials Project `exclude_elements` parameter caps at 60 characters**
  as a comma-joined string, and silently truncates. Fixed by passing it as
  repeated query parameters instead (`requests` does this for a list-valued param).
- **YAML 1.1 boolean coercion bit twice.** `No` (the element symbol for
  Nobelium) parses as boolean `False` unless quoted; `no`/`yes`/`on`/`off` are
  all reserved. Both instances are explicitly quoted in `materials.yaml`.
- **Cluster query batching bug.** An early version joined every chemical-system
  pattern in a cluster into one query and applied `limit_per_query` to the
  *combined* result, starving most sub-queries. Fixed by issuing one query per
  pattern, each with its own limit.
- **Raw-dict vs. DataPoint field-name mismatch.** The similarity vector is
  computed from *raw* MP JSON, where `bulk_modulus`/`shear_modulus` are nested
  `{voigt, reuss, vrh}` dicts and dielectric's key is `e_total` — not the
  flattened names used on the `Material` DataPoint. A naive
  `dict.get(field_name)` would have silently returned `None` for every
  material on those fields. Fixed with an explicit `_similarity_value()` shim.
- **Full crystal `structure` is cached but deliberately not a graph property.**
  It's needed for Phase 3's CGCNN so it's written to `data/raw/`, but it's a
  bulky nested lattice/coordinates blob — not meaningful to embed or traverse.
  The robocrys `description` already captures the same structural facts in
  embeddable text form.

---

## Known gaps (as of PR #1)

Tracked honestly rather than hidden. Ordered by the tier system used in the
hardening review:

**Tier 2 — testing**
- No `tests/` directory exists. It was planned in `docs/PHASE1_PLAN.md` and
  never built. There is zero automated coverage of the classification rules,
  application-domain rules, or similarity math.

**Tier 3 — scalability**
- Adding a new cluster to `materials.yaml` does not trigger a fetch for it;
  `fetch_and_cache_bulk()` only checks whether `_manifest.json` exists, so the
  only way to pick up a config change is `--refresh`, which re-fetches
  *everything*. Needs incremental, per-cluster fetching.
- Cluster fetch requests run sequentially, one HTTP call at a time.
- `compute_similarity_edges()` is a pure-Python O(n²) double loop (~630K pair
  computations at 794 materials). Fine now; should be vectorized with numpy
  before the dataset grows.

**Tier 4 — data coverage (directly affects Phase 2 answer quality)**
- Only 5 `ApplicationDomain` rules exist. Phase 2 retrieval can only answer
  questions the graph already encodes.
- `contains` edges carry no weight. `docs/PHASE1_PLAN.md` originally specified
  `weight = atomic fraction in formula`; the shipped code attaches a bare
  Element list with no edge properties.

**Tier 5 — Phase 2 preparation**
- No node type for logging retrieval interactions / recommendation provenance.
- Cognee's `index_fields` auto-embedding into LanceDB has never been exercised,
  so the vector-search fallback Phase 2 depends on is unverified.

**Tier 6 — operational**
- The pipeline has never been run in the current development environment (no
  `.venv/`, no `data/raw/`, no `.cognee_system/`). The graph statistics quoted
  above come from a previous run on a machine with a real `MP_API_KEY`.
