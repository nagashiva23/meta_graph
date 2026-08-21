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
| **Phase 1** | Knowledge graph construction from Materials Project data | ✅ Complete, and **independently reproduced** on 2026-08-21 |
| **Phase 1.5** | Robustness/scalability hardening before retrieval work | 🔨 Tiers 1, 2, 4, 6 done; Tiers 3 and 5 open |
| **Phase 2** | GraphRAG retrieval + LLM-generated, evidence-backed recommendations | 🔨 Step 1 (deterministic retrieval) done in PR #3; query parsing and generation remain |
| Phase 3 | CGCNN crystal-structure embeddings, structural similarity, element substitution | ⬜ Not started |

**Current graph (after PR #2):** 1,511 nodes (794 Materials, 261
FormulaPatterns, 143 ChemicalSystems, 131 OxidationStates, 106 SpaceGroups,
46 Elements, 12 ApplicationDomains, 11 PropertyClasses, 7 CrystalSystems) and
15,112 edges, built from cached Materials Project data with **zero LLM calls**.

`README.md` was updated to these figures alongside PR #2. The review deck
(`matgraphrag_presentation.tex`, maintained in Overleaf outside this repo) was
updated to match at the same time — schema, results, work-completed and
conclusion slides, plus a new slide on symmetry-based screening. Its traversal
figures (756 O-containing, 80 Li-stable, 143 wide-gap, O²⁻ in 695, Ti⁴⁺ in 107,
44 ABC3) were re-checked against the rebuild and all still hold unchanged.

---

## PR log

### PR #3 — Phase 2, step 1: deterministic retrieval (branch `feat/phase2-retrieval`)

**Why built this way:** Phase 2 splits cleanly into a deterministic half (turn
constraints into candidates plus the evidence backing each) and a generative
half (write prose from that evidence). Building the deterministic half first
means that by the time a language model is involved, everything it is shown has
already been verified against the graph — and no `LLM_API_KEY` was needed to
make real Phase 2 progress.

**`src/matgraph/retrieval.py`** — the query layer.

- **Strict conjunctions, not soft scoring.** Every returned material satisfies
  every constraint. A caller asking for stable lithium cathodes must not receive
  an unstable sodium one ranked 7th. Ranking is a plain sort on a stored value
  (`energy_above_hull` by default), so the ordering is as explainable as the
  filtering — no learned or blended score.
- **Every hit carries its evidence**: the specific edges and attribute values
  that made it match, including the stored rule text for an application domain
  and the atomic fraction on a `contains` edge. Only facts *relevant to the
  query* are collected — dumping all ~19 edges a material has would bury the
  reason it was returned, and this list is intended to be the sole context the
  generation step is given.
- **Traversal runs inwards.** Edges point Material → Element, but a query asks
  "which materials contain iron?", so retrieval starts at the Fe node and walks
  incoming edges. Elements are hubs, so this touches one node and its edge list
  instead of scanning all 794.
- **Crystal system is a real 2-hop walk** (Material → SpaceGroup →
  CrystalSystem). There is no direct edge, because the space group is the finer
  fact and the crystal system is derived from it; storing both directly would
  let them contradict each other.
- **Unknown entities raise rather than return nothing.** "No material contains
  Unobtainium" and "you misspelled Uranium" are different answers; conflating
  them misleads whatever consumes retrieval.

**`src/matgraph/graph_store.py`** — extracted from `verify.py`, which held the
only copy of the graph index and the edge-property reader. Retrieval needs both
and a second copy would drift. Refactor verified behaviour-preserving:
`verify.py` produces identical output against the real graph. It also
centralises the two cognee storage quirks that silently produce wrong answers
(stringified `properties`, string-typed `weight`).

**`scripts/eval_retrieval.py` — the piece that actually earns trust.** Unit
tests prove the traversal logic against a hand-built graph, but they cannot
prove the *real* graph was built correctly, because both the graph and the
expectation would come from the same code. So the harness answers each query
twice: once by traversing the built graph, and once by filtering
`data/raw/*.json` in plain Python, never touching cognee or `enrich.py`. That
second route is an independent oracle over the same Materials Project records
the graph was built from.

**All 14 queries agree exactly** — including the symmetry-screened domains and
the 2-hop crystal-system walk:

| Query | Graph | Oracle |
|---|---|---|
| materials containing lithium | 245 | 245 |
| stable lithium-containing materials | 80 | 80 |
| wide-band-gap materials | 143 | 143 |
| battery cathode candidates | 237 | 237 |
| piezoelectric candidates | 130 | 130 |
| ferroelectric candidates | 94 | 94 |
| photovoltaic absorbers | 22 | 22 |
| materials containing Ti⁴⁺ | 107 | 107 |
| ABC3 perovskite pattern | 44 | 44 |
| cubic materials (2-hop) | 87 | 87 |
| cubic oxides, gap > 2 eV | 19 | 19 |

This validates the whole chain — fetch → enrich → build → store → traverse.

**One investigation worth recording.** Querying Cubic and Tetragonal both
returned exactly 87 materials, which looked like a bug where a lookup was
matching the wrong node. It was genuine coincidence: summing materials across
all seven crystal systems gives exactly 794 and space groups exactly 106, so
every material and space group is accounted for once. Separately, Ti⁴⁺ ∩ Cubic
returned 0, which the oracle confirmed is a true answer — none of the 107 Ti⁴⁺
materials are cubic.

**`scripts/query_graph.py`** — CLI over the same calls the generation step will
make, so whatever the model eventually says can be checked against what this
prints. Sanity results: `--similar-to BaTiO3` returns Ti₃PbO₇, Ba₅Nb₄O₁₅ and
Sr₂Ta₂O₇ (all perovskite-family oxides), and `--connects LiFePO4 Fe2O3` returns
shared Fe, O, the stable class and O²⁻.

**Tests: 132 → 170.** The synthetic graph in `tests/test_retrieval.py`
deliberately reproduces cognee's storage quirks rather than using tidied-up
fixtures — tests against clean fixtures would pass while the real retrieval
path returned nothing.


### PR #2 — Tier 4: domain coverage, symmetry screening, weighted edges (branch `feat/tier4-domain-coverage`)

**Why:** Phase 2 retrieval can only answer questions the graph already encodes.
With five application domains, a query about anything else had nowhere to land.
This widens what can be asked — entirely from fields already cached, with no
additional API calls.

**Application domains: 5 → 12.** Counts measured against the real 794-material
set, so none is dead weight:

| Domain | Materials | Basis |
|---|---|---|
| `piezoelectric` | 130 | non-centrosymmetric symmetry + non-metallic |
| `thermoelectric` | 100 | narrow gap 0.1–0.8 eV |
| `ferroelectric` | 94 | polar space group + non-metallic |
| `uv_transparent` | 33 | gap ≥ 4 eV, non-magnetic |
| `high_k_dielectric` | 33 | dielectric constant ≥ 20, gap > 2 eV |
| `photovoltaic_absorber` | 22 | **direct** gap 1.0–1.8 eV |
| `hard_structural` | 19 | shear ≥ 100 GPa and bulk ≥ 150 GPa |

The two symmetry rules are the most rigorous in the project: piezoelectricity
and ferroelectricity are *forbidden by the crystal's point group* as a matter of
physical law, not by a fitted threshold. `config/space_groups.json` (generated
by `scripts/generate_space_groups.py`) classifies all 230 space groups, and its
counts match the crystallographic literature exactly — 92 centrosymmetric, 68
polar, 32 point groups — including the **432 exception**, the one
non-centrosymmetric class whose symmetry still forces every piezoelectric
tensor component to vanish.

Spot-checked against materials whose behaviour is well known: BaTiO₃ comes out
`ferroelectric` **and** `piezoelectric`, AlN/BN/GaN `piezoelectric`, Al₂O₃
(sapphire) `uv_transparent` and `hard_structural`. Those are the right answers.

**Deliberately NOT added: `solid_electrolyte`.** It looked like an obvious
complement to `battery_cathode`, but the data refused it. The dataset holds only
**3** Li materials with no transition metal, so the rule selects 1 material —
and dropping that condition makes it overlap `battery_cathode` 51 of 52. This is
a **dataset** gap, not a rule-design gap: the 8 clusters were built around
cathodes and never fetch electrolyte chemistries (Li-garnets, LISICON,
sulfides, halides). Shipping it would have been a relabelling presented as new
capability. Logged below as a fetch-config gap.

**Rules restructured into a predicate registry.** Previously five hardcoded
`if` statements in Python, with the human-readable rule text in config —
meaning the description and the logic could drift apart silently. Adding seven
more would have made that worse. Now every threshold lives in
`config/materials.yaml` beside the text describing it, and
`_assert_rules_match_config()` fails the build if config and code disagree
about which domains exist. A configured domain with no predicate never fires;
a predicate with no config entry would produce a `suitable_for` edge carrying
no rule text — an edge asserting something with no stored justification, which
is the one thing this graph must never do. *The refactor was verified
behaviour-preserving: all 84 pre-existing tests passed unchanged.*

**`contains` edges now carry `weight` = atomic fraction.** Specified in
`docs/PHASE1_PLAN.md` originally, never implemented. Without it the
`LiFePO₄ → O` and `LiFePO₄ → Li` edges are indistinguishable, though oxygen is
4/7 of the atoms and lithium 1/7 — so "materials where lithium is a major
constituent" was unanswerable. An unparseable formula produces unweighted edges
rather than an invented number. Verified in the built graph: all 2,269 edges
weighted, none unweighted.

**Resulting graph:** 1,511 nodes (was 1,504 — seven new `ApplicationDomain`
nodes) and 15,112 edges (was 14,681), with `suitable_for` growing 1,506 → 1,937.
All structural checks pass. **These numbers supersede the ones in `README.md`
and the review deck, which still describe the 5-domain graph.**

**Two findings worth keeping:**

- **Cognee does not store edge properties where you would look for them.** An
  `Edge`'s `properties` dict comes back as a *stringified* Python dict under a
  single `"properties"` key, so the rule text sits one level down and
  serialised: `{"properties": "{'rule': '0 < band_gap <= 3.5 eV...'}"}`. A plain
  `edge["rule"]` returns `None` for every edge — which looks exactly like the
  evidence was never stored. Also: `weight` comes back as a **string**, not a
  float, and cognee synthesises an `edge_text` field
  (`"GaAs suitable for 0 < band_gap <= 3.5 eV and not metallic."`) that is
  embeddable. **Phase 2's retrieval layer must know all three of these.**
  Handled by `_edge_rule()` in `verify.py`.
- **`is_metal` is the right test, not `band_gap > 0`.** An early estimate of the
  piezoelectric count was one too high. K₆Ta₁₁O₃₀ has `band_gap = 0.0002 eV` but
  `is_metal = True`; a material with a numerically-tiny gap flagged metallic is
  a metal, and would screen out the internal field. Using MP's own `is_metal`
  determination handles this correctly where a raw threshold does not. It is
  the only such material in the 794.

**Also:** `verify.py` now asserts that **every** `suitable_for` edge carries its
rule text (1,937/1,937 pass) — the project's central explainability claim,
checked on every build rather than assumed. The `battery_cathode` section also
now prints the stored rule, which it previously claimed to show but did not.
New `scripts/verify_graph.py` runs verification against the existing graph
without a full rebuild.

**Tests: 84 → 132.**

---

### PR #1 — Tier 1 robustness hardening (branch `hardening/tier1-robustness`)

**Why:** Phase 2's retrieval layer will be built directly on top of the graph
produced by `enrich.py`. Before putting an LLM on top of it, the
graph-construction path needed to stop failing in all-or-nothing ways, and
verification needed to be able to actually fail. Four failure modes were fixed;
higher tiers were deliberately deferred (see "Known gaps").

**1. One unclassifiable value no longer aborts the whole build** (`enrich.py`)
`_bucket()` raised `ValueError` when a value matched no configured bucket, so a
single malformed material killed the entire run. The realistic trigger is a
`NaN` band gap — `NaN` fails every comparison, falls through all buckets, and
hits the raise. It now returns `None`; the caller logs a warning and skips just
that one classification, matching the convention `classify_ordering` already
used. All existing bucket boundaries verified unchanged.

**2. Incomplete cache records are skipped, not crashed on** (`enrich.py`)
Core fields were read by direct indexing (`raw["band_gap"]`), so one record
missing a key raised `KeyError` and aborted the build. Records are now
validated up front and skipped with a warning naming the material and the
missing fields.

*Deliberately skipped rather than defaulted:* substituting `0.0` for an absent
band gap would classify that material as a metal, and `0.0` for an absent
`energy_above_hull` would classify it as perfectly stable — silently corrupting
every downstream query instead of failing visibly.

*Two subtleties worth remembering:* validation runs **before**
`compute_similarity_edges`, so a `similar_to` edge can never point at a
material that was later skipped (which would `KeyError` when edges are wired
up). And missing-ness is tested with `is None`, not falsiness — otherwise
`band_gap = 0.0` (every metal) and `is_metal = False` would be read as missing
and dropped from the graph.

**3. Comments that contradicted the code** (`materials.yaml`, `mp_client.py`,
`enrich.py`) — `similar_to` was still described as a 4-field vector (it has
been 10 since the elastic/dielectric data landed), and two files still cited
the old, sampling-biased ~84%/~70% coverage estimate instead of the measured
18.0%/19.9%. Comments only, no behaviour change.

**4. Verification actually fails now** (`verify.py`)
It printed `!!` for problems but only asserted that at least one Material
existed — so a build with isolated materials or missing space groups exited 0
and looked successful. Checks are now collected and raise, so
`scripts/build_graph.py` exits non-zero. Added the checks the README already
*claimed* were running: Material count matches the fetch manifest (never
actually verified before), every Material has exactly one `has_space_group`
edge, and at least one `classified_as` edge.

**5. Tier 2 — unit tests** (`tests/`, 84 tests, run in 0.03s)
Covers bucket boundaries (including the tie-break where overlapping buckets
share endpoints), all five application rules either side of their thresholds,
the `similar_to` invariants, and the validation layer. Tests load the **real**
`config/materials.yaml`, so changing a threshold or rule text there fails a
test — the config is as much a part of the behaviour as the Python is.

Two guards against future drift: every `REQUIRED_RAW_FIELDS` entry is asserted
to actually be enforced, and each fired rule's text is asserted to match
`materials.yaml`.

*One test failed on first run, and it was the test's fault, not the code's:*
`_cosine([1,2], [-1,-2])` returns `-0.9999999999999998`, and asserting exact
float equality was simply the wrong assertion. Fixed with `pytest.approx`.

**6. First end-to-end pipeline run** — see "Verification run" below.

---

## Verification run — 2026-08-21 (first end-to-end run of this codebase)

Until this point the pipeline had **never been executed in a development
environment**; the graph statistics in `README.md` and the review deck came
from an earlier run on another machine. This run rebuilt everything from
scratch (`uv sync` → `fetch_materials.py` → `build_graph.py`) against the live
Materials Project API.

**Every published figure reproduced exactly.**

| Quantity | Published | This run |
|---|---|---|
| Materials | 794 | 794 |
| Total nodes | 1,504 | 1,504 |
| Total edges | 14,681 | 14,681 |
| Elements | 46 | 46 |
| Chemical systems | 143 | 143 |
| Formula patterns | 261 | 261 |
| Oxidation states | 131 | 131 |
| Space groups | 106 | 106 |
| Elastic coverage | 18.0% | 18.0% (143/794) |
| Dielectric coverage | 19.9% | 19.9% (158/794) |
| `suitable_for` edges = domain sum | 1,506 | 1,506 |

The fetch pulled 1,692 raw docs across the 8 clusters and deduplicated to 794
unique materials by formula. This makes the Phase 1 numbers **reproducible from
a clean checkout**, not just self-reported.

**All five structural checks passed**, including the manifest check added in
PR #1 that had never actually been running:

```
OK: 794 Material nodes present.
OK: every Material has at least one contains edge (no isolated materials).
OK: Material count matches the fetch manifest exactly (794).
OK: every Material has exactly one space group.
OK: every Material has at least one property classification.
```

**What the Tier 1 fixes did on real data: nothing — which is the right
outcome.** Zero of 794 records were skipped as incomplete, and zero values were
unclassifiable. The fixes are insurance against future data, not silent
behaviour changes to the current build.

**Two previously-unknown things now confirmed:**
- **The vector store is real and usable.** `Material_description` and
  `Material_formula` tables in LanceDB each hold 794 rows of 384-dimensional
  non-zero embeddings, generated locally by `fastembed` with no API key. This
  had never been verified, and Phase 2's vector-search fallback depends on it.
- **Data coverage details:** robocrys descriptions exist for 792/794 (so the
  missing-description fallback path is genuinely exercised), oxidation states
  for 739/794 (93.1%), and the full crystal `structure` blob for **794/794** —
  meaning Phase 3's CGCNN input is completely cached already.

Artifacts produced (all gitignored): `data/raw/` 16 MB, `.cognee_system/`
65 MB, `artifacts/graph.html` 18 MB.

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

**Tier 2 — testing** — ✅ done in PR #2 (84 tests)

> **Correction to a claim made in PR #1.** That PR logged a "blocker": that
> `enrich.py` can't be imported for testing because it pulls in `cognee` via
> `schema.py`, and suggested the module would need splitting first. That was
> overstated. It was only true because dependencies weren't installed at the
> time — the workaround (extracting functions via `ast`) was a symptom of an
> empty environment, not of the module's design. Once `uv sync` ran, `matgraph`
> imports directly as an editable install and the test suite needed no
> restructuring at all. Recorded here rather than deleted, because "the fix is
> to restructure the module" would have been wasted work.

**Tier 3 — scalability**
- Adding a new cluster to `materials.yaml` does not trigger a fetch for it;
  `fetch_and_cache_bulk()` only checks whether `_manifest.json` exists, so the
  only way to pick up a config change is `--refresh`, which re-fetches
  *everything*. Needs incremental, per-cluster fetching.
- Cluster fetch requests run sequentially, one HTTP call at a time.
- `compute_similarity_edges()` is a pure-Python O(n²) double loop (~630K pair
  computations at 794 materials). Fine now; should be vectorized with numpy
  before the dataset grows.

**Tier 4 — data coverage** — mostly done in PR #2
- ~~Only 5 `ApplicationDomain` rules exist.~~ ✅ Now 12.
- ~~`contains` edges carry no weight.~~ ✅ Now weighted by atomic fraction.
- **Still open — the cluster set has chemistry blind spots.** Discovered while
  designing `solid_electrolyte`: the dataset contains only 3 Li materials
  without a transition metal, because all 8 clusters were designed around
  cathodes, semiconductors and oxides. Whole material families are simply
  absent — solid electrolytes (Li-garnets, LISICON, sulfides, halides),
  nitrides beyond III–V, chalcogenide thermoelectrics. Fixing this means new
  clusters in `materials.yaml` and a re-fetch, which costs real API calls.
  Worth doing before Phase 2 if retrieval is meant to answer broadly.
- **Still open — bucket granularity.** `wide_gap` is everything above 3 eV,
  with no finer split. Phase 2 must therefore answer precise numeric questions
  ("band gap between 5 and 6 eV") by filtering the raw `band_gap` attribute on
  the `Material` node, not by traversing to a `PropertyClass` node. That is a
  design decision for the retrieval layer, not necessarily a bucket change.

**Tier 5 — Phase 2 preparation**
- No node type for logging retrieval interactions / recommendation provenance.
- **Next for Phase 2** (in order): natural-language query → `Constraints`
  parsing (the first step needing an `LLM_API_KEY`); vector-search fallback over
  `Material_description` for free-text remainders; then evidence-grounded
  generation, which is deliberately last so every input to the model is already
  verified.
- ~~Cognee's `index_fields` auto-embedding into LanceDB is unverified.~~
  ✅ Resolved by the verification run: 794 rows of 384-dim non-zero embeddings
  in both `Material_description` and `Material_formula`.

**Tier 6 — operational** — ✅ resolved
- ~~The pipeline has never been run in the current development environment.~~
  Run end-to-end on 2026-08-21; every published figure reproduced exactly. See
  "Verification run" above.
