# MatGraphRAG

**An Explainable Graph-Based Retrieval-Augmented Reasoning System for Intelligent Materials Discovery**

> Phase 1 Report — Knowledge Graph Construction Pipeline
> Status: **Phase 1 complete** · Phase 2 (GraphRAG retrieval) and Phase 3 (CGCNN structural embeddings) not yet started

---

## Table of Contents

1. [Overview](#1-overview)
2. [Project Status](#2-project-status)
3. [Dataset](#3-dataset)
4. [Graph Architecture](#4-graph-architecture)
5. [Construction Pipeline](#5-construction-pipeline)
6. [How Materials Interconnect](#6-how-materials-interconnect)
7. [How Materials Are Grouped](#7-how-materials-are-grouped)
8. [Verified Graph Statistics](#8-verified-graph-statistics)
9. [Example Graph Traversals](#9-example-graph-traversals)
10. [Project Structure](#10-project-structure)
11. [Getting Started](#11-getting-started)
12. [Engineering Notes](#12-engineering-notes)
13. [Roadmap](#13-roadmap)
14. [Tech Stack](#14-tech-stack)

---

## 1. Overview

MatGraphRAG is a decision-support system for materials discovery: it turns the
[Materials Project](https://materialsproject.org) database into a knowledge graph
so that materials, their properties, their crystal chemistry, and their
application domains become traversable and explainable, rather than isolated
rows in a table. The eventual goal (Phases 2–3) is graph-based retrieval and
LLM-generated recommendations, where every claim in an answer is traceable to
a concrete graph edge or database record.

This document reports on **Phase 1**: constructing that knowledge graph. No
LLM is involved anywhere in this phase — every node, edge, and classification
is produced by deterministic rules applied to data pulled directly from the
Materials Project API, using [Cognee](https://github.com/topoteretes/cognee)
as the graph/vector storage and pipeline layer.

## 2. Project Status

| Phase | Description | Status |
|---|---|---|
| **Phase 1** | Knowledge graph construction from Materials Project data | ✅ Complete |
| Phase 2 | GraphRAG retrieval + LLM-generated, evidence-backed recommendations | Not started |
| Phase 3 | CGCNN crystal-structure embeddings, structural similarity search, element substitution | Not started |

Phase 1 deliverable, as built and verified: a graph of **794 materials**
spanning **46 elements**, **106 space groups**, **143 chemical systems**,
**131 oxidation states**, **261 stoichiometric formula patterns**, and
**12 application domains**, interconnected by **15,112 edges** — built entirely
from cached, reproducible Materials Project data with zero LLM calls.

Every figure in this document was **reproduced from a clean checkout** on
2026-08-21 (fetch → build → verify against the live Materials Project API),
not carried over from an earlier run.

## 3. Dataset

### Source

All data comes from the [Materials Project REST API](https://api.materialsproject.org)
(`/materials/summary/` for computed properties, `/materials/robocrys/` for
human-readable structure descriptions). Materials Project currently indexes
**154,377 materials** in total.

### Why not the full catalog

The full catalog was deliberately not used for Phase 1. Two reasons:

- **Material–material similarity is quadratic.** Computing pairwise similarity
  over 154,377 materials means ~11.9 billion pairs — not slow, mathematically
  intractable in this design. Full-scale structural similarity is Phase 3's
  job (CGCNN embeddings + an approximate-nearest-neighbour index), not
  Phase 1's.
- **A random or exhaustive pull produces a sparse, disconnected graph.** The
  design goal for Phase 1 is a *densely interconnected* graph — materials
  that actually share elements, structure types, and application relevance —
  not a flat catalog dump.

### Sampling strategy: chemistry-driven clusters

Instead, the dataset is assembled from eight **chemistry-driven clusters**,
each targeting a materials-science domain and each defined declaratively in
[`config/materials.yaml`](config/materials.yaml). A cluster expands into one
or more Materials Project queries (by element composition or exact chemical
system, always sorted by stability), and results are deduplicated **globally
across all clusters** by formula — keeping the lowest-energy-above-hull
(most stable) polymorph whenever a formula is reachable from more than one
cluster.

| Cluster | Domain | Query pattern | Stability cutoff |
|---|---|---|---|
| `battery_cathode_oxide` | Li–transition-metal oxides (layered/spinel cathodes) | Li + O + {Fe, Co, Mn, Ni, V, Ti, Cr, Zn, Cu, Sc, Nb, Mo}, ≤3 elements | E<sub>hull</sub> ≤ 0.08 eV/atom |
| `battery_cathode_phosphate` | Li–transition-metal phosphates (olivines) | Li + P + O + {Fe, Mn, Co, Ni, V, Ti, Cr}, ≤4 elements | ≤ 0.08 eV/atom |
| `semiconductor_iii_v` | III–V semiconductors | {B, Al, Ga, In} × {N, P, As, Sb} binaries | ≤ 0.08 eV/atom |
| `semiconductor_ii_vi` | II–VI semiconductors | {Zn, Cd, Hg} × {O, S, Se, Te} binaries | ≤ 0.08 eV/atom |
| `semiconductor_group_iv` | Group IV elemental/binary semiconductors | Si, Ge, C, Sn + binary combinations | ≤ 0.15 eV/atom |
| `oxide_binary` | Wide-gap and functional binary oxides | O + 33 candidate elements, exactly 2 elements | ≤ 0.08 eV/atom |
| `perovskite_oxide` | ABO₃ perovskites (dielectric/ferroelectric) | {Ba, Sr, Pb, Ca, K, Na} × {Ti, Zr, Nb, Hf, Sn, Ta} + O | ≤ 0.08 eV/atom |
| `magnetic_spinel_oxide` | AB₂O₄ spinel-type magnetic oxides | {Fe, Co, Ni, Mn, Cr} × {Fe, Co, Ni, Mn, Cr, Zn, Ti} + O | ≤ 0.08 eV/atom |

A global `exclude_elements` list drops radioactive and synthetic elements
(Tc, Pm, Po, At, Rn, Fr, Ra, and the actinides) from every query, since they
add no scientific value to a discovery-oriented graph.

**Result:** 8 clusters → ~2,000 raw candidate documents → **794 unique
materials** after cross-cluster deduplication.

### Properties captured

All of the following come from a single `/materials/summary/` query per
cluster sub-pattern — no additional API endpoints or requests are needed,
since Materials Project exposes elastic, dielectric, oxidation-state, and
structural data as extra fields on the same summary document.

| Category | Fields | Coverage (of 794) |
|---|---|---|
| Core (band gap, stability, density, magnetism, symmetry, …) | `band_gap`, `formation_energy_per_atom`, `energy_above_hull`, `density`, `total_magnetization`, `ordering`, `symmetry`, `theoretical`, … | 100% |
| Electronic structure | `cbm`, `vbm`, `efermi`, `is_magnetic`, `num_magnetic_sites` | 100% |
| **Elastic** (PRD: "Elastic Properties") | `bulk_modulus` (VRH), `shear_modulus` (VRH), `universal_anisotropy`, `homogeneous_poisson` | **18.0%** (143/794) |
| **Dielectric** (PRD: "Dielectric Properties") | `e_total`, `e_electronic`, `e_ionic`, refractive index `n` | **19.9%** (158/794) |
| **Oxidation states** (PRD: "Oxidation States") | `possible_species` → `OxidationState` nodes | 93.1% (739/794) |
| Full crystal structure | `structure` (lattice + atomic positions) — cached to disk for Phase 3, not added to the graph (see §12) | 100% |

Elastic and dielectric tensors require an extra DFPT calculation Materials
Project hasn't run for every entry, so real coverage is much lower than core
properties — this was measured directly against the full 794-material set,
not estimated (an earlier spot-check on 50 materials suggested ~84%/~70%
coverage, which turned out to be a sampling-bias artifact; see §12).

## 4. Graph Architecture

### Design principle

> Continuous numeric values stay as **attributes** on the `Material` node
> (exact values, needed for filtering and comparison). Anything used for
> **grouping or traversal** becomes its own **node**. Derived material–material
> relationships become **weighted edges**.

This was a deliberate, revisited decision: an earlier iteration turned every
property (including raw floats like density) into its own node type. That
was reverted — singleton value-nodes for near-unique floats add graph
complexity without adding groupability, which is the entire point of a
knowledge graph over a flat table. `band_gap` and `energy_above_hull`
*are* additionally represented as categorical nodes (`PropertyClass`, see
below) precisely because their **bucketed** form is what's actually shared
and traversable — the raw float isn't. Elastic and dielectric properties
follow the same rule: they stay as attributes (§3), since bucketing them
usefully would need domain-specific thresholds not yet defined, whereas
`possible_species` and `formula_anonymous` *are* inherently categorical
(a small, shared vocabulary) and became nodes directly — no bucketing needed.

### Node types

| Node type | Represents | Identity (dedup key) | Count |
|---|---|---|---|
| `Material` | One Materials Project entry (ground-state polymorph per formula, per cluster) | `material_id` | 794 |
| `Element` | A chemical element and its periodic properties | `symbol` | 46 |
| `SpaceGroup` | A crystallographic space group | `symbol` + `number` | 106 |
| `CrystalSystem` | One of the 7 crystal systems (cubic, hexagonal, …) | `name` | 7 |
| `ChemicalSystem` | A material's element set as a system (e.g. `Fe-Li-O-P`) | `chemsys` | 143 |
| `PropertyClass` | A categorical bucket a material falls into (band-gap class, stability class, magnetic ordering) | `kind` + `name` | 11 |
| `ApplicationDomain` | A materials-science use case (battery cathode, piezoelectric, photovoltaic absorber, …) | `name` | 12 |
| `OxidationState` | An ionic species (e.g. `Fe2+`, `O2-`) a material contains | `species` | 131 |
| `FormulaPattern` | Anonymized stoichiometric pattern (e.g. `ABC3` for perovskites) | `pattern` | 261 |

Every shareable node type declares an identity key, so re-running the
pipeline — or two different materials both containing Oxygen — resolves to
the **same graph node** rather than creating duplicates. This identity
mechanism is what makes the graph's hub structure possible (see §6).

### Relationship types

| Edge | Direction | Meaning | Weight / properties |
|---|---|---|---|
| `contains` | Material → Element | Material's constituent elements | `weight`: atomic fraction (O is 4/7 of LiFePO4, Li 1/7) |
| `has_space_group` | Material → SpaceGroup | Crystallographic space group | — |
| `member_of` | Material → ChemicalSystem | Which chemical system it belongs to | — |
| `classified_as` | Material → PropertyClass | Band-gap / stability / magnetic-ordering bucket | — |
| `suitable_for` | Material → ApplicationDomain | Which application rule(s) it satisfies | `rule`: the exact rule text that fired (explainability) |
| `similar_to` | Material → Material | Property-vector similarity to its nearest neighbours | `weight`: cosine similarity (0–1) |
| `crystal_system` | SpaceGroup → CrystalSystem | Which crystal system a space group belongs to | — |
| `has_oxidation_state` | Material → OxidationState | Ionic species present in the material | — |
| `has_formula_pattern` | Material → FormulaPattern | Which stoichiometric pattern it matches | — |

## 5. Construction Pipeline

```
config/materials.yaml (clusters, thresholds, rules)
        │
        ▼
┌───────────────────────┐
│ 1. FETCH               │  mp_client.py
│  - one query per       │  → Materials Project /materials/summary/
│    cluster sub-pattern │  → sorted by energy_above_hull, capped per query
│  - dedupe by formula    │
│    across all clusters │
│  - batch-fetch robocrys│  → /materials/robocrys/ (up to 100 material_ids/call)
│    structure text      │
└──────────┬─────────────┘
           ▼
  data/raw/<material_id>.json × 794  +  _manifest.json   (cached; reproducible)
           │
           ▼
┌───────────────────────┐
│ 2. ENRICH (no LLM)     │  enrich.py
│  - classify band_gap,  │  → PropertyClass nodes
│    stability, ordering │
│  - apply domain rules  │  → ApplicationDomain edges + fired-rule text
│  - compute similar_to  │  → top-K nearest neighbours per material
│    (cosine, z-scored)  │
│  - build DataPoint     │  → Material/Element/SpaceGroup/... objects
│    graph objects       │     (schema.py)
└──────────┬─────────────┘
           ▼
┌───────────────────────┐
│ 3. WRITE TO COGNEE     │  pipeline.py
│  Task(build_materials) │  → cognee's low-level pipeline
│  Task(add_data_points) │  → writes graph store + vector store together,
│                        │     deduplicating on each node's identity_fields
└──────────┬─────────────┘
           ▼
  Cognee-managed stores: embedded Kuzu (graph) + LanceDB (vectors) + SQLite (metadata)
           │
           ▼
┌───────────────────────┐
│ 4. VERIFY              │  verify.py
│  - node/edge counts    │  → reads the graph directly via get_graph_data(),
│  - structural queries  │     no LLM / no natural-language search needed
│  - visualization       │  → artifacts/graph.html
└────────────────────────┘
```

The entry point is [`scripts/build_graph.py`](scripts/build_graph.py), which
runs steps 2–4 (step 1 is a separate script since fetching is the expensive,
rate-limited part and should be cached independently of graph rebuilds).

## 6. How Materials Interconnect

Interconnection isn't a side effect here — it's what the identity-key +
shared-node design in §4 produces directly. Concretely, in the built graph:

- **Elements act as hubs.** Every material's `contains` edges point at the
  *same* `Element` node for a given symbol. Oxygen, for instance, is one
  node — but **756 of the 794 materials** (95%) point to it, meaning any two
  oxide materials are two hops apart via Oxygen regardless of how different
  their other elements are.
- **Chemical systems group materials by exact composition.** All ternary
  Ba–Ti–O compounds converge on one `ChemicalSystem` node, distinct from
  binary Ba–O materials — this is a finer-grained grouping than individual
  elements.
- **Crystal symmetry groups by structure family.** 106 distinct space groups
  collapse onto just 7 `CrystalSystem` nodes, so e.g. every cubic material —
  regardless of composition — is reachable from every other cubic material
  in two hops.
- **PropertyClass nodes group by behavior, not composition.** A `wide_gap`
  material and another `wide_gap` material may share no elements at all, but
  are still one hop apart through the same bucket node.
- **Oxidation states group by ionic chemistry.** `O2-` alone connects **695
  of 794 materials** (87%); `Li+` connects 227. Two materials with no shared
  elements can still be one hop apart if they share an oxidation state —
  this is also the anchor Phase 3's ionic-substitution model will reason over.
- **Formula patterns group by stoichiometric shape, independent of
  composition.** `BaTiO3` and `CaHfO3` share nothing element-wise but are
  both `ABC3` — 44 materials share that one node. 261 distinct patterns
  cover 794 materials, so this dimension is denser than `ChemicalSystem`
  (143) but sparser than `Element` (46), a genuinely different grouping axis.
- **`similar_to` edges add a direct, weighted shortcut.** Rather than a flat
  similarity threshold (which stops scaling once the dataset is larger than
  a handful of materials — see §12), each material links to its **top-5**
  nearest neighbours by cosine similarity over a **10-dimensional**, z-scored
  property vector spanning electronic (`band_gap`), thermodynamic
  (`formation_energy_per_atom`, `energy_above_hull`), structural-density
  (`density`), magnetic (`total_magnetization`), mechanical (`bulk_modulus_vrh`,
  `shear_modulus_vrh`, `universal_anisotropy`, `homogeneous_poisson`), and
  dielectric (`dielectric_total`) behavior — keeping only pairs with
  similarity ≥ 0.6. Fields with partial coverage (elastic, dielectric) are
  mean-imputed rather than dropped, so a missing value contributes neutrally
  instead of skewing the comparison (see §12 for which fields were
  deliberately left out, and why). This produced 3,964 directed `similar_to`
  edges (2,708 unique undirected pairs) in the current graph.

The net effect: **no material in the graph is an isolated island.** Every
material has at least a `contains` edge (verified structurally on every
build — see §8), and the shared-node design means the graph's connectivity
grows *faster* than its material count as clusters overlap in elements and
chemistry.

## 7. How Materials Are Grouped

Grouping happens through three independent, fully deterministic mechanisms.
The first two are configured in [`config/materials.yaml`](config/materials.yaml),
so every bucket/rule decision is inspectable and traceable to a concrete
threshold; the third comes directly from categorical fields Materials
Project already provides, with no thresholds to define.

### 7.1 Property classification (buckets)

| Property | Buckets | Thresholds |
|---|---|---|
| **Band gap** | `metal` / `narrow_gap` / `semiconductor` / `wide_gap` | 0 / 0–1 / 1–3 / >3 eV (DFT-scale; Materials Project band gaps are systematically underestimated relative to experiment, so thresholds are tuned for DFT values, not experimental ones) |
| **Stability** | `stable` / `metastable` / `unstable` | E<sub>hull</sub> ≤ 1×10⁻⁶ / ≤ 0.05 / > 0.05 eV/atom |
| **Magnetic ordering** | one `PropertyClass` node per distinct value Materials Project reports (FM, AFM, FiM, …) | — (skipped when reported as "Unknown") |

### 7.2 Application domain rules

Each rule fires independently — a material can satisfy several domains at
once — and the exact rule text that fired is stored **on the edge itself**
(`suitable_for.rule`), so a recommendation downstream can point at precisely
why a material was included rather than asserting it opaquely.

Each rule's thresholds live in `config/materials.yaml` **beside the text that
describes them**, and the predicate implementing it lives in
`APPLICATION_RULE_PREDICATES` in `enrich.py`. A startup check
(`_assert_rules_match_config`) fails the build if the two disagree about which
domains exist — a configured domain with no predicate would never fire, and a
predicate with no config entry would produce a `suitable_for` edge carrying no
rule text.

| Domain | Rule |
|---|---|
| `battery_cathode` | Contains Li **and** a transition metal, **and** E<sub>hull</sub> ≤ 0.05 eV/atom |
| `semiconductor_device` | 0 < band gap ≤ 3.5 eV **and** not metallic |
| `photocatalyst` | Contains O **and** 1.5 ≤ band gap ≤ 3.5 eV |
| `dielectric` | Band gap > 2 eV **and** |total magnetization| < 0.1 (non-magnetic) |
| `magnetic_material` | Ordering ∈ {FM, FiM} **and** |total magnetization| ≥ 0.1, **or** ordering = AFM |
| `photovoltaic_absorber` | **Direct** band gap, 1.0 ≤ gap ≤ 1.8 eV, **and** not metallic |
| `piezoelectric` | Non-centrosymmetric space group (excluding point group 432) **and** not metallic |
| `ferroelectric` | Polar space group **and** not metallic |
| `thermoelectric` | 0.1 ≤ band gap ≤ 0.8 eV **and** not metallic |
| `uv_transparent` | Band gap ≥ 4 eV **and** |total magnetization| < 0.1 |
| `hard_structural` | Shear modulus ≥ 100 GPa **and** bulk modulus ≥ 150 GPa (VRH) |
| `high_k_dielectric` | Dielectric constant ≥ 20 **and** band gap > 2 eV |

The transition-metal set used by `battery_cathode` is: Sc, Ti, V, Cr, Mn, Fe,
Co, Ni, Cu, Zn, Y, Zr, Nb, Mo, Tc, Ru, Rh, Pd, Ag, Cd.

**Symmetry rules are categorically stronger than threshold rules.**
`piezoelectric` and `ferroelectric` do not use a tuned cutoff at all:
piezoelectricity is *forbidden* in a centrosymmetric crystal, and
ferroelectricity requires a polar point group — both as a matter of physical
law, not correlation. `config/space_groups.json` (generated by
`scripts/generate_space_groups.py`) classifies all 230 space groups; its counts
match the crystallographic literature exactly — 92 centrosymmetric, 68 polar,
32 point groups — and it encodes the **432 exception**, the one
non-centrosymmetric class whose symmetry still forces every piezoelectric
tensor component to vanish.

These rules screen for what symmetry *permits*: BaTiO₃ correctly comes out both
`ferroelectric` and `piezoelectric`, and AlN, BN and GaN come out
`piezoelectric` — but a qualifying material is a candidate worth examining, not
a confirmed device.

Note also that the metallic test uses Materials Project's own `is_metal` flag
rather than `band_gap > 0`. One material in this set, K₆Ta₁₁O₃₀, has a band gap
of 0.0002 eV yet is flagged metallic — a numerically-tiny gap is still a metal,
and a conductor screens out the internal field these rules depend on.

Note the `battery_cathode` rule's 0.05 eV/atom cutoff is intentionally
*tighter* than the 0.08 eV/atom used to fetch cluster candidates in the
first place — the fetch stage casts a slightly wider net so metastable
near-misses are still present in the graph (reachable via `contains`,
`member_of`, etc.), even when they don't pass the stricter domain-membership
bar.

### 7.3 Categorical fields (no thresholds needed)

Two Materials Project fields are already categorical, so they group
materials directly rather than through a bucketing rule:

- **Oxidation states** (`possible_species`) — e.g. `BaTiO3` groups with every
  other material containing `Ti4+`, independent of what else those materials
  contain.
- **Formula patterns** (`formula_anonymous`) — e.g. every `AB2C4`-pattern
  material groups together regardless of which elements A, B, and C actually
  are, capturing stoichiometric shape (spinel-like 1:2:4 ratios, in this case)
  as a grouping dimension distinct from both composition and symmetry.

## 8. Verified Graph Statistics

Numbers below are from the current build (`uv run python scripts/build_graph.py`),
read directly out of Cognee's graph store — not estimated.

**Nodes — 1,511 total**

| Type | Count |
|---|---|
| Material | 794 |
| FormulaPattern | 261 |
| ChemicalSystem | 143 |
| OxidationState | 131 |
| SpaceGroup | 106 |
| Element | 46 |
| ApplicationDomain | 12 |
| PropertyClass | 11 |
| CrystalSystem | 7 |

**Edges — 15,112 total**

| Relationship | Count |
|---|---|
| `similar_to` | 3,964 |
| `has_oxidation_state` | 2,319 |
| `contains` | 2,269 |
| `classified_as` | 2,135 |
| `suitable_for` | 1,937 |
| `has_formula_pattern` | 794 |
| `has_space_group` | 794 |
| `member_of` | 794 |
| `crystal_system` | 106 |

**Application domain distribution** (a material can belong to more than one; sums to 1,937, matching the `suitable_for` edge count exactly)

| Domain | Materials | Qualifying rule |
|---|---|---|
| `semiconductor_device` | 474 | 0 < gap ≤ 3.5 eV, not metallic |
| `magnetic_material` | 328 | FM/FiM with \|magnetization\| ≥ 0.1, or AFM |
| `photocatalyst` | 249 | contains O, 1.5 ≤ gap ≤ 3.5 eV |
| `battery_cathode` | 237 | contains Li + a transition metal, E_hull ≤ 0.05 |
| `dielectric` | 218 | gap > 2 eV, non-magnetic |
| `piezoelectric` | 130 | non-centrosymmetric space group (excl. 432), not metallic |
| `thermoelectric` | 100 | narrow gap 0.1–0.8 eV, not metallic |
| `ferroelectric` | 94 | polar space group, not metallic |
| `uv_transparent` | 33 | gap ≥ 4 eV, non-magnetic |
| `high_k_dielectric` | 33 | dielectric constant ≥ 20, gap > 2 eV |
| `photovoltaic_absorber` | 22 | **direct** gap 1.0–1.8 eV, not metallic |
| `hard_structural` | 19 | shear ≥ 100 GPa and bulk ≥ 150 GPa |

The last two rules depend on dielectric and elastic data, which exist for only
~20% and ~18% of these materials — so absence of one of those edges means
*"not known"*, never *"not suitable"*.

**Structural integrity checks** (asserted on every build, in `verify.py` — the
build exits non-zero if any fails):

- ✅ Every `Material` node has at least one `contains` edge (no disconnected materials)
- ✅ 794 `Material` nodes present, matching the fetch manifest exactly
- ✅ Every `Material` has exactly one `has_space_group` edge
- ✅ Every `Material` has at least one `classified_as` edge
- ✅ All 1,937 `suitable_for` edges carry the rule text that fired — the
  explainability claim, checked rather than assumed

## 9. Example Graph Traversals

These are real outputs from `verify.py` against the current graph, not
illustrative pseudo-queries:

- **Oxygen hub** — 756 of 794 materials trace back to the same `O` node.
- **"Which Li-containing materials are thermodynamically stable?"** → 80
  materials (`Li2O`, `Li2MnO3`, `Li(CoO2)2`, …), found by intersecting
  `contains → Li` with `classified_as → (stability, stable)`.
- **"What connects LiFePO4 and Fe2O3?"** → shared `Element` nodes `Fe` and
  `O`, found in two hops each.
- **"Which materials are wide-band-gap?"** → 143 materials, top of the list
  led by `Al2O3` (5.87 eV) and `SiO2` (5.68 eV).
- **"Which materials are battery cathode candidates?"** → 237 materials,
  each with the exact rule text that qualified it stored on its edge.
- **"What connects BaTiO3 to CaHfO3 despite sharing no elements?"** → both
  match the `ABC3` `FormulaPattern` node (44 materials total).
- **"Which materials contain Ti4+?"** → 107 materials, found in one hop via
  `has_oxidation_state`.
- **Nearest-neighbour similarity** — e.g. `KNb13O33 ↔ NaNb13O33` at cosine
  similarity 1.000 over the full 10-dimensional property vector (near-identical
  behavior despite being different formulas/polymorphs).

## 10. Project Structure

```
meta_graph/
├── config/
│   ├── materials.yaml      # clusters, buckets, domain rules + thresholds, similarity config
│   ├── elements.json       # full 103-element periodic table (generated, see below)
│   └── space_groups.json   # symmetry class of all 230 space groups (generated)
├── data/raw/                # cached MP records, one JSON per material_id (gitignored)
│   ├── <material_id>.json
│   └── _manifest.json
├── src/matgraph/
│   ├── schema.py            # Cognee DataPoint node/edge type definitions
│   ├── mp_client.py         # Materials Project REST client, cluster query engine
│   ├── enrich.py            # classification rules, similarity, DataPoint graph builder
│   ├── pipeline.py          # Cognee low-level pipeline wiring
│   └── verify.py            # structural verification against the built graph
├── scripts/
│   ├── fetch_materials.py   # step 1: fetch + cache clusters from Materials Project
│   ├── build_graph.py       # steps 2-4: build, write to Cognee, visualize, verify
│   ├── verify_graph.py      # re-run verification alone, without a full rebuild
│   ├── generate_elements.py     # regenerates config/elements.json via pymatgen
│   └── generate_space_groups.py # regenerates config/space_groups.json via pymatgen
├── tests/                   # 132 unit tests (uv run pytest)
├── artifacts/graph.html     # interactive graph visualization (gitignored, regenerable)
├── MEMORY.md                # running project changelog, updated after every PR
├── docs/PHASE1_PLAN.md      # original design research and planning document
└── .cognee_system/          # Cognee's embedded graph/vector/relational stores (gitignored)
```

## 11. Getting Started

### Setup

```bash
uv sync
cp .env.example .env    # fill in MP_API_KEY
```

### Run

```bash
# 0. Only needed if config/elements.json needs regenerating for new elements
uv run python scripts/generate_elements.py

# 1. Fetch + cache the cluster-defined material set from Materials Project
uv run python scripts/fetch_materials.py

# 2. Build the graph in Cognee, render a visualization, run verification
uv run python scripts/build_graph.py
```

`scripts/build_graph.py` prunes and rebuilds from scratch by default; pass
`--no-reset` to add on top of existing state, or `--skip-visualize` to skip
the HTML render (useful for quick verification-only reruns).

No LLM API key is required to build the graph — embeddings use a local
`fastembed` model. An `LLM_API_KEY` becomes necessary only for Phase 2's
natural-language retrieval layer.

## 12. Engineering Notes

A running log of non-obvious decisions and fixes made while building this,
kept here rather than scattered across commit messages:

- **Node-vs-attribute design was tested both ways.** An iteration that made
  every property (including raw floats like density) its own graph node was
  implemented, verified working, then deliberately reverted — see §4's
  design principle. Singleton value-nodes don't add groupability for
  near-unique floats; only their bucketed form does.
- **Flat similarity thresholds don't scale.** The original design used a
  single global cosine cutoff (0.7) for `similar_to` edges. At a few hundred
  materials over only 4 dimensions, that produced tens of thousands of
  "similar enough" pairs — noise, not signal. Switched to top-K nearest
  neighbours per material (K=5), which keeps edge density meaningful
  regardless of dataset size.
- **Cognee's provenance-stamping walk is recursive with no depth cap.**
  Once materials are cross-linked via hundreds of `similar_to` edges, that
  walk exceeds Python's default 1,000-frame recursion limit. Fixed by
  raising `sys.setrecursionlimit(20000)` before invoking the pipeline.
- **The Materials Project `exclude_elements` parameter caps at 60
  characters as a comma-joined string** — silently rejects a full
  radioactive/synthetic element exclusion list if joined naively. Fixed by
  passing it as repeated query parameters instead (`requests` does this
  automatically for a list-valued param).
- **YAML 1.1 boolean coercion bit twice**: `No` (element symbol for
  Nobelium) parses as the boolean `False` unless quoted; `no`/`yes`/`on`/`off`
  are all reserved words. Both instances are now explicitly quoted in
  `materials.yaml`.
- **Cluster query batching bug**: an early version joined every chemical
  system pattern in a cluster into one comma-separated query and applied
  `limit_per_query` to the *combined* result — meaning 16 III-V binary
  systems shared a single limit of 10, starving most of them. Fixed by
  issuing one query per chemical-system pattern, each with its own limit.
- **Robocrystallographer descriptions batch-fetch**: confirmed the
  `/materials/robocrys/` endpoint accepts a comma-separated list of
  `material_ids`, cutting ~794 individual calls down to ~8 batched ones.
- **Elastic/dielectric/oxidation-state/structure data all come from the same
  endpoint already being queried** — widening `_fields=` on the existing
  `/materials/summary/` calls added ~20 new attributes and two new node
  types at zero extra HTTP requests. No new endpoints were needed.
- **Full crystal `structure` is cached but deliberately not a graph node
  property.** It's fetched (needed for Phase 3's CGCNN) and written to
  `data/raw/<material_id>.json`, but never passed into a `Material` node:
  it's a bulky nested lattice/coordinates blob, not something meaningful to
  embed or traverse the way scalar attributes are — and the robocrys
  `description` field already captures the same structural facts in
  embeddable text form, which is what actually serves LLM-facing relational
  reasoning.
- **A 50-material spot-check overestimated elastic/dielectric coverage.**
  An early sample suggested ~84% elastic and ~70% dielectric coverage; the
  real number across all 794 materials is 18.0% and 19.9% respectively (see
  §3). The sample happened to be drawn disproportionately from
  well-characterized battery-cathode compounds fetched first. Lesson: don't
  extrapolate data coverage from an early slice of a clustered fetch order —
  measure against the full set.
- **`similar_to`'s similarity vector was redesigned from 4 to 10 fields**
  once elastic/dielectric/thermodynamic data became available, with each
  inclusion and exclusion justified explicitly (see the comment block above
  `SIMILARITY_FIELDS` in `enrich.py`): intensive/size-normalized fields only
  (excludes `volume`, `nsites`, `num_magnetic_sites`, which are confounded by
  unit-cell choice), fields on a universal scale only (excludes `cbm`/`vbm`/
  `efermi`, referenced to each calculation's own internal zero), and no
  redundant fields (excludes `is_magnetic`, which duplicates
  `total_magnetization`; excludes `e_electronic`/`e_ionic`/refractive index,
  which are derived from/correlated with `dielectric_total`). Missing values
  in the partially-covered fields (elastic, dielectric) are mean-imputed
  rather than dropped, so a material with no elastic data still gets
  meaningful `similar_to` edges from its other 8 dimensions.
- **Raw-dict vs. DataPoint field name mismatch caught before it silently
  broke similarity.** The similarity vector is computed from *raw* MP JSON
  (before `Material` objects exist), where `bulk_modulus`/`shear_modulus`
  are nested `{voigt, reuss, vrh}` dicts and dielectric's raw key is
  `e_total` — not the flattened `bulk_modulus_vrh`/`dielectric_total` names
  used on the `Material` DataPoint. A naive `dict.get(field_name)` would
  have silently returned `None` for every material on these fields. Fixed
  with an explicit `_similarity_value()` extraction shim rather than
  renaming fields to match (the raw JSON shape isn't ours to change).

### Notes from the Tier 4 coverage pass

- **A domain was designed, measured, and then deliberately dropped.**
  `solid_electrolyte` looked like the obvious complement to `battery_cathode`,
  but this dataset holds only **3** Li materials with no transition metal — so
  the rule selects 1 material, and relaxing it makes the domain overlap
  `battery_cathode` 51 of 52. That is a *dataset* gap, not a rule-design gap:
  all 8 clusters were built around cathodes, semiconductors and oxides, and
  none fetch electrolyte chemistries (Li-garnets, LISICON, sulfides, halides).
  Shipping it would have been a relabelling presented as new capability.
- **Cognee does not store edge properties where you would look for them.** An
  `Edge`'s `properties` dict comes back from the graph store as a *stringified*
  Python dict under a single `"properties"` key, so the rule text sits one level
  down and serialised: `{"properties": "{'rule': '0 < band_gap <= 3.5 eV...'}"}`.
  A plain `edge["rule"]` returns `None` for every edge — which is
  indistinguishable from the evidence never having been stored. `weight` also
  comes back as a **string**, not a float, and cognee synthesises an embeddable
  `edge_text` field. All three matter for Phase 2's retrieval layer; the reader
  is `_edge_rule()` in `verify.py`.
- **`is_metal` is the correct test, not `band_gap > 0`.** An early estimate of
  the piezoelectric count came out one too high. K₆Ta₁₁O₃₀ has a band gap of
  0.0002 eV but is flagged metallic; a numerically-tiny gap is still a metal,
  and a conductor screens out the internal field. Using Materials Project's own
  determination handles this where a raw threshold does not.
- **Rules moved from hardcoded `if`s to a predicate registry**, with every
  threshold relocated into `config/materials.yaml` beside the text describing
  it. Previously config held the *description* while Python held the *logic*,
  so the two could drift apart silently — a real risk once the rule count more
  than doubled. The refactor was verified behaviour-preserving: all 84
  pre-existing tests passed unchanged.
- **Verification now asserts the project's central claim** rather than assuming
  it: every `suitable_for` edge must carry the rule text that fired
  (1,937/1,937 pass). Previously `verify.py` printed problems but still exited
  0, so a broken build looked successful.

## 13. Roadmap

- **Phase 2 — GraphRAG retrieval.** Natural-language query → graph traversal
  + vector search → LLM-generated, evidence-backed recommendation, with
  every claim traceable to a specific edge or Materials Project record.
  Requires an `LLM_API_KEY`.
- **Phase 3 — Crystal structure encoding.** Train/reproduce a CGCNN (with a
  pretrained ALIGNN fallback) on a broad Materials Project structure corpus,
  independent of which materials are in this graph. Use it purely for
  inference on the graph's materials to produce structural embeddings,
  replacing the current z-scored-cosine `similar_to` edges with genuine
  structural similarity, and adding an ANN index for scale.
- **Neo4j migration.** Currently on Cognee's embedded Kuzu store for
  frictionless local iteration. Cognee ships a first-class Neo4j adapter —
  the switch is a config-only change (`GRAPH_DATABASE_PROVIDER=neo4j` +
  connection details), with zero changes to `schema.py`, `enrich.py`, or
  `pipeline.py`. Planned for when Phase 2's backend work begins.

## 14. Tech Stack

| Layer | Technology |
|---|---|
| Data source | [Materials Project REST API](https://api.materialsproject.org) |
| Graph/pipeline framework | [Cognee](https://github.com/topoteretes/cognee) (low-level `DataPoint` pipeline) |
| Graph store | Kuzu (embedded; Neo4j-ready via config swap) |
| Vector store | LanceDB (embedded) |
| Embeddings | `fastembed` (local, `BAAI/bge-small-en-v1.5`) — no API key required |
| Element property data | [pymatgen](https://pymatgen.org) (full 103-element periodic table) |
| Language / tooling | Python 3.11–3.14, [`uv`](https://github.com/astral-sh/uv) |
