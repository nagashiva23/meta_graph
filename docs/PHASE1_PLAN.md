# MatGraphRAG — Phase 1 Plan: Knowledge Graph Construction Pipeline

Goal: pull ~10 materials from the Materials Project (MP) API and construct an
interrelated, groupable knowledge graph in **Cognee**, which serves as the memory
layer for later GraphRAG retrieval (Phase 2) and CGCNN structure embeddings (Phase 3).

---

## 1. Key research findings

### Cognee (topoteretes/cognee)
- Cognee's default `cognee.add()` + `cognee.cognify()` flow is **LLM-driven entity
  extraction from unstructured text**. For structured MP records this is the wrong
  tool: it hallucinates schema and loses numeric fidelity.
- The right mechanism is the **low-level custom pipeline**:
  - Define node types as `DataPoint` subclasses (Pydantic models).
    `metadata = {"index_fields": [...]}` marks fields that get embedded into the
    vector store (one collection per `Class_field`).
  - Fields typed as another `DataPoint` (or `list[DataPoint]`) automatically become
    **graph edges** named after the field.
  - `Edge(weight=…, relationship_type=…, properties=…)` via tuple annotation
    `(Edge(...), list[X])` gives weighted, typed edges.
  - Pipeline = `run_pipeline([Task(build_datapoints), Task(add_data_points)], data, dataset_id, user, name)`.
    `add_data_points` writes graph + vector + relational stores at once and
    deduplicates.
  - Reference implementation: `cognee-starter-kit/src/pipelines/low_level.py` in the
    cognee repo (Person/Department/Company example — our template).
- Hybrid trick: we can *also* feed the robocrystallographer text descriptions through
  the normal `add/cognify` path later, but Phase 1 stays deterministic (no LLM in the
  build path; LLM key only needed for embeddings + search).
- Storage defaults (no infra to run): Kuzu (embedded graph DB), LanceDB (embedded
  vectors), SQLite (metadata). Neo4j is a config swap later — the PRD's target —
  without changing pipeline code.
- `visualize_graph(path.html)` renders the built graph for inspection;
  `search(query_type=SearchType.GRAPH_COMPLETION / CHUNKS / INSIGHTS)` queries it.

### Materials Project API (verified live with the provided key)
- REST base `https://api.materialsproject.org`, header `X-API-KEY`.
  (`mp-api` Python client exists but plain `requests` keeps deps light; decide at impl time.)
- `/materials/summary/?formula=…&_fields=…` returns everything Phase 1 needs:
  `material_id, formula_pretty, chemsys, elements, nsites, volume, density,
  band_gap, is_gap_direct, is_metal, formation_energy_per_atom, energy_above_hull,
  is_stable, total_magnetization, ordering, symmetry{crystal_system, symbol, number}, theoretical`.
- `/materials/robocrys/?material_ids=…` returns a **human-readable structure
  description** per material (e.g. "Si is diamond structured and crystallizes in the
  cubic Fd̅3m space group…") — perfect embeddable `description` for the Material node.
- MP now issues new-format IDs (e.g. `mp-aaaaaaft` for Si); old IDs still resolve.
  **Select materials by formula + lowest `energy_above_hull`**, never hard-code IDs.
- A formula can return dozens of polymorphs (LiFePO4 → 70 docs). Picking
  `min(energy_above_hull)` gives the canonical ground-state entry.

---

## 2. The 10 seed materials (chosen for graph connectivity)

Chosen so nodes interrelate through shared elements, crystal systems, property
classes, and application domains — no isolated islands:

| Formula | Why it's in the set | Overlaps |
|---|---|---|
| LiFePO4 | battery cathode | Li, Fe, O; olivine |
| LiCoO2  | battery cathode | Li, O; layered |
| LiMn2O4 | battery cathode | Li, O; spinel |
| Si      | elemental semiconductor | cubic; semiconductor class |
| GaAs    | III-V semiconductor | Ga, cubic; semiconductor |
| GaN     | III-V wide-gap semiconductor | Ga (↔GaAs), N; hexagonal |
| ZnO     | wide-gap oxide semiconductor | O, hexagonal (↔GaN) |
| TiO2    | photocatalyst oxide | O, Ti; wide gap (↔ZnO) |
| Fe2O3   | magnetic oxide | Fe (↔LiFePO4), O; AFM ordering |
| BaTiO3  | dielectric/ferroelectric perovskite | Ti (↔TiO2), O |

Resulting hubs: **O** (7 materials), **Li** (3), **Fe** (2), **Ga** (2), **Ti** (2),
crystal systems (cubic/hexagonal/…), and application groups (cathodes,
semiconductors, oxides). The list lives in a config file — swapping/scaling it later
requires no code change.

## 3. Graph schema (Cognee DataPoints)

**Design rule:** continuous numerics stay as *attributes* on Material (exact values
for filtering/comparison); anything used for *grouping/traversal* becomes a *node*;
derived similarity becomes a *weighted edge*.

### Node types
- **Material** — `material_id`, `formula` (embeddable), `description` (robocrys text,
  embeddable), plus numeric attributes: `band_gap`, `formation_energy_per_atom`,
  `energy_above_hull`, `density`, `total_magnetization`, `volume`, `nsites`,
  `is_stable`, `is_metal`, `is_gap_direct`, `theoretical`.
- **Element** — `symbol`, `name` (embeddable), `group`, `period`,
  `electronegativity`, `atomic_radius`, `category` (alkali metal, transition metal,
  …). Static table in repo (or `pymatgen.core.Element`) — no API call needed.
  These are Phase 3's substitution-model anchors.
- **CrystalSystem** — one of the 7 (cubic, hexagonal, …).
- **SpaceGroup** — `symbol`, `number`; links up to its CrystalSystem.
- **ChemicalSystem** — `chemsys` string (e.g. `Fe-Li-O-P`).
- **PropertyClass** — categorical buckets that make materials groupable:
  - band gap: `metal (=0)`, `narrow_gap (0–1 eV)`, `semiconductor (1–3 eV)`, `wide_gap (>3 eV)`
  - stability: `stable (on hull)`, `metastable (≤0.05 eV/atom)`, `unstable (>0.05)`
  - magnetic ordering: `FM`, `AFM`, `NM`, `FiM`
- **ApplicationDomain** — `battery_cathode`, `semiconductor_device`, `photocatalyst`,
  `dielectric`, `magnetic_material`. Assigned by transparent rules (see §4), each
  edge carrying the rule as `properties` → explainability from day one (PRD's core theme).

### Edge types
| Edge | From → To | Weight / properties |
|---|---|---|
| `contains` | Material → Element | weight = atomic fraction in formula |
| `has_space_group` | Material → SpaceGroup | — |
| `in_crystal_system` | SpaceGroup → CrystalSystem | — (Material reaches it in 2 hops) |
| `member_of` | Material → ChemicalSystem | — |
| `classified_as` | Material → PropertyClass | properties = raw value that triggered the bucket |
| `suitable_for` | Material → ApplicationDomain | weight = rule confidence, properties = rule text |
| `similar_to` | Material → Material | weight = cosine similarity of z-scored property vectors (band_gap, formation_energy, density, magnetization); keep edges > 0.7 threshold |

`similar_to` is the Phase 1 stand-in for Phase 3's CGCNN embedding similarity — the
schema slot (weighted material–material similarity edge) is identical, so Phase 3
only changes how the weight is computed.

Multi-hop queries this enables immediately:
- "Li-containing stable materials with band gap > 3 eV" (Element → Material → PropertyClass)
- "materials in the same crystal system as GaN that are semiconductors" (Material → SpaceGroup → CrystalSystem → … )
- "what connects LiFePO4 and Fe2O3?" (shared Fe, O; both AFM)

## 4. Application-domain rules (deterministic, explainable)

- `battery_cathode`: contains Li AND contains a transition metal AND `energy_above_hull ≤ 0.05`
- `semiconductor_device`: `0 < band_gap ≤ 3.5` AND not metal
- `photocatalyst`: oxide AND `1.5 ≤ band_gap ≤ 3.5`
- `dielectric`: `band_gap > 2` AND not magnetic (|total_magnetization| < 0.1)
- `magnetic_material`: ordering in {FM, FiM, AFM} AND |total_magnetization| ≥ 0.1 (FM/FiM) or AFM ordering

Each fired rule is stored on the `suitable_for` edge (`properties={"rule": "..."}`).

## 5. Pipeline architecture

```
config/materials.yaml ──┐
                        ▼
 [1] fetch_mp.py  ── MP REST API ──► data/raw/{material_id}.json   (cached; offline reruns)
                        ▼
 [2] transform.py ── raw JSON ──► DataPoint objects (schema above)
                        ▼            + derived: property classes, app domains,
                        ▼              similar_to weights, element table join
 [3] build_graph.py ── cognee run_pipeline([Task(load_raw), Task(to_datapoints), Task(add_data_points)])
                        ▼
 [4] verify.py ── visualize_graph(artifacts/graph.html)
                  + structural assertions (node/edge counts, no orphan materials)
                  + 3 smoke searches (GRAPH_COMPLETION / INSIGHTS)
```

Idempotency: `prune.prune_system()` + rebuild on each run (10 materials → seconds).
Raw-JSON cache layer means MP is hit once per material; `--refresh` flag re-fetches.

### Repo layout

```
meta_graph/
├── .env                      # MP_API_KEY, LLM_API_KEY (gitignored)
├── config/materials.yaml     # seed formulas + bucket thresholds + app rules
├── data/raw/                 # cached MP responses (gitignored)
├── src/matgraph/
│   ├── schema.py             # DataPoint classes
│   ├── mp_client.py          # MP REST fetch + cache
│   ├── enrich.py             # buckets, app rules, similarity weights, element table
│   ├── pipeline.py           # cognee tasks + run_pipeline wiring
│   └── verify.py             # assertions + smoke searches
├── scripts/build_graph.py    # entry point: python scripts/build_graph.py [--refresh]
├── artifacts/graph.html      # visualization output
└── tests/                    # transform unit tests on fixture JSON
```

## 6. Environment & keys

- Python ≥3.11, `uv` for env; deps: `cognee`, `requests`, `pyyaml`, `python-dotenv`
  (optionally `pymatgen` only for the element property table — heavy; a static JSON
  table of ~90 elements avoids it in Phase 1).
- Cognee needs an `LLM_API_KEY` for embeddings + search-time completion (build path
  itself is LLM-free). Model/provider set in `.env`.
- **MP API key was pasted in chat — put it in `.env`, never commit it; consider
  rotating it at materialsproject.org after the project is public.**
- Note: `meta_graph/` currently sits inside the old ERP git repo
  (`~/development` is the repo root, with all ERP files showing as deleted).
  Step 0 of implementation: make `meta_graph` its own repository (`git init`)
  so this project has clean history.

## 7. Milestones & acceptance criteria

1. **M1 — Fetch layer**: 10 ground-state records + robocrys descriptions cached as JSON.
   ✓ each file has all summary fields; polymorph selection = min energy_above_hull.
2. **M2 — Schema + transform**: DataPoints built from cache, unit-tested.
   ✓ 10 Materials, ~12 Elements, ≤7 CrystalSystems, ~10 SpaceGroups, 8 PropertyClasses,
   5 ApplicationDomains; every Material has ≥1 `contains`, exactly 1 space group, ≥1 class.
3. **M3 — Cognee build**: pipeline runs green; graph.html renders; O is the biggest hub.
4. **M4 — Verification**: smoke queries return correct answers, e.g.
   "Which stable materials contain lithium?" → LiFePO4/LiCoO2/LiMn2O4 subset;
   "Which materials are wide-band-gap?" → GaN, ZnO(≈), TiO2, BaTiO3 …
   ✓ answers traceable to graph edges (spot-check against raw JSON).

Definition of done for Phase 1: one command builds the graph from a clean checkout
(given `.env`), verification passes, and the graph visualization shows the
interconnected clusters (cathodes / semiconductors / oxides) rather than 10 islands.

## 8. Risks & mitigations

- **DFT band gaps are underestimated** (MP Si = 0.61 eV vs real 1.12 eV) → bucket
  thresholds tuned for DFT values; note stored on PropertyClass description.
- **Cognee API churn** (young project) → pin the version; our template
  (`cognee-starter-kit/low_level.py`) is in-repo upstream, easy to diff on upgrade.
- **Formula ambiguity/polymorphs** → always min-hull selection + `theoretical` flag kept.
- **Scale-up** (10 → 1000s later): `similar_to` is O(n²) — fine now; Phase 3 replaces
  it with ANN over CGCNN embeddings anyway.

## 9. Explicit non-goals for Phase 1

Retrieval/recommendation UX (Phase 2), CGCNN training and structure embeddings
(Phase 3), Neo4j/FastAPI/Next.js deployment, literature ingestion. The schema above
deliberately leaves their slots open (`similar_to` weights, Element nodes for
substitution, robocrys text for hybrid RAG).
