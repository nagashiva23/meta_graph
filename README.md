# MatGraphRAG — Phase 1: Knowledge Graph Construction

Builds an interconnected materials-science knowledge graph from the
[Materials Project](https://materialsproject.org) API using
[Cognee](https://github.com/topoteretes/cognee) as the graph/memory layer.
See [`docs/PHASE1_PLAN.md`](docs/PHASE1_PLAN.md) for the full design writeup.

A curated, chemistry-driven slice of Materials Project (battery cathodes,
semiconductors, perovskite/spinel oxides, magnetic oxides — ~500-2000
materials, see `config/materials.yaml`'s `clusters`) is fetched, deterministically
classified (band gap / stability / magnetic ordering buckets, application-domain
rules), and written into a Cognee graph as `Material`, `Element`, `SpaceGroup`,
`CrystalSystem`, `ChemicalSystem`, `PropertyClass`, and `ApplicationDomain`
nodes — connected so materials share elements, crystal systems, and property
classes rather than sitting as isolated islands. No LLM calls happen in the
build path; it's fully deterministic and reproducible from the cached
Materials Project data.

The full ~154k-material Materials Project catalog was considered and
deliberately not used for Phase 1: pairwise material-material similarity is
O(n²) and becomes intractable at that scale (11.9B pairs), so a curated
subset that stays densely interconnected was chosen instead. Full-scale
structural similarity is Phase 3's job (CGCNN embeddings + an ANN index).

## Setup

```bash
uv sync
cp .env.example .env   # fill in MP_API_KEY
```

## Run

```bash
# 0. (only if config/elements.json needs regenerating for new elements)
uv run python scripts/generate_elements.py

# 1. Fetch + cache the cluster-defined material set from Materials Project
uv run python scripts/fetch_materials.py

# 2. Build the graph in Cognee, render a visualization, run verification queries
uv run python scripts/build_graph.py
```

Output:
- `data/raw/<material_id>.json` + `data/raw/_manifest.json` — cached raw
  Materials Project records, deduped by formula across clusters (gitignored)
- `.cognee_system/` — Cognee's embedded graph/vector/relational stores (gitignored)
- `artifacts/graph.html` — interactive graph visualization (gitignored, regenerate anytime)
- Console output from `verify.py`: node/edge counts and structural sanity queries
  (oxygen hub, Li-cathode cluster, shared-element paths, application-domain
  membership, similarity edges) — sampled rather than printed in full once the
  dataset is large

## Notes

- Embeddings use a local `fastembed` model — no OpenAI/LLM key required to
  build the graph. `COGNEE_SKIP_CONNECTION_TEST=true` skips Cognee's one-time
  startup LLM connectivity self-test (the build path makes no LLM calls, but
  cognee checks for one on first run regardless).
- `similar_to` edges use top-K nearest neighbours per material (config:
  `similarity.top_k`), not a flat similarity threshold — a fixed cosine cutoff
  doesn't scale with dataset size (a few hundred materials over 4 dimensions
  produces tens of thousands of "similar enough" pairs at a threshold tuned
  for 10 materials).
- `matgraph.pipeline.run()` raises Python's recursion limit before calling
  cognee: cognee's internal provenance-stamping walk is recursive with no
  depth cap, and a graph with hundreds of `similar_to`-cross-linked materials
  exceeds the default 1000-frame limit.
- Natural-language graph search (`SearchType.GRAPH_COMPLETION`) needs an
  `LLM_API_KEY` — not required for Phase 1's deliverable (the graph itself),
  planned for Phase 2's retrieval layer.
- Re-running `scripts/build_graph.py` prunes and rebuilds the graph from
  scratch by default; pass `--no-reset` to add on top of existing state.
