# MatGraphRAG — Phase 1: Knowledge Graph Construction

Builds an interconnected materials-science knowledge graph from the
[Materials Project](https://materialsproject.org) API using
[Cognee](https://github.com/topoteretes/cognee) as the graph/memory layer.
See [`docs/PHASE1_PLAN.md`](docs/PHASE1_PLAN.md) for the full design writeup.

10 seed materials (battery cathodes, semiconductors, magnetic/dielectric oxides)
are fetched, deterministically classified (band gap / stability / magnetic
ordering buckets, application-domain rules), and written into a Cognee graph as
`Material`, `Element`, `SpaceGroup`, `CrystalSystem`, `ChemicalSystem`,
`PropertyClass`, and `ApplicationDomain` nodes — connected so materials share
elements, crystal systems, and property classes rather than sitting as
isolated islands. No LLM calls happen in the build path; it's fully
deterministic and reproducible from the cached Materials Project data.

## Setup

```bash
uv sync
cp .env.example .env   # fill in MP_API_KEY
```

## Run

```bash
# 1. Fetch + cache the 10 seed materials from Materials Project
uv run python scripts/fetch_materials.py

# 2. Build the graph in Cognee, render a visualization, run verification queries
uv run python scripts/build_graph.py
```

Output:
- `data/raw/*.json` — cached raw Materials Project records (gitignored)
- `.cognee_system/` — Cognee's embedded graph/vector/relational stores (gitignored)
- `artifacts/graph.html` — interactive graph visualization (gitignored, regenerate anytime)
- Console output from `verify.py`: node/edge counts and structural sanity queries
  (oxygen hub, Li-cathode cluster, shared-element paths, application-domain
  membership, similarity edges)

## Notes

- Embeddings use a local `fastembed` model — no OpenAI/LLM key required to
  build the graph. `COGNEE_SKIP_CONNECTION_TEST=true` skips Cognee's one-time
  startup LLM connectivity self-test (the build path makes no LLM calls, but
  cognee checks for one on first run regardless).
- Natural-language graph search (`SearchType.GRAPH_COMPLETION`) needs an
  `LLM_API_KEY` — not required for Phase 1's deliverable (the graph itself),
  planned for Phase 2's retrieval layer.
- Re-running `scripts/build_graph.py` prunes and rebuilds the graph from
  scratch by default; pass `--no-reset` to add on top of existing state.
