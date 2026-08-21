"""Check graph retrieval against an independent oracle: the raw cached JSON.

The unit tests in tests/test_retrieval.py prove the traversal logic is correct
on a hand-built graph. They cannot prove the *real* graph was built correctly,
because both the graph and the expectation would come from the same code.

This harness closes that gap. For each query it computes the answer twice:

  1. by traversing the built knowledge graph (matgraph.retrieval), and
  2. by filtering data/raw/*.json directly in plain Python - never touching
     cognee, the graph, or enrich.py's builders.

Route 2 is the oracle. It reads the same Materials Project records the graph
was built from, so any disagreement means the graph, the traversal, or the
enrichment lost or invented something between the cache and the query.

Usage:
    uv run python scripts/eval_retrieval.py
Exits non-zero if any case disagrees.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data" / "raw"


def load_raw() -> list[dict]:
    ids = json.loads((DATA_DIR / "_manifest.json").read_text())
    return [json.loads((DATA_DIR / f"{i}.json").read_text()) for i in ids]


def build_cases(config: dict, space_groups: dict):
    """(name, Constraints, oracle) triples.

    Each oracle re-derives the expected formulas straight from raw MP fields,
    deliberately duplicating the *intent* of a rule rather than importing the
    predicate that implements it — importing it would make the check circular.
    """
    from matgraph.retrieval import Constraints, NumericFilter

    transition_metals = set(config["transition_metals"])

    def symmetry(m, key):
        return space_groups[str(m["symmetry"]["number"])][key]

    return [
        (
            "materials containing lithium",
            Constraints(elements=("Li",)),
            lambda m: "Li" in m["elements"],
        ),
        (
            "materials containing both Li and Fe",
            Constraints(elements=("Li", "Fe")),
            lambda m: {"Li", "Fe"} <= set(m["elements"]),
        ),
        (
            "wide-band-gap materials (bucket traversal)",
            Constraints(property_classes=(("band_gap", "wide_gap"),)),
            lambda m: m["band_gap"] > 3.0,
        ),
        (
            "stable materials (bucket traversal)",
            Constraints(property_classes=(("stability", "stable"),)),
            lambda m: m["energy_above_hull"] <= 1.0e-6,
        ),
        (
            "stable lithium-containing materials",
            Constraints(elements=("Li",), property_classes=(("stability", "stable"),)),
            lambda m: "Li" in m["elements"] and m["energy_above_hull"] <= 1.0e-6,
        ),
        (
            "battery cathode candidates",
            Constraints(application_domain="battery_cathode"),
            lambda m: (
                "Li" in m["elements"]
                and bool(set(m["elements"]) & transition_metals)
                and m["energy_above_hull"] <= 0.05
            ),
        ),
        (
            "piezoelectric candidates (symmetry screening)",
            Constraints(application_domain="piezoelectric"),
            lambda m: symmetry(m, "piezoelectric_allowed") and not m["is_metal"],
        ),
        (
            "ferroelectric candidates (polar space groups)",
            Constraints(application_domain="ferroelectric"),
            lambda m: symmetry(m, "polar") and not m["is_metal"],
        ),
        (
            "photovoltaic absorbers (direct gap window)",
            Constraints(application_domain="photovoltaic_absorber"),
            lambda m: m["is_gap_direct"] and 1.0 <= m["band_gap"] <= 1.8 and not m["is_metal"],
        ),
        (
            "materials containing Ti4+ (oxidation state)",
            Constraints(oxidation_states=("Ti4+",)),
            lambda m: "Ti4+" in (m.get("possible_species") or []),
        ),
        (
            "ABC3 perovskite-pattern materials",
            Constraints(formula_pattern="ABC3"),
            lambda m: m.get("formula_anonymous") == "ABC3",
        ),
        (
            "cubic materials (2-hop via space group)",
            Constraints(crystal_system="Cubic"),
            lambda m: m["symmetry"]["crystal_system"] == "Cubic",
        ),
        (
            "band gap between 5 and 6 eV (numeric attribute)",
            Constraints(numeric=(NumericFilter("band_gap", minimum=5.0, maximum=6.0),)),
            lambda m: 5.0 <= m["band_gap"] <= 6.0,
        ),
        (
            "cubic oxides with a gap above 2 eV (structural + numeric)",
            Constraints(
                elements=("O",),
                crystal_system="Cubic",
                numeric=(NumericFilter("band_gap", minimum=2.0),),
            ),
            lambda m: (
                "O" in m["elements"]
                and m["symmetry"]["crystal_system"] == "Cubic"
                and m["band_gap"] >= 2.0
            ),
        ),
    ]


async def main() -> None:
    from cognee import config as cognee_config

    from matgraph.pipeline import COGNEE_SYSTEM_DIR

    cognee_config.system_root_directory(str(COGNEE_SYSTEM_DIR))

    from matgraph.enrich import load_config, load_space_groups
    from matgraph.graph_store import load_graph
    from matgraph.retrieval import find_materials

    graph = await load_graph()
    raw = load_raw()
    cases = build_cases(load_config(), load_space_groups())

    print(f"\nGraph: {len(graph.node)} nodes. Oracle: {len(raw)} cached records.")
    print(f"Cross-checking {len(cases)} queries against raw Materials Project JSON.\n")
    print(f"  {'query':52s} {'graph':>7s} {'oracle':>7s}   result")
    print(f"  {'-' * 52} {'-' * 7} {'-' * 7}   ------")

    failures = []
    for name, constraints, oracle in cases:
        from_graph = {hit.formula for hit in find_materials(graph, constraints)}
        from_raw = {m["formula_pretty"] for m in raw if oracle(m)}

        if from_graph == from_raw:
            print(f"  {name:52s} {len(from_graph):7d} {len(from_raw):7d}   MATCH")
            continue

        failures.append(name)
        print(f"  {name:52s} {len(from_graph):7d} {len(from_raw):7d}   MISMATCH")
        only_graph = sorted(from_graph - from_raw)[:5]
        only_raw = sorted(from_raw - from_graph)[:5]
        if only_graph:
            print(f"      in graph but not oracle: {only_graph}")
        if only_raw:
            print(f"      in oracle but not graph: {only_raw}")

    print()
    if failures:
        raise SystemExit(f"{len(failures)} of {len(cases)} queries disagree with the oracle.")
    print(f"All {len(cases)} queries agree with the raw data exactly.\n")


if __name__ == "__main__":
    asyncio.run(main())
