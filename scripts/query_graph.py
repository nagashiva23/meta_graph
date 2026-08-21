"""Query the knowledge graph from the command line. No LLM involved.

This is Phase 2's retrieval layer with a human at the front instead of a
language model - the same calls the generation step will make later, so
whatever it eventually says can be checked against what this prints.

Examples:
    uv run python scripts/query_graph.py --element Li --domain battery_cathode --limit 5
    uv run python scripts/query_graph.py --class band_gap:wide_gap --min band_gap:5
    uv run python scripts/query_graph.py --oxidation-state Ti4+ --crystal-system Cubic
    uv run python scripts/query_graph.py --similar-to BaTiO3
    uv run python scripts/query_graph.py --connects LiFePO4 Fe2O3
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--element", action="append", default=[], metavar="SYMBOL",
                   help="must contain this element (repeatable, ANDed)")
    p.add_argument("--domain", metavar="NAME", help="application domain, e.g. battery_cathode")
    p.add_argument("--class", dest="classes", action="append", default=[], metavar="KIND:NAME",
                   help="property class, e.g. stability:stable (repeatable)")
    p.add_argument("--oxidation-state", action="append", default=[], metavar="SPECIES",
                   help="e.g. Ti4+ (repeatable)")
    p.add_argument("--formula-pattern", metavar="PATTERN", help="e.g. ABC3")
    p.add_argument("--crystal-system", metavar="NAME", help="e.g. Cubic")
    p.add_argument("--chemical-system", metavar="CHEMSYS", help="e.g. Fe-Li-O-P")
    p.add_argument("--min", dest="minimums", action="append", default=[], metavar="FIELD:VALUE")
    p.add_argument("--max", dest="maximums", action="append", default=[], metavar="FIELD:VALUE")
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--order-by", default="energy_above_hull")
    p.add_argument("--similar-to", metavar="FORMULA", help="nearest neighbours of this material")
    p.add_argument("--connects", nargs=2, metavar=("FORMULA_A", "FORMULA_B"),
                   help="what two materials have in common")
    return p.parse_args()


def build_constraints(args) -> "object":
    from matgraph.retrieval import Constraints, NumericFilter

    def split(pairs, label):
        out = []
        for item in pairs:
            if ":" not in item:
                raise SystemExit(f"--{label} expects FIELD:VALUE, got {item!r}")
            key, value = item.rsplit(":", 1)
            out.append((key, value))
        return out

    bounds: dict[str, dict[str, float]] = {}
    for field, value in split(args.minimums, "min"):
        bounds.setdefault(field, {})["minimum"] = float(value)
    for field, value in split(args.maximums, "max"):
        bounds.setdefault(field, {})["maximum"] = float(value)

    return Constraints(
        elements=tuple(args.element),
        application_domain=args.domain,
        property_classes=tuple((k, v) for k, v in split(args.classes, "class")),
        oxidation_states=tuple(args.oxidation_state),
        formula_pattern=args.formula_pattern,
        crystal_system=args.crystal_system,
        chemical_system=args.chemical_system,
        numeric=tuple(NumericFilter(field=f, **b) for f, b in bounds.items()),
    )


def print_hits(hits, limit: int) -> None:
    shown = hits[:limit]
    header = f"{len(hits)} match(es)"
    if len(shown) < len(hits):
        header += f", showing the {len(shown)} most stable"
    print(f"\n{header}\n")

    for hit in shown:
        gap = hit.band_gap
        hull = hit.energy_above_hull
        line = f"  {hit.formula:16s} {hit.material_id:16s}"
        if gap is not None:
            line += f"  gap={gap:.3f} eV"
        if hull is not None:
            line += f"  E_hull={hull:.4f} eV/atom"
        print(line)
        for item in hit.evidence:
            print(f"      - {item}")
        print()

    if not hits:
        print("  (no material satisfies all of those constraints)\n")


async def main() -> None:
    args = parse_args()

    from cognee import config
    from matgraph.pipeline import COGNEE_SYSTEM_DIR

    # Must precede any cognee store access, or it looks inside the installed
    # package instead of this project and reports an empty graph.
    config.system_root_directory(str(COGNEE_SYSTEM_DIR))

    from matgraph.graph_store import load_graph
    from matgraph.retrieval import (
        UnknownEntity,
        connections_between,
        find_materials,
        similar_materials,
    )

    graph = await load_graph()
    print(f"graph loaded: {len(graph.node)} nodes")

    try:
        if args.connects:
            shared = connections_between(graph, *args.connects)
            print(f"\nWhat connects {args.connects[0]} and {args.connects[1]}:\n")
            for item in shared:
                print(f"  - {item}")
            if not shared:
                print("  (nothing shared)")
            return

        if args.similar_to:
            print_hits(similar_materials(graph, args.similar_to, args.limit), args.limit)
            return

        hits = find_materials(graph, build_constraints(args), order_by=args.order_by)
        print_hits(hits, args.limit)
    except UnknownEntity as exc:
        raise SystemExit(f"\n{exc}") from exc


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")
    asyncio.run(main())
