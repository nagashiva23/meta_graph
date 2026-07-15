"""Structural verification of the built graph.

Deliberately LLM-free: Phase 1's deliverable is the graph itself, so
correctness is checked by pulling the raw node/edge data straight from the
graph engine (``get_graph_data``) and inspecting it in Python, rather than
through cognee's natural-language search (which needs an LLM_API_KEY that
this environment does not have configured) or hand-written Cypher (cognee's
Kuzu/Ladybug backend stores every node/edge in generic ``Node``/``EDGE``
tables with a JSON properties blob, not per-class labels/columns).
"""

from __future__ import annotations

import logging
from collections import defaultdict

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.context_global_variables import set_database_global_context_variables
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.users.methods import get_default_user

from matgraph.pipeline import DATASET_NAME

logger = logging.getLogger(__name__)


class Graph:
    """Thin in-memory index over cognee's (nodes, edges) tuples."""

    def __init__(self, nodes, edges):
        self.node = {node_id: props for node_id, props in nodes}
        self.out: dict[str, list[tuple[str, str, dict]]] = defaultdict(list)
        self.in_: dict[str, list[tuple[str, str, dict]]] = defaultdict(list)
        for source_id, target_id, rel, props in edges:
            self.out[source_id].append((rel, target_id, props))
            self.in_[target_id].append((rel, source_id, props))

    def nodes_of_type(self, type_name: str):
        return [(nid, p) for nid, p in self.node.items() if p.get("type") == type_name]

    def targets(self, node_id: str, rel: str):
        """(target_id, target_node_props) pairs for outgoing edges of type `rel`."""
        return [(tid, self.node[tid]) for r, tid, _edge_props in self.out[node_id] if r == rel]

    def target_edges(self, node_id: str, rel: str):
        """(target_id, edge_props) pairs, when the edge's own properties matter."""
        return [(tid, ep) for r, tid, ep in self.out[node_id] if r == rel]


async def verify() -> None:
    user = await get_default_user()
    datasets = await get_authorized_existing_datasets([DATASET_NAME], "read", user)

    async with set_database_global_context_variables(datasets[0].id, user.id):
        engine = await get_graph_engine()
        raw_nodes, raw_edges = await engine.get_graph_data()
        g = Graph(raw_nodes, raw_edges)

        print("\n== Node counts by type ==")
        by_type: dict[str, int] = defaultdict(int)
        for _, props in g.node.items():
            by_type[props.get("type", "?")] += 1
        for type_name, count in sorted(by_type.items(), key=lambda kv: -kv[1]):
            print(f"  {type_name:20s} {count}")

        print("\n== Edge counts by relationship ==")
        by_rel: dict[str, int] = defaultdict(int)
        for _, _, rel, _ in raw_edges:
            by_rel[rel] += 1
        for rel, count in sorted(by_rel.items(), key=lambda kv: -kv[1]):
            print(f"  {rel:20s} {count}")

        materials = {p["formula"]: (nid, p) for nid, p in g.nodes_of_type("Material")}

        print("\n== Oxygen hub: materials containing O ==")
        for formula, (nid, _) in sorted(materials.items()):
            elements = {ep["symbol"] for _, ep in g.targets(nid, "contains")}
            if "O" in elements:
                print(f"  {formula}")

        print("\n== Lithium-containing materials classified as 'stable' ==")
        for formula, (nid, _) in sorted(materials.items()):
            elements = {ep["symbol"] for _, ep in g.targets(nid, "contains")}
            classes = {(cp["kind"], cp["name"]) for _, cp in g.targets(nid, "classified_as")}
            if "Li" in elements and ("stability", "stable") in classes:
                print(f"  {formula}")

        print("\n== Wide-band-gap materials ==")
        rows = []
        for formula, (nid, p) in materials.items():
            classes = {(cp["kind"], cp["name"]) for _, cp in g.targets(nid, "classified_as")}
            if ("band_gap", "wide_gap") in classes:
                rows.append((formula, p["band_gap"]))
        for formula, band_gap in sorted(rows, key=lambda r: -r[1]):
            print(f"  {formula:12s} band_gap={band_gap:.3f}")

        print("\n== What connects LiFePO4 and Fe2O3 (shared elements) ==")
        nid_a, _ = materials["LiFePO4"]
        nid_b, _ = materials["Fe2O3"]
        elems_a = {ep["symbol"] for _, ep in g.targets(nid_a, "contains")}
        elems_b = {ep["symbol"] for _, ep in g.targets(nid_b, "contains")}
        for symbol in sorted(elems_a & elems_b):
            print(f"  shared element: {symbol}")

        print("\n== Materials suitable for battery_cathode (with rule) ==")
        for formula, (nid, _) in sorted(materials.items()):
            for _, dp in g.targets(nid, "suitable_for"):
                if dp["name"] == "battery_cathode":
                    print(f"  {formula}")

        print("\n== similar_to edges (weight >= threshold) ==")
        seen = set()
        rows = []
        for formula, (nid, _) in materials.items():
            for rel, target_id, props in g.out[nid]:
                if rel != "similar_to":
                    continue
                other_formula = g.node[target_id]["formula"]
                key = tuple(sorted([formula, other_formula]))
                if key in seen:
                    continue
                seen.add(key)
                rows.append((key[0], key[1], props.get("weight", 0.0)))
        for a, b, weight in sorted(rows, key=lambda r: -r[2]):
            print(f"  {a:10s} <-> {b:10s} weight={weight:.3f}")

        print("\n== Structural checks ==")
        isolated = [
            formula for formula, (nid, _) in materials.items() if not g.targets(nid, "contains")
        ]
        if isolated:
            print(f"  !! Materials with no element edges: {isolated}")
        else:
            print("  OK: every Material has at least one contains edge.")

        assert len(materials) == 10, f"expected 10 Material nodes, got {len(materials)}"
        print(f"  OK: {len(materials)} Material nodes present.")
