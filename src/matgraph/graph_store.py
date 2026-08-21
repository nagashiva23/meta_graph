"""Reading the built graph back out of cognee.

Shared by verify.py (structural checks) and retrieval.py (Phase 2 queries).
Both need the same in-memory index and the same edge-property reader, and a
second copy of either would drift from the first.

Everything here works on plain (nodes, edges) tuples, so the query layer built
on top can be tested against a small hand-built Graph without cognee, a
database, or a built graph. Only `load_graph` touches cognee.
"""

from __future__ import annotations

import ast
import logging
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)


def edge_properties(edge_props: dict[str, Any]) -> dict[str, Any]:
    """Unpack the nested `properties` blob cognee stores on an edge.

    Cognee does not keep an Edge's `properties` dict where you would look for
    it. It arrives as a *stringified* Python dict under a single "properties"
    key, so anything put there ends up one level down and serialised:

        {"properties": "{'rule': '0 < band_gap <= 3.5 eV and not metallic'}"}

    A plain edge_props.get("rule") therefore returns None for every edge, which
    is indistinguishable from the property never having been stored - a silent
    failure worth guarding against once retrieval depends on this evidence.

    Parsed with literal_eval, never eval: this is data read back out of a
    database, and literal_eval can only construct literals, never execute code.
    """
    properties = edge_props.get("properties")
    if isinstance(properties, str):
        try:
            properties = ast.literal_eval(properties)
        except (ValueError, SyntaxError):
            logger.warning("could not parse edge properties: %r", properties[:120])
            return {}
    return properties if isinstance(properties, dict) else {}


def edge_rule(edge_props: dict[str, Any]) -> str | None:
    """The rule text stored on a suitable_for edge, or None if absent."""
    rule = edge_properties(edge_props).get("rule")
    return rule if isinstance(rule, str) and rule.strip() else None


def edge_weight(edge_props: dict[str, Any]) -> float | None:
    """The weight on a similar_to or contains edge.

    Cognee returns weights as *strings* ("0.9837107575817106"), so comparing or
    sorting on the raw value silently does the wrong thing - "0.9" > "0.15" is
    True as text and False as numbers.
    """
    weight = edge_props.get("weight")
    if weight is None:
        return None
    try:
        return float(weight)
    except (TypeError, ValueError):
        return None


class Graph:
    """Thin in-memory index over cognee's (nodes, edges) tuples.

    The whole graph is ~1,500 nodes and ~15,000 edges, so indexing it in memory
    once and querying it in Python is both simpler and faster than issuing a
    query per hop against the store.
    """

    def __init__(self, nodes, edges):
        self.node: dict[str, dict] = {node_id: props for node_id, props in nodes}
        self.out: dict[str, list[tuple[str, str, dict]]] = defaultdict(list)
        self.in_: dict[str, list[tuple[str, str, dict]]] = defaultdict(list)
        for source_id, target_id, rel, props in edges:
            self.out[source_id].append((rel, target_id, props))
            self.in_[target_id].append((rel, source_id, props))

        self._by_type: dict[str, list[tuple[str, dict]]] = defaultdict(list)
        for node_id, props in self.node.items():
            self._by_type[props.get("type", "?")].append((node_id, props))

    def nodes_of_type(self, type_name: str) -> list[tuple[str, dict]]:
        return self._by_type.get(type_name, [])

    def targets(self, node_id: str, rel: str) -> list[tuple[str, dict]]:
        """(target_id, target_node_props) for outgoing edges of type `rel`."""
        return [(tid, self.node[tid]) for r, tid, _props in self.out[node_id] if r == rel]

    def target_edges(self, node_id: str, rel: str) -> list[tuple[str, dict]]:
        """(target_id, edge_props) pairs, when the edge's own properties matter."""
        return [(tid, props) for r, tid, props in self.out[node_id] if r == rel]

    def sources(self, node_id: str, rel: str) -> list[tuple[str, dict]]:
        """(source_id, source_node_props) for *incoming* edges of type `rel`.

        This is the direction retrieval traverses. Edges run Material -> Element,
        but a query asks "which materials contain iron?", which walks from the
        Element node back to every Material pointing at it. Because elements are
        hubs (oxygen is one node reached by 756 materials), starting from the
        shared node and walking backwards touches one node instead of scanning
        all 794.
        """
        return [(sid, self.node[sid]) for r, sid, _props in self.in_[node_id] if r == rel]

    def find_node(self, type_name: str, **match: Any) -> tuple[str, dict] | None:
        """First node of `type_name` whose properties equal every kwarg.

        Used to resolve a human-facing name ("Fe", "battery_cathode") to the
        node id that traversal starts from.
        """
        for node_id, props in self.nodes_of_type(type_name):
            if all(props.get(key) == value for key, value in match.items()):
                return node_id, props
        return None


async def load_graph(dataset_name: str | None = None) -> Graph:
    """Read the whole built graph out of cognee into memory.

    The caller must already have pointed cognee at this project's system
    directory (`config.system_root_directory`); without it cognee looks for its
    databases inside the installed package and reports an empty graph rather
    than an error.
    """
    from cognee.context_global_variables import set_database_global_context_variables
    from cognee.infrastructure.databases.graph import get_graph_engine
    from cognee.modules.data.methods import get_authorized_existing_datasets
    from cognee.modules.users.methods import get_default_user

    from matgraph.pipeline import DATASET_NAME

    dataset_name = dataset_name or DATASET_NAME
    user = await get_default_user()
    datasets = await get_authorized_existing_datasets([dataset_name], "read", user)
    if not datasets:
        raise RuntimeError(
            f"no dataset named {dataset_name!r} - run scripts/build_graph.py first"
        )

    async with set_database_global_context_variables(datasets[0].id, user.id):
        engine = await get_graph_engine()
        nodes, edges = await engine.get_graph_data()
        return Graph(nodes, edges)
