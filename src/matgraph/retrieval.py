"""Phase 2, step 1: relationship-aware retrieval over the knowledge graph.

Deliberately contains **no LLM and no embeddings**. Phase 2 splits into a
deterministic half - turn a set of constraints into a candidate list plus the
evidence backing each candidate - and a generative half that writes prose from
that evidence. This module is the deterministic half, and it is built first so
that by the time a language model is involved, everything it is shown has
already been checked against the graph.

Two design commitments, both inherited from Phase 1:

1. **Filters are strict conjunctions.** Every returned material satisfies every
   constraint. No soft scoring, no "close enough" - a caller asking for stable
   lithium cathodes must not receive an unstable sodium one ranked 7th.
2. **Every hit carries its evidence.** A material is never returned as a bare
   answer: it comes with the specific edges and attribute values that made it
   match, including the stored rule text for any application domain. That list
   is what the generation step will be allowed to see, and nothing else.

Traversal runs from the shared node *inwards*. Edges point Material -> Element,
but a query asks "which materials contain iron?", so retrieval starts at the Fe
node and walks incoming edges. Elements are hubs - oxygen is a single node
reached by 756 of 794 materials - so this touches one node and its edge list
rather than scanning every material.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from matgraph.graph_store import Graph, edge_rule, edge_weight

logger = logging.getLogger(__name__)


class UnknownEntity(ValueError):
    """A constraint names something that does not exist in the graph.

    Raised rather than silently returning nothing: "no materials contain
    Unobtainium" and "you misspelled Uranium" are very different answers, and a
    retrieval layer that conflates them will mislead whatever consumes it.
    """


@dataclass(frozen=True)
class NumericFilter:
    """An inclusive range over a numeric attribute stored on the Material node.

    Numeric properties deliberately stayed as attributes rather than becoming
    nodes (Phase 1's design rule), so precise questions - "band gap between 5
    and 6 eV" - are answered by filtering here, not by traversing to a
    PropertyClass node. The buckets are for grouping; this is for precision.
    """

    field: str
    minimum: float | None = None
    maximum: float | None = None

    def describe(self, value: float) -> str:
        if self.minimum is not None and self.maximum is not None:
            bounds = f"between {self.minimum:g} and {self.maximum:g}"
        elif self.minimum is not None:
            bounds = f">= {self.minimum:g}"
        else:
            bounds = f"<= {self.maximum:g}"
        return f"{self.field} = {value:g} ({bounds})"

    def matches(self, value: float | None) -> bool:
        # A material with no value for the field cannot satisfy a range over it.
        # Absent is "not known", never "passes" - the same stance the
        # partial-coverage application rules take.
        if value is None:
            return False
        if self.minimum is not None and value < self.minimum:
            return False
        return not (self.maximum is not None and value > self.maximum)


@dataclass(frozen=True)
class Constraints:
    """What a caller is asking for. Every field is ANDed together.

    Structural fields traverse an edge; `numeric` filters read attributes off
    the Material node.
    """

    elements: tuple[str, ...] = ()  # must contain ALL of these
    application_domain: str | None = None
    property_classes: tuple[tuple[str, str], ...] = ()  # (kind, name)
    oxidation_states: tuple[str, ...] = ()
    formula_pattern: str | None = None
    crystal_system: str | None = None
    chemical_system: str | None = None
    numeric: tuple[NumericFilter, ...] = ()

    def is_empty(self) -> bool:
        return not any(
            (
                self.elements,
                self.application_domain,
                self.property_classes,
                self.oxidation_states,
                self.formula_pattern,
                self.crystal_system,
                self.chemical_system,
                self.numeric,
            )
        )


@dataclass(frozen=True)
class Evidence:
    """One checkable statement about why a material was returned.

    `source` names where it came from - an edge type, or "attribute" - so a
    consumer can tell a traversed relationship apart from a stored number.
    """

    claim: str
    source: str
    detail: str | None = None

    def __str__(self) -> str:
        return f"{self.claim} [{self.source}]" + (f": {self.detail}" if self.detail else "")


@dataclass
class MaterialHit:
    material_id: str
    formula: str
    properties: dict[str, Any] = field(default_factory=dict)
    evidence: list[Evidence] = field(default_factory=list)

    @property
    def band_gap(self) -> float | None:
        return _as_float(self.properties.get("band_gap"))

    @property
    def energy_above_hull(self) -> float | None:
        return _as_float(self.properties.get("energy_above_hull"))


def _as_float(value: Any) -> float | None:
    """Coerce a stored property to float.

    Cognee round-trips some values as strings, so comparing them raw silently
    does the wrong thing ("0.9" > "0.15" is True as text, False as numbers).
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# --- resolving a name to the node traversal starts from ---------------------

# Each structural constraint: the node type it names, the property holding its
# name, and the edge that points at it from a Material.
_STRUCTURAL_LOOKUPS = {
    "element": ("Element", "symbol", "contains"),
    "application_domain": ("ApplicationDomain", "name", "suitable_for"),
    "oxidation_state": ("OxidationState", "species", "has_oxidation_state"),
    "formula_pattern": ("FormulaPattern", "pattern", "has_formula_pattern"),
    "chemical_system": ("ChemicalSystem", "chemsys", "member_of"),
}


def _materials_pointing_at(graph: Graph, node_id: str, relation: str) -> set[str]:
    return {
        source_id
        for source_id, props in graph.sources(node_id, relation)
        if props.get("type") == "Material"
    }


def _resolve(graph: Graph, kind: str, name: str) -> tuple[str, dict]:
    node_type, name_field, _relation = _STRUCTURAL_LOOKUPS[kind]
    found = graph.find_node(node_type, **{name_field: name})
    if found is None:
        available = sorted(props.get(name_field) for _i, props in graph.nodes_of_type(node_type))
        raise UnknownEntity(
            f"no {node_type} named {name!r} in the graph. "
            f"Known values: {', '.join(str(a) for a in available[:12])}"
            + (f" ... (+{len(available) - 12} more)" if len(available) > 12 else "")
        )
    return found


def materials_with(graph: Graph, kind: str, name: str) -> set[str]:
    """Ids of every Material linked to the named node. One hop."""
    node_id, _props = _resolve(graph, kind, name)
    _node_type, _name_field, relation = _STRUCTURAL_LOOKUPS[kind]
    return _materials_pointing_at(graph, node_id, relation)


def materials_in_class(graph: Graph, kind: str, name: str) -> set[str]:
    """Materials classified into a PropertyClass bucket, e.g. ("band_gap", "wide_gap").

    PropertyClass is the one lookup needing two fields to identify it: `kind`
    separates independent bucket families, so "stable" (stability) and
    "wide_gap" (band_gap) cannot collide.
    """
    found = graph.find_node("PropertyClass", kind=kind, name=name)
    if found is None:
        available = sorted(
            f"{p.get('kind')}/{p.get('name')}" for _i, p in graph.nodes_of_type("PropertyClass")
        )
        raise UnknownEntity(
            f"no PropertyClass {kind!r}/{name!r} in the graph. Known: {', '.join(available)}"
        )
    return _materials_pointing_at(graph, found[0], "classified_as")


def materials_in_crystal_system(graph: Graph, name: str) -> set[str]:
    """Materials in a crystal system - two hops, not one.

    Material -> SpaceGroup -> CrystalSystem. There is no direct edge, because
    the space group is the finer-grained fact and the crystal system is derived
    from it; storing both directly would let them contradict each other. This is
    the multi-hop traversal a flat table cannot express.
    """
    found = graph.find_node("CrystalSystem", name=name)
    if found is None:
        available = sorted(p.get("name") for _i, p in graph.nodes_of_type("CrystalSystem"))
        raise UnknownEntity(
            f"no CrystalSystem named {name!r}. Known: {', '.join(str(a) for a in available)}"
        )
    materials: set[str] = set()
    for space_group_id, _props in graph.sources(found[0], "crystal_system"):
        materials |= _materials_pointing_at(graph, space_group_id, "has_space_group")
    return materials


# --- the query itself --------------------------------------------------------


def find_materials(
    graph: Graph,
    constraints: Constraints,
    limit: int | None = None,
    order_by: str = "energy_above_hull",
) -> list[MaterialHit]:
    """Every material satisfying every constraint, with the evidence for each.

    Ordered by `order_by` ascending, defaulting to energy_above_hull so the most
    thermodynamically stable candidates come first. Ranking is a plain sort on a
    stored value rather than a learned or blended score, so the ordering is as
    explainable as the filtering.
    """
    if constraints.is_empty():
        raise ValueError("no constraints given - refusing to return the entire graph")

    # None means "not yet narrowed", which is distinct from an empty set
    # meaning "narrowed to nothing".
    candidates: set[str] | None = None

    def narrow(ids: set[str]) -> None:
        nonlocal candidates
        candidates = ids if candidates is None else (candidates & ids)

    for symbol in constraints.elements:
        narrow(materials_with(graph, "element", symbol))
    if constraints.application_domain:
        narrow(materials_with(graph, "application_domain", constraints.application_domain))
    for species in constraints.oxidation_states:
        narrow(materials_with(graph, "oxidation_state", species))
    if constraints.formula_pattern:
        narrow(materials_with(graph, "formula_pattern", constraints.formula_pattern))
    if constraints.chemical_system:
        narrow(materials_with(graph, "chemical_system", constraints.chemical_system))
    for kind, name in constraints.property_classes:
        narrow(materials_in_class(graph, kind, name))
    if constraints.crystal_system:
        narrow(materials_in_crystal_system(graph, constraints.crystal_system))

    if candidates is None:
        # Only numeric filters were given, so start from every material.
        candidates = {node_id for node_id, _props in graph.nodes_of_type("Material")}

    hits: list[MaterialHit] = []
    for material_id in candidates:
        properties = graph.node[material_id]
        if not all(f.matches(_as_float(properties.get(f.field))) for f in constraints.numeric):
            continue
        hits.append(
            MaterialHit(
                material_id=properties.get("material_id", material_id),
                formula=properties.get("formula", "?"),
                properties=properties,
                evidence=collect_evidence(graph, material_id, constraints),
            )
        )

    # Materials missing the ordering field sort last rather than crashing the
    # comparison or silently jumping to the front.
    hits.sort(key=lambda h: (_as_float(h.properties.get(order_by)) is None,
                             _as_float(h.properties.get(order_by)) or 0.0))
    return hits[:limit] if limit else hits


def collect_evidence(graph: Graph, material_id: str, constraints: Constraints) -> list[Evidence]:
    """The specific graph facts that made this material match.

    Only facts relevant to the constraints asked about - dumping every edge a
    material has would bury the reason it was returned among ~19 others, and
    this list is meant to be the sole context the generation step sees.
    """
    properties = graph.node[material_id]
    evidence: list[Evidence] = []

    for symbol in constraints.elements:
        for target_id, edge_props in graph.target_edges(material_id, "contains"):
            if graph.node[target_id].get("symbol") == symbol:
                fraction = edge_weight(edge_props)
                detail = f"{fraction:.1%} of atoms" if fraction is not None else None
                evidence.append(Evidence(f"contains {symbol}", "contains", detail))

    if constraints.application_domain:
        for target_id, edge_props in graph.target_edges(material_id, "suitable_for"):
            if graph.node[target_id].get("name") == constraints.application_domain:
                # The stored rule text - the audit trail Phase 1 built in.
                evidence.append(
                    Evidence(
                        f"suitable for {constraints.application_domain}",
                        "suitable_for",
                        edge_rule(edge_props),
                    )
                )

    for kind, name in constraints.property_classes:
        for _target_id, class_props in graph.targets(material_id, "classified_as"):
            if class_props.get("kind") == kind and class_props.get("name") == name:
                evidence.append(
                    Evidence(f"classified as {name}", "classified_as", class_props.get("description"))
                )

    for species in constraints.oxidation_states:
        for _target_id, state_props in graph.targets(material_id, "has_oxidation_state"):
            if state_props.get("species") == species:
                evidence.append(Evidence(f"contains {species}", "has_oxidation_state"))

    if constraints.formula_pattern:
        evidence.append(
            Evidence(f"formula pattern {constraints.formula_pattern}", "has_formula_pattern")
        )

    if constraints.chemical_system:
        evidence.append(
            Evidence(f"in chemical system {constraints.chemical_system}", "member_of")
        )

    if constraints.crystal_system:
        for space_group_id, sg_props in graph.targets(material_id, "has_space_group"):
            for _cs_id, cs_props in graph.targets(space_group_id, "crystal_system"):
                if cs_props.get("name") == constraints.crystal_system:
                    evidence.append(
                        Evidence(
                            f"{constraints.crystal_system} crystal system",
                            "has_space_group -> crystal_system",
                            f"space group {sg_props.get('symbol')}",
                        )
                    )

    for numeric in constraints.numeric:
        value = _as_float(properties.get(numeric.field))
        if value is not None:
            evidence.append(Evidence(numeric.describe(value), "attribute"))

    return evidence


def similar_materials(graph: Graph, formula: str, limit: int = 5) -> list[MaterialHit]:
    """Nearest neighbours of a material by its similar_to edges.

    In Phase 1 those weights are cosine similarity over a 10-dimensional
    property vector. Phase 3 replaces the weight with CGCNN structural
    similarity without changing this edge - so this function keeps working, and
    starts answering a better question, with no change here.
    """
    found = graph.find_node("Material", formula=formula)
    if found is None:
        raise UnknownEntity(f"no Material with formula {formula!r} in the graph")

    neighbours = [
        (target_id, edge_weight(edge_props) or 0.0)
        for target_id, edge_props in graph.target_edges(found[0], "similar_to")
    ]
    neighbours.sort(key=lambda pair: -pair[1])

    return [
        MaterialHit(
            material_id=graph.node[target_id].get("material_id", target_id),
            formula=graph.node[target_id].get("formula", "?"),
            properties=graph.node[target_id],
            evidence=[
                Evidence(
                    f"similar to {formula}",
                    "similar_to",
                    f"cosine similarity {weight:.3f} over 10 z-scored properties",
                )
            ],
        )
        for target_id, weight in neighbours[:limit]
    ]


def connections_between(graph: Graph, formula_a: str, formula_b: str) -> list[Evidence]:
    """What two materials have in common: shared elements, classes, domains, etc.

    The "what connects X and Y?" question a flat table cannot answer at all,
    and the clearest demonstration of why this is a graph.
    """
    a = graph.find_node("Material", formula=formula_a)
    b = graph.find_node("Material", formula=formula_b)
    for formula, node in ((formula_a, a), (formula_b, b)):
        if node is None:
            raise UnknownEntity(f"no Material with formula {formula!r} in the graph")

    shared: list[Evidence] = []
    labels = {
        "contains": ("symbol", "element"),
        "classified_as": ("name", "property class"),
        "suitable_for": ("name", "application domain"),
        "has_oxidation_state": ("species", "oxidation state"),
        "has_formula_pattern": ("pattern", "formula pattern"),
        "member_of": ("chemsys", "chemical system"),
        "has_space_group": ("symbol", "space group"),
    }
    for relation, (name_field, label) in labels.items():
        targets_a = {tid for tid, _p in graph.targets(a[0], relation)}
        targets_b = {tid for tid, _p in graph.targets(b[0], relation)}
        for target_id in sorted(targets_a & targets_b):
            shared.append(
                Evidence(
                    f"both are linked to {label} {graph.node[target_id].get(name_field)}",
                    relation,
                )
            )
    return shared
