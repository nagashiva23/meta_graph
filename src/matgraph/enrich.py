"""Deterministic enrichment: raw MP records -> Cognee DataPoint graph.

No LLM calls anywhere in this module. Property classification, application
domain assignment, and material-material similarity are all rule-based so
every edge in the resulting graph is traceable to a concrete threshold or
formula (see config/materials.yaml).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import yaml

from matgraph.schema import (
    ApplicationDomain,
    ChemicalSystem,
    CrystalSystem,
    Edge,
    Element,
    Material,
    PropertyClass,
    SpaceGroup,
)

ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = ROOT / "config" / "materials.yaml"
ELEMENTS_PATH = ROOT / "config" / "elements.json"
DATA_DIR = ROOT / "data" / "raw"


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text())


def load_elements_table() -> dict[str, dict[str, Any]]:
    return json.loads(ELEMENTS_PATH.read_text())


def load_raw_materials(config: dict[str, Any]) -> list[dict[str, Any]]:
    formulas = [m["formula"] for m in config["materials"]]
    raw = []
    for formula in formulas:
        path = DATA_DIR / f"{formula}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"missing cached record for {formula} at {path} "
                "(run scripts/fetch_materials.py first)"
            )
        raw.append(json.loads(path.read_text()))
    return raw


# --- classification -----------------------------------------------------


def _bucket(value: float, buckets: list[dict[str, Any]]) -> dict[str, Any]:
    for bucket in buckets:
        lo = bucket.get("min", -math.inf)
        hi = bucket.get("max", math.inf)
        if lo <= value <= hi:
            return bucket
    raise ValueError(f"value {value} did not match any bucket in {buckets}")


def classify_band_gap(band_gap: float, config: dict[str, Any]) -> dict[str, Any]:
    bucket = _bucket(band_gap, config["band_gap_buckets"])
    return {
        "kind": "band_gap",
        "name": bucket["name"],
        "description": f"band_gap classified as '{bucket['name']}' (value={band_gap:.3f} eV)",
    }


def classify_stability(e_hull: float, config: dict[str, Any]) -> dict[str, Any]:
    bucket = _bucket(e_hull, config["stability_buckets"])
    return {
        "kind": "stability",
        "name": bucket["name"],
        "description": (
            f"stability classified as '{bucket['name']}' "
            f"(energy_above_hull={e_hull:.4f} eV/atom)"
        ),
    }


def classify_ordering(ordering: str) -> dict[str, Any] | None:
    if not ordering or ordering == "Unknown":
        return None
    return {
        "kind": "magnetic_ordering",
        "name": ordering,
        "description": f"magnetic ordering reported by Materials Project as '{ordering}'",
    }


# --- application domain rules --------------------------------------------


def apply_application_rules(
    material: dict[str, Any], config: dict[str, Any]
) -> list[tuple[str, str]]:
    """Return [(domain_name, rule_text), ...] for every rule this material satisfies."""
    elements = set(material["elements"])
    transition_metals = set(config["transition_metals"])
    band_gap = material["band_gap"]
    is_metal = material["is_metal"]
    e_hull = material["energy_above_hull"]
    total_mag = abs(material["total_magnetization"])
    ordering = material["ordering"]

    matches: list[tuple[str, str]] = []
    rules_by_domain = {r["domain"]: r["rule"] for r in config["application_rules"]}

    if "Li" in elements and elements & transition_metals and e_hull <= 0.05:
        matches.append(("battery_cathode", rules_by_domain["battery_cathode"]))

    if 0 < band_gap <= 3.5 and not is_metal:
        matches.append(("semiconductor_device", rules_by_domain["semiconductor_device"]))

    if "O" in elements and 1.5 <= band_gap <= 3.5:
        matches.append(("photocatalyst", rules_by_domain["photocatalyst"]))

    if band_gap > 2 and total_mag < 0.1:
        matches.append(("dielectric", rules_by_domain["dielectric"]))

    if (ordering in {"FM", "FiM"} and total_mag >= 0.1) or ordering == "AFM":
        matches.append(("magnetic_material", rules_by_domain["magnetic_material"]))

    return matches


# --- similarity -----------------------------------------------------------

SIMILARITY_FIELDS = [
    "band_gap",
    "formation_energy_per_atom",
    "density",
    "total_magnetization",
]


def _zscore_vectors(materials: list[dict[str, Any]]) -> dict[str, list[float]]:
    columns = {f: [m[f] for m in materials] for f in SIMILARITY_FIELDS}
    stats = {}
    for f, values in columns.items():
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        std = math.sqrt(variance) or 1.0
        stats[f] = (mean, std)

    vectors = {}
    for m in materials:
        vectors[m["material_id"]] = [
            (m[f] - stats[f][0]) / stats[f][1] for f in SIMILARITY_FIELDS
        ]
    return vectors


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


def compute_similarity_edges(
    materials: list[dict[str, Any]], threshold: float
) -> dict[str, list[tuple[str, float]]]:
    """Return {material_id: [(other_material_id, weight), ...]} above threshold."""
    vectors = _zscore_vectors(materials)
    ids = list(vectors.keys())
    edges: dict[str, list[tuple[str, float]]] = {i: [] for i in ids}
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            sim = _cosine(vectors[ids[i]], vectors[ids[j]])
            if sim >= threshold:
                edges[ids[i]].append((ids[j], sim))
                edges[ids[j]].append((ids[i], sim))
    return edges


# --- graph construction -----------------------------------------------------


def build_graph(
    raw_materials: list[dict[str, Any]],
    config: dict[str, Any],
    elements_table: dict[str, dict[str, Any]],
) -> list[Material]:
    """Build the full set of interconnected DataPoint nodes for all materials."""

    element_cache: dict[str, Element] = {}
    crystal_system_cache: dict[str, CrystalSystem] = {}
    space_group_cache: dict[str, SpaceGroup] = {}
    chemsys_cache: dict[str, ChemicalSystem] = {}
    property_class_cache: dict[tuple[str, str], PropertyClass] = {}
    domain_cache: dict[str, ApplicationDomain] = {}

    def get_element(symbol: str) -> Element:
        if symbol not in element_cache:
            info = elements_table[symbol]
            element_cache[symbol] = Element(
                symbol=symbol,
                name=info["name"],
                atomic_number=info["atomic_number"],
                group=info["group"],
                period=info["period"],
                electronegativity=info["electronegativity"],
                covalent_radius_pm=info["covalent_radius_pm"],
                category=info["category"],
            )
        return element_cache[symbol]

    def get_crystal_system(name: str) -> CrystalSystem:
        if name not in crystal_system_cache:
            crystal_system_cache[name] = CrystalSystem(name=name)
        return crystal_system_cache[name]

    def get_space_group(symbol: str, number: int, crystal_system_name: str) -> SpaceGroup:
        key = f"{symbol}:{number}"
        if key not in space_group_cache:
            space_group_cache[key] = SpaceGroup(
                symbol=symbol,
                number=number,
                crystal_system=get_crystal_system(crystal_system_name),
            )
        return space_group_cache[key]

    def get_chemsys(chemsys: str) -> ChemicalSystem:
        if chemsys not in chemsys_cache:
            chemsys_cache[chemsys] = ChemicalSystem(chemsys=chemsys)
        return chemsys_cache[chemsys]

    def get_property_class(kind: str, name: str, description: str) -> PropertyClass:
        key = (kind, name)
        if key not in property_class_cache:
            property_class_cache[key] = PropertyClass(kind=kind, name=name, description=description)
        return property_class_cache[key]

    def get_domain(name: str) -> ApplicationDomain:
        if name not in domain_cache:
            rules_by_domain = {r["domain"]: r["rule"] for r in config["application_rules"]}
            domain_cache[name] = ApplicationDomain(
                name=name, description=rules_by_domain.get(name, name)
            )
        return domain_cache[name]

    similarity_edges = compute_similarity_edges(
        raw_materials, config["similarity"]["min_edge_weight"]
    )

    material_nodes: dict[str, Material] = {}
    for raw in raw_materials:
        contains = [get_element(sym) for sym in raw["elements"]]
        symmetry = raw["symmetry"]
        space_group = get_space_group(
            symmetry["symbol"], symmetry["number"], symmetry["crystal_system"]
        )
        chemsys = get_chemsys(raw["chemsys"])

        classifications = [
            classify_band_gap(raw["band_gap"], config),
            classify_stability(raw["energy_above_hull"], config),
        ]
        ordering_class = classify_ordering(raw["ordering"])
        if ordering_class:
            classifications.append(ordering_class)
        classified_as = [
            get_property_class(c["kind"], c["name"], c["description"]) for c in classifications
        ]

        domain_matches = apply_application_rules(raw, config)
        suitable_for = [
            (Edge(relationship_type="suitable_for", properties={"rule": rule}), get_domain(name))
            for name, rule in domain_matches
        ]

        material_nodes[raw["material_id"]] = Material(
            material_id=raw["material_id"],
            formula=raw["formula_pretty"],
            description=raw["description"] or f"{raw['formula_pretty']} (no description available)",
            band_gap=raw["band_gap"],
            is_gap_direct=raw["is_gap_direct"],
            is_metal=raw["is_metal"],
            formation_energy_per_atom=raw["formation_energy_per_atom"],
            energy_above_hull=raw["energy_above_hull"],
            is_stable=raw["is_stable"],
            density=raw["density"],
            volume=raw["volume"],
            nsites=raw["nsites"],
            total_magnetization=raw["total_magnetization"],
            ordering=raw["ordering"],
            theoretical=raw["theoretical"],
            contains=contains,
            has_space_group=(Edge(relationship_type="has_space_group"), space_group),
            member_of=(Edge(relationship_type="member_of"), chemsys),
            classified_as=classified_as,
            suitable_for=suitable_for,
            # similar_to filled in below, once every Material node exists.
        )

    for material_id, material in material_nodes.items():
        pairs = similarity_edges.get(material_id, [])
        material.similar_to = [
            (
                Edge(relationship_type="similar_to", weight=weight),
                material_nodes[other_id],
            )
            for other_id, weight in pairs
        ]

    return list(material_nodes.values())
