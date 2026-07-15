"""Cognee DataPoint schema for the materials knowledge graph.

Design rule: continuous numerics stay as attributes on Material (exact values
for filtering/comparison); anything used for grouping or traversal becomes a
node; derived similarity becomes a weighted edge.

Every node type that multiple materials can share (Element, CrystalSystem,
SpaceGroup, ChemicalSystem, PropertyClass, ApplicationDomain) declares
``identity_fields`` so repeated runs and repeated references deduplicate onto
the same graph node instead of creating copies.
"""

from __future__ import annotations

from typing import Any

from pydantic import SkipValidation

from cognee.low_level import DataPoint
from cognee.infrastructure.engine.models.Edge import Edge  # noqa: F401  (re-exported for builders)


class CrystalSystem(DataPoint):
    name: str  # e.g. "Cubic", "Hexagonal", "Orthorhombic"
    metadata: dict = {"index_fields": ["name"], "identity_fields": ["name"]}


class SpaceGroup(DataPoint):
    symbol: str  # e.g. "Fd-3m"
    number: int
    crystal_system: CrystalSystem
    metadata: dict = {"index_fields": ["symbol"], "identity_fields": ["symbol", "number"]}


class ChemicalSystem(DataPoint):
    chemsys: str  # e.g. "Fe-Li-O-P"
    metadata: dict = {"index_fields": ["chemsys"], "identity_fields": ["chemsys"]}


class Element(DataPoint):
    symbol: str
    name: str
    atomic_number: int
    group: int
    period: int
    electronegativity: float
    covalent_radius_pm: float
    category: str
    metadata: dict = {"index_fields": ["name"], "identity_fields": ["symbol"]}


class PropertyClass(DataPoint):
    """A categorical bucket a material can be classified into.

    kind distinguishes independent bucket families (band_gap, stability,
    magnetic_ordering) so e.g. "stable" (stability) and "wide_gap" (band_gap)
    are both reachable without collision.
    """

    kind: str  # "band_gap" | "stability" | "magnetic_ordering"
    name: str  # e.g. "wide_gap", "stable", "AFM"
    description: str
    metadata: dict = {
        "index_fields": ["description"],
        "identity_fields": ["kind", "name"],
    }


class ApplicationDomain(DataPoint):
    name: str  # e.g. "battery_cathode"
    description: str
    metadata: dict = {"index_fields": ["description"], "identity_fields": ["name"]}


class Material(DataPoint):
    material_id: str
    formula: str
    description: str  # robocrystallographer text, embeddable

    band_gap: float
    is_gap_direct: bool
    is_metal: bool
    formation_energy_per_atom: float
    energy_above_hull: float
    is_stable: bool
    density: float
    volume: float
    nsites: int
    total_magnetization: float
    ordering: str
    theoretical: bool

    # These are relationship fields: builders pass a (Edge(...), target(s)) tuple
    # as the value (see build_material in enrich.py). SkipValidation is required
    # because pydantic cannot type-check a tuple-of-(Edge, DataPoint) shape.
    contains: SkipValidation[Any] = None  # (Edge, list[Element])
    has_space_group: SkipValidation[Any] = None  # (Edge, SpaceGroup)
    member_of: SkipValidation[Any] = None  # (Edge, ChemicalSystem)
    classified_as: SkipValidation[Any] = None  # (Edge, list[PropertyClass])
    suitable_for: SkipValidation[Any] = None  # (Edge, list[ApplicationDomain])
    similar_to: SkipValidation[Any] = None  # (Edge, list[Material])

    metadata: dict = {
        "index_fields": ["formula", "description"],
        "identity_fields": ["material_id"],
    }
