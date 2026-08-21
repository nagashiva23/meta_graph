"""Shared fixtures for the enrichment tests.

Every function under test reads *raw* Materials Project JSON (the shape cached
in data/raw/<material_id>.json), not Material DataPoints - so the fixtures here
mirror a real /materials/summary/ response rather than the graph schema.

Tests load the real config/materials.yaml rather than a synthetic config, so a
change to the thresholds or rules in that file is caught here too. The config
is as much a part of the behaviour as the Python is.
"""

from __future__ import annotations

from typing import Any

import pytest

from matgraph.enrich import load_config


def make_raw(**overrides: Any) -> dict[str, Any]:
    """A complete, valid raw MP record. Override any field per test.

    Defaults describe LiFeO2: stable (on hull), non-magnetic, band gap 1.62 eV,
    containing Li + a transition metal + oxygen - chosen because it satisfies
    several application rules at once, which is the realistic case.
    """
    record: dict[str, Any] = {
        "material_id": "mp-test-1",
        "formula_pretty": "LiFeO2",
        "chemsys": "Fe-Li-O",
        "elements": ["Li", "Fe", "O"],
        "symmetry": {"symbol": "Pmmn", "number": 59, "crystal_system": "Orthorhombic"},
        "band_gap": 1.62,
        "is_gap_direct": False,
        "is_metal": False,
        "formation_energy_per_atom": -2.1,
        "energy_above_hull": 0.0,
        "is_stable": True,
        "density": 4.3,
        "volume": 70.2,
        "nsites": 8,
        "total_magnetization": 0.0,
        "ordering": "NM",
        "theoretical": False,
        "description": "LiFeO2 crystallizes in the orthorhombic Pmmn space group.",
        "formula_anonymous": "ABC2",
        "possible_species": ["Li+", "Fe3+", "O2-"],
    }
    record.update(overrides)
    return record


@pytest.fixture
def raw() -> dict[str, Any]:
    return make_raw()


@pytest.fixture(scope="session")
def config() -> dict[str, Any]:
    """The real config/materials.yaml."""
    return load_config()
