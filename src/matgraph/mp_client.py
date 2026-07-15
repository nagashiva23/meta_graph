"""Materials Project REST client: fetch + cache one canonical record per formula."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

MP_BASE_URL = "https://api.materialsproject.org"

SUMMARY_FIELDS = [
    "material_id",
    "formula_pretty",
    "chemsys",
    "elements",
    "nsites",
    "volume",
    "density",
    "band_gap",
    "is_gap_direct",
    "is_metal",
    "formation_energy_per_atom",
    "energy_above_hull",
    "is_stable",
    "total_magnetization",
    "ordering",
    "symmetry",
    "theoretical",
]


class MPClientError(RuntimeError):
    pass


def _api_key() -> str:
    key = os.environ.get("MP_API_KEY")
    if not key:
        raise MPClientError("MP_API_KEY is not set (check your .env file)")
    return key


def _get(path: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    resp = requests.get(
        f"{MP_BASE_URL}{path}",
        params=params,
        headers={"X-API-KEY": _api_key()},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["data"]


def fetch_summary_candidates(formula: str) -> list[dict[str, Any]]:
    """Fetch all summary docs for a formula (may include multiple polymorphs)."""
    return _get(
        "/materials/summary/",
        {"formula": formula, "_fields": ",".join(SUMMARY_FIELDS), "_limit": 1000},
    )


def select_canonical(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick the ground-state polymorph: lowest energy above the convex hull."""
    if not candidates:
        raise MPClientError("no candidates to select from")
    return min(candidates, key=lambda d: d.get("energy_above_hull", float("inf")))


def fetch_robocrys_description(material_id: str) -> str | None:
    """Fetch the human-readable structure description for a material, if available."""
    docs = _get(
        "/materials/robocrys/",
        {"material_ids": material_id, "_fields": "material_id,description", "_limit": 1},
    )
    if not docs:
        return None
    return docs[0].get("description")


def fetch_material(formula: str) -> dict[str, Any]:
    """Fetch and assemble the full raw record for one formula's canonical material."""
    candidates = fetch_summary_candidates(formula)
    material = select_canonical(candidates)
    material["description"] = fetch_robocrys_description(material["material_id"])
    return material


def cache_path(data_dir: Path, formula: str) -> Path:
    return data_dir / f"{formula}.json"


def fetch_and_cache(formula: str, data_dir: Path, refresh: bool = False) -> dict[str, Any]:
    """Fetch a material's raw record, using the on-disk cache unless refresh=True."""
    path = cache_path(data_dir, formula)
    if path.exists() and not refresh:
        logger.info("cache hit: %s", path)
        return json.loads(path.read_text())

    logger.info("fetching from Materials Project: %s", formula)
    material = fetch_material(formula)
    data_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(material, indent=2))
    return material
