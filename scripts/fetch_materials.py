"""Fetch and cache raw Materials Project records for the configured seed set."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from matgraph.mp_client import fetch_and_cache  # noqa: E402

DATA_DIR = ROOT / "data" / "raw"
CONFIG_PATH = ROOT / "config" / "materials.yaml"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="ignore cache, re-fetch from MP")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    config = yaml.safe_load(CONFIG_PATH.read_text())
    formulas = [m["formula"] for m in config["materials"]]

    for formula in formulas:
        material = fetch_and_cache(formula, DATA_DIR, refresh=args.refresh)
        print(
            f"{formula:10s} -> {material['material_id']:12s} "
            f"band_gap={material['band_gap']:.3f} "
            f"E_hull={material['energy_above_hull']:.4f} "
            f"stable={material['is_stable']}"
        )


if __name__ == "__main__":
    main()
