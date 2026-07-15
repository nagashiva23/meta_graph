"""Fetch and cache raw Materials Project records for the configured clusters."""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from matgraph.mp_client import fetch_and_cache_bulk  # noqa: E402

DATA_DIR = ROOT / "data" / "raw"
CONFIG_PATH = ROOT / "config" / "materials.yaml"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="ignore cache, re-fetch from MP")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    config = yaml.safe_load(CONFIG_PATH.read_text())
    materials = fetch_and_cache_bulk(
        config["clusters"], config["exclude_elements"], DATA_DIR, refresh=args.refresh
    )

    by_cluster = Counter(m.get("_cluster", "?") for m in materials)
    print(f"\n{len(materials)} unique materials cached to {DATA_DIR}\n")
    for cluster_name, count in sorted(by_cluster.items(), key=lambda kv: -kv[1]):
        print(f"  {cluster_name:28s} {count}")


if __name__ == "__main__":
    main()
