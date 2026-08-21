"""Verify the already-built graph without rebuilding it.

scripts/build_graph.py runs verification as its last step, but a full rebuild
takes minutes - too slow a loop when the thing being changed is a *check*
rather than the graph. This runs verification alone against whatever is
currently in the store.

Usage:
    uv run python scripts/verify_graph.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")


async def main() -> None:
    from cognee import config

    from matgraph.pipeline import COGNEE_SYSTEM_DIR

    # Must be set before touching any cognee store, or it looks for its
    # databases inside the installed package instead of this project.
    config.system_root_directory(str(COGNEE_SYSTEM_DIR))

    from matgraph.verify import verify

    await verify()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    asyncio.run(main())
