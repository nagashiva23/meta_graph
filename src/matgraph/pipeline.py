"""Cognee low-level pipeline wiring for the materials knowledge graph.

Mirrors cognee's low_level pipeline pattern (Task(build) -> Task(add_data_points))
but the build step is our fully deterministic enrich.build_graph, so no LLM call
happens anywhere in the graph-construction path.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from cognee import config, prune
from cognee.low_level import setup
from cognee.modules.data.methods import create_authorized_dataset
from cognee.modules.pipelines.operations import run_pipeline
from cognee.modules.users.methods import get_default_user
from cognee.pipelines import Task
from cognee.tasks.storage import add_data_points

from matgraph.enrich import build_graph, load_config, load_elements_table, load_raw_materials
from matgraph.schema import Material

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
COGNEE_SYSTEM_DIR = ROOT / ".cognee_system"
DATASET_NAME = "matgraph_phase1"


def build_materials(_: list[Any]) -> list[Material]:
    """Pipeline task: load cached MP records + config, return built DataPoint graph."""
    config_data = load_config()
    elements_table = load_elements_table()
    raw_materials = load_raw_materials(config_data)
    materials = build_graph(raw_materials, config_data, elements_table)
    logger.info("built %d Material nodes from %d cached records", len(materials), len(raw_materials))
    return materials


async def run(reset: bool = True) -> None:
    """Build the knowledge graph and write it into cognee's stores."""
    config.system_root_directory(str(COGNEE_SYSTEM_DIR))

    if reset:
        await prune.prune_system(metadata=True)

    await setup()

    user = await get_default_user()
    dataset = await create_authorized_dataset(DATASET_NAME, user)

    tasks = [Task(build_materials), Task(add_data_points)]
    pipeline = run_pipeline(tasks, [None], [dataset.id], user, "matgraph_phase1_pipeline")

    async for status in pipeline:
        logger.info("pipeline status: %s", status)
