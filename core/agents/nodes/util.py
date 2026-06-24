from __future__ import annotations
from typing import Sequence
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool


def tools_from_runnable_config(runnable_config: RunnableConfig | None) -> list[BaseTool]:
    if not runnable_config:
        return []

    direct_tools = runnable_config.get("tools")
    if isinstance(direct_tools, Sequence):
        return list(direct_tools)  # type: ignore[arg-type]

    configurable = runnable_config.get("configurable")
    if isinstance(configurable, dict):
        configurable_tools = configurable.get("tools")
        if isinstance(configurable_tools, Sequence):
            return list(configurable_tools)  # type: ignore[arg-type]

    return []