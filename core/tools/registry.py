from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from core.model.state import CodingAgentState


ExtractorFunc = Callable[[Any, Dict[str, Any], Optional[CodingAgentState]], Dict[str, Any]]

TOOL_EXTRACTOR_REGISTRY: Dict[str, ExtractorFunc] = {}


def register_extractor(tool_name: str):
    def decorator(func: ExtractorFunc):
        TOOL_EXTRACTOR_REGISTRY[tool_name] = func
        return func

    return decorator