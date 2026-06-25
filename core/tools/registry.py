from __future__ import annotations

from typing import Any, Callable, Dict


ExtractorFunc = Callable[[Any, Dict[str, Any]], Dict[str, Any]]

TOOL_EXTRACTOR_REGISTRY: Dict[str, ExtractorFunc] = {}


def register_extractor(tool_name: str):
    def decorator(func: ExtractorFunc):
        TOOL_EXTRACTOR_REGISTRY[tool_name] = func
        return func

    return decorator