from core.model.state import CodingAgentState
from core.tools.files import read_file, list_files, find_files, get_directory_tree
from core.tools.search import scan_project, grep
from typing import Any, Callable, Dict
from datetime import datetime

ExtractorFunc = Callable[[Any, Dict[str, Any]], Dict[str, Any]]

# Central registry containing all tool extraction strategies
TOOL_EXTRACTOR_REGISTRY: Dict[str, ExtractorFunc] = {}

READ_ONLY_TOOLS = [ list_files, find_files, get_directory_tree, scan_project, grep, read_file ]

WRITE_TOOLS = [ ]

ALL_TOOLS = READ_ONLY_TOOLS + WRITE_TOOLS

TOOLS_DICT = { tool.name: tool for tool in ALL_TOOLS }

def register_extractor(tool_name: str):
    """Decorator to easily register an extraction strategy for a tool."""
    def decorator(func: ExtractorFunc):
        TOOL_EXTRACTOR_REGISTRY[tool_name] = func
        return func
    return decorator

