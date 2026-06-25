from core.tools.files import read_file, list_files, find_files, get_directory_tree
from core.tools.search import scan_project, grep
from core.tools.registry import TOOL_EXTRACTOR_REGISTRY, register_extractor

READ_ONLY_TOOLS = [ list_files, find_files, get_directory_tree, scan_project, grep, read_file ]

WRITE_TOOLS = [ ]

ALL_TOOLS = READ_ONLY_TOOLS + WRITE_TOOLS

TOOLS_DICT = { tool.name: tool for tool in ALL_TOOLS }

