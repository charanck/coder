from core.tools.files import read_file, list_files, find_files
from core.tools.search import scan_project, grep

READ_ONLY_TOOLS = [ list_files, find_files, scan_project, grep, read_file ]

WRITE_TOOLS = [ ]

ALL_TOOLS = READ_ONLY_TOOLS + WRITE_TOOLS

TOOLS_DICT = { tool.name: tool for tool in ALL_TOOLS }

