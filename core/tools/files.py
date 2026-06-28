from __future__ import annotations
import datetime
import fnmatch
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from langchain.tools import tool
from core.common.model import get_model
from langchain_core.prompts import ChatPromptTemplate
from core.tools.registry import register_extractor
from core.client.lsp.manager import lsp_manager
from pydantic import BaseModel, Field


# Mapping standard LSP SymbolKind integers to human-readable identifiers
LSP_SYMBOL_KIND_MAP = {
    1: "File", 2: "Module", 3: "Namespace", 4: "Package", 5: "Class",
    6: "Method", 7: "Property", 8: "Field", 9: "Constructor", 10: "Enum",
    11: "Interface", 12: "Function", 13: "Variable", 14: "Constant",
    15: "String", 16: "Number", 17: "Boolean", 18: "Array", 19: "Object",
    20: "Key", 21: "Null", 22: "EnumMember", 23: "Struct", 24: "Event",
    25: "Operator", 26: "TypeParameter"
}

DEFAULT_IGNORE_PATTERNS = {
    # Python
    "__pycache__",
    "*.pyc",
    "*.pyo",
    "*.pyd",
    "*.egg-info",
    "*.dist-info",
    ".eggs",
    "build",
    "dist",

    # Virtual environments
    ".venv",
    "venv",
    "env",

    # Git
    ".git",
    ".hg",
    ".svn",

    # Node
    "node_modules",
    ".npm",

    # IDE
    ".idea",
    ".vscode",

    # OS
    ".DS_Store",
    "Thumbs.db",

    # Caches
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".coverage",
    "htmlcov",
}

def _should_ignore(
    name: str,
    include_hidden: bool,
    patterns: set[str],
) -> bool:
    if not include_hidden and name.startswith("."):
        return True

    return any(fnmatch.fnmatch(name, p) for p in patterns)


def _format_lines(
    lines: list[str],
    start: int,
) -> str:
    width = len(str(start + len(lines)))

    return "".join(
        f"{i:{width}} | {line}"
        for i, line in enumerate(lines, start=start)
    )

@tool
def read_file(
    file_path: str,
    start_line: int | None = None,
    end_line: int | None = None,
) -> str:
    """
    Read a file.

    Supports partial reads and returns numbered lines.

    Args:
        file_path (str): The path to the file to read.
        start_line (int | None): The starting line number (1-based). If None, starts from the beginning.
        end_line (int | None): The ending line number (1-based). If None, reads to the end of the file.
    """

    try:
        path = Path(file_path)

        if not path.exists():
            return f"Error: File '{file_path}' does not exist."

        if not path.is_file():
            return f"Error: '{file_path}' is not a file."

        with path.open(
            "r",
            encoding="utf-8",
            errors="replace",
        ) as f:
            lines = f.readlines()

        total = len(lines)

        start = 1 if start_line is None else max(1, start_line)
        end = total if end_line is None else min(total, end_line)

        if start > end:
            return "Error: Invalid line range."

        selected = lines[start - 1 : end]

        return (
            f"File: {path}\n"
            f"Lines: {start}-{end} of {total}\n"
            f"{'-'*60}\n"
            f"{_format_lines(selected, start)}"
        )

    except Exception as e:
        return f"Error reading file: {e}"

@register_extractor("read_file")
def extract_read_file(result: Any, args: Dict[str, Any]) -> Dict[str, Any]:
    """Extracts file content metadata using LSP for structural data and invariant 
    generation, reserving the LLM strictly for high-level semantic summarization.
    """
    file_path = args.get("path") or args.get("file_path") or "unknown"
    raw_content = str(result.content) if hasattr(result, "content") else str(result)
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

    file_summary = "Summary extraction unavailable."
    lsp_facts = []
    parsed_symbols = []

    try:
        workspace_dir = str(Path(file_path).parent)
        lsp_client = lsp_manager.get_by_extension(file_path, workspace=workspace_dir)

        if lsp_client is None:
            # TODO: as fallback we should extract symbols using regex or a simple parser for known languages or use llm to extract symbols if lsp is not available
            raise RuntimeError(f"No LSP client available for the file extension of '{file_path}'")

        raw_symbols = lsp_client.extract_document_symbols(str(Path(file_path).resolve().as_uri())) or []
        
        def parse_lsp_symbols(symbols: list[dict], parent_name: str = ""):
            for sym in symbols:
                name = sym.get("name", "unknown")
                kind_id = sym.get("kind", 0)
                kind_name = LSP_SYMBOL_KIND_MAP.get(kind_id, f"Unknown({kind_id})")
                
                sym_range = sym.get("range", {})
                start_line = sym_range.get("start", {}).get("line", 0) + 1
                end_line = sym_range.get("end", {}).get("line", 0) + 1
                
                full_identity = f"{parent_name}.{name}" if parent_name else name

                parsed_symbols.append({
                    "name": name,
                    "kind": kind_name,
                    "start_line": start_line,
                    "end_line": end_line,
                    "identity": full_identity
                })
                
                if kind_name in ["Class", "Interface", "Struct", "Enum"]:
                    lsp_facts.append(
                        f"Defines core structure '{full_identity}' ({kind_name}) from lines {start_line} to {end_line}."
                    )
                elif kind_name in ["Method", "Function"] and not parent_name:
                    lsp_facts.append(
                        f"Exposes top-level executable capability '{full_identity}' ({kind_name}) starting at line {start_line}."
                    )
                
                if "children" in sym and sym["children"]:
                    parse_lsp_symbols(sym["children"], parent_name=full_identity)

        parse_lsp_symbols(raw_symbols)

        model = get_model()
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a core architecture analyzer. Review the provided source code and write a highly concise, "
                "1-2 sentence functional summary describing its business logic and architectural purpose.\n"
                "DO NOT include markdown, code blocks, lists, or structural metadata. Return ONLY the plain text summary."
            ),
            ("human", "File Path: {file_path}\n\nContent:\n{content}")
        ])
        
        chain = prompt | model
        response = chain.invoke({"file_path": file_path, "content": raw_content})
        response_content = response.content if hasattr(response, "content") else response
        file_summary = str(response_content).strip()

    except Exception as e:
        file_summary = f"Systematic analysis completed with degradation. Error mapping structural layout: {str(e)}"

    workspace_payload = {
        "full_content": raw_content,
        "summary": file_summary,
        "lines": len(raw_content.splitlines()),
        "symbols": parsed_symbols,  
        "last_read": timestamp,
    }

    known_facts_payload = [
        {"source": file_path, "fact": fact, "extracted_at": timestamp}
        for fact in lsp_facts
    ]

    return {
        "workspace_update": {file_path: workspace_payload},
        "artifacts_update": {file_path: "read"},
        "known_facts_update": known_facts_payload,
    }

class FileEntry(BaseModel):
    path: str = Field(description="The relative path from the scanned directory root.")
    type: Literal["file", "directory"] = Field(description="The type indicator of the entity.")


class FileListResult(BaseModel):
    directory_path: str = Field(description="The source directory path targets.")
    entries: List[FileEntry] = Field(default_factory=list, description="Collection of structured layout contents.")
    total_count: int = Field(default=0, description="Total matching file counts discoverable.")
    error: Optional[str] = Field(default=None, description="Error track traces encountered during processing.")

    def __str__(self) -> str:
        """Serializes structure cleanly into an indented visualization layout for LLM consumption."""
        if self.error:
            return f"Error: {self.error}"
        if not self.entries:
            return f"The target directory '{self.directory_path}' is completely empty."

        lines = [f"Listing paths inside directory context: '{self.directory_path}'"]
        for entry in self.entries:
            prefix = "[DIR] " if entry.type == "directory" else "      "
            lines.append(f"{prefix}{entry.path}")

        if self.total_count > len(self.entries):
            lines.append(f"\n[Truncated] Showing first {len(self.entries)} out of {self.total_count} total assets.")
        
        return "\n".join(lines)

@tool
def list_files(
    directory_path: str,
    recursive: bool = False,
    include_hidden: bool = False,
    ignore_patterns: Optional[list[str]] = None,
    max_results: int = 400
) -> FileListResult:
    """List directory file layouts and structures returning relative pathways.

    Use this structural verification tracking resource before attempting downstream file sweeps.

    Args:
        directory_path (str): The path to the directory to list.
        recursive (bool): Whether to list files recursively. Defaults to False.
        include_hidden (bool): Whether to include hidden files and directories. Defaults to False.
        ignore_patterns (Optional[list[str]]): List of glob patterns to ignore. Defaults to None.
        max_results (int): Maximum number of results to return before truncation. Defaults to 400.
    """
    try:
        root = Path(directory_path).resolve()
        if not root.is_dir():
            return FileListResult(directory_path=directory_path, error=f"'{directory_path}' is not a directory.")

        patterns = set(ignore_patterns or [])
        # Fallback defaults for global asset safety layers
        if not include_hidden:
            patterns.update([".git", "node_modules", "__pycache__", "venv", ".venv", "dist", "build"])

        results: List[FileEntry] = []
        total_count = 0

        if recursive:
            for current_root, dirs, files in os.walk(root):
                # Apply in-place filtering rules for safety controls
                dirs[:] = sorted([d for d in dirs if not _should_ignore(d, include_hidden, patterns)])
                files = sorted([f for f in files if not _should_ignore(f, include_hidden, patterns)])

                rel_root = Path(os.path.relpath(current_root, root))

                # Aggregate matching directory elements
                for d in dirs:
                    total_count += 1
                    if len(results) < max_results:
                        path_str = str(rel_root / d) if rel_root != Path(".") else d
                        results.append(FileEntry(path=path_str, type="directory"))

                # Aggregate matching file elements
                for f in files:
                    total_count += 1
                    if len(results) < max_results:
                        path_str = str(rel_root / f) if rel_root != Path(".") else f
                        results.append(FileEntry(path=path_str, type="file"))
        else:
            try:
                for child in sorted(root.iterdir()):
                    if _should_ignore(child.name, include_hidden, patterns):
                        continue
                    
                    total_count += 1
                    if len(results) < max_results:
                        e_type = "directory" if child.is_dir() else "file"
                        results.append(FileEntry(path=child.name, type=e_type))
            except PermissionError:
                return FileListResult(directory_path=directory_path, error=f"Permission denied accessing path workspace location.")

        return FileListResult(directory_path=str(root), entries=results, total_count=total_count)

    except Exception as e:
        return FileListResult(directory_path=directory_path, error=f"Unexpected layout exploration crash: {str(e)}")

@register_extractor("list_files")
def extract_list_files(result: Any, args: Dict[str, Any]) -> Dict[str, Any]:
    """Intercepts file list outputs to register persistent structural layout mapping schemas 

    into out-of-band workspace metrics caches.
    """
    if not isinstance(result, FileListResult) or result.error:
        return {}

    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    target_dir = result.directory_path

    # Construct safe localized indexing profiles inside our workspace structure
    workspace_update = {}
    manifest_entries = []

    for entry in result.entries:
        # Resolve target location paths cleanly relative to target contexts
        full_path = str(Path(target_dir) / entry.path)
        
        manifest_entries.append({
            "relative_path": entry.path,
            "full_path": full_path,
            "type": entry.type
        })

        # Initialize lightweight stub boundaries inside workspace if not previously configured
        # This gives the system out-of-band layout visibility without reading content streams prematurely
        if entry.type == "file":
            workspace_update[full_path] = {
                "exists": True,
                "type": "file",
                "last_listed": timestamp
            }

    # Store a centralized snapshot index of the target folder's directory structure
    manifest_key = f"__directory_manifest__:{target_dir}"
    workspace_update[manifest_key] = {
        "directory": target_dir,
        "total_tracked_assets": result.total_count,
        "contents": manifest_entries,
        "scanned_at": timestamp
    }

    # Register read telemetry actions over tracked structures
    artifacts_update = {target_dir: "read"}

    return {
        "workspace_update": workspace_update,
        "artifacts_update": artifacts_update
    }

class FindFilesResult(BaseModel):
    pattern: str = Field(description="The glob pattern applied to the file search match.")
    root_path: str = Field(description="The resolved base directory path of the search.")
    matches: List[str] = Field(default_factory=list, description="List of matched relative file pathways.")
    total_found: int = Field(default=0, description="Total match count discovered before safety truncation limit.")
    error: Optional[str] = Field(default=None, description="Detailed trace message if a failure occurs.")

    def __str__(self) -> str:
        """Saves LLM token overhead by printing a clean, scannable summary layout."""
        if self.error:
            return f"Error executing find_files: {self.error}"
        if not self.matches:
            return f"No matching files found for pattern '{self.pattern}' under directory '{self.root_path}'."

        lines = [f"Found {self.total_found} file(s) matching pattern '{self.pattern}':"]
        for match in self.matches:
            lines.append(f"  - {match}")

        if self.total_found > len(self.matches):
            lines.append(f"\n[Truncated] Showing first {len(self.matches)} out of {self.total_found} matches.")
        
        return "\n".join(lines)


@tool
def find_files(
    pattern: str,
    root: str = ".",
    include_hidden: bool = False,
    ignore_patterns: Optional[list[str]] = None,
    max_results: int = 250
) -> FindFilesResult:
    """Find files matching a specific glob expression pattern recursively under a root directory.

    Examples: '*.py', 'test_*.go', 'config.json', 'index.ts'
    Use this tool when locating specific files across complex codebase trees.

    Args:
        pattern (str): The glob pattern to match files against.
        root (str): The root directory to start the search from. Defaults to the current directory.
        include_hidden (bool): Whether to include hidden files and directories in the search. Defaults to False.
        ignore_patterns (Optional[list[str]]): List of glob patterns to ignore during the search. Defaults to None.
        max_results (int): Maximum number of results to return before truncation. Defaults to 250.
    """
    try:
        root_path = Path(root).resolve()
        if not root_path.is_dir():
            return FindFilesResult(pattern=pattern, root_path=root, error=f"'{root}' is not a valid directory.")

        custom_ignores = set(ignore_patterns or [])
        matches = []
        total_found = 0

        for current_root, dirs, files in os.walk(root_path):
            # In-place directory pruning to avoid scanning ignored paths entirely
            dirs[:] = sorted([
                d for d in dirs 
                if not _should_ignore(d, include_hidden, custom_ignores)
            ])

            rel_root = Path(current_root).relative_to(root_path)

            for file in sorted(files):
                if _should_ignore(file, include_hidden, custom_ignores):
                    continue

                # Smart evaluation: match against just the filename OR the relative path layout
                rel_file_path = rel_root / file if rel_root != Path(".") else Path(file)
                rel_file_str = str(rel_file_path)

                if fnmatch.fnmatch(file, pattern) or fnmatch.fnmatch(rel_file_str, pattern):
                    total_found += 1
                    if len(matches) < max_results:
                        matches.append(rel_file_str)

        return FindFilesResult(
            pattern=pattern,
            root_path=str(root_path),
            matches=matches,
            total_found=total_found
        )

    except Exception as e:
        return FindFilesResult(pattern=pattern, root_path=root, error=str(e))

@register_extractor("find_files")
def extract_find_files(result: Any, args: Dict[str, Any]) -> Dict[str, Any]:
    """Intercepts file glob pattern discovery outputs to log search actions 

    and seed structural workspace tracking indexes.
    """
    if not isinstance(result, FindFilesResult) or result.error:
        return {}

    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    search_pattern = result.pattern
    base_dir = result.root_path

    # Formulate structured historical telemetry profiles for state['searches']
    search_entry_matches = []
    workspace_update = {}

    for match_path in result.matches:
        full_resolved_path = str(Path(base_dir) / match_path)
        
        search_entry_matches.append({
            "relative_path": match_path,
            "full_path": full_resolved_path,
            "matched_at": timestamp
        })

        # Initialize lightweight structural stubs out-of-band inside the workspace state tracking
        # This informs the routing graph that the asset verified-exists without reading heavy data blocks yet
        workspace_update[full_resolved_path] = {
            "exists": True,
            "type": "file",
            "last_discovered": timestamp
        }

    return {
        # Stores files grouped explicitly under the key of the search glob pattern utilized
        "searches_update": {
            f"glob:{search_pattern}": search_entry_matches
        },
        # Updates the out-of-band context index mapping
        "workspace_update": workspace_update,
        # Registers read tracking indicators over the base root targeted
        "artifacts_update": {
            base_dir: "read"
        }
    }

class NodeEntry(BaseModel):
    relative_path: str = Field(description="The relative path from the tree generation root.")
    name: str = Field(description="The basename of the item.")
    type: Literal["file", "directory"] = Field(description="The cataloged entity variant classification.")
    depth: int = Field(description="The depth level of recursion inside the repository branch.")


class DirectoryTreeResult(BaseModel):
    root_path: str = Field(description="The absolute base directory where the tree visualization originated.")
    tree_string: str = Field(description="The generated string drawing visualization representation.")
    nodes: List[NodeEntry] = Field(default_factory=list, description="A flat listing of every structured entity explored.")
    total_directories: int = Field(default=0, description="Total directories traversed.")
    total_files: int = Field(default=0, description="Total files captured.")
    error: Optional[str] = Field(default=None, description="Error notes generated if operations fail.")

    def __str__(self) -> str:
        """Outputs the classic ASCII visualization directly to the LLM console view."""
        if self.error:
            return f"Error mapping tree: {self.error}"
        
        metrics_header = (
            f"Directory Structure for: {self.root_path}\n"
            f"({self.total_directories} directories, {self.total_files} files captured)\n"
            f"--------------------------------------------------\n"
        )
        return f"{metrics_header}{self.tree_string}"


@tool
def get_directory_tree(
    root: str = ".",
    max_depth: int = 3,
    include_hidden: bool = False,
    ignore_patterns: Optional[list[str]] = None,
) -> DirectoryTreeResult:
    """Return an ASCII visual tree layout representation of a directory structure.

    Use this tool to evaluate unfamiliar subfolder landscapes and quickly spot project layouts.

    Args:
        root (str): The root directory to start the tree visualization from. Defaults to the current directory.
        max_depth (int): The maximum depth of recursion for the tree. Defaults to 3.
        include_hidden (bool): Whether to include hidden files and directories in the tree. Defaults to False.
        ignore_patterns (Optional[list[str]]): List of glob patterns to ignore during the tree generation. Defaults to None.
    """
    try:
        # Standard fallbacks for isolated cross-language analysis
        TREE_DEFAULT_IGNORE = {
            "node_modules", ".git", ".venv", "venv", "env", "__pycache__",
            ".pytest_cache", ".mypy_cache", "dist", "build", "target"
        }
        root_path = Path(root).resolve()
        if not root_path.exists():
            return DirectoryTreeResult(root_path=root, tree_string="", error=f"'{root}' does not exist.")
        if not root_path.is_dir():
            return DirectoryTreeResult(root_path=root, tree_string="", error=f"'{root}' is not a directory.")

        # Compute ignore policies
        active_patterns = set(TREE_DEFAULT_IGNORE)
        if ignore_patterns is not None:
            if len(ignore_patterns) == 0:
                active_patterns = set()
            else:
                active_patterns.update(ignore_patterns)

        def should_ignore(name: str) -> bool:
            if not include_hidden and name.startswith(".") and name not in (".github", ".vscode"):
                return True
            return any(fnmatch.fnmatch(name, p) for p in active_patterns)

        lines = [root_path.name + "/"]
        nodes_list: List[NodeEntry] = []
        metrics = {"dirs": 0, "files": 0}

        def walk(path: Path, prefix: str, depth: int):
            if depth > max_depth:
                return

            try:
                # Group folders together on top, then sort alphabetically for deterministic tracing
                children = sorted(
                    path.iterdir(),
                    key=lambda p: (not p.is_dir(), p.name.lower()),
                )
            except PermissionError:
                lines.append(prefix + "└── [Permission Denied]")
                return

            filtered_children = [c for c in children if not should_ignore(c.name)]
            total_children = len(filtered_children)

            for index, child in enumerate(filtered_children):
                is_last = (index == total_children - 1)
                connector = "└── " if is_last else "├── "
                
                # Append string line visualization formatting parameters
                name_suffix = "/" if child.is_dir() else ""
                lines.append(f"{prefix}{connector}{child.name}{name_suffix}")

                # Build structural meta out-of-band context maps
                rel_path = str(child.relative_to(root_path))
                entity_type = "directory" if child.is_dir() else "file"
                
                metrics["dirs" if child.is_dir() else "files"] += 1
                nodes_list.append(NodeEntry(
                    relative_path=rel_path,
                    name=child.name,
                    type=entity_type,
                    depth=depth
                ))

                if child.is_dir():
                    next_extension = "    " if is_last else "│   "
                    walk(child, prefix + next_extension, depth + 1)

        walk(root_path, "", 1)

        return DirectoryTreeResult(
            root_path=str(root_path),
            tree_string="\n".join(lines),
            nodes=nodes_list,
            total_directories=metrics["dirs"],
            total_files=metrics["files"]
        )

    except Exception as e:
        return DirectoryTreeResult(root_path=root, tree_string="", error=str(e))
    
@register_extractor("get_directory_tree")
def extract_get_directory_tree(result: Any, args: Dict[str, Any]) -> Dict[str, Any]:
    """Intercepts visual tree compilation data to inject deep structural topology maps 

    directly inside out-of-band agent workspace caches.
    """
    if not isinstance(result, DirectoryTreeResult) or result.error:
        return {}

    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    base_root = result.root_path

    workspace_update = {}
    flat_manifest_summary = []

    # Iterate structural element objects collected systematically during layout recursion
    for node in result.nodes:
        full_resolved_path = str(Path(base_root) / node.relative_path)
        
        flat_manifest_summary.append({
            "relative_path": node.relative_path,
            "type": node.type,
            "depth": node.depth
        })

        # Pre-seed workspace validation metadata out-of-band.
        # This tells the state graph that these files are verified to exist without reading content strings yet.
        if node.type == "file":
            workspace_update[full_resolved_path] = {
                "exists": True,
                "type": "file",
                "last_validated_via_tree": timestamp
            }

    # Store a global structural representation overview context node for lookup references
    tree_manifest_key = f"__tree_layout__:{base_root}"
    workspace_update[tree_manifest_key] = {
        "root": base_root,
        "total_directories": result.total_directories,
        "total_files": result.total_files,
        "flat_layout_manifest": flat_manifest_summary,
        "scanned_at": timestamp
    }

    return {
        "workspace_update": workspace_update,
        "artifacts_update": {base_root: "read"}
    }