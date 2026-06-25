from __future__ import annotations
import datetime
import fnmatch
import os
from pathlib import Path
from typing import Dict, Any
from langchain.tools import tool
import json
from core.common.model import get_executor_model
from langchain_core.prompts import ChatPromptTemplate
from core.tools.registry import register_extractor
from core.client.lsp.manager import lsp_manager

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


def _build_ignore_patterns(
    ignore_patterns: list[str] | None,
) -> set[str]:
    """
    None -> use defaults

    [] -> disable defaults

    ["*.log"] -> defaults + *.log
    """

    if ignore_patterns is None:
        return DEFAULT_IGNORE_PATTERNS

    if len(ignore_patterns) == 0:
        return set()

    return DEFAULT_IGNORE_PATTERNS | set(ignore_patterns)


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

        model = get_executor_model()
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

@tool
def list_files(
    directory_path: str,
    recursive: bool = False,
    include_hidden: bool = False,
    ignore_patterns: list[str] | None = None,
) -> str:
    """
    List files and directories.

    Returns relative paths.
    """

    try:
        root = Path(directory_path)

        if not root.is_dir():
            return f"Error: '{directory_path}' is not a directory."

        patterns = _build_ignore_patterns(ignore_patterns)

        results = []

        if recursive:

            for current_root, dirs, files in os.walk(root):

                dirs[:] = sorted(
                    d
                    for d in dirs
                    if not _should_ignore(
                        d,
                        include_hidden,
                        patterns,
                    )
                )

                files = sorted(
                    f
                    for f in files
                    if not _should_ignore(
                        f,
                        include_hidden,
                        patterns,
                    )
                )

                rel_root = Path(
                    os.path.relpath(
                        current_root,
                        root,
                    )
                )

                for d in dirs:
                    results.append(
                        str(rel_root / d)
                        if rel_root != Path(".")
                        else d
                    )

                for f in files:
                    results.append(
                        str(rel_root / f)
                        if rel_root != Path(".")
                        else f
                    )

        else:

            for child in sorted(root.iterdir()):

                if _should_ignore(
                    child.name,
                    include_hidden,
                    patterns,
                ):
                    continue

                results.append(child.name)

        return "\n".join(results) if results else "Directory is empty."

    except Exception as e:
        return f"Error listing files: {e}"

@tool
def find_files(
    pattern: str,
    root: str = ".",
    include_hidden: bool = False,
    ignore_patterns: list[str] | None = None,
) -> str:
    """
    Find files matching a glob pattern.

    Example:

    *.py

    test_*.go

    *.md
    """

    try:

        root_path = Path(root)

        if not root_path.is_dir():
            return f"Error: '{root}' is not a directory."

        patterns = _build_ignore_patterns(ignore_patterns)

        matches = []

        for current_root, dirs, files in os.walk(root_path):

            dirs[:] = sorted(
                d
                for d in dirs
                if not _should_ignore(
                    d,
                    include_hidden,
                    patterns,
                )
            )

            for file in sorted(files):

                if _should_ignore(
                    file,
                    include_hidden,
                    patterns,
                ):
                    continue

                if fnmatch.fnmatch(file, pattern):

                    matches.append(
                        str(
                            Path(
                                os.path.relpath(
                                    current_root,
                                    root_path,
                                )
                            )
                            / file
                        )
                    )

        return "\n".join(matches) if matches else "No matching files found."

    except Exception as e:
        return f"Error finding files: {e}"
    
@tool
def get_directory_tree(
    root: str = ".",
    max_depth: int = 3,
    include_hidden: bool = False,
    ignore_patterns: list[str] | None = None,
) -> str:
    """
    Return a tree representation of a directory.

    This tool is intended for understanding project structure,
    not listing every file in large repositories.

    Args:
        root: Root directory.
        max_depth: Maximum recursion depth.
        include_hidden: Include hidden files/directories.
        ignore_patterns:
            None -> use default ignores
            [] -> disable all ignores
            [...] -> defaults + additional patterns

    Returns:
        Tree representation of the directory.
    """

    try:

        root_path = Path(root)

        if not root_path.exists():
            return f"Error: '{root}' does not exist."

        if not root_path.is_dir():
            return f"Error: '{root}' is not a directory."


        if ignore_patterns is None:
            active_patterns = DEFAULT_IGNORE_PATTERNS
        elif len(ignore_patterns) == 0:
            active_patterns = set()
        else:
            active_patterns = DEFAULT_IGNORE_PATTERNS | set(ignore_patterns)

        def should_ignore(name: str) -> bool:
            if not include_hidden and name.startswith("."):
                return True

            return any(
                fnmatch.fnmatch(name, p)
                for p in active_patterns
            )

        lines = []

        def walk(path: Path, prefix: str, depth: int):

            if depth > max_depth:
                return

            try:
                children = sorted(
                    path.iterdir(),
                    key=lambda p: (
                        not p.is_dir(),
                        p.name.lower(),
                    ),
                )
            except PermissionError:
                lines.append(prefix + "└── [Permission Denied]")
                return

            children = [
                c
                for c in children
                if not should_ignore(c.name)
            ]

            for index, child in enumerate(children):

                last = index == len(children) - 1

                connector = "└── " if last else "├── "

                name = child.name + ("/" if child.is_dir() else "")

                lines.append(prefix + connector + name)

                if child.is_dir():
                    extension = "    " if last else "│   "

                    walk(
                        child,
                        prefix + extension,
                        depth + 1,
                    )

        lines.append(root_path.name + "/")

        walk(root_path, "", 1)

        return "\n".join(lines)

    except Exception as e:
        return f"Error generating directory tree: {e}"