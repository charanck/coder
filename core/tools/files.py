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
from core.tools.tools import register_extractor


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
    """Extracts file content metadata and leverages an execution LLM out-of-band to distill architectural facts and functional summaries.
    """
    file_path = args.get("path", "unknown")
    raw_content = str(result)
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

    file_summary = "Summary extraction unavailable."
    extracted_facts = []

    try:
        model = get_executor_model()
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a core architecture analyzer. Inspect the following source file code.\n"
                    "Extract:\n"
                    "1. A highly concise 1-2 sentence functional summary of its purpose.\n"
                    "2. A list of critical code invariants, core structural patterns, or system truths.\n\n"
                    "Return your analysis STRICTLY as a raw JSON object matching this schema:\n"
                    '{{"summary": "string", "facts": ["string", "string"]}}',
                ),
                ("human", "File Path: {file_path}\n\nContent:\n{content}"),
            ]
        )
        chain = prompt | model
        response = chain.invoke({"file_path": file_path, "content": raw_content})
        response_content = response.content if hasattr(response, "content") else response
        response_text: str = response_content if isinstance(response_content, str) else str(response_content)

        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]

        parsed_data = json.loads(response_text.strip())
        file_summary = parsed_data.get("summary", file_summary)
        extracted_facts = parsed_data.get("facts", [])
        # TODO: use LSP to extract structured symbols, classes, functions, and dependencies for the workspace update
    except Exception as e:
        file_summary = (
            f"Failed to systematically extract summary due to exception: {str(e)}"
        )

    workspace_payload = {
        "full_content": raw_content,
        "summary": file_summary,
        "lines": len(raw_content.splitlines()),
        "last_read": timestamp,
    }

    known_facts_payload = [
        {"source": file_path, "fact": fact, "extracted_at": timestamp}
        for fact in extracted_facts
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