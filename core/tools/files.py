from __future__ import annotations

import fnmatch
import os
from pathlib import Path

from langchain.tools import tool


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


@tool
def read_files(
    paths: list[str],
) -> dict[str, str]:
    """
    Read multiple files.

    Returns:

    {
        "main.py": "...",
        "config.py": "...",
    }
    """

    results = {}

    for path in paths:
        results[path] = read_file.invoke(
            {
                "file_path": path,
            }
        )

    return results


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