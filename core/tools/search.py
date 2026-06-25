import datetime
import re
import os
import collections
from pathlib import Path
from langchain_core.tools import tool
import fnmatch
from typing import Any, Dict
from core.model.search import ProjectSummary
from core.tools.files import DEFAULT_IGNORE_PATTERNS
from langchain.tools import tool

from core.tools.registry import register_extractor

def should_ignore(name: str, include_hidden: bool):
    if not include_hidden and name.startswith("."):
        return True

    return any(
        fnmatch.fnmatch(name, pattern)
        for pattern in DEFAULT_IGNORE_PATTERNS
    )


DEFAULT_LANGUAGE_MAP = {
    ".py": "Python", ".ts": "TypeScript", ".js": "JavaScript", 
    ".go": "Go", ".rs": "Rust", ".cpp": "C++", ".c": "C",
    ".java": "Java", ".rb": "Ruby", ".php": "PHP", ".cs": "C#"
}

DEFAULT_FRAMEWORK_FILES = {
    "package.json": "Node.js", "requirements.txt": "Python",
    "pyproject.toml": "Python", "go.mod": "Go Modules",
    "Cargo.toml": "Rust", "Gemfile": "Ruby", "composer.json": "PHP"
}

IGNORE_DIRS = {
    "node_modules", ".git", ".venv", "venv", "env", "__pycache__",
    ".pytest_cache", ".mypy_cache", "dist", "build", "target"
}

def default_should_ignore(name: str, include_hidden: bool) -> bool:
    if not include_hidden and name.startswith("."):
        return name not in (".github", ".vscode") 
    return name in IGNORE_DIRS

@tool
def scan_project(
    root: str = ".",
    include_hidden: bool = False,
    max_depth: int = 4
) -> ProjectSummary:
    """Scan a project directory structure and return a high-level architectural summary.

    Use this tool at the initialization phase before invoking targeted file code read sweeps.

    Args:
        root (str): The root directory of the project to be scanned.
        include_hidden (bool): Whether to include hidden files and directories in the scan.
        max_depth (int): Maximum directory depth to scan to avoid deep recursion.
    """
    try:
        root_path = Path(root).resolve()
        if not root_path.is_dir():
            raise ValueError(f"'{root_path}' is not a valid directory.")

        language_counter = collections.Counter()
        frameworks = set()
        top_dirs = []
        config_files = []
        test_dirs = []
        source_dirs = set()
        total_files = 0

        root_depth = len(root_path.parts)

        for current_root, dirs, files in os.walk(root_path):
            current_path = Path(current_root)
            current_depth = len(current_path.parts) - root_depth
            
            if current_depth >= max_depth:
                dirs[:] = [] 
                continue

            dirs[:] = sorted([
                d for d in dirs 
                if not default_should_ignore(d, include_hidden)
            ])

            rel_root = current_path.relative_to(root_path)

            if rel_root == Path("."):
                top_dirs.extend(sorted(dirs))

            parts_lower = [p.lower() for p in rel_root.parts]
            if any(x in parts_lower for x in ("test", "tests", "__tests__", "spec")):
                test_dirs.append(str(rel_root))

            if any(x in parts_lower for x in ("src", "app", "internal", "cmd", "pkg", "lib", "source")):
                source_dirs.add(str(rel_root))

            for file in files:
                if default_should_ignore(file, include_hidden):
                    continue

                total_files += 1
                ext = Path(file).suffix.lower()

                if ext in DEFAULT_LANGUAGE_MAP:
                    language_counter[DEFAULT_LANGUAGE_MAP[ext]] += 1

                if file in DEFAULT_FRAMEWORK_FILES:
                    frameworks.add(DEFAULT_FRAMEWORK_FILES[file])
                    config_files.append(str(rel_root / file) if rel_root != Path(".") else file)

        package_manager = next(
            (fw for fw in frameworks if fw in ("Node.js", "Python", "Go Modules", "Rust")), 
            None
        )

        return ProjectSummary(
            root=str(root_path),
            total_files=total_files,
            languages=dict(language_counter),
            frameworks=sorted(list(frameworks)),
            package_manager=package_manager,
            top_level_directories=top_dirs,
            source_directories=sorted(list(source_dirs)),
            test_directories=sorted(list(set(test_dirs))),
            config_files=sorted(config_files),
        )

    except Exception as e:
        raise ValueError(f"Error scanning project layout: {str(e)}")
                         
@register_extractor("scan_project")
def extract_scan_project(result: Any, args: Dict[str, Any]) -> Dict[str, Any]:
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    
    if hasattr(result, "model_dump"):
        data = result.model_dump()
    elif hasattr(result, "dict"):
        data = result.dict()
    elif isinstance(result, dict):
        data = result
    else:
        return {}

    root_dir = data.get("root", "unknown")
    
    # 1. Structure out-of-band persistent repository metadata inside the workspace state tracking
    workspace_update = {
        "__repository_manifest__": {
            "total_files": data.get("total_files", 0),
            "languages_distribution": data.get("languages", {}),
            "frameworks": data.get("frameworks", []),
            "package_manager": data.get("package_manager"),
            "top_level_layout": data.get("top_level_directories", []),
            "source_roots": data.get("source_directories", []),
            "test_roots": data.get("test_directories", []),
            "scanned_at": timestamp
        }
    }

    # 2. Derive system truths to append directly to the global `known_facts` list reducer
    known_facts_update = []
    
    if data.get("package_manager"):
        known_facts_update.append({
            "source": f"scan_project:{root_dir}",
            "fact": f"Project target runtime package manager environment is explicitly identified as {data['package_manager']}.",
            "extracted_at": timestamp
        })

    primary_langs = sorted(data.get("languages", {}).items(), key=lambda x: x[1], reverse=True)
    if primary_langs:
        top_lang = primary_langs[0][0]
        known_facts_update.append({
            "source": f"scan_project:{root_dir}",
            "fact": f"The dominant project codebase development language is tracked as {top_lang}.",
            "extracted_at": timestamp
        })

    for cfg in data.get("config_files", []):
        known_facts_update.append({
            "source": f"scan_project:{root_dir}",
            "fact": f"Detected fundamental project execution configuration map resource: {cfg}",
            "extracted_at": timestamp
        })

    return {
        "workspace_update": workspace_update,
        "artifacts_update": {root_dir: "read"},
        "known_facts_update": known_facts_update
    }

@tool
def grep(pattern: str, file_path: str) -> str:
    """Search for a pattern in a file and return matching lines.
    
    Args:
        pattern (str): The regex pattern to search for.
        file_path (str): The path to the file to be searched.

    Returns:
        str: Newline-separated lines that match the pattern,
             or an error message if the file cannot be read.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        matches = [line for line in lines if re.search(pattern, line)]

        if not matches:
            return f"No matches found for pattern '{pattern}' in '{file_path}'."

        return "".join(matches)  # lines already contain \n

    except FileNotFoundError:
        return f"Error: file not found: '{file_path}'"
    except re.error as e:
        return f"Error: invalid regex pattern '{pattern}': {e}"
    except Exception as e:
        return f"Error reading file: {e}"


@tool
def grep_in_project(pattern: str, root: str = ".", include_hidden: bool = False) -> dict[str, list[str]]:
    """
    Search for a pattern in all files of a project and return matching lines.

    Args:
        pattern (str): The regex pattern to search for.
        root (str): The root directory of the project to be searched.
        include_hidden (bool): Whether to include hidden files and directories.

    Returns:
        dict: A dictionary where keys are file paths and values are lists of matching lines.
    """
    matches = {}
    try:
        root_path = Path(root)

        if not root_path.is_dir():
            raise ValueError(f"'{root_path}' is not a directory.")

        for current_root, dirs, files in os.walk(root_path):
            dirs[:] = sorted(
                d
                for d in dirs
                if not should_ignore(d, include_hidden)
            )

            for file in files:
                if should_ignore(file, include_hidden):
                    continue

                file_path = Path(current_root) / file
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()

                    file_matches = [line for line in lines if re.search(pattern, line)]
                    if file_matches:
                        matches[str(file_path)] = file_matches

                except Exception as e:
                    matches[str(file_path)] = [f"Error reading file: {e}"]

        return matches

    except Exception as e:
        raise ValueError(f"Error searching project: {e}")