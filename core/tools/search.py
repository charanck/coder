import datetime
import re
import os
import collections
import logging
from pathlib import Path
import datetime
from langchain_core.tools import tool
import fnmatch
from typing import Any, Dict, List
from core.model.search import ProjectSummary
from core.model.state import CodingAgentState
from core.tools.files import DEFAULT_IGNORE_PATTERNS
from langchain.tools import tool
from pydantic import BaseModel, Field
from core.tools.registry import register_extractor
from config import PROJECT_FRAMEWORK_FILES, PROJECT_LANGUAGE_MAP
from core.common.tracing import langfuse_observe

logger = logging.getLogger(__name__)

def should_ignore(name: str, include_hidden: bool):
    if not include_hidden and name.startswith("."):
        return True

    return any(
        fnmatch.fnmatch(name, pattern)
        for pattern in DEFAULT_IGNORE_PATTERNS
    )


IGNORE_DIRS = {
    "node_modules", ".git", ".venv", "venv", "env", "__pycache__",
    ".pytest_cache", ".mypy_cache", "dist", "build", "target"
}

def default_should_ignore(name: str, include_hidden: bool) -> bool:
    if not include_hidden and name.startswith("."):
        return name not in (".github", ".vscode") 
    return name in IGNORE_DIRS

@tool
@langfuse_observe
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
    logger.info(f"Scanning project at {root} with max_depth={max_depth}")
    
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

                if ext in PROJECT_LANGUAGE_MAP:
                    language_counter[PROJECT_LANGUAGE_MAP[ext]] += 1

                if file in PROJECT_FRAMEWORK_FILES:
                    frameworks.add(PROJECT_FRAMEWORK_FILES[file])
                    config_files.append(str(rel_root / file) if rel_root != Path(".") else file)

        package_manager = next(
            (fw for fw in frameworks if fw in ("Node.js", "Python", "Go Modules", "Rust")), 
            None
        )

        result = ProjectSummary(
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
        
        logger.info(f"Project scan completed: {total_files} files, languages={len(language_counter)}, frameworks={len(frameworks)}")
        return result

    except Exception as e:
        logger.exception(f"Error scanning project: {str(e)}")
        raise ValueError(f"Error scanning project layout: {str(e)}")
                         
@register_extractor("scan_project")
@langfuse_observe
def extract_scan_project(result: Any, args: Dict[str, Any], state: CodingAgentState | None = None) -> Dict[str, Any]:
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

class MatchLine(BaseModel):
    line_number: int = Field(description="The 1-indexed line number where the pattern was found.")
    content: str = Field(description="The raw text content of the matching line.")

class GrepResult(BaseModel):
    pattern: str = Field(description="The regex pattern used for the search.")
    file_path: str = Field(description="The path of the file searched.")
    matches: List[MatchLine] = Field(default_factory=list, description="List of structured matches.")
    total_found: int = Field(default=0, description="Total match count before truncation.")
    error: str | None = Field(default=None, description="Error message if the execution failed.")

    def __str__(self) -> str:
        """Formats the output into a ultra-scannable format for the LLM context window."""
        if self.error:
            return self.error
        if not self.matches:
            return f"No matches found for pattern '{self.pattern}' in '{self.file_path}'."
        
        output = []
        for m in self.matches:
            # Ensure line content ends with a newline safely
            clean_content = m.content if m.content.endswith("\n") else f"{m.content}\n"
            output.append(f"{m.line_number}:{clean_content}")
            
        if self.total_found > len(self.matches):
            output.append(f"\n[Truncated] Found {self.total_found} total matches. Showing first {len(self.matches)}.")
        return "".join(output)

@tool
@langfuse_observe
def grep(pattern: str, file_path: str, max_matches: int = 250) -> GrepResult:
    """Search for a regex pattern in a file and return matching lines with line numbers.

    Use this tool to find specific references, declarations, or functions inside a target file.

    Args:
        pattern (str): The regex pattern to search for.
        file_path (str): The path to the file to search.
        max_matches (int): Maximum number of matches to return before truncation.
    """
    logger.debug(f"Grep search: pattern={pattern}, file={file_path}, max_matches={max_matches}")
    
    try:
        path = Path(file_path)
        if not path.is_file():
            logger.warning(f"File not found: {file_path}")
            return GrepResult(pattern=pattern, file_path=file_path, error=f"Error: file not found: '{file_path}'")

        matches = []
        total_found = 0
        compiled_regex = re.compile(pattern)

        # Using errors="replace" to prevent unexpected decoding crashes on non-UTF-8 binary/generated files
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for idx, line in enumerate(f, start=1):
                if compiled_regex.search(line):
                    total_found += 1
                    if len(matches) < max_matches:
                        matches.append(MatchLine(line_number=idx, content=line))

        logger.info(f"Grep completed: found {total_found} matches in {file_path} (showing {len(matches)})")
        
        return GrepResult(
            pattern=pattern,
            file_path=file_path,
            matches=matches,
            total_found=total_found
        )

    except re.error as e:
        logger.error(f"Invalid regex pattern: {pattern}")
        return GrepResult(pattern=pattern, file_path=file_path, error=f"Error: invalid regex pattern '{pattern}': {e}")
    except Exception as e:
        logger.exception(f"Error reading file {file_path}")
        return GrepResult(pattern=pattern, file_path=file_path, error=f"Error reading file: {str(e)}")

@register_extractor("grep")
@langfuse_observe
def extract_grep(result: Any, args: Dict[str, Any], state: CodingAgentState | None = None) -> Dict[str, Any]:
    """Extracts structural matches from a grep invocation and logs them 

    into historical searches and artifact tracking layers.
    """
    # Guard clause if the execution crashed completely before creating a model
    if not isinstance(result, GrepResult) or result.error:
        return {}

    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    pattern_key = result.pattern
    file_path = result.file_path

    # Convert matches into flat, loose dictionaries matching your CodingAgentState paradigm
    extracted_matches = [
        {
            "file_path": file_path,
            "line_number": match.line_number,
            "content": match.content.strip(),
            "found_at": timestamp
        }
        for match in result.matches
    ]

    return {
        "searches_update": {
            pattern_key: extracted_matches
        },
        "artifacts_update": {
            file_path: "read"
        }
    }

