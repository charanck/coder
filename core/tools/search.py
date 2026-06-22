import re
from pathlib import Path
from collections import Counter
import os
import fnmatch
from core.model.search import ProjectSummary
from core.tools.files import DEFAULT_IGNORE_PATTERNS
from langchain.tools import tool


LANGUAGE_MAP = {
    ".py": "Python",
    ".go": "Go",
    ".js": "JavaScript",
    ".ts": "TypeScript",
    ".tsx": "TypeScript",
    ".jsx": "JavaScript",
    ".java": "Java",
    ".kt": "Kotlin",
    ".rs": "Rust",
    ".cpp": "C++",
    ".cc": "C++",
    ".c": "C",
    ".cs": "C#",
    ".swift": "Swift",
    ".php": "PHP",
    ".rb": "Ruby",
    ".scala": "Scala",
    ".dart": "Dart",
    ".yaml": "YAML",
    ".yml": "YAML",
    ".json": "JSON",
    ".toml": "TOML",
}


FRAMEWORK_FILES = {
    "package.json": "Node.js",
    "go.mod": "Go Modules",
    "Cargo.toml": "Rust",
    "pyproject.toml": "Python",
    "requirements.txt": "Python",
    "poetry.lock": "Poetry",
    "Pipfile": "Pipenv",
    "pom.xml": "Maven",
    "build.gradle": "Gradle",
    "build.gradle.kts": "Gradle",
    "composer.json": "Composer",
    "Gemfile": "Ruby",
    "mix.exs": "Elixir",
    "pubspec.yaml": "Flutter",
}


def should_ignore(name: str, include_hidden: bool):
    if not include_hidden and name.startswith("."):
        return True

    return any(
        fnmatch.fnmatch(name, pattern)
        for pattern in DEFAULT_IGNORE_PATTERNS
    )


@tool
def scan_project(
    root: str = ".",
    include_hidden: bool = False,
) -> ProjectSummary:
    """
    Scan a project and return a high-level summary.

    Use this before searching the codebase.

    Returns:
        Repository summary including languages,
        frameworks, directory layout, configs,
        tests and project size.
    """

    try:
        root_path = Path(root)

        if not root_path.is_dir():
            raise ValueError(f"'{root_path}' is not a directory.")

        language_counter = Counter()

        frameworks = set()

        top_dirs = []

        config_files = []

        test_dirs = []

        total_files = 0

        source_dirs = set()

        for current_root, dirs, files in os.walk(root_path):

            dirs[:] = sorted(
                d
                for d in dirs
                if not should_ignore(d, include_hidden)
            )

            rel_root = Path(
                os.path.relpath(current_root, root_path)
            )

            if rel_root == Path("."):
                top_dirs.extend(sorted(dirs))

            if any(
                x in rel_root.parts
                for x in ("test", "tests", "__tests__")
            ):
                test_dirs.append(str(rel_root))

            if any(
                x in rel_root.parts
                for x in ("src", "app", "internal", "cmd", "pkg", "lib")
            ):
                source_dirs.add(str(rel_root))

            for file in files:

                if should_ignore(file, include_hidden):
                    continue

                total_files += 1

                ext = Path(file).suffix.lower()

                if ext in LANGUAGE_MAP:
                    language_counter[
                        LANGUAGE_MAP[ext]
                    ] += 1

                if file in FRAMEWORK_FILES:
                    frameworks.add(
                        FRAMEWORK_FILES[file]
                    )

                    config_files.append(
                        str(rel_root / file)
                        if rel_root != Path(".")
                        else file
                    )

        package_manager = next(
            (fw for fw in frameworks 
             if fw in ("Node.js", "Python", "Go Modules", "Rust")), 
            None
        )

        return ProjectSummary(
            root=str(root_path.resolve()),
            total_files=total_files,
            languages=dict(language_counter),
            frameworks=sorted(frameworks),
            package_manager=package_manager,
            top_level_directories=top_dirs,
            source_directories=sorted(source_dirs),
            test_directories=sorted(set(test_dirs)),
            config_files=sorted(config_files),
        )

    except Exception as e:
        raise ValueError(f"Error scanning project: {e}")
    

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
