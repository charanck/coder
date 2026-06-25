from __future__ import annotations

from core.tools.files import find_files, get_directory_tree, list_files, read_file
from core.tools.search import grep, grep_in_project, scan_project

from tests.support.helpers import invoke_tool


def test_file_tools(sample_workspace):
    root = sample_workspace
    cases = [
        (read_file, {"file_path": str(root / "src" / "app.py")}, "def run()"),
        (list_files, {"directory_path": str(root)}, "pyproject.toml"),
        (find_files, {"pattern": "*.py", "root": str(root)}, "src/app.py"),
        (get_directory_tree, {"root": str(root), "max_depth": 2}, "src/"),
        (grep, {"pattern": "run", "file_path": str(root / "src" / "app.py")}, "def run()"),
        (grep_in_project, {"pattern": "assert", "root": str(root)}, "test_app.py"),
    ]

    for tool_obj, kwargs, expected in cases:
        result = invoke_tool(tool_obj, **kwargs)
        assert expected in str(result).replace("\\", "/")


def test_scan_project_tool(sample_workspace):
    project = invoke_tool(scan_project, root=str(sample_workspace))

    assert project.root == str(sample_workspace.resolve())
    assert project.package_manager == "Python"
    assert project.languages["Python"] == 2
    assert "src" in project.top_level_directories
    assert "tests" in project.test_directories