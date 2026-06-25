from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from core.model.search import ProjectSummary
from core.tools.files import extract_read_file
from core.tools.registry import TOOL_EXTRACTOR_REGISTRY
from core.tools.search import extract_scan_project

from tests.support.helpers import make_tool_result


def test_registry_covers_all_current_extractors():
    assert set(TOOL_EXTRACTOR_REGISTRY) == {"read_file", "scan_project"}


def test_scan_project_extractor():
    summary = ProjectSummary(
        root="/repo",
        total_files=3,
        languages={"Python": 2},
        frameworks=["Python"],
        package_manager="Python",
        top_level_directories=["src"],
        source_directories=["src"],
        test_directories=["tests"],
        config_files=["pyproject.toml"],
    )

    updates = extract_scan_project(summary, {})

    assert updates["artifacts_update"] == {"/repo": "read"}
    assert "__repository_manifest__" in updates["workspace_update"]
    assert len(updates["known_facts_update"]) >= 2


def test_read_file_extractor():
    result = make_tool_result("class App:\n    pass\n")
    fake_lsp_client = SimpleNamespace(
        extract_document_symbols=lambda _path: [
            {
                "name": "App",
                "kind": 5,
                "range": {"start": {"line": 0}, "end": {"line": 1}},
            }
        ]
    )
    fake_model = SimpleNamespace(invoke=lambda _payload: SimpleNamespace(content="Concise summary."))

    with patch("core.tools.files.lsp_manager.get_by_extension", return_value=fake_lsp_client), patch(
        "core.tools.files.get_executor_model", return_value=fake_model
    ):
        updates = extract_read_file(result, {"path": str(Path("sample.py"))})

    assert updates["artifacts_update"] == {str(Path("sample.py")): "read"}
    assert str(Path("sample.py")) in updates["workspace_update"]
    assert updates["known_facts_update"]