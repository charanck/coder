from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from langchain_core.runnables import RunnableLambda

from core.model.search import ProjectSummary
from core.tools.files import extract_read_file
from core.tools.registry import TOOL_EXTRACTOR_REGISTRY
from core.tools.search import extract_scan_project
from config import SUPPORTED_LSP_LANGUAGE_CASES

from tests.support.helpers import make_tool_result


def test_registry_covers_all_current_extractors():
    assert {"read_file", "scan_project"}.issubset(set(TOOL_EXTRACTOR_REGISTRY))


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


def _build_fake_lsp_client(case):
    return SimpleNamespace(
        extract_document_symbols=lambda _path: [
            {
                "name": case.symbol_name,
                "kind": case.symbol_kind,
                "range": {"start": {"line": 0}, "end": {"line": 1}},
            }
        ]
    )


@pytest.mark.parametrize("case", SUPPORTED_LSP_LANGUAGE_CASES)
def test_read_file_extractor_uses_language_routed_lsp(case, tmp_path):
    file_path = tmp_path / case.file_name
    result = make_tool_result(case.source_code)
    fake_lsp_client = _build_fake_lsp_client(case)
    fake_model = RunnableLambda(lambda _payload: SimpleNamespace(content="Concise summary."))

    with patch("core.tools.files.lsp_manager.get_by_extension", return_value=fake_lsp_client) as get_client, patch(
        "core.tools.files.get_executor_model", return_value=fake_model
    ):
        updates = extract_read_file(result, {"path": str(file_path)})

    get_client.assert_called_once_with(str(file_path), workspace=str(tmp_path))
    assert updates["artifacts_update"] == {str(file_path): "read"}
    assert str(file_path) in updates["workspace_update"]
    assert updates["workspace_update"][str(file_path)]["summary"] == "Concise summary."
    assert updates["workspace_update"][str(file_path)]["symbols"] == [
        {
            "name": case.symbol_name,
            "kind": "Class" if case.symbol_kind == 5 else "Struct" if case.symbol_kind == 23 else "Function",
            "start_line": 1,
            "end_line": 2,
            "identity": case.symbol_name,
        }
    ]
    assert any(
        fact["source"] == str(file_path) and case.expected_fact_fragment in fact["fact"]
        for fact in updates["known_facts_update"]
    )