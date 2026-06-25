from __future__ import annotations

import logging
import os
import shutil

import pytest

from core.client.lsp.manager import lsp_manager
from core.common.model import get_executor_model
from core.tools.files import extract_read_file, read_file
from core.tools.search import scan_project


def _has_python_lsp() -> bool:
    return shutil.which("basedpyright-langserver") is not None



@pytest.fixture()
def integration_workspace(tmp_path):
    (tmp_path / "src").mkdir(parents=True, exist_ok=True)
    (tmp_path / "src" / "sample.py").write_text("class Sample:\n    def run(self):\n        return 1\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'sample'\n", encoding="utf-8")
    return tmp_path


def test_scan_project_tool_real(integration_workspace):
    summary = scan_project.invoke({"root": str(integration_workspace)})

    assert summary.package_manager == "Python"
    assert "src" in summary.top_level_directories


def test_lsp_client_real(integration_workspace):
    if not _has_python_lsp():
        pytest.skip("basedpyright-langserver is not installed")

    file_path = integration_workspace / "src" / "sample.py"
    client = lsp_manager.get_by_extension(str(file_path), workspace=str(integration_workspace))
    symbols = client.extract_document_symbols(str(file_path.resolve().as_uri()))

    assert isinstance(symbols, list)


def test_read_file_extractor_real(integration_workspace):
    if not _has_python_lsp():
        pytest.fail("basedpyright-langserver is not installed. Please install it to run this test.")

    model = get_executor_model()
    response = model.invoke("Reply with one word: ok")
    assert str(response.content if hasattr(response, "content") else response).strip()

    file_path = integration_workspace / "src" / "sample.py"
    tool_output = read_file.invoke({"file_path": str(file_path)})
    updates = extract_read_file(tool_output, {"file_path": str(file_path)})

    assert str(file_path) in updates["workspace_update"]
    assert updates["known_facts_update"] == []