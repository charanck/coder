from __future__ import annotations
import shutil
from pathlib import Path

import pytest

from core.client.lsp.manager import lsp_manager
from core.common.model import get_executor_model
from core.tools.files import extract_read_file, read_file
from core.tools.search import scan_project
from config import SUPPORTED_LSP_LANGUAGE_CASES, get_lsp_server_command


@pytest.fixture()
def integration_workspace(tmp_path):
    (tmp_path / "src").mkdir(parents=True, exist_ok=True)
    (tmp_path / "src" / "sample.py").write_text("class Sample:\n    def run(self):\n        return 1\n", encoding="utf-8")
    (tmp_path / "src" / "sample.ts").write_text("export class App {}\n", encoding="utf-8")
    (tmp_path / "src" / "sample.js").write_text("function run() {}\n", encoding="utf-8")
    (tmp_path / "src" / "sample.go").write_text("package main\n\ntype App struct{}\n", encoding="utf-8")
    (tmp_path / "src" / "sample.rs").write_text("struct App;\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'sample'\n", encoding="utf-8")
    (tmp_path / "go.mod").write_text("module sample\n\ngo 1.22\n", encoding="utf-8")
    (tmp_path / "Cargo.toml").write_text("[package]\nname = 'sample'\nversion = '0.1.0'\nedition = '2021'\n", encoding="utf-8")
    (tmp_path / "package.json").write_text("{\n  \"name\": \"sample\"\n}\n", encoding="utf-8")
    (tmp_path / "tsconfig.json").write_text("{\n  \"compilerOptions\": {\n    \"target\": \"ES2020\"\n  }\n}\n", encoding="utf-8")
    return tmp_path


def test_scan_project_tool_real(integration_workspace):
    summary = scan_project.invoke({"root": str(integration_workspace)})

    assert summary.package_manager
    assert "src" in summary.top_level_directories
    assert {"pyproject.toml", "go.mod", "Cargo.toml", "package.json"}.issubset(summary.config_files)


def test_lsp_client_real(integration_workspace):
    file_path = integration_workspace / "src" / "sample.py"
    client = lsp_manager.get_by_extension(str(file_path), workspace=str(integration_workspace))
    if client is None:
        pytest.skip("No LSP client available for the given file extension")
    symbols = client.extract_document_symbols(str(file_path.resolve().as_uri()))

    assert isinstance(symbols, list)


@pytest.mark.parametrize("case", SUPPORTED_LSP_LANGUAGE_CASES)
def test_lsp_client_real_for_all_supported_languages(integration_workspace, case):
    command = get_lsp_server_command(case.language)
    if not command:
        pytest.skip(f"No LSP command configured for {case.language}")

    binary = command[0]
    if shutil.which(binary) is None:
        pytest.skip(f"{binary} is not installed")

    file_path = Path(integration_workspace) / "src" / case.file_name
    client = lsp_manager.get_by_extension(str(file_path), workspace=str(integration_workspace))
    if client is None:
        pytest.skip(f"No LSP client available for the given file extension {file_path.suffix}")
    symbols = client.extract_document_symbols(str(file_path.resolve().as_uri()))

    assert isinstance(symbols, list)


def test_read_file_extractor_real(integration_workspace):
    file_path = integration_workspace / "src" / "sample.py"
    client = lsp_manager.get_by_extension(str(file_path), workspace=str(integration_workspace))
    if client is None:
        pytest.skip(f"No LSP client available for the given file extension {file_path.suffix}")

    model = get_executor_model()
    response = model.invoke("Reply with one word: ok")
    assert str(response.content if hasattr(response, "content") else response).strip()

    file_path = integration_workspace / "src" / "sample.py"
    tool_output = read_file.invoke({"file_path": str(file_path)})
    updates = extract_read_file(tool_output, {"file_path": str(file_path)})

    assert str(file_path) in updates["workspace_update"]
    assert updates["known_facts_update"] == []