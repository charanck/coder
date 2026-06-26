from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from core.client.lsp.manager import lsp_manager

from tests.support.language_cases import SUPPORTED_LSP_LANGUAGE_CASES


@pytest.mark.parametrize("case", SUPPORTED_LSP_LANGUAGE_CASES)
def test_get_by_extension_routes_supported_languages(case, tmp_path):
    file_path = tmp_path / case.file_name
    fake_client = SimpleNamespace()

    with patch("core.client.lsp.manager.LSPFactory.create", return_value=fake_client) as create:
        client = lsp_manager.get_by_extension(str(file_path), workspace=str(tmp_path))

    assert client is fake_client
    create.assert_called_once_with(case.language, str(tmp_path))


def test_get_by_extension_rejects_unknown_extensions(tmp_path):
    file_path = tmp_path / "sample.txt"

    with pytest.raises(ValueError, match="No LSP configured for files with extension .txt"):
        lsp_manager.get_by_extension(str(file_path), workspace=str(tmp_path))