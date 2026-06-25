from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

from core.client.lsp.client import LSPClient


def test_initialize_server_sends_initialized_as_notification():
    fake_process = SimpleNamespace(stdin=io.BytesIO(), stdout=io.BytesIO())

    with patch("core.client.lsp.client.subprocess.Popen", return_value=fake_process), patch.object(
        LSPClient,
        "_send_request",
        return_value={"result": {"capabilities": {}}},
    ) as send_request, patch.object(LSPClient, "_send_notification") as send_notification:
        client = LSPClient(["fake-server"], Path(".").resolve().as_uri())

    send_request.assert_called_once()
    send_notification.assert_called_once_with("initialized", {})


def test_read_message_parses_lsp_headers_and_body():
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"ok": True}}).encode("utf-8")
    raw_message = (
        b"Content-Type: application/vscode-jsonrpc; charset=utf-8\r\n"
        + f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
        + body
    )

    client = LSPClient.__new__(LSPClient)
    client.process = cast(Any, SimpleNamespace(stdout=io.BytesIO(raw_message)))

    message = client._read_message()

    assert message == {"jsonrpc": "2.0", "id": 1, "result": {"ok": True}}