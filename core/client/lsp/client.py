import json
import subprocess
from pathlib import Path
from typing import Any


SERVERS = {
    "go": {
        "cmd": ["gopls"],
        "language_id": "go",
    },
    "python": {
        "cmd": ["basedpyright-langserver", "--stdio"],
        "language_id": "python",
    },
    "typescript": {
        "cmd": ["typescript-language-server", "--stdio"],
        "language_id": "typescript",
    },
    "javascript": {
        "cmd": ["typescript-language-server", "--stdio"],
        "language_id": "javascript",
    },
    "rust": {
        "cmd": ["rust-analyzer"],
        "language_id": "rust",
    },
}

class LSPFactory:

    @staticmethod
    def create(language: str, workspace: str):
        config = SERVERS.get(language)

        if not config:
            raise ValueError(f"No LSP configured for {language}")

        client = LSPClient(
            server_command=config["cmd"],
            root_uri=Path(workspace).resolve().as_uri(),
        )
        return client




class LSPClient:
    def __init__(self, server_command: list[str], root_uri: str):
        # Spin up the language server as a background subprocess
        self.process = subprocess.Popen(
            server_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=False,
        )
        self.request_id = 1
        self.root_uri = root_uri
        self._initialize_server()

    def _send_request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Formats and sends a standard JSON-RPC request over stdin."""
        if not self.process.stdin:
            raise RuntimeError("LSP server stdin is not available.")
        request_id = self.request_id
        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }
        self._write_message(request)
        self.process.stdin.flush()

        self.request_id += 1
        while True:
            response = self._read_message()
            if not response:
                return {}
            if response.get("id") == request_id:
                return response

    def _send_notification(self, method: str, params: dict[str, Any]) -> None:
        if not self.process.stdin:
            raise RuntimeError("LSP server stdin is not available.")
        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }
        self._write_message(notification)
        self.process.stdin.flush()

    def _write_message(self, message: dict[str, Any]) -> None:
        if not self.process.stdin:
            raise RuntimeError("LSP server stdin is not available.")

        json_body = json.dumps(message).encode("utf-8")
        header = f"Content-Length: {len(json_body)}\r\n\r\n".encode("ascii")

        # Write the exact LSP frame: headers, blank line, then the JSON payload.
        self.process.stdin.write(header + json_body)

    def _read_message(self) -> dict[str, Any]:
        if not self.process.stdout:
            return {}
        """Reads and parses a single JSON-RPC message from stdout."""
        headers: dict[str, str] = {}

        while True:
            line = self.process.stdout.readline()
            if not line:
                return {}

            if line in (b"\r\n", b"\n", b""):
                break

            decoded = line.decode("ascii", errors="ignore").strip()
            if ":" not in decoded:
                continue

            key, value = decoded.split(":", 1)
            headers[key.strip().lower()] = value.strip()

        content_length = headers.get("content-length")
        if not content_length:
            return {}

        body = self.process.stdout.read(int(content_length))
        if not body:
            return {}

        return json.loads(body.decode("utf-8"))

    def _initialize_server(self):
        """Required handshake to wake up the LSP server."""
        self._send_request(
            "initialize",
            {
                "processId": None,
                "rootUri": self.root_uri,
                "capabilities": {},
            },
        )
        self._send_notification("initialized", {})

    def extract_document_symbols(self, file_uri: str) -> list[dict[str, Any]]:
        """Queries the LSP for a complete, precise structural map of a file."""
        response = self._send_request(
            "textDocument/documentSymbol", {"textDocument": {"uri": file_uri}}
        )
        # Returns an array of symbols (Name, Kind: Function/Class, Range: Start/End Lines)
        return response.get("result", [])


# --- Example Usage inside your Extractor Node ---
# client = LSPClient(server_command=["pyright-langserver", "--stdio"], root_uri="file:///path/to/project")
# symbols = client.extract_document_symbols("file:///path/to/project/main.py")
            # return await self.request(
            #     "textDocument/definition",
            #     {
            #         "textDocument": {"uri": uri},
            #         "position": {
            #             "line": line,
            #             "character": char,
            #         },
            #     },
            # )