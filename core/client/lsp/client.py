import json
import json
import subprocess
from typing import Any, Dict


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
            root_uri=workspace,
        )
        return client




class LSPClient:
    def __init__(self, server_command: list[str], root_uri: str):
        # Spin up the language server as a background subprocess
        self.process = subprocess.Popen(
            server_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.request_id = 1
        self.root_uri = root_uri
        self._initialize_server()

    def _send_request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Formats and sends a standard JSON-RPC request over stdin."""
        if not self.process.stdin:
            raise RuntimeError("LSP server stdin is not available.")
        request = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": method,
            "params": params,
        }
        json_body = json.dumps(request)
        content_length = len(json_body.encode("utf-8"))

        # Write using the strict LSP HTTP-like header format
        self.process.stdin.write(
            f"Content-Length: {content_length}\r\n\r\n{json_body}"
        )
        self.process.stdin.flush()

        self.request_id += 1
        return self._read_response()

    def _read_response(self) -> dict[str, Any]:
        if not self.process.stdout:
            return {}
        """Reads and parses the JSON-RPC response from stdout."""
        # Simple parsing of Content-Length header
        line = self.process.stdout.readline().strip()
        if not line.startswith("Content-Length:"):
            return {}

        content_length = int(line.split(":")[1].strip())
        self.process.stdout.readline()  # Consume the empty \r\n line

        # Read the exact body payload
        body = self.process.stdout.read(content_length)
        return json.loads(body)

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
        self._send_request("initialized", {})

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