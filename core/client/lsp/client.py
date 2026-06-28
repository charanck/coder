import json
import os
import subprocess
import shutil
import logging
import time
from urllib.parse import urlparse
from pathlib import Path
from typing import Any

from config import get_lsp_server_command

logger = logging.getLogger(__name__)

class LSPFactory:

    @staticmethod
    def create(language: str, workspace: str):
        logger.info(f"[LSPFactory.create] Creating LSP client for language={language}, workspace={workspace}")
        
        server_command = get_lsp_server_command(language)

        if not server_command:
            logger.error(f"[LSPFactory.create] No LSP configured for {language}")
            raise ValueError(f"No LSP configured for {language}")

        if shutil.which(server_command[0]) is None:
            logger.warning(f"[LSPFactory.create] Server command not found: {server_command[0]}")
            return None

        logger.info(f"[LSPFactory.create] Server command found: {server_command}")
        
        client = LSPClient(
            server_command=_resolve_server_command(server_command),
            root_uri=Path(workspace).resolve().as_uri(),
        )
        logger.info(f"[LSPFactory.create] LSP client created successfully")
        return client


def _resolve_server_command(server_command: list[str]) -> list[str]:
    executable = shutil.which(server_command[0])

    if executable is None:
        return server_command

    if os.name == "nt" and Path(executable).suffix.lower() in {".cmd", ".bat"}:
        return ["cmd.exe", "/c", executable, *server_command[1:]]

    return [executable, *server_command[1:]]




class LSPClient:
    def __init__(self, server_command: list[str], root_uri: str):
        logger.info(f"[LSPClient.__init__] Starting LSP server with command: {server_command}")
        logger.info(f"[LSPClient.__init__] Root URI: {root_uri}")
        
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
        logger.info(f"[LSPClient.__init__] Process started with PID: {self.process.pid}")
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
        logger.debug(f"[_send_request] Sending {method} (id={request_id})")
        self._write_message(request)
        self.process.stdin.flush()

        self.request_id += 1
        while True:
            response = self._read_message()
            logger.debug(f"[_send_request] Received response: {response}")
            
            if not response:
                logger.warning(f"[_send_request] Empty response for {method}")
                return {}
            
            if response.get("id") == request_id:
                logger.debug(f"[_send_request] Got matching response for id={request_id}")
                return response
            else:
                logger.debug(f"[_send_request] Got response with mismatched id: {response.get('id')} vs {request_id}")

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
                logger.debug(f"[_read_message] EOF reached on stdout")
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
            logger.debug(f"[_read_message] No content-length header. Headers: {headers}")
            return {}

        body = self.process.stdout.read(int(content_length))
        if not body:
            return {}

        return json.loads(body.decode("utf-8"))

    def _initialize_server(self):
        """Required handshake to wake up the LSP server."""
        logger.debug(f"[LSPClient._initialize_server] Initializing server with root URI: {self.root_uri}")

        capabilities = {
                        "textDocument": {
                            "documentSymbol": {
                                "hierarchicalDocumentSymbolSupport": True,
                                "symbolKind": {
                                    "valueSet": list(range(1, 27))
                                },
                                "labelSupport": True,
                            }
                        },
                        "workspace": {
                            "workspaceFolders": True
                        }
                    }
        
        response = self._send_request(
            "initialize",
            {
                "processId": None,
                "rootUri": self.root_uri,
                "capabilities": capabilities,
            },
        )
        logger.debug(f"[LSPClient._initialize_server] Initialize response: {response}")
        
        self._send_notification("initialized", {})
        logger.info(f"[LSPClient._initialize_server] Server initialized successfully")

    def extract_document_symbols(self, file_uri: str) -> list[dict[str, Any]]:
        """Queries the LSP for a complete, precise structural map of a file."""
        logger.debug(f"[extract_document_symbols] Requesting symbols for: {file_uri}")
        
        # Convert file URI to local path
        try:
            # Parse the URI to extract the path
            parsed = urlparse(file_uri)
            file_path = parsed.path
            
            # On Windows, remove leading slash from /C:/path -> C:/path
            if os.name == "nt" and len(file_path) > 2 and file_path[0] == "/" and file_path[2] == ":":
                file_path = file_path[1:]
            
            logger.debug(f"[extract_document_symbols] Converted URI to path: {file_path}")
            
            # Open the document in the LSP server so it can parse and index it
            with open(file_path, "r", encoding="utf-8") as f:
                file_content = f.read()
            
            logger.debug(f"[extract_document_symbols] Opening document in LSP server (size: {len(file_content)} bytes)")
            self._send_notification(
                "textDocument/didOpen",
                {
                    "textDocument": {
                        "uri": file_uri,
                        "languageId": "python",
                        "version": 1,
                        "text": file_content,
                    }
                }
            )
            logger.debug(f"[extract_document_symbols] Document opened, waiting for indexing...")
            
            # Give the LSP server a moment to index the document
            time.sleep(0.5)
            
        except Exception as e:
            logger.warning(f"[extract_document_symbols] Failed to open document: {e}")
        
        response = self._send_request(
            "textDocument/documentSymbol", {"textDocument": {"uri": file_uri}}
        )
        logger.debug(f"[extract_document_symbols] Response: {response}")
        
        result = response.get("result", [])
        logger.info(f"[extract_document_symbols] Got {len(result)} symbols from LSP")
        logger.debug(f"[extract_document_symbols] Symbols: {result}")
        
        # Returns an array of symbols (Name, Kind: Function/Class, Range: Start/End Lines)
        return result
