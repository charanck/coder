from pathlib import Path
from core.client.lsp.client import LSPFactory

class LSPManager:
    def __init__(self):
        self._clients = {}

    def get(self, language: str, workspace: str):
        key = (Path(workspace).resolve(), language)

        if key not in self._clients:
            self._clients[key] = LSPFactory.create(
                language,
                workspace,
            )

        return self._clients[key]
    
    def get_by_extension(self, file_path: str, workspace: str):
        extension_to_language = {
            ".py": "python",
            ".ts": "typescript",
            ".js": "javascript",
            ".go": "go",
            ".rs": "rust",
        }

        ext = Path(file_path).suffix
        language = extension_to_language.get(ext)

        if not language:
            raise ValueError(f"No LSP configured for files with extension {ext}")

        return self.get(language, workspace)
    

lsp_manager = LSPManager()