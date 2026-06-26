from pathlib import Path
from config import get_lsp_language_for_extension
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
        ext = Path(file_path).suffix
        language = get_lsp_language_for_extension(ext)

        if not language:
            raise ValueError(f"No LSP configured for files with extension {ext}")

        return self.get(language, workspace)
    

lsp_manager = LSPManager()