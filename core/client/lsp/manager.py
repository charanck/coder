from pathlib import Path
import logging
from config import get_lsp_language_for_extension, get_lsp_server_command
from core.client.lsp.client import LSPClient, LSPFactory

logger = logging.getLogger(__name__)

class LSPManager:
    def __init__(self):
        self._clients = {}

    def get(self, language: str, workspace: str):
        key = (Path(workspace).resolve(), language)
        logger.debug(f"[LSPManager.get] Cache key: {key}")

        if key not in self._clients:
            logger.info(f"[LSPManager.get] Creating new client for language={language}, workspace={workspace}")
            self._clients[key] = LSPFactory.create(
                language,
                workspace,
            )
        else:
            logger.debug(f"[LSPManager.get] Using cached client for {key}")

        client = self._clients[key]
        logger.info(f"[LSPManager.get] Returning client: {client is not None}")
        return client
    
    def get_by_extension(self, file_path: str, workspace: str) -> LSPClient | None:
        ext = Path(file_path).suffix
        logger.info(f"[LSPManager.get_by_extension] File extension: {ext}, workspace: {workspace}")
        
        language = get_lsp_language_for_extension(ext)
        logger.debug(f"[LSPManager.get_by_extension] Mapped extension {ext} to language: {language}")

        if not language:
            raise ValueError(f"No LSP configured for files with extension {ext}")

        server_cmd = get_lsp_server_command(language)
        if not server_cmd:
            logger.warning(f"No LSP server command configured for language {language}")
            return None

        logger.info(f"[LSPManager.get_by_extension] Getting client for language={language}")
        return self.get(language, workspace)
    

lsp_manager = LSPManager()