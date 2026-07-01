import logging
import os
import importlib
from urllib.parse import urlparse
from tree_sitter import Parser
from typing import Any, Dict

from config import PROJECT_LANGUAGE_MAP
from core.model.search import FindReferencesResult, Reference

logger = logging.getLogger(__name__)


_PARSERS: Dict[str, Parser] = {}


def _load_parser(extension: str):
    """Load the tree-sitter parser for the given extension."""
    logger.debug(f"[_load_parser] Loading parser for extension: {extension}")
    
    language_name = PROJECT_LANGUAGE_MAP.get(extension)
    if not language_name:
        logger.warning(f"[_load_parser] Unknown extension: {extension}")
        return
    
    try:
        # Convert language name to module name (e.g., "Python" -> "python", "TypeScript" -> "typescript")
        language_lower = language_name.lower()
        
        # Replace spaces with underscores for C++ and C#
        module_name = language_lower.replace(" ", "_").replace("+", "plus").replace("#", "sharp")
        
        # Try to import the tree-sitter language module
        logger.debug(f"[_load_parser] Attempting to import tree_sitter_{module_name}")
        language_module = importlib.import_module(f"tree_sitter_{module_name}")
        
        # Get the language object from the module
        if hasattr(language_module, "language"):
            language = language_module.language
        else:
            logger.warning(f"[_load_parser] Module tree_sitter_{module_name} does not have 'language' attribute")
            return
        
        # Create a parser with this language
        parser = Parser()
        parser.set_language(language)  # type: ignore
        
        # Cache the parser
        _PARSERS[extension] = parser
        logger.info(f"[_load_parser] Successfully loaded parser for {language_name} ({extension})")
        
    except ImportError as e:
        logger.warning(f"[_load_parser] Could not import tree_sitter_{module_name}: {e}. Make sure tree-sitter-{language_lower} is installed.") # type: ignore
        return
    except AttributeError as e:
        logger.warning(f"[_load_parser] Error setting language for {language_name}: {e}")
        return
    except Exception as e:
        logger.error(f"[_load_parser] Unexpected error loading parser for {language_name}: {e}")
        return
    

class TreeSitterService:
    def __init__(self, extension: str):
        self.extension = extension
        self.parser = _PARSERS.get(extension)
        
        if not self.parser and extension in PROJECT_LANGUAGE_MAP:
            _load_parser(extension)
            self.parser = _PARSERS.get(extension)

    def extract_document_symbols(self, file_uri: str) -> list[dict[str, Any]]:
        """Extracts document symbols (functions, classes, etc.) from a file using tree-sitter."""
        logger.debug(f"[extract_document_symbols] Extracting symbols from: {file_uri}")
        
        if not self.parser:
            logger.warning(f"[extract_document_symbols] Parser not available for extension {self.extension}")
            return []
        
        try:
            # Convert URI to file path
            parsed = urlparse(file_uri)
            file_path = parsed.path
            
            # On Windows, remove leading slash from /C:/path -> C:/path
            if os.name == "nt" and len(file_path) > 2 and file_path[0] == "/" and file_path[2] == ":":
                file_path = file_path[1:]
            
            logger.debug(f"[extract_document_symbols] Reading file: {file_path}")
            
            # Read the file content
            with open(file_path, "r", encoding="utf-8") as f:
                file_content = f.read()
            
            # Parse the file into an AST
            tree = self.parser.parse(file_content.encode("utf-8"))
            logger.debug(f"[extract_document_symbols] Successfully parsed file, extracting symbols...")
            
            # Extract symbols from the AST
            symbols = self._extract_symbols_from_node(tree.root_node, file_content)
            logger.info(f"[extract_document_symbols] Extracted {len(symbols)} symbols")
            
            return symbols
        except FileNotFoundError as e:
            logger.error(f"[extract_document_symbols] File not found: {file_path}: {e}")  # type: ignore
            return []
        except Exception as e:
            logger.error(f"[extract_document_symbols] Error extracting symbols: {e}")
            return []
    
    def _extract_symbols_from_node(self, node: Any, file_content: str, symbols: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        """Recursively extracts symbols from AST nodes."""
        if symbols is None:
            symbols = []
        
        # Define symbol kinds for common language structures
        symbol_kinds = {
            "function_definition": "Function",
            "class_definition": "Class",
            "method_definition": "Method",
            "module": "Module",
            "interface": "Interface",
            "struct": "Struct",
            "enum": "Enum",
            "const_declaration": "Constant",
            "variable_declarator": "Variable",
        }
        
        node_type = node.type
        
        # Check if this node is a symbol we care about
        if node_type in symbol_kinds:
            try:
                # Extract symbol name from the node's first child or the node itself
                symbol_name = self._extract_symbol_name(node, file_content)
                if symbol_name:
                    symbol = {
                        "name": symbol_name,
                        "kind": symbol_kinds[node_type],
                        "range": {
                            "start": {"line": node.start_point[0], "character": node.start_point[1]},
                            "end": {"line": node.end_point[0], "character": node.end_point[1]},
                        },
                    }
                    symbols.append(symbol)
                    logger.debug(f"[_extract_symbols_from_node] Found {symbol_kinds[node_type]}: {symbol_name} at line {node.start_point[0]}")
            except Exception as e:
                logger.debug(f"[_extract_symbols_from_node] Error extracting symbol from {node_type}: {e}")
        
        # Recursively process child nodes
        for child in node.children:
            self._extract_symbols_from_node(child, file_content, symbols)
        
        return symbols
    
    def _extract_symbol_name(self, node: Any, file_content: str) -> str:
        """Extracts the name of a symbol from a node."""
        try:
            # For most symbols, the name is the text of a child identifier node
            for child in node.children:
                if child.type in ("identifier", "name"):
                    return file_content[child.start_byte:child.end_byte].decode("utf-8") if isinstance(file_content, bytes) else file_content[child.start_byte:child.end_byte] # type: ignore
            
            # Fallback: if no identifier child, use the node's first child
            if node.children:
                return file_content[node.children[0].start_byte:node.children[0].end_byte].decode("utf-8") if isinstance(file_content, bytes) else file_content[node.children[0].start_byte:node.children[0].end_byte] # type: ignore
        except Exception as e:
            logger.debug(f"[_extract_symbol_name] Error extracting name: {e}")
        
        return ""

    def find_references(self, file_uri: str, symbol_name: str) -> FindReferencesResult:
        """Finds all references to a symbol in a file using tree-sitter."""
        logger.debug(f"[find_references] Finding references for symbol '{symbol_name}' in {file_uri}")
        
        if not self.parser:
            logger.warning(f"[find_references] Parser not available for extension {self.extension}")
            return FindReferencesResult(references=[], count=0)
        
        try:
            # Convert URI to file path
            parsed = urlparse(file_uri)
            file_path = parsed.path
            
            # On Windows, remove leading slash from /C:/path -> C:/path
            if os.name == "nt" and len(file_path) > 2 and file_path[0] == "/" and file_path[2] == ":":
                file_path = file_path[1:]
            
            logger.debug(f"[find_references] Reading file: {file_path}")
            
            # Read the file content
            with open(file_path, "r", encoding="utf-8") as f:
                file_content = f.read()
                lines = file_content.splitlines(keepends=True)
            
            # Parse the file into an AST
            tree = self.parser.parse(file_content.encode("utf-8"))
            logger.debug(f"[find_references] Successfully parsed file, searching for references...")
            
            # Find all references by traversing the AST
            references = []
            self._find_references_in_node(tree.root_node, symbol_name, file_content, lines, file_uri, references)
            
            logger.info(f"[find_references] Found {len(references)} references for symbol '{symbol_name}'")
            
            return FindReferencesResult(references=references, count=len(references))
        except FileNotFoundError as e:
            logger.error(f"[find_references] File not found: {file_path}: {e}") # type: ignore
            return FindReferencesResult(references=[], count=0)
        except Exception as e:
            logger.error(f"[find_references] Error finding references: {e}")
            return FindReferencesResult(references=[], count=0)
    
    def _find_references_in_node(self, node: Any, symbol_name: str, file_content: str, lines: list[str], file_uri: str, references: list[Reference]) -> None:
        """Recursively finds all references to a symbol in AST nodes."""
        # Check if this node is an identifier matching the symbol name
        if node.type == "identifier":
            try:
                # Extract the identifier text
                node_text = file_content[node.start_byte:node.end_byte] if isinstance(file_content, str) else file_content[node.start_byte:node.end_byte].decode("utf-8")
                
                if node_text == symbol_name:
                    # Get the line text
                    line_num = node.start_point[0]
                    text = ""
                    if 0 <= line_num < len(lines):
                        text = lines[line_num].strip()
                    
                    reference = Reference(
                        file_path=file_uri,
                        line=line_num + 1,  # Convert from 0-based to 1-based
                        column=node.start_point[1] + 1,  # Convert from 0-based to 1-based
                        text=text,
                    )
                    references.append(reference)
                    logger.debug(f"[_find_references_in_node] Found reference at line {line_num + 1}, column {node.start_point[1] + 1}")
            except Exception as e:
                logger.debug(f"[_find_references_in_node] Error processing identifier node: {e}")
        
        # Recursively process child nodes
        for child in node.children:
            self._find_references_in_node(child, symbol_name, file_content, lines, file_uri, references)
    