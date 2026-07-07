import logging
import os
from urllib.parse import urlparse
from tree_sitter_language_pack import get_parser
from typing import Any, Dict

from config import PROJECT_LANGUAGE_MAP, TREE_SITTER_LANGUAGE_MAP
from core.model.search import FindReferencesResult, Reference

logger = logging.getLogger(__name__)


_PARSERS: Dict[str, Any] = {}


def _normalize_language_name(language_name: str) -> str:
    """Normalize language names to tree-sitter-languages format using TREE_SITTER_LANGUAGE_MAP."""
    return TREE_SITTER_LANGUAGE_MAP.get(language_name, language_name.lower())


def _load_parser(extension: str):
    """Load the tree-sitter parser for the given extension."""
    logger.debug(f"[_load_parser] Loading parser for extension: {extension}")
    
    # Check if already cached
    if extension in _PARSERS:
        logger.debug(f"[_load_parser] Parser for {extension} already cached")
        return
    
    language_name = PROJECT_LANGUAGE_MAP.get(extension)
    if not language_name:
        logger.warning(f"[_load_parser] Unknown extension: {extension}")
        return
    
    try:
        # Normalize language name to tree-sitter-language-pack format
        language_normalized = _normalize_language_name(language_name)
        
        # Try to get the parser from tree_sitter_language_pack
        logger.debug(f"[_load_parser] Attempting to load parser for language: {language_normalized}")
        parser = get_parser(language_normalized)
        
        # Cache the parser
        _PARSERS[extension] = parser
        logger.info(f"[_load_parser] Successfully loaded parser for {language_name} ({extension})")
        
    except Exception as e:
        logger.warning(f"[_load_parser] Could not load language '{language_name}': {e}. Make sure tree-sitter-language-pack is installed.") # type: ignore
    

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
            
            # Read the file content as bytes to preserve exact line endings (CRLF vs LF)
            # This ensures tree-sitter byte offsets match the actual file bytes
            with open(file_path, "rb") as f:
                file_bytes = f.read()
            
            # Parse the file into an AST using parse_bytes for accurate byte offsets
            tree = self.parser.parse_bytes(file_bytes)
            logger.debug("[extract_document_symbols] Successfully parsed file, extracting symbols...")
            
            # Extract symbols from the AST
            symbols = self._extract_symbols_from_node(tree.root_node(), file_bytes)
            logger.info(f"[extract_document_symbols] Extracted {len(symbols)} symbols")
            
            return symbols
        except FileNotFoundError as e:
            logger.error(f"[extract_document_symbols] File not found: {file_path}: {e}")  # type: ignore
            return []
        except Exception as e:
            import traceback
            logger.error(f"[extract_document_symbols] Error extracting symbols: {e}\n{traceback.format_exc()}")
            return []
    
    def _extract_symbols_from_node(self, node: Any, file_bytes: bytes, symbols: list[dict[str, Any]] | None = None, depth: int = 0) -> list[dict[str, Any]]:
        """Recursively extracts symbols from AST nodes.
        
        Extracts:
        - Functions (including async functions)
        - Classes
        - Variables and Constants (from assignments)
        - Module/Package imports
        - Properties and methods within classes
        """
        if symbols is None:
            symbols = []
        
        # Define symbol kinds mapping from tree-sitter node types to LSP symbol kinds
        symbol_kinds_mapping = {
            "function_definition": "Function",
            "class_definition": "Class",
            "async_function_definition": "Function",
            "assignment": "Variable",  # Will be upgraded to Constant if all caps
            "import_from_statement": "Module",
            "decorated_definition": None,  # Handle specially - check what's inside
        }
        
        node_kind = node.kind()
        
        # Handle decorated_definition by extracting the inner function or class
        if node_kind == "decorated_definition":
            for i in range(node.child_count()):
                child = node.child(i)
                if child.kind() in ("function_definition", "class_definition"):
                    # Process the decorated function/class
                    self._extract_symbols_from_node(child, file_bytes, symbols, depth)
                    # Don't recurse further for decorated_definition
                    return symbols
        
        # Check if this node is a symbol we care about
        if node_kind in symbol_kinds_mapping and symbol_kinds_mapping[node_kind] is not None:
            try:
                # Extract symbol name from the node
                symbol_name = None
                symbol_kind = symbol_kinds_mapping[node_kind]
                
                if node_kind == "assignment":
                    # For assignments, get the left-hand side identifier
                    symbol_name = self._extract_assignment_target(node, file_bytes)
                    # Determine if it's a constant (all uppercase) or variable
                    if symbol_name and symbol_name.isupper() and len(symbol_name) > 1:
                        symbol_kind = "Constant"
                elif node_kind == "import_from_statement":
                    # For imports, extract the module name being imported from
                    symbol_name = self._extract_import_module(node, file_bytes)
                else:
                    # For functions and classes, find the identifier child
                    symbol_name = self._extract_symbol_name(node, file_bytes)
                
                if symbol_name and symbol_name.isidentifier():  # Validate it's a valid identifier
                    start_pos = node.start_position()
                    end_pos = node.end_position()
                    
                    symbol = {
                        "name": symbol_name,
                        "kind": symbol_kind,
                        "range": {
                            "start": {"line": start_pos.row, "character": start_pos.column},
                            "end": {"line": end_pos.row, "character": end_pos.column},
                        },
                    }
                    symbols.append(symbol)
                    logger.debug(f"[_extract_symbols_from_node] Found {symbol_kind}: {symbol_name} at line {start_pos.row}")
            except Exception as e:
                logger.debug(f"[_extract_symbols_from_node] Error extracting symbol from {node_kind}: {e}")
        
        # Recursively process child nodes (for nested classes/functions)
        for i in range(node.child_count()):
            self._extract_symbols_from_node(node.child(i), file_bytes, symbols, depth + 1)
        
        return symbols
    
    def _extract_symbol_name(self, node: Any, file_bytes: bytes) -> str:
        """Extracts the name of a symbol from a node."""
        try:
            # For most symbols, the name is the text of a child identifier node
            for i in range(node.child_count()):
                child = node.child(i)
                if child.kind() == "identifier":
                    start, end = child.start_byte(), child.end_byte()
                    return file_bytes[start:end].decode("utf-8")
            
            # Fallback: return empty string if no identifier found
            logger.debug(f"[_extract_symbol_name] No identifier found in node of kind {node.kind()}")
        except Exception as e:
            logger.debug(f"[_extract_symbol_name] Error extracting name: {e}")
        
        return ""
    
    def _extract_assignment_target(self, node: Any, file_bytes: bytes) -> str:
        """Extracts the target name from an assignment node (left-hand side)."""
        try:
            # For assignments, the target is usually the first identifier on the left side
            for i in range(node.child_count()):
                child = node.child(i)
                if child.kind() == "identifier":
                    start, end = child.start_byte(), child.end_byte()
                    return file_bytes[start:end].decode("utf-8")
                elif child.kind() == "=":
                    # Stop at assignment operator - we only want left side
                    break
            
            logger.debug("[_extract_assignment_target] No target found in assignment node")
        except Exception as e:
            logger.debug(f"[_extract_assignment_target] Error extracting target: {e}")
        
        return ""
    
    def _extract_import_module(self, node: Any, file_bytes: bytes) -> str:
        """Extracts the module name from an import statement."""
        try:
            # For import_from_statement, find the module name (after 'from' keyword)
            # Structure: 'from' module_name 'import' items
            found_from = False
            for i in range(node.child_count()):
                child = node.child(i)
                
                if child.kind() == "from":
                    found_from = True
                elif found_from and child.kind() in ("identifier", "dotted_name"):
                    start, end = child.start_byte(), child.end_byte()
                    return file_bytes[start:end].decode("utf-8")
                elif found_from and child.kind() == "import":
                    # Stop at import keyword - we only want the module part
                    break
            
            logger.debug("[_extract_import_module] No module found in import statement")
        except Exception as e:
            logger.debug(f"[_extract_import_module] Error extracting module: {e}")
        
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
            
            # Read the file as bytes to preserve exact line endings and get correct byte offsets
            with open(file_path, "rb") as f:
                file_bytes = f.read()
            
            # Decode for text operations
            file_content = file_bytes.decode("utf-8")
            lines = file_content.splitlines(keepends=True)
            
            # Parse the file into an AST using parse_bytes for accurate byte offsets
            tree = self.parser.parse_bytes(file_bytes)
            logger.debug("[find_references] Successfully parsed file, searching for references...")
            
            # Find all references by traversing the AST
            references = []
            self._find_references_in_node(tree.root_node(), symbol_name, file_bytes, lines, file_uri, references)
            
            logger.info(f"[find_references] Found {len(references)} references for symbol '{symbol_name}'")
            
            return FindReferencesResult(references=references, count=len(references))
        except FileNotFoundError as e:
            logger.error(f"[find_references] File not found: {file_path}: {e}") # type: ignore
            return FindReferencesResult(references=[], count=0)
        except Exception as e:
            logger.error(f"[find_references] Error finding references: {e}")
            return FindReferencesResult(references=[], count=0)
    
    def _find_references_in_node(self, node: Any, symbol_name: str, file_bytes: bytes, lines: list[str], file_uri: str, references: list[Reference]) -> None:
        """Recursively finds all references to a symbol in AST nodes."""
        # Check if this node is an identifier matching the symbol name
        if node.kind() == "identifier":
            try:
                # Extract the identifier text
                start, end = node.start_byte(), node.end_byte()
                node_text = file_bytes[start:end].decode("utf-8")
                
                if node_text == symbol_name:
                    # Get the line text
                    start_pos = node.start_position()
                    line_num = start_pos.row
                    text = ""
                    if 0 <= line_num < len(lines):
                        text = lines[line_num].strip()
                    
                    reference = Reference(
                        file_path=file_uri,
                        line=line_num + 1,  # Convert from 0-based to 1-based
                        column=start_pos.column + 1,  # Convert from 0-based to 1-based
                        text=text,
                    )
                    references.append(reference)
                    logger.debug(f"[_find_references_in_node] Found reference at line {line_num + 1}, column {start_pos.column + 1}")
            except Exception as e:
                logger.debug(f"[_find_references_in_node] Error processing identifier node: {e}")
        
        # Recursively process child nodes
        for i in range(node.child_count()):
            self._find_references_in_node(node.child(i), symbol_name, file_bytes, lines, file_uri, references)
    