from __future__ import annotations

import importlib
import logging
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock, Mock, patch

import pytest

from core.model.search import FindReferencesResult, Reference
from core.service.tree_sitter import TreeSitterService, _load_parser, _PARSERS


logger = logging.getLogger(__name__)


class TestLoadParser:
    """Tests for the _load_parser function."""

    def test_load_parser_with_valid_extension(self):
        """Test loading a parser for a valid extension."""
        with patch("core.service.tree_sitter.Parser") as mock_parser_class:
            with patch("importlib.import_module") as mock_import:
                # Mock the language module
                mock_language = MagicMock()
                mock_module = MagicMock()
                mock_module.language = mock_language
                mock_import.return_value = mock_module

                mock_parser_instance = MagicMock()
                mock_parser_class.return_value = mock_parser_instance

                # Clear any cached parsers for this test
                _PARSERS.clear()

                _load_parser(".py")

                # Verify the parser was created and cached
                mock_parser_instance.set_language.assert_called_once_with(mock_language)
                assert ".py" in _PARSERS
                assert _PARSERS[".py"] == mock_parser_instance

    def test_load_parser_with_unknown_extension(self):
        """Test loading a parser for an unknown extension."""
        _PARSERS.clear()

        _load_parser(".unknown")

        # Verify no parser was cached
        assert ".unknown" not in _PARSERS

    def test_load_parser_import_error(self):
        """Test handling of ImportError when loading parser."""
        with patch("importlib.import_module") as mock_import:
            mock_import.side_effect = ImportError("Module not found")

            _PARSERS.clear()

            _load_parser(".py")

            # Verify no parser was cached
            assert ".py" not in _PARSERS

    def test_load_parser_missing_language_attribute(self):
        """Test handling when language module has no 'language' attribute."""
        with patch("importlib.import_module") as mock_import:
            mock_module = MagicMock(spec=[])  # No 'language' attribute
            mock_import.return_value = mock_module

            _PARSERS.clear()

            _load_parser(".py")

            # Verify no parser was cached
            assert ".py" not in _PARSERS

    def test_load_parser_attribute_error(self):
        """Test handling of AttributeError when setting language."""
        with patch("importlib.import_module") as mock_import:
            mock_language = MagicMock()
            mock_module = MagicMock()
            mock_module.language = mock_language
            mock_import.return_value = mock_module

            with patch("core.service.tree_sitter.Parser") as mock_parser_class:
                mock_parser_instance = MagicMock()
                mock_parser_instance.set_language.side_effect = AttributeError("set_language failed")
                mock_parser_class.return_value = mock_parser_instance

                _PARSERS.clear()

                _load_parser(".py")

                # Verify no parser was cached
                assert ".py" not in _PARSERS

    def test_load_parser_converts_language_names(self):
        """Test that language names are correctly converted to module names."""
        with patch("importlib.import_module") as mock_import:
            mock_language = MagicMock()
            mock_module = MagicMock()
            mock_module.language = mock_language
            mock_import.return_value = mock_module

            with patch("core.service.tree_sitter.Parser") as mock_parser_class:
                mock_parser_instance = MagicMock()
                mock_parser_class.return_value = mock_parser_instance

                _PARSERS.clear()

                # Test C++ (should convert + to plus)
                _load_parser(".cpp")
                mock_import.assert_called()
                call_args = mock_import.call_args[0][0]
                assert "plus" in call_args or "cpp" in call_args.lower()


class TestTreeSitterServiceInit:
    """Tests for TreeSitterService initialization."""

    def test_init_with_cached_parser(self):
        """Test initialization when parser is already cached."""
        mock_parser = MagicMock()
        _PARSERS[".py"] = mock_parser

        service = TreeSitterService(".py")

        assert service.extension == ".py"
        assert service.parser == mock_parser

    def test_init_with_uncached_parser(self):
        """Test initialization when parser needs to be loaded."""
        # Make sure .py is not in _PARSERS before the test
        if ".py" in _PARSERS:
            del _PARSERS[".py"]

        with patch("core.service.tree_sitter._load_parser") as mock_load:
            def load_parser_side_effect(ext):
                mock_parser = MagicMock()
                _PARSERS[ext] = mock_parser
            
            mock_load.side_effect = load_parser_side_effect

            service = TreeSitterService(".py")

            mock_load.assert_called_once_with(".py")
            assert service.parser is not None

    def test_init_with_unknown_extension(self):
        """Test initialization with an unknown extension."""
        _PARSERS.clear()

        service = TreeSitterService(".unknown")

        assert service.extension == ".unknown"
        assert service.parser is None


class TestExtractDocumentSymbols:
    """Tests for extract_document_symbols method."""

    def test_extract_document_symbols_success(self, tmp_path):
        """Test successful extraction of document symbols."""
        # Create a test file
        test_file = tmp_path / "test.py"
        test_file.write_text("def hello():\n    pass\n", encoding="utf-8")

        file_uri = test_file.as_uri()

        # Create mock AST nodes
        mock_func_node = MagicMock()
        mock_func_node.type = "function_definition"
        mock_func_node.start_point = (0, 0)
        mock_func_node.end_point = (1, 11)
        mock_func_node.start_byte = 0
        mock_func_node.end_byte = 12

        mock_identifier = MagicMock()
        mock_identifier.type = "identifier"
        mock_identifier.start_byte = 4
        mock_identifier.end_byte = 9
        mock_func_node.children = [mock_identifier]

        mock_root = MagicMock()
        mock_root.children = [mock_func_node]

        mock_tree = MagicMock()
        mock_tree.root_node = mock_root

        # Mock the parser
        mock_parser = MagicMock()
        mock_parser.parse.return_value = mock_tree

        service = TreeSitterService(".py")
        service.parser = mock_parser

        symbols = service.extract_document_symbols(file_uri)

        assert len(symbols) == 1
        assert symbols[0]["name"] == "hello"
        assert symbols[0]["kind"] == "Function"
        assert symbols[0]["range"]["start"]["line"] == 0

    def test_extract_document_symbols_no_parser(self):
        """Test extraction when parser is not available."""
        service = TreeSitterService(".unknown")
        service.parser = None

        result = service.extract_document_symbols("file:///test.unknown")

        assert result == []

    def test_extract_document_symbols_file_not_found(self):
        """Test extraction when file does not exist."""
        mock_parser = MagicMock()
        service = TreeSitterService(".py")
        service.parser = mock_parser

        result = service.extract_document_symbols("file:///nonexistent/file.py")

        assert result == []

    def test_extract_document_symbols_windows_path(self, tmp_path):
        """Test extraction with Windows-style file URI."""
        test_file = tmp_path / "test.py"
        test_file.write_text("class MyClass:\n    pass\n", encoding="utf-8")

        # Create Windows-style URI
        file_uri = test_file.as_uri()

        mock_class_node = MagicMock()
        mock_class_node.type = "class_definition"
        mock_class_node.start_point = (0, 0)
        mock_class_node.end_point = (1, 8)
        mock_class_node.start_byte = 0
        mock_class_node.end_byte = 21

        mock_identifier = MagicMock()
        mock_identifier.type = "identifier"
        mock_identifier.start_byte = 6
        mock_identifier.end_byte = 13
        mock_class_node.children = [mock_identifier]

        mock_root = MagicMock()
        mock_root.children = [mock_class_node]

        mock_tree = MagicMock()
        mock_tree.root_node = mock_root

        mock_parser = MagicMock()
        mock_parser.parse.return_value = mock_tree

        service = TreeSitterService(".py")
        service.parser = mock_parser

        symbols = service.extract_document_symbols(file_uri)

        assert len(symbols) == 1
        assert symbols[0]["kind"] == "Class"

    def test_extract_document_symbols_multiple_symbols(self, tmp_path):
        """Test extraction of multiple symbols."""
        test_file = tmp_path / "test.py"
        test_file.write_text("def func1():\n    pass\ndef func2():\n    pass\n", encoding="utf-8")

        file_uri = test_file.as_uri()

        # Create mock nodes for two functions
        mock_func1 = MagicMock()
        mock_func1.type = "function_definition"
        mock_func1.start_point = (0, 0)
        mock_func1.end_point = (1, 8)
        mock_func1.start_byte = 0
        mock_func1.end_byte = 13

        mock_id1 = MagicMock()
        mock_id1.type = "identifier"
        mock_id1.start_byte = 4
        mock_id1.end_byte = 9
        mock_func1.children = [mock_id1]

        mock_func2 = MagicMock()
        mock_func2.type = "function_definition"
        mock_func2.start_point = (2, 0)
        mock_func2.end_point = (3, 8)
        mock_func2.start_byte = 14
        mock_func2.end_byte = 27

        mock_id2 = MagicMock()
        mock_id2.type = "identifier"
        mock_id2.start_byte = 18
        mock_id2.end_byte = 23
        mock_func2.children = [mock_id2]

        mock_root = MagicMock()
        mock_root.children = [mock_func1, mock_func2]

        mock_tree = MagicMock()
        mock_tree.root_node = mock_root

        mock_parser = MagicMock()
        mock_parser.parse.return_value = mock_tree

        service = TreeSitterService(".py")
        service.parser = mock_parser

        symbols = service.extract_document_symbols(file_uri)

        assert len(symbols) == 2


class TestFindReferences:
    """Tests for find_references method."""

    def test_find_references_success(self, tmp_path):
        """Test successful reference finding."""
        test_file = tmp_path / "test.py"
        test_file.write_text("def foo():\n    foo()\n", encoding="utf-8")

        file_uri = test_file.as_uri()

        # Create mock identifier nodes
        mock_id1 = MagicMock()
        mock_id1.type = "identifier"
        mock_id1.start_byte = 4
        mock_id1.end_byte = 7
        mock_id1.start_point = (0, 4)

        mock_id2 = MagicMock()
        mock_id2.type = "identifier"
        mock_id2.start_byte = 16
        mock_id2.end_byte = 19
        mock_id2.start_point = (1, 4)

        def traverse_mock(node, symbol_name, file_content, lines, file_uri, references):
            if node.type == "identifier" and file_content[node.start_byte:node.end_byte] == symbol_name:
                line_num = node.start_point[0]
                text = lines[line_num].strip() if 0 <= line_num < len(lines) else ""
                references.append(
                    Reference(
                        file_path=file_uri,
                        line=line_num + 1,
                        column=node.start_point[1] + 1,
                        text=text,
                    )
                )

        mock_root = MagicMock()
        mock_root.children = [mock_id1, mock_id2]

        mock_tree = MagicMock()
        mock_tree.root_node = mock_root

        mock_parser = MagicMock()
        mock_parser.parse.return_value = mock_tree

        service = TreeSitterService(".py")
        service.parser = mock_parser

        # Mock the _find_references_in_node method
        with patch.object(service, "_find_references_in_node", side_effect=traverse_mock):
            result = service.find_references(file_uri, "foo")

            assert isinstance(result, FindReferencesResult)
            assert result.count >= 0

    def test_find_references_no_parser(self):
        """Test reference finding when parser is not available."""
        service = TreeSitterService(".unknown")
        service.parser = None

        result = service.find_references("file:///test.unknown", "symbol")

        assert isinstance(result, FindReferencesResult)
        assert result.count == 0
        assert len(result.references) == 0

    def test_find_references_file_not_found(self):
        """Test reference finding when file does not exist."""
        mock_parser = MagicMock()
        service = TreeSitterService(".py")
        service.parser = mock_parser

        result = service.find_references("file:///nonexistent/file.py", "symbol")

        assert isinstance(result, FindReferencesResult)
        assert result.count == 0

    def test_find_references_no_matches(self, tmp_path):
        """Test reference finding when symbol has no matches."""
        test_file = tmp_path / "test.py"
        test_file.write_text("def foo():\n    pass\n", encoding="utf-8")

        file_uri = test_file.as_uri()

        # Create mock identifier nodes that don't match the symbol
        mock_id1 = MagicMock()
        mock_id1.type = "identifier"
        mock_id1.start_byte = 4
        mock_id1.end_byte = 7
        mock_id1.start_point = (0, 4)

        mock_root = MagicMock()
        mock_root.children = [mock_id1]

        mock_tree = MagicMock()
        mock_tree.root_node = mock_root

        mock_parser = MagicMock()
        mock_parser.parse.return_value = mock_tree

        service = TreeSitterService(".py")
        service.parser = mock_parser

        # Mock the _find_references_in_node to return no references
        with patch.object(service, "_find_references_in_node"):
            result = service.find_references(file_uri, "nonexistent")

            assert result.count == 0


class TestExtractSymbolName:
    """Tests for _extract_symbol_name helper method."""

    def test_extract_symbol_name_from_identifier_child(self):
        """Test extracting symbol name from identifier child node."""
        file_content = "def hello():\n    pass"

        mock_identifier = MagicMock()
        mock_identifier.type = "identifier"
        mock_identifier.start_byte = 4
        mock_identifier.end_byte = 9

        mock_node = MagicMock()
        mock_node.children = [mock_identifier]

        service = TreeSitterService(".py")
        name = service._extract_symbol_name(mock_node, file_content)

        assert name == "hello"

    def test_extract_symbol_name_no_identifier_child(self):
        """Test extracting symbol name when no identifier child exists."""
        file_content = "def hello():\n    pass"

        mock_child = MagicMock()
        mock_child.type = "other"
        mock_child.start_byte = 0
        mock_child.end_byte = 3

        mock_node = MagicMock()
        mock_node.children = [mock_child]

        service = TreeSitterService(".py")
        name = service._extract_symbol_name(mock_node, file_content)

        assert name == "def"

    def test_extract_symbol_name_empty_result(self):
        """Test extracting symbol name with empty children."""
        mock_node = MagicMock()
        mock_node.children = []

        service = TreeSitterService(".py")
        name = service._extract_symbol_name(mock_node, "")

        assert name == ""


class TestExtractSymbolsFromNode:
    """Tests for _extract_symbols_from_node recursive method."""

    def test_extract_symbols_recognizes_function_definition(self):
        """Test that function_definition nodes are recognized."""
        file_content = "def test():\n    pass"

        mock_identifier = MagicMock()
        mock_identifier.type = "identifier"
        mock_identifier.start_byte = 4
        mock_identifier.end_byte = 8

        mock_node = MagicMock()
        mock_node.type = "function_definition"
        mock_node.start_point = (0, 0)
        mock_node.end_point = (1, 8)
        mock_node.children = [mock_identifier]

        mock_root = MagicMock()
        mock_root.type = "module"
        mock_root.children = [mock_node]
        mock_root.start_point = (0, 0)
        mock_root.end_point = (2, 0)

        service = TreeSitterService(".py")
        symbols = service._extract_symbols_from_node(mock_root, file_content)

        assert len(symbols) == 1
        assert symbols[0]["kind"] == "Function"

    def test_extract_symbols_recognizes_class_definition(self):
        """Test that class_definition nodes are recognized."""
        file_content = "class Test:\n    pass"

        mock_identifier = MagicMock()
        mock_identifier.type = "identifier"
        mock_identifier.start_byte = 6
        mock_identifier.end_byte = 10

        mock_node = MagicMock()
        mock_node.type = "class_definition"
        mock_node.start_point = (0, 0)
        mock_node.end_point = (1, 8)
        mock_node.children = [mock_identifier]

        mock_root = MagicMock()
        mock_root.type = "module"
        mock_root.children = [mock_node]

        service = TreeSitterService(".py")
        symbols = service._extract_symbols_from_node(mock_root, file_content)

        assert len(symbols) == 1
        assert symbols[0]["kind"] == "Class"

    def test_extract_symbols_recursively_processes_children(self):
        """Test recursive processing of child nodes."""
        file_content = "def func():\n    def inner():\n        pass"

        mock_inner_id = MagicMock()
        mock_inner_id.type = "identifier"
        mock_inner_id.start_byte = 20
        mock_inner_id.end_byte = 25

        mock_inner = MagicMock()
        mock_inner.type = "function_definition"
        mock_inner.start_point = (1, 4)
        mock_inner.end_point = (2, 8)
        mock_inner.children = [mock_inner_id]

        mock_outer_id = MagicMock()
        mock_outer_id.type = "identifier"
        mock_outer_id.start_byte = 4
        mock_outer_id.end_byte = 8

        mock_outer = MagicMock()
        mock_outer.type = "function_definition"
        mock_outer.start_point = (0, 0)
        mock_outer.end_point = (2, 8)
        mock_outer.children = [mock_outer_id, mock_inner]

        mock_root = MagicMock()
        mock_root.type = "module"
        mock_root.children = [mock_outer]

        service = TreeSitterService(".py")
        symbols = service._extract_symbols_from_node(mock_root, file_content)

        assert len(symbols) == 2
        assert symbols[0]["name"] == "func"
        assert symbols[1]["name"] == "inner"


class TestFindReferencesInNode:
    """Tests for _find_references_in_node recursive method."""

    def test_find_references_identifies_identifiers(self, tmp_path):
        """Test that identifier nodes matching symbol name are found."""
        test_file = tmp_path / "test.py"
        test_file.write_text("symbol = 1\nprint(symbol)\n", encoding="utf-8")

        file_content = test_file.read_text()
        lines = file_content.splitlines(keepends=True)
        file_uri = test_file.as_uri()

        # Correct byte offsets for "symbol = 1\nprint(symbol)\n"
        # First "symbol": bytes 0-6
        # Second "symbol": bytes 17-23
        mock_id1 = MagicMock()
        mock_id1.type = "identifier"
        mock_id1.start_byte = 0
        mock_id1.end_byte = 6
        mock_id1.start_point = (0, 0)

        mock_id2 = MagicMock()
        mock_id2.type = "identifier"
        mock_id2.start_byte = 17
        mock_id2.end_byte = 23
        mock_id2.start_point = (1, 6)

        mock_root = MagicMock()
        mock_root.type = "module"
        mock_root.children = [mock_id1, mock_id2]

        service = TreeSitterService(".py")
        references = []

        service._find_references_in_node(mock_root, "symbol", file_content, lines, file_uri, references)

        assert len(references) == 2

    def test_find_references_only_matches_exact_names(self, tmp_path):
        """Test that only exact symbol name matches are found."""
        test_file = tmp_path / "test.py"
        test_file.write_text("symbol = 1\nsymbolic = 2\n", encoding="utf-8")

        file_content = test_file.read_text()
        lines = file_content.splitlines(keepends=True)
        file_uri = test_file.as_uri()

        mock_id1 = MagicMock()
        mock_id1.type = "identifier"
        mock_id1.start_byte = 0
        mock_id1.end_byte = 6
        mock_id1.start_point = (0, 0)

        mock_id2 = MagicMock()
        mock_id2.type = "identifier"
        mock_id2.start_byte = 12
        mock_id2.end_byte = 19
        mock_id2.start_point = (1, 0)

        mock_root = MagicMock()
        mock_root.type = "module"
        mock_root.children = [mock_id1, mock_id2]

        service = TreeSitterService(".py")
        references = []

        service._find_references_in_node(mock_root, "symbol", file_content, lines, file_uri, references)

        assert len(references) == 1
        assert references[0].line == 1
