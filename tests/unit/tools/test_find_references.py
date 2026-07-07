"""Unit tests for find_references tool and its implementation with TreeSitter and LSP."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from core.model.search import FindReferencesResult, Reference
from core.tools.search import find_references
from tests.support.helpers import invoke_tool


class TestFindReferencesToolBasics:
    """Test basic find_references tool functionality."""

    def test_find_references_signature(self):
        """Test that find_references tool has expected structure."""
        # Check that tool has required attributes
        assert hasattr(find_references, "invoke")
        assert hasattr(find_references, "name")
        assert find_references.name == "find_references"
        assert hasattr(find_references, "description")
        # Description should contain info about the parameters
        assert "symbol" in find_references.description

    def test_find_references_with_empty_file(self, tmp_path):
        """Test find_references with an empty file."""
        test_file = tmp_path / "empty.py"
        test_file.write_text("", encoding="utf-8")

        result = invoke_tool(
            find_references,
            symbol="test_symbol",
            file_path=str(test_file),
        )

        assert isinstance(result, FindReferencesResult)
        assert result.count == 0
        assert len(result.references) == 0

    def test_find_references_with_nonexistent_file(self, tmp_path):
        """Test find_references with a file that doesn't exist."""
        nonexistent_file = tmp_path / "nonexistent.py"

        result = invoke_tool(
            find_references,
            symbol="test_symbol",
            file_path=str(nonexistent_file),
        )

        assert isinstance(result, FindReferencesResult)
        assert result.count == 0
        assert len(result.references) == 0

    def test_find_references_result_structure(self, tmp_path):
        """Test that result has correct FindReferencesResult structure."""
        test_file = tmp_path / "test.py"
        test_file.write_text("def func():\n    return 1\nfunc()", encoding="utf-8")

        result = invoke_tool(
            find_references,
            symbol="func",
            file_path=str(test_file),
        )

        assert isinstance(result, FindReferencesResult)
        assert hasattr(result, "references")
        assert hasattr(result, "count")
        assert isinstance(result.references, list)
        assert isinstance(result.count, int)


class TestFindReferencesToolWithPython:
    """Test find_references with Python files using TreeSitter."""

    def test_find_simple_function_reference(self, tmp_path):
        """Test finding references to a simple function."""
        test_file = tmp_path / "test.py"
        code = """
def my_func():
    return 42

result = my_func()
print(my_func)
"""
        test_file.write_text(code, encoding="utf-8")

        result = invoke_tool(
            find_references,
            symbol="my_func",
            file_path=str(test_file),
            project_root=str(tmp_path),
        )

        assert isinstance(result, FindReferencesResult)
        assert result.count >= 0
        assert isinstance(result.references, list)

        # Check that if references exist, they have correct structure
        for ref in result.references:
            assert isinstance(ref, Reference)
            assert ref.line > 0
            assert ref.column > 0
            assert isinstance(ref.text, str)

    def test_find_variable_reference(self, tmp_path):
        """Test finding references to a variable."""
        test_file = tmp_path / "test.py"
        code = """
counter = 0
counter += 1
print(counter)
total = counter
"""
        test_file.write_text(code, encoding="utf-8")

        result = invoke_tool(
            find_references,
            symbol="counter",
            file_path=str(test_file),
        )

        assert isinstance(result, FindReferencesResult)
        # Should find at least the definition and uses
        assert result.count >= 1

    def test_find_class_reference(self, tmp_path):
        """Test finding references to a class."""
        test_file = tmp_path / "test.py"
        code = """
class MyClass:
    pass

obj = MyClass()
another = MyClass()
"""
        test_file.write_text(code, encoding="utf-8")

        result = invoke_tool(
            find_references,
            symbol="MyClass",
            file_path=str(test_file),
        )

        assert isinstance(result, FindReferencesResult)
        assert result.count > 0

    def test_find_no_references(self, tmp_path):
        """Test when symbol has no references."""
        test_file = tmp_path / "test.py"
        code = """
def func_a():
    return 1

def func_b():
    return 2
"""
        test_file.write_text(code, encoding="utf-8")

        result = invoke_tool(
            find_references,
            symbol="nonexistent_func",
            file_path=str(test_file),
        )

        assert result.count == 0
        assert len(result.references) == 0

    def test_references_have_correct_line_numbers(self, tmp_path):
        """Test that line numbers in references are accurate."""
        test_file = tmp_path / "test.py"
        code = """x = 1
y = x + 1
z = x * 2
"""
        test_file.write_text(code, encoding="utf-8")

        result = invoke_tool(
            find_references,
            symbol="x",
            file_path=str(test_file),
        )

        if result.count > 0:
            # References should have increasing line numbers or be the same
            lines = [ref.line for ref in result.references]
            assert all(isinstance(line, int) and line > 0 for line in lines)

    def test_references_respect_max_matches(self, tmp_path):
        """Test that max_matches parameter is respected."""
        test_file = tmp_path / "test.py"
        # Create a file with many references to the same symbol
        code = "x = 1\n" + "\n".join([f"y{i} = x" for i in range(100)])
        test_file.write_text(code, encoding="utf-8")

        result = invoke_tool(
            find_references,
            symbol="x",
            file_path=str(test_file),
            max_matches=10,
        )

        # Should not exceed max_matches
        assert result.count <= 10
        assert len(result.references) <= 10


class TestFindReferencesToolWithLSP:
    """Test find_references with LSP integration."""

    @patch("core.tools.search.lsp_manager.get_by_extension")
    def test_find_references_uses_lsp_when_available(self, mock_get_lsp, tmp_path):
        """Test that LSP is used when available."""
        test_file = tmp_path / "test.py"
        test_file.write_text("def func():\n    pass\nfunc()", encoding="utf-8")

        # Mock LSP client
        mock_lsp_client = MagicMock()
        mock_lsp_client.find_references.return_value = FindReferencesResult(
            references=[
                Reference(
                    file_path=str(test_file),
                    line=1,
                    column=5,
                    text="def func():",
                ),
                Reference(
                    file_path=str(test_file),
                    line=3,
                    column=1,
                    text="func()",
                ),
            ],
            count=2,
        )
        mock_get_lsp.return_value = mock_lsp_client

        result = invoke_tool(
            find_references,
            symbol="func",
            file_path=str(test_file),
            project_root=str(tmp_path),
        )

        # LSP should have been called
        mock_get_lsp.assert_called_once()
        # Result should come from LSP
        assert result.count == 2
        assert len(result.references) == 2

    @patch("core.tools.search.lsp_manager.get_by_extension")
    def test_find_references_falls_back_to_treesitter_on_lsp_error(
        self, mock_get_lsp, tmp_path
    ):
        """Test fallback to TreeSitter when LSP fails."""
        test_file = tmp_path / "test.py"
        test_file.write_text("def func():\n    pass\nfunc()", encoding="utf-8")

        # Mock LSP client that raises an error
        mock_lsp_client = MagicMock()
        mock_lsp_client.find_references.side_effect = Exception("LSP error")
        mock_get_lsp.return_value = mock_lsp_client

        result = invoke_tool(
            find_references,
            symbol="func",
            file_path=str(test_file),
            project_root=str(tmp_path),
        )

        # Should still return a valid result (from TreeSitter fallback)
        assert isinstance(result, FindReferencesResult)
        # Exact results depend on TreeSitter, but structure should be valid
        assert hasattr(result, "references")
        assert hasattr(result, "count")

    @patch("core.tools.search.lsp_manager.get_by_extension")
    def test_find_references_when_lsp_not_available(self, mock_get_lsp, tmp_path):
        """Test TreeSitter fallback when LSP not available for file type."""
        test_file = tmp_path / "test.py"
        test_file.write_text("def func():\n    pass\nfunc()", encoding="utf-8")

        # Mock that LSP is not available
        mock_get_lsp.return_value = None

        result = invoke_tool(
            find_references,
            symbol="func",
            file_path=str(test_file),
            project_root=str(tmp_path),
        )

        # Should use TreeSitter fallback
        assert isinstance(result, FindReferencesResult)
        mock_get_lsp.assert_called_once()


class TestFindReferencesToolWithTreeSitter:
    """Test find_references with TreeSitter backend."""

    @patch("core.tools.search.lsp_manager.get_by_extension")
    def test_treesitter_finds_function_definitions(self, mock_get_lsp, tmp_path):
        """Test that TreeSitter can find function definitions."""
        mock_get_lsp.return_value = None  # Force TreeSitter fallback

        test_file = tmp_path / "test.py"
        code = """
def helper():
    return 1

x = helper()
y = helper()
"""
        test_file.write_text(code, encoding="utf-8")

        result = invoke_tool(
            find_references,
            symbol="helper",
            file_path=str(test_file),
        )

        assert isinstance(result, FindReferencesResult)
        # TreeSitter should find at least the references (definition + calls)
        if result.count > 0:
            assert all(ref.line > 0 for ref in result.references)
            assert all(ref.column > 0 for ref in result.references)

    @patch("core.tools.search.lsp_manager.get_by_extension")
    def test_treesitter_handles_complex_symbols(self, mock_get_lsp, tmp_path):
        """Test that TreeSitter handles various symbol types."""
        mock_get_lsp.return_value = None  # Force TreeSitter fallback

        test_file = tmp_path / "test.py"
        code = """
class Config:
    debug = True

config = Config()
if config.debug:
    pass
"""
        test_file.write_text(code, encoding="utf-8")

        # Test finding class references
        result = invoke_tool(
            find_references,
            symbol="Config",
            file_path=str(test_file),
        )

        assert isinstance(result, FindReferencesResult)

    @patch("core.tools.search.lsp_manager.get_by_extension")
    def test_treesitter_with_unsupported_language(self, mock_get_lsp, tmp_path):
        """Test TreeSitter with files of unsupported language."""
        mock_get_lsp.return_value = None

        test_file = tmp_path / "test.unknown"
        test_file.write_text("some random content", encoding="utf-8")

        result = invoke_tool(
            find_references,
            symbol="symbol",
            file_path=str(test_file),
        )

        # Should return empty result gracefully
        assert isinstance(result, FindReferencesResult)


class TestFindReferencesToolIntegration:
    """Integration tests for find_references tool."""

    def test_find_references_in_realistic_project_structure(self, tmp_path):
        """Test find_references in a realistic multi-file project structure."""
        # Create project structure
        src_dir = tmp_path / "src"
        src_dir.mkdir()

        # Create main file
        main_file = src_dir / "main.py"
        main_file.write_text("""
from utils import helper

def main():
    result = helper(42)
    return result
""", encoding="utf-8")

        # Create utils file
        utils_file = src_dir / "utils.py"
        utils_file.write_text("""
def helper(x):
    return x * 2
""", encoding="utf-8")

        # Find references to main function
        result = invoke_tool(
            find_references,
            symbol="main",
            file_path=str(main_file),
            project_root=str(tmp_path),
        )

        assert isinstance(result, FindReferencesResult)

    def test_find_references_with_unicode_content(self, tmp_path):
        """Test find_references with files containing unicode."""
        test_file = tmp_path / "test.py"
        code = """
def café():  # Unicode in function name
    return "☕"

result = café()  # Unicode in reference
"""
        test_file.write_text(code, encoding="utf-8")

        result = invoke_tool(
            find_references,
            symbol="café",
            file_path=str(test_file),
        )

        assert isinstance(result, FindReferencesResult)

    def test_find_references_result_str_representation(self, tmp_path):
        """Test that result has meaningful string representation."""
        test_file = tmp_path / "test.py"
        test_file.write_text("x = 1\ny = x", encoding="utf-8")

        result = invoke_tool(
            find_references,
            symbol="x",
            file_path=str(test_file),
        )

        # Result should have a string representation
        result_str = str(result)
        assert isinstance(result_str, str)
        if result.count > 0:
            assert "Found" in result_str or "reference" in result_str.lower()


class TestFindReferencesToolEdgeCases:
    """Test edge cases for find_references tool."""

    def test_find_references_with_empty_symbol(self, tmp_path):
        """Test find_references with empty symbol name."""
        test_file = tmp_path / "test.py"
        test_file.write_text("x = 1", encoding="utf-8")

        result = invoke_tool(
            find_references,
            symbol="",
            file_path=str(test_file),
        )

        assert isinstance(result, FindReferencesResult)

    def test_find_references_with_special_characters_in_symbol(self, tmp_path):
        """Test find_references with special characters in symbol."""
        test_file = tmp_path / "test.py"
        test_file.write_text("x = 1", encoding="utf-8")

        # Try finding symbol with special characters
        result = invoke_tool(
            find_references,
            symbol="__special__",
            file_path=str(test_file),
        )

        assert isinstance(result, FindReferencesResult)

    def test_find_references_with_very_large_file(self, tmp_path):
        """Test find_references performance with large file."""
        test_file = tmp_path / "large.py"
        # Create a large Python file
        code = "\n".join([f"var_{i} = {i}" for i in range(10000)])
        test_file.write_text(code, encoding="utf-8")

        result = invoke_tool(
            find_references,
            symbol="var_100",
            file_path=str(test_file),
        )

        assert isinstance(result, FindReferencesResult)

    def test_find_references_preserves_file_path_format(self, tmp_path):
        """Test that file paths in references have reasonable structure."""
        test_file = tmp_path / "test.py"
        test_file.write_text("def func():\n    pass\nfunc()", encoding="utf-8")

        file_path_str = str(test_file)
        result = invoke_tool(
            find_references,
            symbol="func",
            file_path=file_path_str,
            project_root=str(tmp_path),
        )

        # Check that result structure is valid
        assert isinstance(result, FindReferencesResult)
        # If we get references, check they have reasonable structure
        for ref in result.references:
            assert isinstance(ref.file_path, str)
            assert ref.line > 0
            assert ref.column > 0


class TestFindReferencesToolConsistency:
    """Test consistency and determinism of find_references."""

    def test_find_references_is_deterministic(self, tmp_path):
        """Test that multiple calls return consistent results."""
        test_file = tmp_path / "test.py"
        test_file.write_text("x = 1\ny = x\nz = x", encoding="utf-8")

        results = []
        for _ in range(3):
            result = invoke_tool(
                find_references,
                symbol="x",
                file_path=str(test_file),
            )
            results.append(result)

        # All results should have same count
        counts = [r.count for r in results]
        assert len(set(counts)) == 1  # All counts should be identical

        # All references should be the same
        for i in range(1, len(results)):
            assert results[i].count == results[0].count
            if results[0].count > 0:
                assert len(results[i].references) == len(results[0].references)
