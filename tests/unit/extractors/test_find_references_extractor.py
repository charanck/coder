"""Unit tests for find_references extractor."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from core.model.search import FindReferencesResult, Reference
from core.tools.registry import TOOL_EXTRACTOR_REGISTRY
from core.tools.search import extract_find_references


class TestFindReferencesExtractorRegistry:
    """Test that find_references extractor is registered."""

    def test_find_references_registered_in_extractor_registry(self):
        """Verify find_references is registered in the extractor registry."""
        assert "find_references" in TOOL_EXTRACTOR_REGISTRY
        assert callable(TOOL_EXTRACTOR_REGISTRY["find_references"])

    def test_registry_contains_required_extractors(self):
        """Ensure all core search extractors are registered."""
        required_extractors = {
            "scan_project",
            "grep",
            "find_references",
        }
        assert required_extractors.issubset(set(TOOL_EXTRACTOR_REGISTRY))


class TestFindReferencesExtractor:
    """Test extract_find_references function."""

    def test_extract_empty_result(self):
        """Test extraction when no references found."""
        result = FindReferencesResult(references=[], count=0)
        args = {"symbol": "test_symbol"}

        updates = extract_find_references(result, args)

        assert updates == {}

    def test_extract_none_result(self):
        """Test extraction when result is None."""
        updates = extract_find_references(None, {"symbol": "test_symbol"})
        assert updates == {}

    def test_extract_invalid_result_type(self):
        """Test extraction with non-FindReferencesResult object."""
        result = {"invalid": "result"}
        args = {"symbol": "test_symbol"}

        updates = extract_find_references(result, args)

        assert updates == {}

    def test_extract_single_reference(self):
        """Test extraction of a single reference."""
        ref = Reference(
            file_path="src/module.py",
            line=10,
            column=5,
            text="test_symbol = value",
        )
        result = FindReferencesResult(references=[ref], count=1)
        args = {"symbol": "test_symbol"}

        updates = extract_find_references(result, args)

        assert "searches_update" in updates
        assert "workspace_update" in updates
        assert "artifacts_update" in updates

        # Check searches_update structure
        assert "references:test_symbol" in updates["searches_update"]
        search_entries = updates["searches_update"]["references:test_symbol"]
        assert len(search_entries) == 1
        assert search_entries[0]["file_path"] == "src/module.py"
        assert search_entries[0]["line"] == 10
        assert search_entries[0]["column"] == 5
        assert search_entries[0]["text"] == "test_symbol = value"
        assert "found_at" in search_entries[0]

        # Check artifacts_update structure
        assert updates["artifacts_update"]["src/module.py"] == "read"

        # Check workspace_update structure
        assert "src/module.py" in updates["workspace_update"]
        workspace_entry = updates["workspace_update"]["src/module.py"]
        assert workspace_entry["exists"] is True
        assert workspace_entry["type"] == "file"
        assert workspace_entry["last_referenced_symbol"] == "test_symbol"
        assert "last_validated" in workspace_entry

    def test_extract_multiple_references(self):
        """Test extraction of multiple references."""
        refs = [
            Reference(
                file_path="src/module.py",
                line=10,
                column=5,
                text="test_symbol = value",
            ),
            Reference(
                file_path="src/utils.py",
                line=25,
                column=12,
                text="result = test_symbol + 1",
            ),
            Reference(
                file_path="src/main.py",
                line=1,
                column=1,
                text="from module import test_symbol",
            ),
        ]
        result = FindReferencesResult(references=refs, count=3)
        args = {"symbol": "test_symbol"}

        updates = extract_find_references(result, args)

        # Check search entries
        search_entries = updates["searches_update"]["references:test_symbol"]
        assert len(search_entries) == 3

        # Verify all files are tracked
        assert len(updates["artifacts_update"]) == 3
        assert "src/module.py" in updates["artifacts_update"]
        assert "src/utils.py" in updates["artifacts_update"]
        assert "src/main.py" in updates["artifacts_update"]

        # Verify all files in workspace_update
        assert len(updates["workspace_update"]) == 3
        for file_path in ["src/module.py", "src/utils.py", "src/main.py"]:
            assert updates["workspace_update"][file_path]["exists"] is True
            assert updates["workspace_update"][file_path]["last_referenced_symbol"] == "test_symbol"

    def test_extract_reference_with_whitespace_text(self):
        """Test that text is stripped of leading/trailing whitespace."""
        ref = Reference(
            file_path="src/test.py",
            line=5,
            column=1,
            text="  some_symbol = value  \n",
        )
        result = FindReferencesResult(references=[ref], count=1)
        args = {"symbol": "some_symbol"}

        updates = extract_find_references(result, args)

        search_entries = updates["searches_update"]["references:some_symbol"]
        assert search_entries[0]["text"] == "some_symbol = value"

    def test_extract_unknown_symbol_in_args(self):
        """Test extraction when symbol is not provided in args."""
        ref = Reference(
            file_path="src/module.py",
            line=10,
            column=5,
            text="unknown = value",
        )
        result = FindReferencesResult(references=[ref], count=1)
        args = {}

        updates = extract_find_references(result, args)

        # Should still work with "unknown" as default symbol
        assert "references:unknown" in updates["searches_update"]

    def test_extract_complex_file_paths(self):
        """Test extraction with complex relative file paths."""
        refs = [
            Reference(
                file_path="../../project/src/core/module.py",
                line=1,
                column=1,
                text="symbol",
            ),
            Reference(
                file_path="src/deeply/nested/dir/file.py",
                line=50,
                column=10,
                text="symbol usage",
            ),
        ]
        result = FindReferencesResult(references=refs, count=2)
        args = {"symbol": "symbol"}

        updates = extract_find_references(result, args)

        # Both paths should be present
        assert "../../project/src/core/module.py" in updates["workspace_update"]
        assert "src/deeply/nested/dir/file.py" in updates["workspace_update"]

    def test_extract_timestamp_is_valid(self):
        """Test that extracted timestamp is valid ISO format."""
        ref = Reference(
            file_path="src/test.py",
            line=1,
            column=1,
            text="test",
        )
        result = FindReferencesResult(references=[ref], count=1)
        args = {"symbol": "test"}

        updates = extract_find_references(result, args)

        search_entry = updates["searches_update"]["references:test"][0]
        found_at = search_entry["found_at"]

        # Should parse as valid ISO format datetime
        parsed_dt = datetime.fromisoformat(found_at)
        assert parsed_dt is not None
        assert parsed_dt.tzinfo is not None  # Should have timezone info

    def test_extract_preserves_line_and_column_numbers(self):
        """Test that line and column numbers are correctly preserved."""
        test_cases = [
            (1, 1),
            (100, 50),
            (9999, 999),
        ]

        for line, col in test_cases:
            ref = Reference(
                file_path="src/test.py",
                line=line,
                column=col,
                text="symbol",
            )
            result = FindReferencesResult(references=[ref], count=1)
            args = {"symbol": "symbol"}

            updates = extract_find_references(result, args)

            search_entry = updates["searches_update"]["references:symbol"][0]
            assert search_entry["line"] == line
            assert search_entry["column"] == col


class TestFindReferencesExtractorWithTreeSitter:
    """Test extraction with TreeSitter-sourced references."""

    def test_extract_treesitter_results(self):
        """Test extraction of references found by TreeSitter."""
        refs = [
            Reference(
                file_path="file:///home/user/project/src/main.py",
                line=5,
                column=10,
                text="my_function()",
            ),
            Reference(
                file_path="file:///home/user/project/src/utils.py",
                line=15,
                column=3,
                text="my_function(x, y)",
            ),
        ]
        result = FindReferencesResult(references=refs, count=2)
        args = {"symbol": "my_function"}

        updates = extract_find_references(result, args)

        search_entries = updates["searches_update"]["references:my_function"]
        assert len(search_entries) == 2
        assert search_entries[0]["text"] == "my_function()"
        assert search_entries[1]["text"] == "my_function(x, y)"


class TestFindReferencesExtractorWithLSP:
    """Test extraction with LSP-sourced references."""

    def test_extract_lsp_results(self):
        """Test extraction of references found by LSP."""
        refs = [
            Reference(
                file_path="/workspace/project/src/module.py",
                line=20,
                column=8,
                text="class MyClass:",
            ),
            Reference(
                file_path="/workspace/project/src/main.py",
                line=45,
                column=12,
                text="obj = MyClass()",
            ),
        ]
        result = FindReferencesResult(references=refs, count=2)
        args = {"symbol": "MyClass"}

        updates = extract_find_references(result, args)

        search_entries = updates["searches_update"]["references:MyClass"]
        assert len(search_entries) == 2
        assert all(entry["found_at"] for entry in search_entries)


class TestFindReferencesExtractorDataStructure:
    """Test the data structure integrity of extracted updates."""

    def test_searches_update_structure(self):
        """Test that searches_update has correct structure."""
        ref = Reference(
            file_path="src/test.py",
            line=1,
            column=1,
            text="test",
        )
        result = FindReferencesResult(references=[ref], count=1)
        args = {"symbol": "test_func"}

        updates = extract_find_references(result, args)

        # Check key structure
        assert "searches_update" in updates
        searches = updates["searches_update"]
        assert "references:test_func" in searches
        assert isinstance(searches["references:test_func"], list)

        # Check entry structure
        entry = searches["references:test_func"][0]
        required_keys = {"file_path", "line", "column", "text", "found_at"}
        assert set(entry.keys()) == required_keys

    def test_artifacts_update_structure(self):
        """Test that artifacts_update is simple path->action mapping."""
        refs = [
            Reference(
                file_path="src/a.py",
                line=1,
                column=1,
                text="test",
            ),
            Reference(
                file_path="src/b.py",
                line=1,
                column=1,
                text="test",
            ),
        ]
        result = FindReferencesResult(references=refs, count=2)
        args = {"symbol": "test"}

        updates = extract_find_references(result, args)

        artifacts = updates["artifacts_update"]
        assert artifacts == {"src/a.py": "read", "src/b.py": "read"}

    def test_workspace_update_structure(self):
        """Test that workspace_update has correct metadata structure."""
        ref = Reference(
            file_path="src/test.py",
            line=1,
            column=1,
            text="test",
        )
        result = FindReferencesResult(references=[ref], count=1)
        args = {"symbol": "my_symbol"}

        updates = extract_find_references(result, args)

        workspace = updates["workspace_update"]
        entry = workspace["src/test.py"]

        assert entry["exists"] is True
        assert entry["type"] == "file"
        assert entry["last_referenced_symbol"] == "my_symbol"
        assert "last_validated" in entry


class TestFindReferencesExtractorIntegration:
    """Integration tests for find_references extractor."""

    def test_extractor_can_be_retrieved_from_registry(self):
        """Test that the extractor can be retrieved and called from registry."""
        ref = Reference(
            file_path="src/test.py",
            line=1,
            column=1,
            text="test",
        )
        result = FindReferencesResult(references=[ref], count=1)
        args = {"symbol": "test"}

        extractor_func = TOOL_EXTRACTOR_REGISTRY["find_references"]
        updates = extractor_func(result, args)

        assert "searches_update" in updates
        assert "workspace_update" in updates
        assert "artifacts_update" in updates

    def test_extractor_handles_result_from_find_references_tool(self):
        """Test extraction of result from actual find_references tool structure."""
        # Simulating a result structure from the find_references tool
        refs = [
            Reference(
                file_path="src/app.py",
                line=10,
                column=5,
                text="def process():",
            ),
            Reference(
                file_path="src/main.py",
                line=50,
                column=12,
                text="process()",
            ),
        ]
        result = FindReferencesResult(references=refs, count=2)

        # Simulate tool invocation args
        tool_args = {
            "symbol": "process",
            "file_path": "src/app.py",
            "max_matches": 250,
            "project_root": "/project",
        }

        updates = extract_find_references(result, tool_args)

        # Verify extracted state
        assert len(updates["searches_update"]["references:process"]) == 2
        assert len(updates["artifacts_update"]) == 2
        assert len(updates["workspace_update"]) == 2

    def test_multiple_sequential_extractions(self):
        """Test that multiple sequential extractions produce consistent results."""
        symbols = ["func_a", "func_b", "func_c"]
        all_updates = []

        for symbol in symbols:
            ref = Reference(
                file_path=f"src/{symbol}.py",
                line=1,
                column=1,
                text=symbol,
            )
            result = FindReferencesResult(references=[ref], count=1)
            args = {"symbol": symbol}

            updates = extract_find_references(result, args)
            all_updates.append(updates)

        # Each should be independent and complete
        for updates, symbol in zip(all_updates, symbols):
            assert f"references:{symbol}" in updates["searches_update"]
            assert f"src/{symbol}.py" in updates["workspace_update"]
