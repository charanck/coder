from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LanguageCase:
    extension: str
    language: str
    file_name: str
    symbol_name: str
    symbol_kind: int
    source_code: str
    expected_fact_fragment: str


SUPPORTED_LSP_LANGUAGE_CASES = [
    LanguageCase(
        extension=".py",
        language="python",
        file_name="sample.py",
        symbol_name="App",
        symbol_kind=5,
        source_code="class App:\n    pass\n",
        expected_fact_fragment="Defines core structure 'App' (Class)",
    ),
    LanguageCase(
        extension=".ts",
        language="typescript",
        file_name="sample.ts",
        symbol_name="App",
        symbol_kind=5,
        source_code="export class App {}\n",
        expected_fact_fragment="Defines core structure 'App' (Class)",
    ),
    LanguageCase(
        extension=".js",
        language="javascript",
        file_name="sample.js",
        symbol_name="run",
        symbol_kind=12,
        source_code="function run() {}\n",
        expected_fact_fragment="Exposes top-level executable capability 'run' (Function)",
    ),
    LanguageCase(
        extension=".go",
        language="go",
        file_name="sample.go",
        symbol_name="App",
        symbol_kind=23,
        source_code="type App struct{}\n",
        expected_fact_fragment="Defines core structure 'App' (Struct)",
    ),
    LanguageCase(
        extension=".rs",
        language="rust",
        file_name="sample.rs",
        symbol_name="App",
        symbol_kind=23,
        source_code="struct App;\n",
        expected_fact_fragment="Defines core structure 'App' (Struct)",
    ),
]