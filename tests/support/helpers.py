from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace


def build_sample_workspace(root: Path) -> None:
    (root / "src").mkdir(parents=True, exist_ok=True)
    (root / "tests").mkdir(parents=True, exist_ok=True)
    (root / ".hidden").mkdir(parents=True, exist_ok=True)

    (root / "pyproject.toml").write_text("[project]\nname = 'sample'\n", encoding="utf-8")
    (root / "src" / "app.py").write_text("def run():\n    return 'ok'\n", encoding="utf-8")
    (root / "tests" / "test_app.py").write_text("def test_run():\n    assert True\n", encoding="utf-8")
    (root / "README.md").write_text("# Sample\n", encoding="utf-8")
    (root / ".hidden" / "secret.txt").write_text("hidden\n", encoding="utf-8")


def make_tool_result(content: str = "tool output", **attrs) -> SimpleNamespace:
    return SimpleNamespace(content=content, **attrs)


def invoke_tool(tool_obj, **kwargs):
    return tool_obj.invoke(kwargs)