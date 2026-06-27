from __future__ import annotations
from pathlib import Path
import os
import platform
import shutil
import subprocess
from typing import Dict, List
from config import LSP_SERVER_COMMANDS, get_lsp_language_for_extension

INSTALL_MAP: Dict[str, Dict[str, str]] = {
    "gopls": {"brew": "brew install gopls", "apt": "sudo apt install -y golang-gopls", "go": "go install golang.org/x/tools/gopls@latest"},
    "pyright-langserver": {"npm": "npm install -g pyright", "brew": "npm install -g pyright"},
    "basedpyright-langserver": {"npm": "npm install -g pyright"},
    "typescript-language-server": {"npm": "npm install -g typescript typescript-language-server", "brew": "npm install -g typescript typescript-language-server"},
    "rust-analyzer": {"brew": "brew install rust-analyzer", "apt": "sudo apt install -y rust-analyzer", "dnf": "sudo dnf install -y rust-analyzer", "choco": "choco install rust-analyzer"},
}

PREFERRED_MANAGER = {
    "darwin": ["brew", "npm"],
    "linux": ["apt", "dnf", "pacman", "npm", "go"],
    "windows": ["choco", "scoop", "npm"],
}


def detect_platform() -> str:
    return platform.system().lower()


def detect_managers() -> List[str]:
    candidates = ["brew", "apt", "dnf", "pacman", "npm", "go", "choco", "scoop"]
    return [c for c in candidates if shutil.which(c)]


def _pick_install_command(executable: str, managers: List[str], system: str) -> str | None:
    entry = INSTALL_MAP.get(executable)
    if not entry:
        return None

    preferred = PREFERRED_MANAGER.get(system, [])
    ordered = [m for m in preferred if m in managers]
    ordered += [m for m in managers if m not in ordered]

    for m in ordered:
        if m in entry:
            return entry[m]

    return next(iter(entry.values())) if entry else None


def build_plan(languages: List[str] | None = None) -> Dict[str, str]:
    system = detect_platform()
    managers = detect_managers()

    if languages is None:
        languages = list(LSP_SERVER_COMMANDS.keys())

    plan: Dict[str, str] = {}
    for lang in languages:
        cmd_list = LSP_SERVER_COMMANDS.get(lang)
        if not cmd_list:
            plan[lang] = f"No server command configured for language '{lang}'"
            continue
        executable = cmd_list[0]
        cmd = _pick_install_command(executable, managers, system)
        plan[lang] = cmd or f"No installer known for server '{executable}'"

    return plan


def apply_plan(plan: Dict[str, str], auto_confirm: bool = False) -> Dict[str, int]:
    """Run install commands. Returns mapping language -> exit code (0 ok)."""
    if not auto_confirm:
        raise RuntimeError("apply_plan requires auto_confirm=True to run destructive commands")

    results: Dict[str, int] = {}
    for lang, cmd in plan.items():
        if cmd.startswith("No "):
            results[lang] = 0
            continue
        print(f"> {cmd}")
        rc = subprocess.run(cmd, shell=True).returncode
        results[lang] = rc

    return results


def detect_workspace_languages(workspace: str | None = None) -> List[str]:
    """Scan the workspace for file extensions and map them to supported LSP languages.

    This is conservative: it looks for file extensions that map to configured
    LSP languages. It skips common large directories like .git, node_modules, venv.
    """
    root = Path(workspace or os.getcwd())
    if not root.exists():
        return []

    skip_dirs = {'.git', '__pycache__', 'venv', '.venv', 'node_modules'}
    found: set[str] = set()

    for dirpath, dirnames, filenames in os.walk(root):
        # prune traversal
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for fn in filenames:
            ext = Path(fn).suffix
            if not ext:
                continue
            lang = get_lsp_language_for_extension(ext)
            if lang:
                found.add(lang)

    return sorted(found)
