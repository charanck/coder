from functools import lru_cache
from dataclasses import dataclass
from typing import Literal
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel
import os


@dataclass(frozen=True, slots=True)
class SupportedLanguageCase:
    extension: str
    language: str
    file_name: str
    symbol_name: str
    symbol_kind: int
    source_code: str
    expected_fact_fragment: str


SUPPORTED_LSP_LANGUAGE_CASES = (
    SupportedLanguageCase(
        extension=".py",
        language="python",
        file_name="sample.py",
        symbol_name="App",
        symbol_kind=5,
        source_code="class App:\n    pass\n",
        expected_fact_fragment="Defines core structure 'App' (Class)",
    ),
    SupportedLanguageCase(
        extension=".ts",
        language="typescript",
        file_name="sample.ts",
        symbol_name="App",
        symbol_kind=5,
        source_code="export class App {}\n",
        expected_fact_fragment="Defines core structure 'App' (Class)",
    ),
    SupportedLanguageCase(
        extension=".js",
        language="javascript",
        file_name="sample.js",
        symbol_name="run",
        symbol_kind=12,
        source_code="function run() {}\n",
        expected_fact_fragment="Exposes top-level executable capability 'run' (Function)",
    ),
    SupportedLanguageCase(
        extension=".go",
        language="go",
        file_name="sample.go",
        symbol_name="App",
        symbol_kind=23,
        source_code="type App struct{}\n",
        expected_fact_fragment="Defines core structure 'App' (Struct)",
    ),
    SupportedLanguageCase(
        extension=".rs",
        language="rust",
        file_name="sample.rs",
        symbol_name="App",
        symbol_kind=23,
        source_code="struct App;\n",
        expected_fact_fragment="Defines core structure 'App' (Struct)",
    ),
)

LSP_EXTENSION_TO_LANGUAGE = {
    case.extension: case.language for case in SUPPORTED_LSP_LANGUAGE_CASES
}

LSP_SERVER_COMMANDS = {
    "go": ["gopls"],
    "python": ["pyright-langserver", "--stdio"],
    "typescript": ["typescript-language-server", "--stdio"],
    "javascript": ["typescript-language-server", "--stdio"],
    "rust": ["rust-analyzer"],
}

PROJECT_LANGUAGE_MAP = {
    ".py": "Python",
    ".ts": "TypeScript",
    ".js": "JavaScript",
    ".go": "Go",
    ".rs": "Rust",
    ".cpp": "C++",
    ".c": "C",
    ".java": "Java",
    ".rb": "Ruby",
    ".php": "PHP",
    ".cs": "C#",
}

PROJECT_FRAMEWORK_FILES = {
    "package.json": "Node.js",
    "requirements.txt": "Python",
    "pyproject.toml": "Python",
    "go.mod": "Go Modules",
    "Cargo.toml": "Rust",
    "Gemfile": "Ruby",
    "composer.json": "PHP",
}


def get_supported_lsp_language_cases() -> tuple[SupportedLanguageCase, ...]:
    return SUPPORTED_LSP_LANGUAGE_CASES


def get_lsp_language_for_extension(extension: str) -> str | None:
    return LSP_EXTENSION_TO_LANGUAGE.get(extension)


def get_lsp_server_command(language: str) -> list[str] | None:
    command = LSP_SERVER_COMMANDS.get(language)
    return list(command) if command is not None else None


class ModelConfig(BaseModel):
    """Configuration for the model."""
    model_provider: Literal["google", "openai", "local"] = "google" 
    model_name: str = "gemma-3-31b-it"
    base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    api_key: str = "your-api-key-here"


class LangfuseConfig(BaseModel):
    """Configuration for Langfuse tracing."""
    enabled: bool = False
    public_key: str = ""
    secret_key: str = ""
    base_url: str = "http://localhost:3000"

class Config(BaseModel):
    """Configuration for the application."""
    model: ModelConfig = ModelConfig(
        model_name="gemma-3-31b-it",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key="your-api-key-here"
    ) # Model configuration
    langfuse: LangfuseConfig = LangfuseConfig()
    model_timeout: int = 60 * 5  # Timeout for model calls in seconds
    planner_step_timeout: int = 60  # Per-step timeout for the planner agent in seconds
    planner_agent_timeout: int = 60 * 2  # Total timeout for the planner agent in seconds
    planner_tool_call_limit: int = 20  # Maximum number of tool calls allowed per planner run


def _build_config_from_env(*, emit_summary: bool) -> Config:
    load_dotenv(find_dotenv())
    model_provider = os.getenv("MODEL_PROVIDER", "google")
    model_name = os.getenv("MODEL_NAME", "gemma-3-31b-it")
    base_url = os.getenv("BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
    api_key = os.getenv("API_KEY", "your-api-key-here")

    langfuse_enabled = os.getenv("LANGFUSE_ENABLED", "false").lower() == "true"
    langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    langfuse_secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")
    langfuse_base_url = os.getenv("LANGFUSE_BASE_URL", "http://localhost:3000")

    model_timeout = int(os.getenv("MODEL_TIMEOUT", 60 * 10))
    planner_step_timeout = int(os.getenv("PLANNER_STEP_TIMEOUT", 60 * 10))
    planner_agent_timeout = int(os.getenv("PLANNER_AGENT_TIMEOUT", 60 * 10))
    planner_tool_call_limit = int(os.getenv("PLANNER_TOOL_CALL_LIMIT", 200))

    config = Config(
        model=ModelConfig(
            model_provider=model_provider,  # type: ignore
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
        ),
        langfuse=LangfuseConfig(
            enabled=langfuse_enabled,
            public_key=langfuse_public_key,
            secret_key=langfuse_secret_key,
            base_url=langfuse_base_url,
        ),
        model_timeout=model_timeout,
        planner_step_timeout=planner_step_timeout,
        planner_agent_timeout=planner_agent_timeout,
        planner_tool_call_limit=planner_tool_call_limit,
    )

    if emit_summary:
        print("Configuration loaded successfully.")
        print(
            {
                "model_timeout": config.model_timeout,
                "planner_step_timeout": config.planner_step_timeout,
                "planner_agent_timeout": config.planner_agent_timeout,
                "planner_tool_call_limit": config.planner_tool_call_limit,
                "langfuse_enabled": config.langfuse.enabled,
                "langfuse_base_url": config.langfuse.base_url,
            }
        )

    return config


def load_unit_test_config() -> Config:
    """Load a fast, deterministic config for unit tests."""
    return Config(
        model=ModelConfig(
            model_provider="local",
            model_name="unit-test-model",
            base_url="http://localhost:11434/v1",
            api_key="unit-test-api-key",
        ),
        langfuse=LangfuseConfig(),
        model_timeout=5,
        planner_step_timeout=5,
        planner_agent_timeout=15,
        planner_tool_call_limit=5,
    )


def load_integration_test_config() -> Config:
    """Load environment-backed config for integration tests."""
    return _build_config_from_env(emit_summary=False)


@lru_cache(maxsize=1)
def load_runtime_config() -> Config:
    """Load configuration from environment variables for normal runtime use."""
    return _build_config_from_env(emit_summary=True)


def _current_config_profile() -> str:
    import os

    profile = os.getenv("APP_CONFIG_PROFILE", "").strip().lower()
    if profile:
        return profile

    current_test = os.getenv("PYTEST_CURRENT_TEST", "")
    normalized_test = current_test.replace("\\", "/")

    if "tests/integration/" in normalized_test:
        return "integration"
    if "tests/unit/" in normalized_test:
        return "unit"

    return "runtime"


def load_config() -> Config:
    """Load the appropriate config for the current execution context."""
    profile = _current_config_profile()

    if profile == "unit":
        return load_unit_test_config()
    if profile == "integration":
        return load_integration_test_config()
    return load_runtime_config()

