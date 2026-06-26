from functools import lru_cache
from typing import Literal
from pydantic import BaseModel


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
    planner_model: ModelConfig = ModelConfig(
        model_name="gemma-3-31b-it",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key="your-api-key-here"
    ) # Model configuration for the planner
    executor_model: ModelConfig = ModelConfig(
        model_name="gemma-3-27b-it",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key="your-api-key-here"
    ) # Model configuration for the executor
    langfuse: LangfuseConfig = LangfuseConfig()
    planner_timeout: int = 60 * 5  # Timeout for planner in seconds
    planner_step_timeout: int = 60  # Per-step timeout for the planner agent in seconds
    planner_agent_timeout: int = 60 * 2  # Total timeout for the planner agent in seconds
    planner_tool_call_limit: int = 20  # Maximum number of tool calls allowed per planner run
    executor_timeout: int = 60 * 10  # Timeout for executor in seconds


def _build_config_from_env(*, emit_summary: bool) -> Config:
    import os

    from dotenv import load_dotenv

    load_dotenv()

    planner_model_provider = os.getenv("PLANNER_MODEL_PROVIDER", "google")
    planner_model_name = os.getenv("PLANNER_MODEL_NAME", "gemma-3-31b-it")
    planner_base_url = os.getenv("PLANNER_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
    planner_api_key = os.getenv("PLANNER_API_KEY", "your-api-key-here")

    executor_model_provider = os.getenv("EXECUTOR_MODEL_PROVIDER", "google")
    executor_model_name = os.getenv("EXECUTOR_MODEL_NAME", "gemma-3-27b-it")
    executor_base_url = os.getenv("EXECUTOR_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
    executor_api_key = os.getenv("EXECUTOR_API_KEY", "your-api-key-here")

    langfuse_enabled = os.getenv("LANGFUSE_ENABLED", "false").lower() == "true"
    langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    langfuse_secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")
    langfuse_base_url = os.getenv("LANGFUSE_BASE_URL", "http://localhost:3000")

    planner_timeout = int(os.getenv("PLANNER_TIMEOUT", 60 * 10))
    planner_step_timeout = int(os.getenv("PLANNER_STEP_TIMEOUT", 60 * 10))
    planner_agent_timeout = int(os.getenv("PLANNER_AGENT_TIMEOUT", 60 * 10))
    planner_tool_call_limit = int(os.getenv("PLANNER_TOOL_CALL_LIMIT", 200))
    executor_timeout = int(os.getenv("EXECUTOR_TIMEOUT", 60 * 10))

    config = Config(
        planner_model=ModelConfig(
            model_provider=planner_model_provider,  # type: ignore
            model_name=planner_model_name,
            base_url=planner_base_url,
            api_key=planner_api_key,
        ),
        executor_model=ModelConfig(
            model_provider=executor_model_provider,  # type: ignore
            model_name=executor_model_name,
            base_url=executor_base_url,
            api_key=executor_api_key,
        ),
        langfuse=LangfuseConfig(
            enabled=langfuse_enabled,
            public_key=langfuse_public_key,
            secret_key=langfuse_secret_key,
            base_url=langfuse_base_url,
        ),
        planner_timeout=planner_timeout,
        planner_step_timeout=planner_step_timeout,
        planner_agent_timeout=planner_agent_timeout,
        planner_tool_call_limit=planner_tool_call_limit,
        executor_timeout=executor_timeout,
    )

    if emit_summary:
        print("Configuration loaded successfully.")
        print(
            {
                "planner_timeout": config.planner_timeout,
                "planner_step_timeout": config.planner_step_timeout,
                "planner_agent_timeout": config.planner_agent_timeout,
                "planner_tool_call_limit": config.planner_tool_call_limit,
                "executor_timeout": config.executor_timeout,
                "langfuse_enabled": config.langfuse.enabled,
                "langfuse_base_url": config.langfuse.base_url,
            }
        )

    return config


def load_unit_test_config() -> Config:
    """Load a fast, deterministic config for unit tests."""
    return Config(
        planner_model=ModelConfig(
            model_provider="local",
            model_name="unit-test-planner",
            base_url="http://localhost:11434/v1",
            api_key="unit-test-api-key",
        ),
        executor_model=ModelConfig(
            model_provider="local",
            model_name="unit-test-executor",
            base_url="http://localhost:11434/v1",
            api_key="unit-test-api-key",
        ),
        langfuse=LangfuseConfig(),
        planner_timeout=5,
        planner_step_timeout=5,
        planner_agent_timeout=15,
        planner_tool_call_limit=5,
        executor_timeout=5,
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

