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
    executor_timeout: int = 60 * 10  # Timeout for executor in seconds


@lru_cache(maxsize=1)
def load_config() -> Config:
    """Load configuration from environment variables."""
    import os

    # load .env file if it exists
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

    planner_timeout = int(os.getenv("PLANNER_TIMEOUT", 60 * 5))
    executor_timeout = int(os.getenv("EXECUTOR_TIMEOUT", 60 * 10))

    config =  Config(
        planner_model=ModelConfig(
            model_provider=planner_model_provider,  # type: ignore
            model_name=planner_model_name,
            base_url=planner_base_url,
            api_key=planner_api_key
        ),
        executor_model=ModelConfig(
            model_provider=executor_model_provider,  # type: ignore
            model_name=executor_model_name,
            base_url=executor_base_url,
            api_key=executor_api_key
        ),
        langfuse=LangfuseConfig(
            enabled=langfuse_enabled,
            public_key=langfuse_public_key,
            secret_key=langfuse_secret_key,
            base_url=langfuse_base_url,
        ),
        planner_timeout=planner_timeout,
        executor_timeout=executor_timeout
    )
    print("Configuration loaded successfully.")
    print(
        {
            "planner_timeout": config.planner_timeout,
            "executor_timeout": config.executor_timeout,
            "langfuse_enabled": config.langfuse.enabled,
            "langfuse_base_url": config.langfuse.base_url,
        }
    )
    return config

