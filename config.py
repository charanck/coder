from functools import lru_cache
from typing import Literal
from pydantic import BaseModel


class ModelConfig(BaseModel):
    """Configuration for the model."""
    model_provider: Literal["google", "openai", "local"] = "google" 
    model_name: str = "gemma-3-31b-it"
    base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    api_key: str = "your-api-key-here"

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
        planner_timeout=planner_timeout,
        executor_timeout=executor_timeout
    )
    print("Configuration loaded successfully.")
    print(config)
    return config

