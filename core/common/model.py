from pydantic import SecretStr
from langchain.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from config import load_config


def get_planner_model() -> BaseChatModel:
    """Get the planner model based on the configuration."""
    config = load_config()

    if config.planner_model.model_provider == "google":
        return ChatGoogleGenerativeAI(
            model=config.planner_model.model_name,
            api_key=SecretStr(config.planner_model.api_key),
            timeout=config.planner_timeout
        )
    return ChatOpenAI(
            model=config.planner_model.model_name,
            base_url=config.planner_model.base_url,
            api_key=SecretStr(config.planner_model.api_key),
            timeout=config.planner_timeout
        )