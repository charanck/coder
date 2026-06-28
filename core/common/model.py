from pydantic import SecretStr
from langchain.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from config import load_config


def get_model() -> BaseChatModel:
    """Get the model based on the configuration."""
    config = load_config()

    if config.model.model_provider == "google":
        return ChatGoogleGenerativeAI(
            model=config.model.model_name,
            api_key=SecretStr(config.model.api_key),
            timeout=config.model_timeout
        )
    return ChatOpenAI(
            model=config.model.model_name,
            base_url=config.model.base_url,
            api_key=SecretStr(config.model.api_key),
            timeout=config.model_timeout
        )
