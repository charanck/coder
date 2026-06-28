from pydantic import SecretStr
import logging
from langchain.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from config import load_config
from core.common.tracing import langfuse_observe

logger = logging.getLogger(__name__)


@langfuse_observe
def get_model() -> BaseChatModel:
    """Get the model based on the configuration."""
    config = load_config()
    logger.debug(f"Getting model with provider={config.model.model_provider}, model={config.model.model_name}")

    if config.model.model_provider == "google":
        logger.info(f"Initializing ChatGoogleGenerativeAI with model={config.model.model_name}")
        return ChatGoogleGenerativeAI(
            model=config.model.model_name,
            api_key=SecretStr(config.model.api_key),
            timeout=config.model_timeout
        )
    logger.info(f"Initializing ChatOpenAI with model={config.model.model_name}, base_url={config.model.base_url}")
    return ChatOpenAI(
            model=config.model.model_name,
            base_url=config.model.base_url,
            api_key=SecretStr(config.model.api_key),
            timeout=config.model_timeout
        )
