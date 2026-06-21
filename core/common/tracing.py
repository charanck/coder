from functools import lru_cache

from langfuse import Langfuse, get_client
from langfuse.langchain import CallbackHandler

from config import load_config


@lru_cache(maxsize=1)
def get_langfuse_callback_handler() -> CallbackHandler | None:
    """Create a Langfuse callback handler when tracing is enabled."""
    langfuse_config = load_config().langfuse

    if not langfuse_config.enabled:
        return None

    if not langfuse_config.public_key or not langfuse_config.secret_key:
        return None

    Langfuse(
        public_key=langfuse_config.public_key,
        secret_key=langfuse_config.secret_key,
        base_url=langfuse_config.base_url,
    )
    return CallbackHandler()


def flush_langfuse_traces() -> None:
    """Flush queued Langfuse events when tracing is enabled."""
    if not load_config().langfuse.enabled:
        return

    get_client().flush()