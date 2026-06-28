from functools import lru_cache, wraps
from typing import Callable, TypeVar, Any
import logging

from langfuse import Langfuse, get_client, observe
from langfuse.langchain import CallbackHandler

from config import load_config

logger = logging.getLogger(__name__)

F = TypeVar('F', bound=Callable[..., Any])


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
    logger.info("Langfuse callback handler initialized")
    return CallbackHandler()


def flush_langfuse_traces() -> None:
    """Flush queued Langfuse events when tracing is enabled."""
    if not load_config().langfuse.enabled:
        return

    logger.debug("Flushing Langfuse traces")
    get_client().flush()


def langfuse_observe(func: F) -> F:
    """
    Decorator to add Langfuse @observe to functions when tracing is enabled.
    Falls back to identity decorator when tracing is disabled.
    """
    config = load_config()
    if config.langfuse.enabled:
        return observe()(func)
    
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)
    
    return wrapper  # type: ignore