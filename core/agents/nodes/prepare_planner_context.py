"""
Context preparation node for the planner agent.

This node prepares and injects context information into the state,
following the SOLID principle by separating context preparation
from context injection.
"""

from __future__ import annotations
from typing import Any, cast
import logging
from langchain_core.runnables import RunnableConfig

from core.agents.nodes.context import ContextFormatter
from core.agents.nodes.util import tools_from_runnable_config
from core.model.state import CodingAgentState
from core.common.tracing import langfuse_observe

logger = logging.getLogger(__name__)


def get_planner_system_prompt() -> str:
    from core.agents.planner_graph import PLANNER_SYSTEM_PROMPT
    return PLANNER_SYSTEM_PROMPT


@langfuse_observe
def prepare_planner_context_node(state: CodingAgentState, config: RunnableConfig) -> dict[str, Any]:
    """
    Prepare context values and inject them into the state.
    
    This node:
    1. Retrieves the system prompt
    2. Formats state into state context
    3. Formats tools into tools context
    4. Updates the state with these context values
    
    Args:
        state: The current coding agent state
        config: Optional runnable configuration (contains tools)
    
    Returns:
        Updated state with system_prompt and state_context
    """
    logger.debug("Preparing planner context")
    
    # Extract tools from config
    available_tools = tools_from_runnable_config(config)
    logger.debug(f"Found {len(available_tools)} available tools")
    
    # Build context components
    system_prompt = get_planner_system_prompt()
    state_context = ContextFormatter.build_state_context(cast(dict[str, Any], state)) or ""
    logger.debug(f"Built state context ({len(state_context)} chars)")
    
    logger.debug("Planner context preparation completed")
    
    return {
        "system_prompt": system_prompt or "",
        "state_context": state_context,
    }
