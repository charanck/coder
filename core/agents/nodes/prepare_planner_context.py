"""
Context preparation node for the planner agent.

This node prepares and injects context information into the state,
following the SOLID principle by separating context preparation
from context injection.
"""

from __future__ import annotations
from typing import Any, cast
from langchain_core.runnables import RunnableConfig

from core.agents.nodes.context import ContextFormatter
from core.agents.nodes.util import tools_from_runnable_config
from core.model.state import CodingAgentState


def get_planner_system_prompt() -> str:
    from core.agents.planner_graph import PLANNER_SYSTEM_PROMPT
    return PLANNER_SYSTEM_PROMPT


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
    # Extract tools from config
    available_tools = tools_from_runnable_config(config)
    
    # Build context components
    system_prompt = get_planner_system_prompt()
    state_context = ContextFormatter.build_state_context(cast(dict[str, Any], state)) or ""
    
    
    return {
        "system_prompt": system_prompt or "",
        "state_context": state_context,
    }
