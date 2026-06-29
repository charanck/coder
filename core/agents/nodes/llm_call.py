from __future__ import annotations
from typing import Any, cast
import logging
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from core.agents.nodes.util import tools_from_runnable_config
from core.agents.nodes.context import MessageHistoryManager
from core.common.model import get_model
from core.model.state import CodingAgentState
from core.common.tracing import langfuse_observe

logger = logging.getLogger(__name__)


@langfuse_observe
def llm_call_node(state: CodingAgentState, config: RunnableConfig) -> dict[str, Any]:
    """
    LLM call node that invokes the model with prepared context from state.
    """
    logger.info("LLM call node invoked")
    
    model = get_model()
    available_tools = tools_from_runnable_config(config)
    logger.debug(f"LLM call with {len(available_tools)} available tools")
    
    model_with_tools = model.bind_tools(available_tools) if available_tools else model

    # Get messages and context from state
    messages: list[BaseMessage] = state.get("messages") or []
    system_prompt = state.get("system_prompt") or ""
    state_context = state.get("state_context") or ""
    summary = state.get("summary") or ""
    
    logger.debug(f"Building message history with {len(messages)} messages")
    
    # Inject context into messages
    enhanced_messages = MessageHistoryManager.build(
        messages,
        system_prompt,
        state_context,
        summary,
    )

    
    # Track runtime statistics
    runtime_stats = state.get("runtime", {}).copy() or {
        "iterations": 0, "max_iterations": 50, "retry_count": 0, "max_retries": 5, "planner_calls": 0
    }
    runtime_stats["planner_calls"] = runtime_stats.get("planner_calls", 0) + 1
    runtime_stats["retry_count"] = runtime_stats.get("retry_count", 0) + 1

    logger.debug(f"Invoking model (planner_calls={runtime_stats['planner_calls']})")
    
    # Invoke the model
    response = model_with_tools.invoke(enhanced_messages)
    logger.info(f"Model invocation completed, response type={type(response).__name__}")
    
    return {
        "messages": [response],
        "runtime": runtime_stats
    }