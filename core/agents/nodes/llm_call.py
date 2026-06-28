from __future__ import annotations
from typing import Any, cast
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from core.agents.nodes.util import tools_from_runnable_config
from core.agents.nodes.context import MessageHistoryManager
from core.common.model import get_model
from core.model.state import CodingAgentState


def llm_call_node(state: CodingAgentState, config: RunnableConfig) -> dict[str, Any]:
    """
    LLM call node that invokes the model with prepared context from state.
    """
    model = get_model()
    available_tools = tools_from_runnable_config(config)
    model_with_tools = model.bind_tools(available_tools) if available_tools else model

    # Get messages and context from state
    messages: list[BaseMessage] = state.get("messages") or []
    system_prompt = state.get("system_prompt") or ""
    state_context = state.get("state_context") or ""
    
    # Inject context into messages
    enhanced_messages = MessageHistoryManager.build(
        messages,
        system_prompt,
        state_context
    )

    
    # Track runtime statistics
    runtime_stats = state.get("runtime", {}).copy() or {
        "iterations": 0, "max_iterations": 50, "retry_count": 0, "max_retries": 5, "planner_calls": 0
    }
    runtime_stats["planner_calls"] = runtime_stats.get("planner_calls", 0) + 1
    runtime_stats["retry_count"] = runtime_stats.get("retry_count", 0) + 1

    # Invoke the model
    response = model_with_tools.invoke(enhanced_messages)
    
    return {
        "messages": [response],
        "runtime": runtime_stats
    }