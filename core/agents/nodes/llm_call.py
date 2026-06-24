from __future__ import annotations
from typing import Any
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from core.agents.nodes.util import tools_from_runnable_config
from core.common.model import get_planner_model
from core.model.state import CodingAgentState

def llm_call_node(state: CodingAgentState, runnable_config: RunnableConfig | None = None) -> dict[str, Any]:
    model = get_planner_model()
    available_tools = tools_from_runnable_config(runnable_config)
    model_with_tools = model.bind_tools(available_tools) if available_tools else model

    messages: list[BaseMessage] = state["messages"] or []
    runtime_stats = state.get("runtime", {}).copy() or {
        "iterations": 0, "max_iterations": 50, "retry_count": 0, "max_retries": 5, "planner_calls": 0
    }
    runtime_stats["planner_calls"] = runtime_stats.get("planner_calls", 0) + 1
    runtime_stats["retry_count"] = runtime_stats.get("retry_count", 0) + 1

    response = model_with_tools.invoke(messages)
    
    return {
        "messages": [response],
        "runtime": runtime_stats
    }