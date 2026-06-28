import time
from datetime import datetime
from typing import Any
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage, ToolCall, ToolMessage

from core.agents.nodes.util import tools_from_runnable_config
from core.model.state import CodingAgentState
from core.tools.registry import TOOL_EXTRACTOR_REGISTRY


def tools_to_run_from_history(state: CodingAgentState) -> list[ToolCall]:
    """Extract the tools to run from the agent's history."""
    messages = state.get("messages", [])
    latest_ai_message = next((msg for msg in reversed(messages) if isinstance(msg, AIMessage)), None)
    if not latest_ai_message:
        return []
    return latest_ai_message.tool_calls 

def save_tool_cache(state: CodingAgentState, tool_name: str, tool_args: dict[str, Any], result: Any) -> None:
    """Save the result of a tool call to the tool cache in the agent's state."""
    if "tool_cache" not in state:
        state["tool_cache"] = {}
    cache_key = f"{tool_name}:{str(tool_args)}"
    state["tool_cache"][cache_key] = {
        "result": result,
        "timestamp": datetime.now().isoformat()
    }

def tool_node(state: CodingAgentState, config: RunnableConfig) -> dict[str, Any]:
    available_tools = tools_from_runnable_config(config)
    tools_to_run = tools_to_run_from_history(state)
    
    # Map tool names to their execution functions/objects
    tool_map = {tool.name: tool for tool in available_tools}
    
    new_messages: list[ToolMessage] = []
    new_tool_history: list[dict[str, Any]] = []
    
    # Track metrics inside runtime state
    runtime_stats = state.get("runtime", {}).copy()
    tool_calls_count = runtime_stats.get("tool_calls", 0)

    # Deep copies of the existing state collections to update safely
    workspace = state.get("workspace", {}).copy()
    artifacts = state.get("artifacts", {}).copy()
    searches = state.get("searches", {}).copy()
    known_facts = [] # This field is Annotated[list, add], so we just collect new ones
    
    for tool_call in tools_to_run:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]
        
        tool = tool_map.get(tool_name)
        start_time = time.perf_counter()
        tool_calls_count += 1
        
        if not tool:
            # Handle cases where the LLM hallucinations a non-existent tool
            error_msg = f"Error: Tool '{tool_name}' not found or is unavailable."
            new_messages.append(ToolMessage(content=error_msg, tool_call_id=tool_id, name=tool_name))
            new_tool_history.append({
                "tool": tool_name,
                "arguments": tool_args,
                "successful": False,
                "timestamp": datetime.now().isoformat(),
                "duration_ms": int((time.perf_counter() - start_time) * 1000)
            })
            continue

        try:
            # Execute the tool synchronously (swap with await tool.ainvoke if working with async graphs)
            result = tool.invoke(tool_args)
            
            # Formulating output message (str convert ensure validation handles cleanly)
            content = str(result.content) if hasattr(result, "content") else str(result)
            
            # Ensure content is never None, empty, or whitespace-only
            if not content or not content.strip():
                content = f"Tool '{tool_name}' executed successfully with no output."
            
            extractor = TOOL_EXTRACTOR_REGISTRY.get(tool_name)
            if extractor:
                try:
                    updates = extractor(result, tool_args)
                    # Apply changes safely to state trackers
                    if "workspace_update" in updates:
                        workspace.update(updates["workspace_update"])
                    if "artifacts_update" in updates:
                        artifacts.update(updates["artifacts_update"])
                    if "searches_update" in updates:
                        searches.update(updates["searches_update"])
                    if "known_facts_update" in updates:
                        known_facts.extend(updates["known_facts_update"])
                        
                except Exception as e:
                    print(f"Metadata extraction failed for tool '{tool_name}': {e}")
            save_tool_cache(state, tool_name, tool_args, result)
            successful = True
        except Exception as e:
            content = f"Error executing tool '{tool_name}': {str(e)}"
            successful = False

        duration_ms = int((time.perf_counter() - start_time) * 1000)
        
        # Ensure content is always a valid non-empty string for Jinja template rendering
        if not isinstance(content, str) or not content.strip():
            content = f"Tool '{tool_name}' executed (no output available)."
        
        # 1. Append the ToolMessage structured execution result
        new_messages.append(ToolMessage(content=content, tool_call_id=tool_id, name=tool_name))
        
        # 2. Append metrics payload to history
        new_tool_history.append({
            "tool": tool_name,
            "arguments": tool_args,
            "successful": successful,
            "timestamp": datetime.now().isoformat(),
            "duration_ms": duration_ms
        })

    runtime_stats["tool_calls"] = tool_calls_count

    return {
        "messages": new_messages,
        "tool_history": new_tool_history,
        "runtime": runtime_stats,
        "workspace": workspace,
        "artifacts": artifacts,
        "searches": searches,
        "known_facts": known_facts
    }