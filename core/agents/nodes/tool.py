import time
from datetime import datetime
from typing import Any
import logging
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage, ToolCall, ToolMessage

from core.agents.nodes.util import tools_from_runnable_config
from core.model.state import CodingAgentState
from core.tools.registry import TOOL_EXTRACTOR_REGISTRY
from core.common.tracing import langfuse_observe

logger = logging.getLogger(__name__)

# Maximum length for tool output content before truncation
MAX_TOOL_OUTPUT_LENGTH = 10000


def tools_to_run_from_history(state: CodingAgentState) -> list[ToolCall]:
    """Extract the tools to run from the agent's history."""
    messages = state.get("messages", [])
    latest_ai_message = next((msg for msg in reversed(messages) if isinstance(msg, AIMessage)), None)
    if not latest_ai_message:
        logger.debug("No AI message found in history, no tools to run")
        return []
    
    tool_calls = latest_ai_message.tool_calls
    logger.debug(f"Found {len(tool_calls)} tool calls in latest AI message")
    return tool_calls

def save_tool_cache(state: CodingAgentState, tool_name: str, tool_args: dict[str, Any], result: Any) -> None:
    """Save the result of a tool call to the tool cache in the agent's state."""
    if "tool_cache" not in state:
        state["tool_cache"] = {}
    cache_key = f"{tool_name}:{str(tool_args)}"
    state["tool_cache"][cache_key] = {
        "result": result,
        "timestamp": datetime.now().isoformat()
    }
    logger.debug(f"Cached tool result for {tool_name}")

def _truncate_content(content: str, max_length: int = MAX_TOOL_OUTPUT_LENGTH) -> str:
    """
    Truncate content to max_length and add ellipsis if truncated.
    This helps reduce context window usage while preserving information.
    """
    if len(content) <= max_length:
        return content
    
    truncated = content[:max_length]
    logger.warning(f"Tool output truncated from {len(content)} to {max_length} chars")
    return f"{truncated}\n... [output truncated, total length: {len(content)} chars]"

@langfuse_observe
def tool_node(state: CodingAgentState, config: RunnableConfig) -> dict[str, Any]:
    available_tools = tools_from_runnable_config(config)
    tools_to_run = tools_to_run_from_history(state)
    
    logger.info(f"Tool node executing {len(tools_to_run)} tools")
    
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
        
        logger.debug(f"Executing tool: {tool_name} with args: {list(tool_args.keys())}")
        
        tool = tool_map.get(tool_name)
        start_time = time.perf_counter()
        tool_calls_count += 1
        
        if not tool:
            # Handle cases where the LLM hallucinations a non-existent tool
            error_msg = f"Error: Tool '{tool_name}' not found or is unavailable."
            logger.error(f"Tool '{tool_name}' not found")
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
            # Inject project_root from state if the tool supports it
            tool_args_with_context = tool_args.copy()
            
            # Check if tool accepts project_root parameter
            try:
                if hasattr(tool, "args_schema") and tool.args_schema:
                    schema = tool.args_schema
                    # Handle both Pydantic models and dict-based schemas
                    schema_params = getattr(schema, "__fields__", None) or getattr(schema, "properties", {})
                    if "project_root" in schema_params:
                        project_root = state.get("project_root")
                        tool_args_with_context["project_root"] = project_root
            except (AttributeError, TypeError):
                pass
            
            # Execute the tool synchronously (swap with await tool.ainvoke if working with async graphs)
            result = tool.invoke(tool_args_with_context)
            
            # Formulating output message (str convert ensure validation handles cleanly)
            content = str(result.content) if hasattr(result, "content") else str(result)
            
            # Ensure content is never None, empty, or whitespace-only
            if not content or not content.strip():
                content = f"Tool '{tool_name}' executed successfully with no output."
            
            # Truncate content to reduce context window usage
            content = _truncate_content(content)
            
            logger.debug(f"Tool '{tool_name}' executed successfully, output length: {len(content)}")
            
            extractor = TOOL_EXTRACTOR_REGISTRY.get(tool_name)
            if extractor:
                try:
                    updates = extractor(result, tool_args, state)
                    # Apply changes safely to state trackers
                    if "workspace_update" in updates:
                        workspace.update(updates["workspace_update"])
                    if "artifacts_update" in updates:
                        artifacts.update(updates["artifacts_update"])
                    if "searches_update" in updates:
                        searches.update(updates["searches_update"])
                    if "known_facts_update" in updates:
                        known_facts.extend(updates["known_facts_update"])
                    logger.debug(f"Extracted metadata from tool '{tool_name}' output")
                        
                except Exception as e:
                    logger.exception(f"Metadata extraction failed for tool '{tool_name}'")
            save_tool_cache(state, tool_name, tool_args, result)
            successful = True
        except Exception as e:
            content = f"Error executing tool '{tool_name}': {str(e)}"
            logger.exception(f"Tool execution failed for '{tool_name}'")
            successful = False

        duration_ms = int((time.perf_counter() - start_time) * 1000)
        logger.info(f"Tool '{tool_name}' completed in {duration_ms}ms, successful={successful}")
        
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
    logger.info(f"Tool node completed with {len(new_messages)} messages, total tool calls: {tool_calls_count}")

    return {
        "messages": new_messages,
        "tool_history": new_tool_history,
        "runtime": runtime_stats,
        "workspace": workspace,
        "artifacts": artifacts,
        "searches": searches,
        "known_facts": known_facts
    }