from typing import TypedDict, Annotated, Any, Literal
from operator import add
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class CodingAgentState(TypedDict):
    # ==========================================
    # PROJECT CONFIGURATION
    # ==========================================
    project_root: str  # The root directory of the project (where the agent is started from)
    
    # ==========================================
    # CONVERSATION STATE
    # ==========================================
    messages: Annotated[list[BaseMessage], add_messages]
    summary: str  # Rolling summary of past chat messages to save context windows

    # ==========================================
    # PLANNER & PROGRESS STATE
    # ==========================================
    goal: str  # The high-level objective assigned by the user
    current_task: str  # ID of the task currently being executed by the agent
    
    tasks: list[dict[str, Any]]
    # Schema: [{
    #     "id": str,
    #     "title": str,
    #     "description": str,
    #     "status": "pending" | "in_progress" | "completed" | "blocked",
    #     "depends_on": list[str],  # List of task IDs
    #     "notes": str | None
    # }]
    
    # ==========================================
    # SHARED REPOSITORY STATE
    # ==========================================
    workspace: dict[str, dict[str, Any]]
    # Schema: { 
    #     file_path (str): {
    #         "summary": str, "language": str, "classes": list[str],
    #         "functions": list[str], "dependencies": list[str]
    #     }
    # }
    
    known_facts: Annotated[list[dict[str, Any]], add]
    # Schema: [{ "fact": str, "source_files": list[str], "confidence": float }]
    
    artifacts: dict[str, Literal["read", "created", "modified", "deleted"]]
    # Schema: { file_path (str): status_string }
    
    searches: dict[str, list[dict[str, Any]]]
    # Schema: { 
    #     query_string: [{ "file": str, "line": int, "symbol": str | None, "preview": str }] 
    # }

    # ==========================================
    # TOOLS & RUNTIME
    # ==========================================
    tool_history: Annotated[list[dict[str, Any]], add]
    # Schema: [{ 
    #     "tool": str, "arguments": dict, "successful": bool, 
    #     "timestamp": str, "duration_ms": int, "cached": bool 
    # }]
    
    tool_cache: dict[str, dict[str, Any]]
    # Schema: { 
    #     cache_key (str): { "tool": str, "workspace_hash": str, "result": Any } 
    # }
    
    runtime: dict[str, int]
    # Schema: { 
    #     "iterations": int, "max_iterations": int, 
    #     "planner_calls": int, "tool_calls": int 
    # }
    
    # ==========================================
    # CONTEXT & PROMPTS
    # ==========================================
    system_prompt: str  # System prompt to inject into messages
    state_context: str  # Formatted state context from CodingAgentState