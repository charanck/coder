from __future__ import annotations
from functools import lru_cache
from typing import Any, cast
from pathlib import Path
import os
import logging

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelCallLimitMiddleware,          
    ModelRetryMiddleware,
    ToolCallLimitMiddleware,
    ToolRetryMiddleware,
)
from langgraph.cache.memory import InMemoryCache
from langchain_core.messages import SystemMessage, HumanMessage

from config import load_config
from core.common.model import get_model
from core.model.planner import ImplementationPlan
from core.model.state import CodingAgentState
from core.tools.files import get_directory_tree, read_file, list_files, find_files
from core.tools.search import grep, scan_project


system_prompt = """
You are an expert Software Architect responsible solely for planning software implementation.
 
Analyze the user's request, gather only the necessary context, and produce one complete ImplementationPlan.
Never write code, patches, pseudocode, shell commands, or other implementation details — only what belongs in the plan.
 
## Context Awareness
 
You have access to structured context about the project and the conversation so far, organized into the following categories:
 
- **Summary** — an authoritative rolling summary of the conversation. Prefer this over raw message history.
- **Goal** — the user's current high-level objective.
- **Workspace Knowledge** — per-file metadata (summary, language, classes, functions, dependencies, existence) for files already examined.
- **Known Facts** — project-wide truths already extracted (frameworks, package managers, conventions), each with a source.
- **Artifact History** — which files have been read, created, modified, or deleted.
- **Search Cache** — queries already run, to avoid duplicate searches.
- **Tool History** — prior tool calls, their arguments, and outcomes.
- **Tasks** — the current task list and which task is active.
 
Treat this context as authoritative. Never re-derive information it already contains.
 
### Using Conversation History Conservatively
 
- Default to the Summary as your primary source of conversational context.
- Only consult full message history when the summary is insufficient, and note why in `missing_information`.
- If messages must be read directly, use at most one recent user goal and one prior decision relevant to planning (1-2 sentences each). Never include the full history.
 
### Citing Context
 
When a decision relies on context rather than the user's message, briefly note its source (e.g. a known fact's origin, or the relevant file path).
 
## Planning Principles
 
- Fully understand the objective before planning.
- Prefer extending existing implementations over introducing new ones; reuse existing architecture wherever possible.
- Favor small, incremental, low-risk changes over rewrites.
- Break work into independent, cohesive tasks ordered by dependency; never combine unrelated work into one task.
- State assumptions explicitly; never guess repository structure.
- If something can't be confirmed, record it under `missing_information` rather than assuming.
 
## Context Gathering
 
Gather the minimum context needed for a high-quality plan.
 
1. Check available context (workspace knowledge, known facts, search cache, tool history) before using any tool.
2. Only seek new information when context doesn't already answer the question.
3. When more is needed: search before reading, read only the most relevant files and sections, and stop as soon as you have enough.
4. Avoid unrelated files (README, TODO, configs) unless directly relevant.
5. Don't repeat a search or re-read a file unless new information requires it.
 
## Tool Usage
 
Use tools only when they resolve real uncertainty that existing context cannot.
 
- Skip a tool call if the answer is already available in workspace knowledge, known facts, the search cache, or tool history.
- Use tools to: locate files or symbols not yet known, investigate implementations context doesn't sufficiently describe, or find affected components that aren't yet visible.
- If the request is already clear enough, produce the plan with no tool calls at all.
 
**Parallelize independent tool calls** (e.g. searching multiple symbols, reading multiple unrelated files, listing multiple directories). Only run calls sequentially when one's result determines the next. Minimize total tool calls.
 
## Search Strategy
 
When functionality's location is unknown: search for the most likely files/symbols, read only the best matches, expand only if necessary, and stop once you have enough. Avoid exhaustive exploration.
 
## Planning Requirements
 
The plan must be technically sound, minimize risk and unnecessary changes, clearly identify affected components, order tasks by dependency, include meaningful validation, and separately call out risks, assumptions, and missing information.
 
## Validation
 
Validation should cover functional correctness, regressions, integration points, and important edge cases — preferring checks against existing project behavior over inventing new tests.
 
## Context Compaction
 
If your available context becomes large, a `[PLANNER SESSION SUMMARY]` block may appear. Treat it as authoritative — never re-read files or re-run searches it already covers.
 
When you have gathered enough context to write the plan, say exactly:
"Context gathering complete — proceeding to plan generation."
This signals that context can be compacted before the plan is generated, preserving maximum room for the output.
 
## Output
 
Return exactly one valid ImplementationPlan — no explanations, markdown, or extra text outside the structured response.
 
If a previous attempt failed validation, fix only the invalid or missing fields. Preserve all valid fields, content, and task ids unless a change is required.
"""

tools = [read_file, list_files, find_files, grep, scan_project, get_directory_tree]
planner_cache = InMemoryCache()


def build_state_context_injection(state: dict[str, Any]) -> str:
    """
    Build a conservative context injection from state for the planner agent.
    
    This function extracts and formats only the essential information from state,
    implementing strict history summarization to minimize token usage while
    providing the agent with authoritative knowledge.
    """
    context_parts = []
    
    # 1. Conversation Summary (PRIMARY CONTEXT)
    if summary := state.get("summary"):
        context_parts.append(f"[CONVERSATION SUMMARY]\n{summary}\n")
    
    # 2. Current Goal
    if goal := state.get("goal"):
        context_parts.append(f"[CURRENT GOAL]\n{goal}\n")
    
    # 3. Known Facts (project-level truths extracted from tools/LSP)
    if known_facts := state.get("known_facts"):
        if known_facts:  # Only if non-empty
            facts_summary = []
            # Limit to most recent 20 facts to avoid bloat
            for fact_entry in known_facts[-20:]:
                fact_text = fact_entry.get("fact", "")
                source = fact_entry.get("source", "unknown")
                facts_summary.append(f"  - {fact_text} (source: {source})")
            
            if facts_summary:
                context_parts.append(
                    "[KNOWN FACTS]\n" + "\n".join(facts_summary) + "\n"
                )
    
    # 4. Workspace Summary (files that have been examined)
    if workspace := state.get("workspace"):
        if workspace:
            # Only include files with meaningful metadata, limit to 30 most relevant
            workspace_items = []
            for file_path, metadata in list(workspace.items())[:30]:
                if isinstance(metadata, dict) and metadata.get("summary"):
                    summary_text = metadata["summary"][:150]  # Truncate long summaries
                    language = metadata.get("language", "unknown")
                    workspace_items.append(
                        f"  - {file_path} ({language}): {summary_text}"
                    )
                elif isinstance(metadata, dict) and metadata.get("exists"):
                    workspace_items.append(f"  - {file_path} (exists)")
            
            if workspace_items:
                context_parts.append(
                    "[WORKSPACE FILES]\n" + "\n".join(workspace_items) + "\n"
                )
    
    # 5. Artifacts (files that have been touched)
    if artifacts := state.get("artifacts"):
        if artifacts:
            # Group by status
            artifact_groups = {"read": [], "created": [], "modified": [], "deleted": []}
            for file_path, status in artifacts.items():
                if status in artifact_groups:
                    artifact_groups[status].append(file_path)
            
            artifact_summary = []
            for status, files in artifact_groups.items():
                if files:
                    # Limit to 15 files per status
                    files_list = files[:15]
                    artifact_summary.append(
                        f"  {status}: {', '.join(files_list)}"
                        + (f" (+ {len(files) - 15} more)" if len(files) > 15 else "")
                    )
            
            if artifact_summary:
                context_parts.append(
                    "[ARTIFACTS]\n" + "\n".join(artifact_summary) + "\n"
                )
    
    # 6. Recent Tool History (last 10 successful tool calls)
    if tool_history := state.get("tool_history"):
        recent_tools = [
            t for t in tool_history[-10:] 
            if isinstance(t, dict) and t.get("successful", False)
        ]
        if recent_tools:
            tool_summary = []
            for tool_entry in recent_tools:
                tool_name = tool_entry.get("tool", "unknown")
                args = tool_entry.get("arguments", {})
                # Simplify args display
                args_str = str(args)[:80]
                tool_summary.append(f"  - {tool_name}({args_str})")
            
            if tool_summary:
                context_parts.append(
                    "[RECENT TOOL CALLS]\n" + "\n".join(tool_summary) + "\n"
                )
    
    # 7. Search Cache (to avoid redundant searches)
    if searches := state.get("searches"):
        if searches:
            search_queries = list(searches.keys())[:10]  # Show only recent 10 queries
            if search_queries:
                context_parts.append(
                    "[CACHED SEARCHES]\n"
                    + "The following searches have already been executed:\n"
                    + "\n".join(f"  - {q}" for q in search_queries)
                    + "\n"
                )
    
    # Combine all parts with clear separation
    if context_parts:
        return (
            "=== STATE CONTEXT ===\n\n"
            + "\n".join(context_parts)
            + "=== END STATE CONTEXT ===\n\n"
            "Use the above state context to inform your planning. "
            "Avoid redundant tool calls and prefer information already extracted.\n"
        )
    
    return ""


@lru_cache(maxsize=1)
def get_planner_agent():
    config = load_config()
    model = get_model()

    middleware = cast(
        list[AgentMiddleware[Any, None, Any]],
        [
            ModelRetryMiddleware(
                max_retries=2,
                backoff_factor=1.5,
                on_failure="continue",
            ),
            ToolRetryMiddleware(
                max_retries=2,
                backoff_factor=1.5,
                on_failure="continue",
            ),
            ModelCallLimitMiddleware(                                            
                run_limit=25,
                exit_behavior="error",
            ),
            ToolCallLimitMiddleware(
                run_limit=config.planner_tool_call_limit,
                exit_behavior="continue",
            ),
        ],
    )

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        response_format=ImplementationPlan,
        middleware=middleware,
        cache=planner_cache,
    )

    # Auto-install LSP servers for this workspace on first agent initialization.
    # Can be disabled by setting environment variable NO_AUTO_LSP_INSTALL=1
    try:
        from core.tools.lsp_installer import (
                detect_workspace_languages,
                build_plan,
                apply_plan,
            )
        workspace_root = str(Path.cwd())
        langs = detect_workspace_languages(workspace_root)
        logger = logging.getLogger(__name__)
        if langs:
            plan = build_plan(langs)
            # Only run installers for entries that look like real commands
            real_commands = {l: c for l, c in plan.items() if not str(c).startswith("No ")}
            if real_commands:
                logger.info("Auto-installing LSP servers for detected languages: %s", ", ".join(real_commands.keys()))
                try:
                    results = apply_plan(plan, auto_confirm=True)
                    logger.info("LSP install results: %s", results)
                except Exception as exc:
                    logger.exception("LSP auto-install failed: %s", exc)
        else:
            logging.getLogger(__name__).info("No supported languages detected for LSP installation.")
            
    except Exception:
        logging.getLogger(__name__).exception("Unexpected error during LSP auto-install step")

    return agent


def invoke_planner_with_state(state: dict[str, Any], **kwargs: Any) -> Any:
    """
    Invoke the planner agent with state-aware context injection.
    
    This wrapper injects conservative state context into the messages before
    invoking the agent, enabling state-based memory and avoiding redundant
    context gathering.
    
    Args:
        state: The LangGraph state containing workspace knowledge, facts, etc.
        **kwargs: Additional arguments to pass to the agent's invoke method
    
    Returns:
        The agent's response
    """
    agent = get_planner_agent()
    
    # Build state context injection
    state_context = build_state_context_injection(state)
    
    # Get existing messages from state
    messages = state.get("messages", [])
    
    # Inject state context as a system message at the beginning
    # This ensures the agent sees the state context before processing user messages
    enhanced_messages = []
    
    if state_context:
        enhanced_messages.append(
            SystemMessage(content=state_context)
        )
    
    # Add existing messages
    enhanced_messages.extend(messages)
    
    # Invoke agent with enhanced messages
    return agent.invoke({"messages": enhanced_messages}, **kwargs)


async def ainvoke_planner_with_state(state: dict[str, Any], **kwargs: Any) -> Any:
    """
    Async version of invoke_planner_with_state.
    
    Args:
        state: The LangGraph state containing workspace knowledge, facts, etc.
        **kwargs: Additional arguments to pass to the agent's ainvoke method
    
    Returns:
        The agent's response
    """
    agent = get_planner_agent()
    
    # Build state context injection
    state_context = build_state_context_injection(state)
    
    # Get existing messages from state
    messages = state.get("messages", [])
    
    # Inject state context as a system message at the beginning
    enhanced_messages = []
    
    if state_context:
        enhanced_messages.append(
            SystemMessage(content=state_context)
        )
    
    # Add existing messages
    enhanced_messages.extend(messages)
    
    # Invoke agent with enhanced messages
    return await agent.ainvoke({"messages": enhanced_messages}, **kwargs)


# Usage Example:
# ===============
#
# Traditional usage (without state):
#   agent = get_planner_agent()
#   response = agent.invoke({"messages": [HumanMessage(content="...")]})
#
# New state-aware usage:
#   from core.model.state import CodingAgentState
#   from langchain_core.messages import HumanMessage
#
#   # Initialize or load state
#   state: dict[str, Any] = {
#       "messages": [HumanMessage(content="Implement user authentication")],
#       "summary": "User requested authentication feature implementation",
#       "goal": "Add secure user authentication to the application",
#       "workspace": {
#           "/app/auth.py": {
#               "summary": "Contains basic user model",
#               "language": "python",
#               "classes": ["User"],
#               "functions": ["create_user"]
#           }
#       },
#       "known_facts": [
#           {
#               "fact": "Project uses FastAPI framework",
#               "source": "scan_project:/app",
#               "extracted_at": "2026-06-27T10:00:00Z"
#           }
#       ],
#       "artifacts": {
#           "/app/auth.py": "read",
#           "/app/models.py": "read"
#       },
#       "tool_history": [],
#       "searches": {},
#       "tasks": [],
#       "current_task": "",
#       "tool_cache": {},
#       "runtime": {"iterations": 0, "max_iterations": 50}
#   }
#
#   # Invoke with state context injection
#   response = await ainvoke_planner_with_state(state, config=runnable_config)
#
#   # The agent will receive state context automatically and avoid redundant tool calls