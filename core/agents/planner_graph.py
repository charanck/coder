"""
LangGraph implementation of the planner agent.

This module provides a minimal, extensible graph-based architecture for the planner,
replacing the previous agent-based implementation while maintaining full compatibility.
"""

from __future__ import annotations
from typing import Any, Literal, cast
from pathlib import Path
import logging
import os
import uuid

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from core.agents.nodes.summarizer import summarizer_node
from core.model.state import CodingAgentState
from core.model.planner import ImplementationPlan
from core.agents.nodes.prepare_planner_context import prepare_planner_context_node
from core.agents.nodes.llm_call import llm_call_node
from core.agents.nodes.tool import tool_node
from core.agents.nodes.util import tools_from_runnable_config
from core.common.tracing import langfuse_observe

logger = logging.getLogger(__name__)


PLANNER_SYSTEM_PROMPT = """
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

def should_continue(state: CodingAgentState) -> Literal["tools", "end"]:
    """
    Routing function to determine next step after LLM call.
    
    Returns:
        "tools" if the LLM made tool calls
        "end" if the LLM produced a final response
    """
    messages = state.get("messages", [])
    if not messages:
        logger.debug("No messages in state, routing to end")
        return "end"
    
    last_message = messages[-1]
    
    # Check if the last message has tool calls
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        logger.debug(f"Last message has {len(last_message.tool_calls)} tool calls, routing to tools")
        return "tools"
    
    logger.debug("Last message has no tool calls, routing to end")
    return "end"

@langfuse_observe
def create_planner_graph(
    checkpointer: Any | None = None
) -> CompiledStateGraph:
    """
    Create the planner agent graph.
    
    Args:
        checkpointer: Optional checkpointer for persistence (defaults to MemorySaver)
    
    Returns:
        Compiled StateGraph ready for invocation
    """
    logger.info("Creating planner graph")
    
    # Create the graph
    graph = StateGraph(CodingAgentState)
    
    graph.add_node("prepare_planner_context", prepare_planner_context_node)
    graph.add_node("llm", llm_call_node)
    graph.add_node("tools", tool_node)
    graph.add_node("summarizer", summarizer_node)
    
    # Add edges
    graph.add_edge(START, "prepare_planner_context")
    graph.add_edge("prepare_planner_context", "llm")
    graph.add_conditional_edges(
        "llm",
        should_continue,
        {
            "tools": "tools",
            "end": END
        }
    )
    graph.add_edge("tools", "summarizer")
    graph.add_edge("summarizer", "prepare_planner_context")
    
    # Compile with checkpointer
    if checkpointer is None:
        checkpointer = MemorySaver()
    
    logger.debug("Planner graph compiled successfully")
    return graph.compile(checkpointer=checkpointer)



@langfuse_observe
def invoke_planner(
    state: CodingAgentState,
    config: RunnableConfig | None = None,
    checkpointer: Any | None = None,
    langfuse_handler: Any | None = None,
    tools: list[Any] | None = None
) -> dict[str, Any]:
    """
    Invoke the planner graph with the given state.
    
    Args:
        state: The current coding agent state
        config: Optional runnable configuration
        checkpointer: Optional checkpointer for persistence
        langfuse_handler: Optional Langfuse callback handler for tracing
        tools: Optional list of tools to make available to the LLM
    
    Returns:
        Updated state after graph execution
    """
    logger.info(f"Invoking planner with {len(tools) if tools else 0} tools")
    
    graph = create_planner_graph(checkpointer=checkpointer)
    
    # Build mutable config dict
    mutable_config: dict[str, Any] = {}
    if config:
        mutable_config.update(config)
    
    if "configurable" not in mutable_config:
        mutable_config["configurable"] = {}
    if "thread_id" not in mutable_config["configurable"]:
        thread_id = uuid.uuid4().hex
        mutable_config["configurable"]["thread_id"] = thread_id
        logger.debug(f"Created new thread_id: {thread_id}")
    
    # Add tools to config if provided
    if tools:
        mutable_config["configurable"]["tools"] = tools
        logger.debug(f"Added {len(tools)} tools to config")
    
    if langfuse_handler is not None:
        mutable_config["callbacks"] = [langfuse_handler]
        logger.debug("Added Langfuse callback handler to config")
    
    logger.debug("Invoking planner graph")
    result = graph.invoke(state, config=cast(RunnableConfig, mutable_config))
    logger.info("Planner graph invocation completed")
    
    return result

@langfuse_observe
async def ainvoke_planner(
    state: CodingAgentState,
    config: RunnableConfig | None = None,
    checkpointer: Any | None = None,
    langfuse_handler: Any | None = None,
    tools: list[Any] | None = None
) -> dict[str, Any]:
    """
    Async invoke the planner graph with the given state.
    
    Args:
        state: The current coding agent state
        config: Optional runnable configuration
        checkpointer: Optional checkpointer for persistence
        langfuse_handler: Optional Langfuse callback handler for tracing
        tools: Optional list of tools to make available to the LLM
    
    Returns:
        Updated state after graph execution
    """
    logger.info(f"Async invoking planner with {len(tools) if tools else 0} tools")
    
    graph = create_planner_graph(checkpointer=checkpointer)
    
    # Build mutable config dict
    mutable_config: dict[str, Any] = {}
    if config:
        mutable_config.update(config)
    
    if "configurable" not in mutable_config:
        mutable_config["configurable"] = {}
    if "thread_id" not in mutable_config["configurable"]:
        thread_id = uuid.uuid4().hex
        mutable_config["configurable"]["thread_id"] = thread_id
        logger.debug(f"Created new thread_id: {thread_id}")
    
    # Add tools to config if provided
    if tools:
        mutable_config["configurable"]["tools"] = tools
        logger.debug(f"Added {len(tools)} tools to config")
    
    if langfuse_handler is not None:
        mutable_config["callbacks"] = [langfuse_handler]
        logger.debug("Added Langfuse callback handler to config")
    
    logger.debug("Async invoking planner graph")
    result = await graph.ainvoke(state, config=cast(RunnableConfig, mutable_config))
    logger.info("Async planner graph invocation completed")
    
    return result
