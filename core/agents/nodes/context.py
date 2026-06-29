"""
Context management for the planner agent.

This module provides utilities to format and manage context from state,
inserting it into the prompt in a way that avoids duplicate system messages
and properly integrates with LangChain's message history.
"""

from __future__ import annotations
from typing import Any
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage


SUMMARY_ANCHOR_PREFIX = "[PLANNER SESSION SUMMARY]"


class ContextFormatter:
    """Format context from state into structured prompt injections."""
    
    @staticmethod
    def build_state_context(state: dict[str, Any]) -> str:
        """
        Build conservative context injection from state for the planner.
        
        Extracts and formats only essential information from state,
        implementing strict history summarization to minimize token usage.
        """
        context_parts = []
        
        # 1. Conversation Summary (PRIMARY CONTEXT)
        if summary := state.get("summary"):
            context_parts.append(f"[CONVERSATION SUMMARY]\n{summary}\n")
        
        # 2. Current Goal
        if goal := state.get("goal"):
            context_parts.append(f"[CURRENT GOAL]\n{goal}\n")
        
        # 3. Known Facts
        if known_facts := state.get("known_facts"):
            if known_facts:
                facts_summary = []
                for fact_entry in known_facts[-20:]:
                    fact_text = fact_entry.get("fact", "")
                    source = fact_entry.get("source", "unknown")
                    facts_summary.append(f"  - {fact_text} (source: {source})")
                
                if facts_summary:
                    context_parts.append(
                        "[KNOWN FACTS]\n" + "\n".join(facts_summary) + "\n"
                    )
        
        # 4. Workspace Summary
        if workspace := state.get("workspace"):
            if workspace:
                workspace_items = []
                for file_path, metadata in list(workspace.items())[:30]:
                    if isinstance(metadata, dict) and metadata.get("summary"):
                        summary_text = metadata["summary"][:150]
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
        
        # 5. Artifacts
        if artifacts := state.get("artifacts"):
            if artifacts:
                artifact_groups = {"read": [], "created": [], "modified": [], "deleted": []}
                for file_path, status in artifacts.items():
                    if status in artifact_groups:
                        artifact_groups[status].append(file_path)
                
                artifact_summary = []
                for status, files in artifact_groups.items():
                    if files:
                        files_list = files[:15]
                        artifact_summary.append(
                            f"  {status}: {', '.join(files_list)}"
                            + (f" (+ {len(files) - 15} more)" if len(files) > 15 else "")
                        )
                
                if artifact_summary:
                    context_parts.append(
                        "[ARTIFACTS]\n" + "\n".join(artifact_summary) + "\n"
                    )
        
        # 6. Recent Tool History
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
                    args_str = str(args)[:80]
                    tool_summary.append(f"  - {tool_name}({args_str})")
                
                if tool_summary:
                    context_parts.append(
                        "[RECENT TOOL CALLS]\n" + "\n".join(tool_summary) + "\n"
                    )
        
        # 7. Search Cache
        if searches := state.get("searches"):
            if searches:
                search_queries = list(searches.keys())[:10]
                if search_queries:
                    context_parts.append(
                        "[CACHED SEARCHES]\n"
                        + "The following searches have already been executed:\n"
                        + "\n".join(f"  - {q}" for q in search_queries)
                        + "\n"
                    )
        
        if context_parts:
            return (
                "=== STATE CONTEXT ===\n\n"
                + "\n".join(context_parts)
                + "=== END STATE CONTEXT ===\n\n"
                "Use the above state context to inform your planning. "
                "Avoid redundant tool calls and prefer information already extracted.\n"
            )
        
        return ""
    

class MessageHistoryManager:
    """Build the message list for an LLM invocation."""

    @staticmethod
    def _sanitize_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
        """
        Sanitize messages to ensure no None or empty content exists.
        """
        sanitized = []
        for msg in messages:
            if isinstance(msg, ToolMessage):
                # Ensure ToolMessage has valid string content
                content = msg.content
                if not isinstance(content, str) or not content.strip():
                    # Replace with a safe placeholder
                    content = f"Tool {msg.name} executed with no output available."
                # Create a new ToolMessage with sanitized content
                sanitized.append(ToolMessage(
                    content=content,
                    tool_call_id=msg.tool_call_id,
                    name=msg.name
                ))
            elif isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                sanitized.append(msg)
            elif msg.content is None or (isinstance(msg.content, str) and not msg.content.strip()):
                # Skip messages with None or empty content
                continue
            else:
                sanitized.append(msg)
        return sanitized

    @staticmethod
    def build(
        messages: list[BaseMessage],
        system_prompt: str,
        state_context: str | None = None,
        summary: str | None = None,
    ) -> list[BaseMessage]:
        history = [
            msg
            for msg in messages
            if not isinstance(msg, SystemMessage)
        ]

        result: list[BaseMessage] = [
            SystemMessage(content=system_prompt or "")
        ]

        if state_context:
            result.append(SystemMessage(content=state_context))

        # Gemini function-calling requires an assistant tool-call turn to follow
        # a user or tool-response turn. After summarization, retained history may
        # begin with an assistant tool-call, so anchor it with the session summary.
        if (
            summary
            and history
            and isinstance(history[0], AIMessage)
            and bool(history[0].tool_calls)
        ):
            result.append(HumanMessage(content=f"{SUMMARY_ANCHOR_PREFIX}\n{summary.strip()}"))

        result.extend(history)

        return MessageHistoryManager._sanitize_messages(result)