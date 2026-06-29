from __future__ import annotations
import logging
from typing import Any
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph.message import REMOVE_ALL_MESSAGES, RemoveMessage
from core.model.state import CodingAgentState
from core.common.model import get_model


logger = logging.getLogger(__name__)


KEEP_RECENT_MESSAGES = 10

SUMMARY_TRIGGER_MESSAGES = 30

SUMMARY_TRIGGER_AI_TURNS = 12

SUMMARY_TRIGGER_ESTIMATED_TOKENS = 6000

SUMMARY_TRIGGER_SINGLE_MESSAGE = 2500


def _select_recent_messages(messages: list[Any]) -> list[Any]:
    start_index = max(0, len(messages) - KEEP_RECENT_MESSAGES)

    # A retained suffix cannot start with a tool response because the matching
    # assistant tool-call would be missing from history.
    while start_index > 0 and isinstance(messages[start_index], ToolMessage):
        start_index -= 1

    return messages[start_index:]


def _stringify_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                text = item.strip()
            elif isinstance(item, dict):
                text = str(item)
            else:
                text = str(item).strip()

            if text:
                chunks.append(text)

        return "\n".join(chunks).strip()

    return str(content).strip()


def _format_tool_calls(message: AIMessage) -> str:
    tool_calls = getattr(message, "tool_calls", None) or []
    if not tool_calls:
        return ""

    formatted_calls = []
    for tool_call in tool_calls:
        name = tool_call.get("name", "unknown")
        args = tool_call.get("args", {})
        formatted_calls.append(f"{name}({args})")

    return "; ".join(formatted_calls)


def _normalize_summary_output(response: Any) -> str:
    raw_content = response.content if hasattr(response, "content") else response
    summary = _stringify_content(raw_content)
    if not summary:
        return ""

    words = summary.split()
    if len(words) > 500:
        summary = " ".join(words[:500])

    return summary.strip()

def estimate_tokens(text: str) -> int:
    """Fast token estimate (~4 chars/token)."""
    return max(1, len(text) // 4)

def message_to_text(message) -> str:
    """Convert a LangChain message into compact text."""

    if isinstance(message, HumanMessage):
        role = "User"

    elif isinstance(message, AIMessage):
        role = "Assistant"

    elif isinstance(message, ToolMessage):
        role = f"Tool({message.name})"

    elif isinstance(message, SystemMessage):
        role = "System"

    else:
        role = type(message).__name__

    content = _stringify_content(message.content)
    details: list[str] = []

    if content:
        details.append(content)

    if isinstance(message, AIMessage):
        tool_calls = _format_tool_calls(message)
        if tool_calls:
            details.append(f"Tool calls: {tool_calls}")

    body = "\n".join(details).strip() or "[no textual content]"
    return f"{role}: {body}"

def should_summarize(messages: list) -> bool:
    """Determine if summarization should run."""

    if len(messages) <= KEEP_RECENT_MESSAGES:
        return False

    if len(messages) >= SUMMARY_TRIGGER_MESSAGES:
        return True

    ai_turns = sum(isinstance(m, AIMessage) for m in messages)
    if ai_turns >= SUMMARY_TRIGGER_AI_TURNS:
        return True

    total_tokens = 0

    for msg in messages:
        text = message_to_text(msg)

        tokens = estimate_tokens(text)

        total_tokens += tokens

        if tokens >= SUMMARY_TRIGGER_SINGLE_MESSAGE:
            return True

    return total_tokens >= SUMMARY_TRIGGER_ESTIMATED_TOKENS


def summarizer_node(
    state: CodingAgentState,
    config: RunnableConfig,
) -> dict[str, Any]:

    logger.info("Running rolling conversation summarizer")

    messages = state.get("messages", [])

    if not should_summarize(messages):
        logger.debug("Skipping summarization.")
        return {}

    previous_summary = state.get("summary", "")

    recent_messages = _select_recent_messages(messages)
    messages_to_summarize = messages[: len(messages) - len(recent_messages)]

    conversation = "\n\n".join(
        message_to_text(msg)
        for msg in messages_to_summarize
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You maintain the long-term memory of a software engineering agent.

                Update the existing summary using the newly provided conversation.

                Preserve:

                - user goals
                - architecture decisions
                - implementation decisions
                - discovered project structure
                - completed work
                - pending work
                - constraints
                - assumptions

                Remove:

                - greetings
                - acknowledgements
                - duplicated information
                - temporary reasoning
                - repeated tool output
                - verbose code snippets

                Write a concise summary under 500 words.

                Return plain text only.
                """,
            ),
            (
                "human",
                """
                Previous Summary:

                {previous_summary}

                Conversation To Compress:

                {conversation}
                """,
                    ),
        ]
    )

    model = get_model()

    chain = prompt | model

    try:
        response = chain.invoke(
            {
                "previous_summary": previous_summary,
                "conversation": conversation,
            }
        )
    except Exception:
        logger.exception("Summarizer model invocation failed; preserving full message history.")
        return {}

    summary = _normalize_summary_output(response)
    if not summary:
        logger.warning("Summarizer returned empty content; preserving full message history.")
        return {}

    logger.info(
        "Conversation summarized. "
        "Compressed %d messages -> summary.",
        len(messages_to_summarize),
    )

    return {
        "summary": summary,
        "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *recent_messages],
    }