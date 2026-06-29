from __future__ import annotations

from typing import Any, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.graph.message import add_messages

from core.agents.nodes import summarizer
from core.model.state import CodingAgentState


def _build_messages(count: int) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    for index in range(count):
        if index % 2 == 0:
            messages.append(HumanMessage(content=f"user-{index}"))
        else:
            messages.append(AIMessage(content=f"assistant-{index}"))
    return messages


def _build_state(messages: list[BaseMessage], summary: str = "") -> CodingAgentState:
    return {
        "project_root": ".",
        "messages": messages,
        "summary": summary,
        "goal": "",
        "current_task": "",
        "tasks": [],
        "workspace": {},
        "known_facts": [],
        "artifacts": {},
        "searches": {},
        "tool_history": [],
        "tool_cache": {},
        "runtime": {},
        "system_prompt": "",
        "state_context": "",
    }


def test_message_to_text_includes_ai_tool_calls():
    message = AIMessage(
        content="",
        tool_calls=[
            {
                "id": "call_1",
                "name": "read_file",
                "args": {"file_path": "core/agents/nodes/summarizer.py"},
            }
        ],
    )

    text = summarizer.message_to_text(message)

    assert "Assistant:" in text
    assert "Tool calls:" in text
    assert "read_file" in text


def test_should_summarize_on_large_message_count():
    messages = _build_messages(summarizer.SUMMARY_TRIGGER_MESSAGES)

    assert summarizer.should_summarize(messages) is True


def test_summarizer_replaces_old_history_with_recent_messages(monkeypatch):
    messages = _build_messages(summarizer.SUMMARY_TRIGGER_MESSAGES + 2)
    monkeypatch.setattr(
        summarizer,
        "get_model",
        lambda: RunnableLambda(lambda _: AIMessage(content="compressed summary")),
    )

    result = summarizer.summarizer_node(_build_state(messages), config={})
    updated_messages = cast(
        list[BaseMessage],
        add_messages(cast(Any, messages), cast(Any, result["messages"])),
    )

    assert result["summary"] == "compressed summary"
    assert len(updated_messages) == summarizer.KEEP_RECENT_MESSAGES
    assert [message.content for message in updated_messages] == [
        message.content for message in messages[-summarizer.KEEP_RECENT_MESSAGES :]
    ]


def test_select_recent_messages_rolls_back_when_cut_would_start_on_tool():
    messages: list[BaseMessage] = [HumanMessage(content="request"), HumanMessage(content="follow-up")]
    for index in range(5):
        messages.append(
            AIMessage(
                content="",
                tool_calls=[
                    {"id": f"call_{index}", "name": "scan_project", "args": {"root": "."}},
                ],
            )
        )
        messages.append(
            ToolMessage(content=f"result-{index}", tool_call_id=f"call_{index}", name="scan_project")
        )

    retained = summarizer._select_recent_messages(messages)

    assert isinstance(retained[0], AIMessage)
    assert retained[0].tool_calls


def test_summarizer_preserves_history_when_summary_is_empty(monkeypatch):
    messages = _build_messages(summarizer.SUMMARY_TRIGGER_MESSAGES + 1)
    monkeypatch.setattr(
        summarizer,
        "get_model",
        lambda: RunnableLambda(lambda _: AIMessage(content="   ")),
    )

    result = summarizer.summarizer_node(_build_state(messages, summary="existing summary"), config={})

    assert result == {}


def test_message_to_text_formats_tool_messages():
    message = ToolMessage(content="tool output", tool_call_id="call_1", name="grep")

    text = summarizer.message_to_text(message)

    assert text == "Tool(grep): tool output"