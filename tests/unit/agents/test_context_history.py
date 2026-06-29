from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from core.agents.nodes.context import MessageHistoryManager, SUMMARY_ANCHOR_PREFIX


def test_build_injects_summary_anchor_before_leading_tool_call():
    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "call_1",
                    "name": "scan_project",
                    "args": {"root": "."},
                }
            ],
        ),
        ToolMessage(content="scan result", tool_call_id="call_1", name="scan_project"),
    ]

    built = MessageHistoryManager.build(
        messages=messages,
        system_prompt="system",
        state_context="context",
        summary="project summary",
    )

    assert isinstance(built[0], SystemMessage)
    assert isinstance(built[1], SystemMessage)
    assert isinstance(built[2], HumanMessage)
    assert built[2].content == f"{SUMMARY_ANCHOR_PREFIX}\nproject summary"
    assert isinstance(built[3], AIMessage)
    assert isinstance(built[4], ToolMessage)