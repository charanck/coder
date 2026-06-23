from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState, Runtime
from langchain_core.messages import AIMessage, AnyMessage, RemoveMessage, ToolMessage

logger = logging.getLogger(__name__)

_REMOVE_ALL: str = "__remove_all__"

_STUB_TEMPLATE = "[{tool_name} result compressed — captured in session summary]"


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------

@dataclass
class _ToolPair:
    """One AIMessage and all ToolMessages that must immediately follow it."""
    ai_message: AIMessage
    tool_messages: list[ToolMessage] = field(default_factory=list)

    @property
    def expected_ids(self) -> set[str]:
        return {str(tc["id"]) for tc in self.ai_message.tool_calls if tc.get("id")}

    @property
    def found_ids(self) -> set[str]:
        return {str(tm.tool_call_id) for tm in self.tool_messages if tm.tool_call_id}

    @property
    def orphaned_ids(self) -> set[str]:
        return self.expected_ids - self.found_ids

    def tool_name_for(self, call_id: str) -> str:
        return next(
            (tc["name"] for tc in self.ai_message.tool_calls if tc.get("id") == call_id),
            "tool",
        )


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

class ToolResultCompressionMiddleware(AgentMiddleware):
    """
    Compresses stale ToolMessage bodies before every model call and
    guarantees the message sequence is structurally valid for strict
    providers such as Gemini (every AIMessage with tool_calls must be
    immediately followed by a ToolMessage for each call_id).

    Parameters
    ----------
    keep_last_n:
        Number of most-recent ToolMessages to leave untouched.
    min_chars_to_compress:
        Minimum content length (chars) before a stale result is stubbed.
        Short results (e.g. small grep hits) are kept verbatim.
    """

    def __init__(
        self,
        *,
        keep_last_n: int = 3,
        min_chars_to_compress: int = 500,
    ) -> None:
        super().__init__()
        self.keep_last_n = keep_last_n
        self.min_chars_to_compress = min_chars_to_compress

    # ------------------------------------------------------------------
    # AgentMiddleware interface
    # ------------------------------------------------------------------

    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        original: list[AnyMessage] = list(state["messages"])
        result, n_stubbed, n_injected = self._process(original)

        if n_stubbed == 0 and n_injected == 0 and result == original:
            return None

        logger.debug(
            "[ToolResultCompressor] stubbed=%d injected=%d "
            "(keep_last_n=%d min_chars=%d)",
            n_stubbed, n_injected, self.keep_last_n, self.min_chars_to_compress,
        )

        return {"messages": [RemoveMessage(id=_REMOVE_ALL), *result]}

    async def abefore_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        return self.before_model(state, runtime)

    # ------------------------------------------------------------------
    # Core pipeline  (single pass)
    # ------------------------------------------------------------------

    def _process(
        self, messages: list[AnyMessage]
    ) -> tuple[list[AnyMessage], int, int]:
        """
        Single-pass pipeline:
          1. Parse messages into segments (paired AI+Tools and lone messages).
          2. Determine which ToolMessages are stale.
          3. Compress stale large results.
          4. Inject synthetic ToolMessages for any orphaned tool_call_ids.
          5. Assign ids to any message missing one.

        Returns (processed_messages, n_stubbed, n_injected).
        """
        tool_name_lookup = _build_tool_name_lookup(messages)
        stale_indices = self._stale_indices(messages)

        result: list[AnyMessage] = []
        n_stubbed = 0
        n_injected = 0
        i = 0

        while i < len(messages):
            msg = messages[i]

            if isinstance(msg, AIMessage) and msg.tool_calls:
                # consume this AI message + all immediately-following ToolMessages
                # as a single atomic pair block
                pair, i = self._consume_pair(messages, i)

                result.append(pair.ai_message)

                for tm in pair.tool_messages:
                    j = messages.index(tm)  # original index for staleness check
                    if j in stale_indices and len(str(tm.content)) >= self.min_chars_to_compress:
                        result.append(self._stub(tm, tool_name_lookup))
                        n_stubbed += 1
                    else:
                        result.append(tm)

                # inject synthetic responses for orphaned call ids
                for call_id in pair.orphaned_ids:
                    tool_name = pair.tool_name_for(call_id)
                    result.append(self._synthetic_stub(call_id, tool_name))
                    n_injected += 1
                    logger.debug(
                        "[ToolResultCompressor] Injected synthetic ToolMessage "
                        "for orphaned tool_call_id=%s (%s)",
                        call_id, tool_name,
                    )

            else:
                result.append(msg)
                i += 1

        _ensure_ids(result)
        return result, n_stubbed, n_injected

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _stale_indices(self, messages: list[AnyMessage]) -> set[int]:
        """Indices of ToolMessages old enough to be candidates for compression."""
        tool_indices = [i for i, m in enumerate(messages) if isinstance(m, ToolMessage)]
        cutoff = max(0, len(tool_indices) - self.keep_last_n)
        return set(tool_indices[:cutoff])

    @staticmethod
    def _consume_pair(
        messages: list[AnyMessage], ai_index: int
    ) -> tuple[_ToolPair, int]:
        """
        Starting at ai_index (an AIMessage with tool_calls), consume all
        immediately-following ToolMessages and return (_ToolPair, next_index).
        """
        ai_msg = messages[ai_index]
        assert isinstance(ai_msg, AIMessage)
        pair = _ToolPair(ai_message=ai_msg)

        j = ai_index + 1
        while j < len(messages) and isinstance(messages[j], ToolMessage):
            pair.tool_messages.append(messages[j]) # type: ignore[arg-type]
            j += 1

        return pair, j

    @staticmethod
    def _stub(tm: ToolMessage, tool_name_lookup: dict[str, str]) -> ToolMessage:
        tool_name = tool_name_lookup.get(tm.tool_call_id or "", "tool")
        return ToolMessage(
            content=_STUB_TEMPLATE.format(tool_name=tool_name),
            tool_call_id=tm.tool_call_id,
            id=tm.id,
        )

    @staticmethod
    def _synthetic_stub(call_id: str, tool_name: str) -> ToolMessage:
        return ToolMessage(
            content=_STUB_TEMPLATE.format(tool_name=tool_name),
            tool_call_id=call_id,
            id=str(uuid.uuid4()),
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _build_tool_name_lookup(messages: list[AnyMessage]) -> dict[str, str]:
    """Map tool_call_id → tool_name from all AIMessages in the history."""
    lookup: dict[str, str] = {}
    for msg in messages:
        if isinstance(msg, AIMessage):
            for tc in msg.tool_calls:
                if tc.get("id"):
                    lookup[tc["id"]] = tc.get("name", "tool")  # type: ignore[arg-type]
    return lookup


def _ensure_ids(messages: list[AnyMessage]) -> None:
    """Assign a uuid to any message that is missing an id (mutates in place)."""
    for msg in messages:
        if msg.id is None:
            msg.id = str(uuid.uuid4())