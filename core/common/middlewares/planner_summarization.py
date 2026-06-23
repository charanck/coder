import logging
import threading
from typing import Any

from langchain.chat_models import BaseChatModel
from langchain.agents.middleware import AgentState, Runtime, SummarizationMiddleware
from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, ToolMessage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

PLANNER_SUMMARY_PROMPT = """\
<role>
Planner Agent Context Extractor
</role>

<primary_objective>
Extract and compress the planning session history below into a structured summary.
This summary REPLACES the conversation history — it must be complete enough for the
agent to continue planning without re-reading files it has already examined.
</primary_objective>

<instructions>
Populate every section below. Write "None" if a section has nothing to report.

## SESSION INTENT
What is the user's overall goal? What codebase or feature is being planned?

## FILES EXAMINED
List every file path the agent read, with a one-line purpose for each.
Format: `path/to/file.py` — <purpose>
Never omit a file; re-reading wastes tokens.

## KEY FINDINGS
Architectural patterns, module boundaries, critical dependencies, design constraints
discovered so far. Include class names, function names, and data flow where relevant.

## DECISIONS MADE
Any design or implementation decisions the agent has explicitly committed to.
Include the reasoning. These must NOT be revisited unless the user says so.

## OPEN QUESTIONS
Anything flagged as uncertain or requiring further investigation.

## NEXT STEPS
Concrete remaining tasks to reach the final ImplementationPlan output.
</instructions>

Compress all internal monologue and reasoning.
Never compress file paths, class/function names, concrete findings, or decisions.

<messages>
{messages}
</messages>
"""

# ---------------------------------------------------------------------------
# Phase detection
# ---------------------------------------------------------------------------

_PHASE_SIGNALS: dict[str, list[str]] = {
    "exploration_complete": [
        "finished exploring",
        "completed exploration",
        "i have a clear picture of the codebase",
        "ready to move on to analysis",
        "exploration complete",
        "done reading the relevant files",
    ],
    "analysis_complete": [
        "analysis complete",
        "ready to generate the plan",
        "ready to produce the implementation plan",
        "i now have all the information needed",
        "proceeding to plan generation",
        "i have enough context to",
    ],
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SUMMARY_PREFIX = (
    "[PLANNER SESSION SUMMARY — treat as ground truth, "
    "do not re-examine files listed here]\n\n"
)


def _safe_split(
    messages: list[AnyMessage],
    keep_last_n: int,
) -> tuple[list[AnyMessage], list[AnyMessage]]:
    """
    Split *messages* into (to_summarize, to_keep) such that:

    1. The last *keep_last_n* messages are kept verbatim.
    2. The split point is walked **backwards** until it does not land
       inside an AI+Tool pair — i.e. the first message in `to_keep`
       is never a ToolMessage whose paired AIMessage is in `to_summarize`.

    This guarantees the kept tail always starts at a clean boundary,
    so providers that enforce strict turn alternation (Gemini, etc.)
    never receive an orphaned ToolMessage at the head of the history.
    """
    if keep_last_n <= 0 or keep_last_n >= len(messages):
        return messages, []

    split = len(messages) - keep_last_n

    # Walk split forward until it does not land inside a pair.
    # A "dangerous" position is one where messages[split] is a ToolMessage
    # whose AIMessage partner sits before the split.
    while split < len(messages) and isinstance(messages[split], ToolMessage):
        split += 1

    # Edge case: if we walked past the end everything must be summarised.
    if split >= len(messages):
        return messages, []

    return messages[:split], messages[split:]


def _last_ai_text(messages: list[AnyMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return msg.content if isinstance(msg.content, str) else ""
    return ""


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

class PlannerSummarizationMiddleware(SummarizationMiddleware):
    """
    Summarization middleware for planner agents.

    Differences from the base class
    --------------------------------
    - Uses a structured planner prompt (files / findings / decisions / next steps).
    - Injects the summary as a SystemMessage so the model treats it as
      authoritative ground truth rather than chat content.
    - Accumulates summaries across compression cycles so no findings are lost.
    - Detects phase-boundary signals in after_model() and forces immediate
      summarization, independent of the token / message threshold.
    - Splits the message list at a *pair-safe* boundary so a ToolMessage is
      never left without its paired AIMessage in the kept tail, which would
      violate Gemini's strict turn-alternation requirement.
    """

    def __init__(
        self,
        model: str | BaseChatModel,
        *,
        trigger: Any = ("tokens", 24_000),
        keep: Any = ("messages", 8),
        summary_prompt: str = PLANNER_SUMMARY_PROMPT,
        enable_phase_detection: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model,
            trigger=trigger,
            keep=keep,
            summary_prompt=summary_prompt,
            **kwargs,
        )
        self._enable_phase_detection = enable_phase_detection
        self._accumulated_summary: str | None = None
        self._force_summarize: bool = False
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # AgentMiddleware hooks
    # ------------------------------------------------------------------

    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        if not self._enable_phase_detection:
            return None

        text = _last_ai_text(state.get("messages", [])).lower() 
        if not text:
            return None

        for phase, signals in _PHASE_SIGNALS.items():
            if any(sig in text for sig in signals):
                logger.info(
                    "[PlannerSummarizer] Phase boundary detected: %r — "
                    "forcing summarization before next model call.",
                    phase,
                )
                with self._lock:
                    self._force_summarize = True
                break

        return None

    async def aafter_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        return self.after_model(state, runtime)

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def _should_summarize(self, messages: list[AnyMessage], total_tokens: int) -> bool:
        with self._lock:
            forced, self._force_summarize = self._force_summarize, False

        if forced:
            logger.info("[PlannerSummarizer] Force-summarize triggered (phase boundary).")
            return True

        return super()._should_summarize(messages, total_tokens)

    def _split_messages(
        self, messages: list[AnyMessage]
    ) -> tuple[list[AnyMessage], list[AnyMessage]]:
        """
        Override the base-class split with a pair-safe variant.

        The base class typically keeps the last N messages by a simple index
        slice, which can land mid-pair.  We instead delegate to _safe_split()
        which walks the boundary forward until it clears any open AI+Tool pair.
        """
        keep_n = self._resolve_keep_n(messages)
        return _safe_split(messages, keep_n)

    def _create_summary(self, messages_to_summarize: list[AnyMessage]) -> str:
        new_summary = super()._create_summary(messages_to_summarize)
        return self._fold_summary(new_summary)

    async def _acreate_summary(self, messages_to_summarize: list[AnyMessage]) -> str:
        new_summary = await super()._acreate_summary(messages_to_summarize)
        return self._fold_summary(new_summary)

    @staticmethod
    def _build_new_messages(summary: str) -> list[SystemMessage]:  # type: ignore[override]
        """
        Emit the summary as a SystemMessage so the model treats it as
        authoritative ground truth, not as user-provided chat content.
        """
        return [
            SystemMessage(
                content=_SUMMARY_PREFIX + summary,
                additional_kwargs={"lc_source": "planner_summarization"},
            )
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fold_summary(self, new_summary: str) -> str:
        """
        Fold *new_summary* into the running accumulated summary.

        Each compression cycle prepends the previous accumulated summary so
        findings discovered in earlier cycles are never discarded.
        """
        if self._accumulated_summary:
            folded = (
                f"[PREVIOUS SESSION SUMMARY]\n{self._accumulated_summary}\n\n"
                f"[NEW FINDINGS]\n{new_summary}"
            )
            self._accumulated_summary = folded
            logger.debug("[PlannerSummarizer] Folded new summary into accumulated history.")
            return folded

        self._accumulated_summary = new_summary
        return new_summary

    def _resolve_keep_n(self, messages: list[AnyMessage]) -> int:
        """
        Derive a concrete integer from self._keep, which may be expressed as
        ("messages", N) or a plain int, depending on the base class version.
        """
        keep = getattr(self, "_keep", None) or getattr(self, "keep", None)
        if isinstance(keep, tuple) and keep[0] == "messages":
            return int(keep[1])
        if isinstance(keep, int):
            return keep
        # Fallback: keep the last 8 messages.
        return 8

    # ------------------------------------------------------------------
    # Debug / observability
    # ------------------------------------------------------------------

    @property
    def accumulated_summary(self) -> str | None:
        """Current accumulated summary (useful for debugging / logging)."""
        return self._accumulated_summary