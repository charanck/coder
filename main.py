import asyncio
from typing import cast

from config import load_config
from core.agents.planner import get_planner_agent
from core.common.tracing import flush_langfuse_traces, get_langfuse_callback_handler
from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphRecursionError
from langchain.messages import HumanMessage



def main():
    config = load_config()
    langfuse_handler = get_langfuse_callback_handler()

    invoke_config = cast(RunnableConfig, {
        "recursion_limit": max(3, (config.planner_tool_call_limit * 2) + 1),
    })
    if langfuse_handler is not None:
        invoke_config["callbacks"] = [langfuse_handler]

    try:
        response = asyncio.run(
            asyncio.wait_for(
                get_planner_agent().ainvoke(
                    {
                        "messages": [
                            HumanMessage(
                                content=(
                                    "go through the codebase and give plan for implementing codebase indexing functionality."
                                )
                            )
                        ]
                    },
                    config=invoke_config,
                ),
                timeout=config.planner_agent_timeout,
            )
        )
    except TimeoutError as exc:
        raise RuntimeError(
            f"Planner agent exceeded the total timeout of {config.planner_agent_timeout} seconds."
        ) from exc
    except GraphRecursionError as exc:
        raise RuntimeError(
            "Planner agent exceeded the configured step limit. "
            f"Reduce tool usage or increase PLANNER_TOOL_CALL_LIMIT from {config.planner_tool_call_limit}."
        ) from exc
    finally:
        flush_langfuse_traces()

    print(response)


if __name__ == "__main__":
    main()


