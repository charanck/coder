from core.agents.planner import planner_agent
from core.common.tracing import flush_langfuse_traces, get_langfuse_callback_handler
from langchain.messages import HumanMessage
import logging

# logging.basicConfig(level=logging.DEBUG)
# logging.getLogger("langchain").setLevel(logging.DEBUG)
# logging.getLogger("langgraph").setLevel(logging.DEBUG)
# logging.getLogger("httpx").setLevel(logging.DEBUG)

def main():
    langfuse_handler = get_langfuse_callback_handler()

    invoke_kwargs = {}
    if langfuse_handler is not None:
        invoke_kwargs["config"] = {"callbacks": [langfuse_handler]}

    try:
        response = planner_agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            "Update the system prompt for the planner agent to limit the total number of tool calls from 8 to 5"
                        )
                    )
                ]
            },
            **invoke_kwargs,
        )
    finally:
        flush_langfuse_traces()

    print(response)


if __name__ == "__main__":
    main()


