import asyncio
import logging
from typing import cast

from config import load_config
from core.agents.planner import get_planner_agent
from core.agents.planner_graph import invoke_planner
from core.common.tracing import flush_langfuse_traces, get_langfuse_callback_handler
from core.tools.tools import ALL_TOOLS, READ_ONLY_TOOLS
from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphRecursionError
from langchain.messages import HumanMessage

from core.model.state import CodingAgentState

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main():
    logger.info("Starting main execution")
    config = load_config()
    langfuse_handler = get_langfuse_callback_handler()

    invoke_config = cast(RunnableConfig, {
        "recursion_limit": max(3, (config.planner_tool_call_limit * 2) + 1),
    })
    if langfuse_handler is not None:
        invoke_config["callbacks"] = [langfuse_handler]

    try:
        logger.info("Invoking example_basic_usage")
        result = example_basic_usage()
    except TimeoutError as exc:
        logger.error(f"Planner agent exceeded timeout of {config.planner_agent_timeout} seconds")
        raise RuntimeError(
            f"Planner agent exceeded the total timeout of {config.planner_agent_timeout} seconds."
        ) from exc
    except GraphRecursionError as exc:
        logger.error(f"Planner agent exceeded recursion limit of {config.planner_tool_call_limit}")
        raise RuntimeError(
            "Planner agent exceeded the configured step limit. "
            f"Reduce tool usage or increase PLANNER_TOOL_CALL_LIMIT from {config.planner_tool_call_limit}."
        ) from exc
    finally:
        logger.debug("Flushing langfuse traces")
        flush_langfuse_traces()

    logger.info("Main execution completed")
    print(result)

def example_basic_usage():
    logger.info("Starting example_basic_usage")
    langfuse_handler = get_langfuse_callback_handler()
    
    # Initialize state with a user request
    state: CodingAgentState = {
        "messages": [
            HumanMessage(content="go through the tool node implementation and without breaking anything if there is anything to be improved improve and also while adding the tool output to the message history need to truncate it to reduce context count")
        ],
        "summary": "",
        "goal": "",
        "current_task": "",
        "tasks": [],
        "workspace": {},
        "known_facts": [],
        "artifacts": {},
        "searches": {},
        "tool_history": [],
        "tool_cache": {},
        "runtime": {
            "iterations": 0,
            "max_iterations": 50,
            "planner_calls": 0,
            "tool_calls": 0
        },
        "state_context": "",
        "system_prompt": "",
    }
    
    # Create and invoke the planner graph with optional langfuse tracing and tools
    logger.info("Invoking planner with READ_ONLY_TOOLS")
    result = invoke_planner(
        state,
        langfuse_handler=langfuse_handler,
        tools=READ_ONLY_TOOLS
    )
    
    # Access the plan from the final AI message
    final_message = result["messages"][-1]
    logger.info(f"Plan generated with {len(result['messages'])} messages")
    print(f"Plan generated: {final_message.content}")
    
    return result


if __name__ == "__main__":
    main()


