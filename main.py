from core.agents.planner import planner_agent
from langchain.messages import HumanMessage
import logging

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("langchain").setLevel(logging.DEBUG)
logging.getLogger("langgraph").setLevel(logging.DEBUG)
logging.getLogger("httpx").setLevel(logging.DEBUG)

def main():
    response = planner_agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "Update the system prompt for the planner agent to have more info about the tools"
                    )
                )
            ]
        }
    )

    print(response)


if __name__ == "__main__":
    main()


