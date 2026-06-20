from langchain_protocol import Command

from core.agents.planner import planner_agent

from langchain.messages import HumanMessage

def main():
    response = planner_agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "Implement a new feature that allows users login "
                        "The feature should include secure token "
                        "generation, and token validation."
                    )
                )
            ]
        }
    )

    print(response)


if __name__ == "__main__":
    main()


