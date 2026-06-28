"""
Example usage of the LangGraph-based planner implementation.

This demonstrates how to use the new planner_graph module as a drop-in
replacement for the legacy agent-based planner.
"""

from langchain_core.messages import HumanMessage
from core.agents.planner_graph import create_planner_graph, invoke_planner, ainvoke_planner
from core.model.state import CodingAgentState
from core.tools.files import read_file, list_files, find_files
from core.tools.search import grep, scan_project


# ============================================================================
# EXAMPLE 1: Basic Usage with State
# ============================================================================

def example_basic_usage():
    """Demonstrate basic planner usage with minimal state."""
    
    # Initialize state with a user request
    state: CodingAgentState = {
        "messages": [
            HumanMessage(content="Implement user authentication with JWT tokens")
        ],
        "summary": "",
        "goal": "Add secure user authentication to the application",
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
        }
    }
    
    # Create and invoke the planner graph
    result = invoke_planner(state)
    
    # Access the plan from the final AI message
    final_message = result["messages"][-1]
    print(f"Plan generated: {final_message.content}")
    
    return result


# ============================================================================
# EXAMPLE 2: Usage with Custom Tools Configuration
# ============================================================================

def example_with_custom_tools():
    """Demonstrate planner usage with custom tool configuration."""
    
    from langchain_core.runnables import RunnableConfig
    
    # Define available tools
    tools = [read_file, list_files, find_files, grep, scan_project]
    
    # Create runnable config with tools
    config = RunnableConfig(
        configurable={
            "tools": tools
        }
    )
    
    state: CodingAgentState = {
        "messages": [
            HumanMessage(content="Add input validation to the user registration endpoint")
        ],
        "summary": "User requested input validation for registration",
        "goal": "Implement robust input validation",
        "current_task": "",
        "tasks": [],
        "workspace": {
            "/app/routes/auth.py": {
                "summary": "Contains authentication routes",
                "language": "python",
                "classes": ["AuthRouter"],
                "functions": ["register", "login"],
                "exists": True
            }
        },
        "known_facts": [
            {
                "fact": "Project uses FastAPI framework",
                "source": "scan_project:/app"
            }
        ],
        "artifacts": {
            "/app/routes/auth.py": "read"
        },
        "searches": {},
        "tool_history": [],
        "tool_cache": {},
        "runtime": {
            "iterations": 0,
            "max_iterations": 50,
            "planner_calls": 0,
            "tool_calls": 0
        }
    }
    
    # Invoke with configuration
    result = invoke_planner(state, config=config)
    
    return result


# ============================================================================
# EXAMPLE 3: Streaming Usage
# ============================================================================

async def example_streaming():
    """Demonstrate streaming planner execution."""
    
    from core.agents.planner_graph import create_planner_graph
    
    state: CodingAgentState = {
        "messages": [
            HumanMessage(content="Refactor the database connection logic to use a connection pool")
        ],
        "summary": "",
        "goal": "Optimize database connections",
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
        }
    }
    
    # Create the graph
    graph = create_planner_graph()
    
    # Stream the execution
    final_chunk = None
    async for chunk in graph.astream(state):
        print(f"Step: {chunk}")
        final_chunk = chunk
    
    return final_chunk


# ============================================================================
# EXAMPLE 4: With Persistence (Checkpointing)
# ============================================================================

async def example_with_persistence():
    """Demonstrate planner usage with state persistence."""
    
    from langgraph.checkpoint.memory import MemorySaver
    
    # Create a checkpointer for persistence
    checkpointer = MemorySaver()
    
    state: CodingAgentState = {
        "messages": [
            HumanMessage(content="Add API rate limiting middleware")
        ],
        "summary": "",
        "goal": "Implement rate limiting",
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
        }
    }
    
    # Invoke with persistence
    result = await ainvoke_planner(state, checkpointer=checkpointer)
    
    return result


# ============================================================================
# EXAMPLE 5: Direct Graph Usage (Advanced)
# ============================================================================

def example_direct_graph_usage():
    """Demonstrate direct graph manipulation for advanced use cases."""
    
    from core.agents.planner_graph import create_planner_graph
    from langchain_core.runnables import RunnableConfig
    from core.tools.files import read_file, list_files
    
    # Create the graph once (can be reused)
    graph = create_planner_graph(auto_install_lsp=False)
    
    # Prepare state
    state: CodingAgentState = {
        "messages": [
            HumanMessage(content="Add logging to all API endpoints")
        ],
        "summary": "",
        "goal": "Improve observability",
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
        }
    }
    
    # Configure tools
    config = RunnableConfig(
        configurable={
            "tools": [read_file, list_files]
        }
    )
    
    # Invoke the graph multiple times with different inputs
    result = graph.invoke(state, config=config)
    
    return result


# ============================================================================
# MIGRATION GUIDE
# ============================================================================

"""
MIGRATION FROM LEGACY PLANNER TO LANGGRAPH

OLD CODE (legacy agent-based):
    from core.agents.planner import invoke_planner_with_state
    
    result = invoke_planner_with_state(state, config=runnable_config)

NEW CODE (LangGraph-based):
    from core.agents.planner_graph import invoke_planner
    
    result = invoke_planner(state, config=runnable_config)

KEY DIFFERENCES:
1. The graph-based implementation is stateless by default - use checkpointers for persistence
2. The graph exposes more granular control via create_planner_graph()
3. All nodes are reusable and can be composed into different graphs
4. Better debugging via graph visualization and step-by-step streaming

BENEFITS:
- Minimal: Only 3 nodes (inject_context, llm, tools)
- Extensible: Easy to add new nodes or modify the graph structure
- Debuggable: Built-in support for visualization and streaming
- Testable: Each node can be tested independently
- Maintainable: Clear separation of concerns
"""


if __name__ == "__main__":
    # Run basic example
    result = example_basic_usage()
    print(f"Runtime stats: {result.get('runtime')}")
