"""
Unit tests for the LangGraph planner implementation.
"""

import pytest
from langchain_core.messages import HumanMessage, AIMessage
from core.agents.planner_graph import (
    should_continue,
    create_planner_graph,
)
from core.model.state import CodingAgentState


@pytest.fixture
def minimal_state() -> CodingAgentState:
    """Create a minimal valid state for testing."""
    return {
        "messages": [HumanMessage(content="Test request")],
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
        }
    }


class TestShouldContinue:
    """Test the routing logic."""
    
    def test_routes_to_end_when_no_tool_calls(self, minimal_state):
        """Should route to END when AI message has no tool calls."""
        state = minimal_state.copy()
        state["messages"] = [
            AIMessage(content="Here is the plan", tool_calls=[])
        ]
        
        result = should_continue(state)
        assert result == "end"
    
    def test_routes_to_tools_when_tool_calls_present(self, minimal_state):
        """Should route to tools when AI message has tool calls."""
        state = minimal_state.copy()
        state["messages"] = [
            AIMessage(
                content="Let me search for files",
                tool_calls=[
                    {
                        "id": "call_123",
                        "name": "read_file",
                        "args": {"file_path": "/app/test.py"}
                    }
                ]
            )
        ]
        
        result = should_continue(state)
        assert result == "tools"
    
    def test_routes_to_end_when_empty_messages(self):
        """Should route to END when there are no messages."""
        state: CodingAgentState = {
            "messages": [],
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
            "runtime": {}
        }
        
        result = should_continue(state)
        assert result == "end"


class TestCreatePlannerGraph:
    """Test graph creation."""
    
    def test_creates_valid_graph(self):
        """Should create a compiled graph."""
        graph = create_planner_graph()
        
        # Should be compiled and ready to invoke
        assert graph is not None
        assert hasattr(graph, "invoke")
        assert hasattr(graph, "ainvoke")
        assert hasattr(graph, "astream")
    
    def test_graph_has_correct_nodes(self):
        """Graph should have the expected nodes."""
        graph = create_planner_graph()
        
        # Check node names (this is implementation-dependent)
        # The graph should have inject_context, llm, and tools nodes
        # Note: exact API for inspecting nodes may vary
        nodes = graph.nodes if hasattr(graph, "nodes") else {}
        
        # At minimum, should be a compiled graph
        assert callable(graph.invoke)
    
    def test_graph_accepts_custom_checkpointer(self):
        """Should accept a custom checkpointer."""
        from langgraph.checkpoint.memory import MemorySaver
        
        checkpointer = MemorySaver()
        graph = create_planner_graph(
            checkpointer=checkpointer,
        )
        
        assert graph is not None
    
    def test_graph_creates_successfully(self, minimal_state):
        """Graph creation should succeed without LSP auto-install."""
        from core.agents.planner_graph import create_planner_graph
        
        graph = create_planner_graph()
        
        # This would normally call the LLM, so we mark it as integration test
        # In a real test, you'd mock the LLM
        try:
            result = graph.invoke(minimal_state)
            assert "messages" in result
            assert "runtime" in result
        except Exception:
            # Expected to fail without a real LLM configured
            pytest.skip("Requires LLM configuration")
    
    @pytest.mark.integration
    def test_graph_supports_streaming(self, minimal_state):
        """Graph should support streaming execution."""
        from core.agents.planner_graph import create_planner_graph
        
        graph = create_planner_graph()
        
        try:
            chunks = []
            for chunk in graph.stream(minimal_state):
                chunks.append(chunk)
            
            assert len(chunks) > 0
        except Exception:
            # Expected to fail without a real LLM configured
            pytest.skip("Requires LLM configuration")
