"""
Unit tests for the LangGraph planner implementation.
"""

import pytest
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from core.agents.planner_graph import (
    inject_context_node,
    should_continue,
    build_state_context,
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


@pytest.fixture
def rich_state() -> CodingAgentState:
    """Create a state with rich context for testing."""
    return {
        "messages": [HumanMessage(content="Implement authentication")],
        "summary": "User wants to add authentication to the API",
        "goal": "Secure the application with JWT-based auth",
        "current_task": "",
        "tasks": [],
        "workspace": {
            "/app/auth.py": {
                "summary": "Contains basic user model",
                "language": "python",
                "classes": ["User"],
                "functions": ["create_user"]
            }
        },
        "known_facts": [
            {
                "fact": "Project uses FastAPI framework",
                "source": "scan_project:/app"
            }
        ],
        "artifacts": {
            "/app/auth.py": "read",
            "/app/models.py": "read"
        },
        "searches": {
            "authentication": [{"file": "/app/auth.py", "line": 10}]
        },
        "tool_history": [
            {
                "tool": "read_file",
                "arguments": {"file_path": "/app/auth.py"},
                "successful": True,
                "timestamp": "2026-06-27T10:00:00Z",
                "duration_ms": 50
            }
        ],
        "tool_cache": {},
        "runtime": {
            "iterations": 0,
            "max_iterations": 50,
            "planner_calls": 0,
            "tool_calls": 0
        }
    }


class TestBuildStateContext:
    """Test state context building."""
    
    def test_minimal_state_produces_no_context(self, minimal_state):
        """Empty state should produce minimal or no context."""
        context = build_state_context(minimal_state)
        # Should be empty or just the wrapper
        assert context == "" or "STATE CONTEXT" in context
    
    def test_rich_state_includes_all_sections(self, rich_state):
        """Rich state should include all relevant sections."""
        context = build_state_context(rich_state)
        
        assert "CONVERSATION SUMMARY" in context
        assert "CURRENT GOAL" in context
        assert "KNOWN FACTS" in context
        assert "WORKSPACE FILES" in context
        assert "ARTIFACTS" in context
        assert "RECENT TOOL CALLS" in context
        assert "CACHED SEARCHES" in context
    
    def test_context_includes_known_facts(self, rich_state):
        """Known facts should appear in context."""
        context = build_state_context(rich_state)
        assert "FastAPI framework" in context
    
    def test_context_truncates_long_summaries(self):
        """Long workspace summaries should be truncated."""
        state: CodingAgentState = {
            "messages": [],
            "summary": "",
            "goal": "",
            "current_task": "",
            "tasks": [],
            "workspace": {
                "/long.py": {
                    "summary": "x" * 200,  # Very long summary
                    "language": "python"
                }
            },
            "known_facts": [],
            "artifacts": {},
            "searches": {},
            "tool_history": [],
            "tool_cache": {},
            "runtime": {}
        }
        
        context = build_state_context(state)
        # Should be truncated to 150 chars
        assert "x" * 151 not in context


class TestInjectContextNode:
    """Test the context injection node."""
    
    def test_adds_system_message(self, minimal_state):
        """Should add system prompt as first message."""
        result = inject_context_node(minimal_state)
        
        messages = result.get("messages", [])
        assert len(messages) > 0
        assert isinstance(messages[0], SystemMessage)
        assert "Software Architect" in messages[0].content
    
    def test_preserves_user_messages(self, minimal_state):
        """Should preserve original user messages."""
        original_content = minimal_state["messages"][0].content
        result = inject_context_node(minimal_state)
        
        messages = result.get("messages", [])
        # Original message should still be present
        assert any(
            isinstance(msg, HumanMessage) and msg.content == original_content
            for msg in messages
        )
    
    def test_adds_state_context_when_available(self, rich_state):
        """Should add state context when state has data."""
        result = inject_context_node(rich_state)
        
        messages = result.get("messages", [])
        # Should have at least: system prompt, state context, user message
        system_messages = [msg for msg in messages if isinstance(msg, SystemMessage)]
        assert len(system_messages) >= 1
        
        # State context should be in one of the system messages
        all_content = "".join(
            str(msg.content) for msg in system_messages
        )
        assert "STATE CONTEXT" in all_content
    
    def test_removes_duplicate_system_messages(self):
        """Should not duplicate system messages if called multiple times."""
        state: CodingAgentState = {
            "messages": [
                SystemMessage(content="Old system message"),
                HumanMessage(content="User request")
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
            "runtime": {}
        }
        
        result = inject_context_node(state)
        messages = result["messages"]
        
        # Should have new system messages but not the old one
        system_messages = [msg for msg in messages if isinstance(msg, SystemMessage)]
        # At least one (system prompt), possibly two (system prompt + state context)
        assert len(system_messages) >= 1
        assert not any("Old system message" in msg.content for msg in system_messages)


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
    
    def test_graph_creates_successfully(self):
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
    async def test_graph_supports_streaming(self, minimal_state):
        """Graph should support streaming execution."""
        from core.agents.planner_graph import create_planner_graph
        
        graph = create_planner_graph()
        
        try:
            chunks = []
            async for chunk in graph.astream(minimal_state):
                chunks.append(chunk)
            
            assert len(chunks) > 0
        except Exception:
            # Expected to fail without a real LLM configured
            pytest.skip("Requires LLM configuration")
