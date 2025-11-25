"""
Comprehensive test cases for LangGraph multi-agent system
"""

import pytest
from typing import Dict, Any
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph_multiagent import (
    LangGraphMultiAgentSystem,
    SimpleLangGraphWorkflow,
    LangGraphConfig,
    AgentState,
    create_langgraph_system,
    create_simple_langgraph_workflow
)


class TestLangGraphConfig:
    """Test cases for LangGraphConfig"""

    def test_default_config(self):
        """Test default configuration values"""
        config = LangGraphConfig()
        assert config.model == "gpt-4"
        assert config.temperature == 0.7
        assert config.api_key is None
        assert config.max_iterations == 10

    def test_custom_config(self):
        """Test custom configuration"""
        config = LangGraphConfig(
            api_key="test-key",
            model="gpt-3.5-turbo",
            temperature=0.5,
            max_iterations=20
        )
        assert config.api_key == "test-key"
        assert config.model == "gpt-3.5-turbo"
        assert config.temperature == 0.5
        assert config.max_iterations == 20

    def test_get_llm(self):
        """Test LLM instance creation"""
        config = LangGraphConfig(api_key="test-key", model="gpt-4")
        llm = config.get_llm()
        assert llm is not None


class TestLangGraphMultiAgentSystem:
    """Test cases for LangGraphMultiAgentSystem"""

    @pytest.fixture
    def config(self):
        """Create test configuration"""
        return LangGraphConfig(api_key="test-key")

    @pytest.fixture
    def system(self, config):
        """Create test system"""
        return LangGraphMultiAgentSystem(config)

    def test_system_initialization(self, system):
        """Test system initialization"""
        assert system.config is not None
        assert system.llm is not None
        assert system.graph is not None

    def test_coordinator_node(self, system):
        """Test coordinator node function"""
        state: AgentState = {
            "task": "Test task",
            "messages": [],
            "research_data": None,
            "draft_content": None,
            "feedback": None,
            "final_output": None,
            "next_agent": "coordinator",
            "iteration": 0,
            "max_iterations": 10
        }

        result = system._coordinator_node(state)

        assert isinstance(result, dict)
        assert "messages" in result
        assert "next_agent" in result
        assert len(result["messages"]) > 0
        assert result["iteration"] == 1

    def test_coordinator_routing_to_researcher(self, system):
        """Test coordinator routes to researcher when no research data"""
        state: AgentState = {
            "task": "Test",
            "messages": [],
            "research_data": None,
            "draft_content": None,
            "feedback": None,
            "final_output": None,
            "next_agent": "coordinator",
            "iteration": 0,
            "max_iterations": 10
        }

        result = system._coordinator_node(state)
        assert result["next_agent"] == "researcher"

    def test_coordinator_routing_to_writer(self, system):
        """Test coordinator routes to writer when research is done"""
        state: AgentState = {
            "task": "Test",
            "messages": [],
            "research_data": "Research complete",
            "draft_content": None,
            "feedback": None,
            "final_output": None,
            "next_agent": "coordinator",
            "iteration": 0,
            "max_iterations": 10
        }

        result = system._coordinator_node(state)
        assert result["next_agent"] == "writer"

    def test_coordinator_routing_to_critic(self, system):
        """Test coordinator routes to critic when draft is done"""
        state: AgentState = {
            "task": "Test",
            "messages": [],
            "research_data": "Research complete",
            "draft_content": "Draft complete",
            "feedback": None,
            "final_output": None,
            "next_agent": "coordinator",
            "iteration": 0,
            "max_iterations": 10
        }

        result = system._coordinator_node(state)
        assert result["next_agent"] == "critic"

    def test_coordinator_routing_to_finalizer(self, system):
        """Test coordinator routes to finalizer when all done"""
        state: AgentState = {
            "task": "Test",
            "messages": [],
            "research_data": "Research complete",
            "draft_content": "Draft complete",
            "feedback": "Feedback provided",
            "final_output": None,
            "next_agent": "coordinator",
            "iteration": 0,
            "max_iterations": 10
        }

        result = system._coordinator_node(state)
        assert result["next_agent"] == "finalizer"

    def test_researcher_node(self, system):
        """Test researcher node function"""
        state: AgentState = {
            "task": "Test research task",
            "messages": [],
            "research_data": None,
            "draft_content": None,
            "feedback": None,
            "final_output": None,
            "next_agent": "researcher",
            "iteration": 0,
            "max_iterations": 10
        }

        result = system._researcher_node(state)

        assert "messages" in result
        assert "research_data" in result
        assert result["research_data"] is not None
        assert len(result["messages"]) > 0
        assert result["next_agent"] == "coordinator"

    def test_writer_node(self, system):
        """Test writer node function"""
        state: AgentState = {
            "task": "Test writing task",
            "messages": [],
            "research_data": "Test research",
            "draft_content": None,
            "feedback": None,
            "final_output": None,
            "next_agent": "writer",
            "iteration": 0,
            "max_iterations": 10
        }

        result = system._writer_node(state)

        assert "messages" in result
        assert "draft_content" in result
        assert result["draft_content"] is not None
        assert len(result["messages"]) > 0
        assert result["next_agent"] == "coordinator"

    def test_critic_node(self, system):
        """Test critic node function"""
        state: AgentState = {
            "task": "Test task",
            "messages": [],
            "research_data": "Research",
            "draft_content": "Draft",
            "feedback": None,
            "final_output": None,
            "next_agent": "critic",
            "iteration": 0,
            "max_iterations": 10
        }

        result = system._critic_node(state)

        assert "messages" in result
        assert "feedback" in result
        assert result["feedback"] is not None
        assert len(result["messages"]) > 0
        assert result["next_agent"] == "coordinator"

    def test_finalizer_node(self, system):
        """Test finalizer node function"""
        state: AgentState = {
            "task": "Test task",
            "messages": [],
            "research_data": "Research",
            "draft_content": "Draft",
            "feedback": "Feedback",
            "final_output": None,
            "next_agent": "finalizer",
            "iteration": 0,
            "max_iterations": 10
        }

        result = system._finalizer_node(state)

        assert "messages" in result
        assert "final_output" in result
        assert result["final_output"] is not None
        assert result["next_agent"] == "end"

    def test_route_agent_function(self, system):
        """Test routing function"""
        state: AgentState = {
            "task": "Test",
            "messages": [],
            "research_data": None,
            "draft_content": None,
            "feedback": None,
            "final_output": None,
            "next_agent": "researcher",
            "iteration": 0,
            "max_iterations": 10
        }

        route = system._route_agent(state)
        assert route == "researcher"

    def test_route_agent_max_iterations(self, system):
        """Test routing stops at max iterations"""
        state: AgentState = {
            "task": "Test",
            "messages": [],
            "research_data": None,
            "draft_content": None,
            "feedback": None,
            "final_output": None,
            "next_agent": "researcher",
            "iteration": 15,
            "max_iterations": 10
        }

        route = system._route_agent(state)
        assert route == "end"

    def test_run_returns_dict(self, system):
        """Test run method returns dictionary"""
        result = system.run("Test task")
        assert isinstance(result, dict)

    def test_run_result_structure(self, system):
        """Test run result has expected structure"""
        result = system.run("Test task", max_iterations=5)

        assert "task" in result
        assert "status" in result
        assert "conversation" in result
        assert "final_output" in result
        assert "iterations" in result

        assert result["task"] == "Test task"
        assert result["status"] == "completed"

    def test_run_with_different_max_iterations(self, system):
        """Test run with different max_iterations values"""
        result1 = system.run("Test", max_iterations=5)
        result2 = system.run("Test", max_iterations=10)

        assert isinstance(result1["iterations"], int)
        assert isinstance(result2["iterations"], int)

    def test_get_graph_visualization(self, system):
        """Test graph visualization method"""
        viz = system.get_graph_visualization()

        assert isinstance(viz, str)
        assert "Coordinator" in viz
        assert "Researcher" in viz
        assert "Writer" in viz
        assert "Critic" in viz
        assert "Finalizer" in viz


class TestSimpleLangGraphWorkflow:
    """Test cases for SimpleLangGraphWorkflow"""

    @pytest.fixture
    def workflow(self):
        """Create test workflow"""
        config = LangGraphConfig(api_key="test-key")
        return SimpleLangGraphWorkflow(config)

    def test_workflow_initialization(self, workflow):
        """Test workflow initialization"""
        assert workflow.config is not None
        assert workflow.workflow is not None

    def test_start_node(self, workflow):
        """Test start node"""
        state: AgentState = {
            "task": "Test task",
            "messages": [],
            "research_data": None,
            "draft_content": None,
            "feedback": None,
            "final_output": None,
            "next_agent": "start",
            "iteration": 0,
            "max_iterations": 10
        }

        result = workflow._start_node(state)
        assert "messages" in result
        assert len(result["messages"]) > 0

    def test_research_node(self, workflow):
        """Test research node"""
        state: AgentState = {
            "task": "Test task",
            "messages": [],
            "research_data": None,
            "draft_content": None,
            "feedback": None,
            "final_output": None,
            "next_agent": "research",
            "iteration": 0,
            "max_iterations": 10
        }

        result = workflow._research_node(state)
        assert result["research_data"] is not None

    def test_write_node(self, workflow):
        """Test write node"""
        state: AgentState = {
            "task": "Test task",
            "messages": [],
            "research_data": "Research",
            "draft_content": None,
            "feedback": None,
            "final_output": None,
            "next_agent": "write",
            "iteration": 0,
            "max_iterations": 10
        }

        result = workflow._write_node(state)
        assert result["draft_content"] is not None

    def test_review_node(self, workflow):
        """Test review node"""
        state: AgentState = {
            "task": "Test task",
            "messages": [],
            "research_data": "Research",
            "draft_content": "Draft",
            "feedback": None,
            "final_output": None,
            "next_agent": "review",
            "iteration": 0,
            "max_iterations": 10
        }

        result = workflow._review_node(state)
        assert result["feedback"] is not None

    def test_finalize_node(self, workflow):
        """Test finalize node"""
        state: AgentState = {
            "task": "Test task",
            "messages": [],
            "research_data": "Research",
            "draft_content": "Draft",
            "feedback": "Feedback",
            "final_output": None,
            "next_agent": "finalize",
            "iteration": 0,
            "max_iterations": 10
        }

        result = workflow._finalize_node(state)
        assert result["final_output"] is not None

    def test_run(self, workflow):
        """Test workflow run"""
        result = workflow.run("Test task")

        assert isinstance(result, dict)
        assert result["task"] == "Test task"
        assert result["status"] == "completed"

    def test_run_conversation(self, workflow):
        """Test workflow produces conversation"""
        result = workflow.run("Test task")

        assert "conversation" in result
        assert len(result["conversation"]) > 0

    def test_run_final_output(self, workflow):
        """Test workflow produces final output"""
        result = workflow.run("Test task")

        assert "final_output" in result
        assert result["final_output"] is not None


class TestFactoryFunctions:
    """Test cases for factory functions"""

    def test_create_langgraph_system(self):
        """Test create_langgraph_system factory"""
        system = create_langgraph_system(api_key="test-key")

        assert isinstance(system, LangGraphMultiAgentSystem)
        assert system.config.api_key == "test-key"

    def test_create_langgraph_system_without_key(self, monkeypatch):
        """Test create_langgraph_system without explicit key"""
        monkeypatch.setenv("OPENAI_API_KEY", "env-test-key")
        system = create_langgraph_system()

        assert isinstance(system, LangGraphMultiAgentSystem)

    def test_create_simple_langgraph_workflow(self):
        """Test create_simple_langgraph_workflow factory"""
        workflow = create_simple_langgraph_workflow(api_key="test-key")

        assert isinstance(workflow, SimpleLangGraphWorkflow)
        assert workflow.config.api_key == "test-key"

    def test_create_simple_langgraph_workflow_without_key(self, monkeypatch):
        """Test create_simple_langgraph_workflow without explicit key"""
        monkeypatch.setenv("OPENAI_API_KEY", "env-test-key")
        workflow = create_simple_langgraph_workflow()

        assert isinstance(workflow, SimpleLangGraphWorkflow)


class TestIntegration:
    """Integration tests for the complete workflow"""

    def test_end_to_end_simple_workflow(self):
        """Test complete workflow from creation to execution"""
        # Create workflow
        workflow = create_simple_langgraph_workflow(api_key="test-key")

        # Run task
        task = "Research quantum computing applications"
        result = workflow.run(task)

        # Verify complete result
        assert result["task"] == task
        assert result["status"] == "completed"
        assert len(result["conversation"]) > 0
        assert result["final_output"]

    def test_end_to_end_full_system(self):
        """Test full system execution"""
        system = create_langgraph_system(api_key="test-key")

        task = "Analyze machine learning trends"
        result = system.run(task, max_iterations=15)

        assert result["task"] == task
        assert result["status"] == "completed"
        assert result["final_output"]
        assert result["iterations"] > 0

    def test_workflow_with_different_tasks(self):
        """Test workflow with various task types"""
        workflow = create_simple_langgraph_workflow(api_key="test-key")

        tasks = [
            "Research blockchain technology",
            "Write about neural networks",
            "Analyze data visualization"
        ]

        for task in tasks:
            result = workflow.run(task)
            assert result["status"] == "completed"
            assert result["task"] == task

    def test_system_configuration_propagation(self):
        """Test that configuration is properly propagated"""
        config = LangGraphConfig(
            api_key="test-key",
            model="gpt-3.5-turbo",
            temperature=0.3,
            max_iterations=20
        )
        system = LangGraphMultiAgentSystem(config)

        assert system.config.model == "gpt-3.5-turbo"
        assert system.config.temperature == 0.3
        assert system.config.max_iterations == 20


class TestStatePersistence:
    """Test state persistence through workflow"""

    def test_state_accumulation(self):
        """Test that state accumulates data through nodes"""
        system = create_langgraph_system(api_key="test-key")

        result = system.run("Test task")

        # State should have accumulated data
        assert len(result["conversation"]) > 0

    def test_state_isolation(self):
        """Test that different runs have isolated state"""
        workflow = create_simple_langgraph_workflow(api_key="test-key")

        result1 = workflow.run("Task 1")
        result2 = workflow.run("Task 2")

        assert result1["task"] != result2["task"]


class TestErrorHandling:
    """Test error handling and edge cases"""

    def test_empty_task(self):
        """Test handling of empty task"""
        workflow = create_simple_langgraph_workflow(api_key="test-key")
        result = workflow.run("")

        assert result["task"] == ""
        assert result["status"] == "completed"

    def test_very_long_task(self):
        """Test handling of very long task description"""
        workflow = create_simple_langgraph_workflow(api_key="test-key")
        long_task = "A" * 1000
        result = workflow.run(long_task)

        assert result["task"] == long_task
        assert result["status"] == "completed"

    def test_special_characters_in_task(self):
        """Test handling of special characters"""
        workflow = create_simple_langgraph_workflow(api_key="test-key")
        task = "Research <AI> & ML: \"quotes\" & 'apostrophes'"
        result = workflow.run(task)

        assert result["task"] == task
        assert result["status"] == "completed"


@pytest.mark.parametrize("max_iterations", [1, 5, 10, 20])
def test_different_max_iterations(max_iterations):
    """Parametrized test for different max_iterations values"""
    system = create_langgraph_system(api_key="test-key")
    result = system.run("Test task", max_iterations=max_iterations)

    assert result["status"] == "completed"
    assert result["iterations"] <= max_iterations


@pytest.mark.parametrize("temperature", [0.0, 0.5, 0.7, 1.0])
def test_different_temperatures(temperature):
    """Parametrized test for different temperature values"""
    config = LangGraphConfig(temperature=temperature)
    assert config.temperature == temperature


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
