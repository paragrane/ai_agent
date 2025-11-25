"""
Comprehensive test cases for Autogen multi-agent system
"""

import pytest
from typing import Dict, Any
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autogen_multiagent import (
    AutogenMultiAgentSystem,
    SimpleAutogenWorkflow,
    AgentConfig,
    create_autogen_system,
    create_simple_workflow
)


class TestAgentConfig:
    """Test cases for AgentConfig"""

    def test_default_config(self):
        """Test default configuration values"""
        config = AgentConfig()
        assert config.model == "gpt-4"
        assert config.temperature == 0.7
        assert config.api_key is None

    def test_custom_config(self):
        """Test custom configuration"""
        config = AgentConfig(
            api_key="test-key",
            model="gpt-3.5-turbo",
            temperature=0.5
        )
        assert config.api_key == "test-key"
        assert config.model == "gpt-3.5-turbo"
        assert config.temperature == 0.5

    def test_get_llm_config(self):
        """Test LLM config generation"""
        config = AgentConfig(api_key="test-key", model="gpt-4")
        llm_config = config.get_llm_config()

        assert "model" in llm_config
        assert "temperature" in llm_config
        assert "api_key" in llm_config
        assert llm_config["model"] == "gpt-4"
        assert llm_config["api_key"] == "test-key"

    def test_get_llm_config_without_api_key(self):
        """Test LLM config without API key"""
        config = AgentConfig(model="gpt-4")
        llm_config = config.get_llm_config()

        assert "model" in llm_config
        assert "temperature" in llm_config
        # api_key should not be in config if not provided
        assert "api_key" not in llm_config or llm_config["api_key"] is None


class TestAutogenMultiAgentSystem:
    """Test cases for AutogenMultiAgentSystem"""

    @pytest.fixture
    def config(self):
        """Create test configuration"""
        return AgentConfig(api_key="test-key")

    @pytest.fixture
    def system(self, config):
        """Create test system"""
        return AutogenMultiAgentSystem(config)

    def test_system_initialization(self, system):
        """Test system initialization"""
        assert system.coordinator is not None
        assert system.researcher is not None
        assert system.writer is not None
        assert system.critic is not None
        assert system.user_proxy is not None

    def test_agent_names(self, system):
        """Test agent names are correct"""
        assert system.coordinator.name == "Coordinator"
        assert system.researcher.name == "Researcher"
        assert system.writer.name == "Writer"
        assert system.critic.name == "Critic"
        assert system.user_proxy.name == "User"

    def test_agents_list(self, system):
        """Test agents list contains all agents"""
        assert len(system.agents) == 5
        agent_names = [agent.name for agent in system.agents]
        assert "Coordinator" in agent_names
        assert "Researcher" in agent_names
        assert "Writer" in agent_names
        assert "Critic" in agent_names
        assert "User" in agent_names

    def test_create_group_chat(self, system):
        """Test group chat creation"""
        groupchat = system.create_group_chat(max_round=5)
        assert groupchat is not None
        assert groupchat.max_round == 5
        assert len(groupchat.agents) == 5

    def test_run_returns_dict(self, system):
        """Test run method returns dictionary"""
        result = system.run("Test task")
        assert isinstance(result, dict)

    def test_run_result_structure(self, system):
        """Test run result has expected structure"""
        result = system.run("Test task", max_round=5)

        assert "task" in result
        assert "agents" in result
        assert "max_round" in result
        assert "status" in result
        assert "conversation" in result

        assert result["task"] == "Test task"
        assert result["max_round"] == 5
        assert result["status"] == "completed"

    def test_run_with_different_max_rounds(self, system):
        """Test run with different max_round values"""
        result1 = system.run("Test", max_round=5)
        result2 = system.run("Test", max_round=10)

        assert result1["max_round"] == 5
        assert result2["max_round"] == 10


class TestSimpleAutogenWorkflow:
    """Test cases for SimpleAutogenWorkflow"""

    @pytest.fixture
    def workflow(self):
        """Create test workflow"""
        config = AgentConfig(api_key="test-key")
        return SimpleAutogenWorkflow(config)

    def test_workflow_initialization(self, workflow):
        """Test workflow initialization"""
        assert workflow.config is not None
        assert workflow.llm_config is not None
        assert workflow.conversation_history == []

    def test_add_message(self, workflow):
        """Test adding messages to conversation"""
        workflow.add_message("TestAgent", "Test message")

        assert len(workflow.conversation_history) == 1
        assert workflow.conversation_history[0]["agent"] == "TestAgent"
        assert workflow.conversation_history[0]["message"] == "Test message"

    def test_add_multiple_messages(self, workflow):
        """Test adding multiple messages"""
        workflow.add_message("Agent1", "Message 1")
        workflow.add_message("Agent2", "Message 2")
        workflow.add_message("Agent3", "Message 3")

        assert len(workflow.conversation_history) == 3
        assert workflow.conversation_history[0]["agent"] == "Agent1"
        assert workflow.conversation_history[1]["agent"] == "Agent2"
        assert workflow.conversation_history[2]["agent"] == "Agent3"

    def test_process_task(self, workflow):
        """Test task processing"""
        result = workflow.process_task("Test research task")

        assert isinstance(result, dict)
        assert "task" in result
        assert "status" in result
        assert "conversation" in result
        assert "final_output" in result

    def test_process_task_result_structure(self, workflow):
        """Test process_task result structure"""
        task = "Research AI frameworks"
        result = workflow.process_task(task)

        assert result["task"] == task
        assert result["status"] == "completed"
        assert isinstance(result["conversation"], list)
        assert len(result["conversation"]) > 0
        assert task in result["final_output"]

    def test_process_task_conversation_flow(self, workflow):
        """Test that conversation follows expected flow"""
        result = workflow.process_task("Test task")
        agents = [msg["agent"] for msg in result["conversation"]]

        # Check that expected agents participate
        assert "Coordinator" in agents
        assert "Researcher" in agents
        assert "Writer" in agents
        assert "Critic" in agents

    def test_process_task_conversation_order(self, workflow):
        """Test conversation follows logical order"""
        result = workflow.process_task("Test task")
        agents = [msg["agent"] for msg in result["conversation"]]

        # Coordinator should appear first
        assert agents[0] == "Coordinator"

        # Researcher should appear before Writer
        researcher_idx = agents.index("Researcher")
        writer_idx = agents.index("Writer")
        assert researcher_idx < writer_idx

    def test_get_conversation_history(self, workflow):
        """Test getting conversation history"""
        workflow.process_task("Test task")
        history = workflow.get_conversation_history()

        assert isinstance(history, list)
        assert len(history) > 0
        assert all("agent" in msg and "message" in msg for msg in history)

    def test_multiple_tasks(self, workflow):
        """Test processing multiple tasks"""
        result1 = workflow.process_task("Task 1")
        result2 = workflow.process_task("Task 2")

        # Each task should clear previous history
        assert result1["task"] == "Task 1"
        assert result2["task"] == "Task 2"

        # Current history should be from task 2
        history = workflow.get_conversation_history()
        assert any("Task 2" in msg["message"] for msg in history)


class TestFactoryFunctions:
    """Test cases for factory functions"""

    def test_create_autogen_system(self):
        """Test create_autogen_system factory"""
        system = create_autogen_system(api_key="test-key")

        assert isinstance(system, AutogenMultiAgentSystem)
        assert system.config.api_key == "test-key"

    def test_create_autogen_system_without_key(self, monkeypatch):
        """Test create_autogen_system without explicit key"""
        monkeypatch.setenv("OPENAI_API_KEY", "env-test-key")
        system = create_autogen_system()

        assert isinstance(system, AutogenMultiAgentSystem)

    def test_create_simple_workflow(self):
        """Test create_simple_workflow factory"""
        workflow = create_simple_workflow(api_key="test-key")

        assert isinstance(workflow, SimpleAutogenWorkflow)
        assert workflow.config.api_key == "test-key"

    def test_create_simple_workflow_without_key(self, monkeypatch):
        """Test create_simple_workflow without explicit key"""
        monkeypatch.setenv("OPENAI_API_KEY", "env-test-key")
        workflow = create_simple_workflow()

        assert isinstance(workflow, SimpleAutogenWorkflow)


class TestIntegration:
    """Integration tests for the complete workflow"""

    def test_end_to_end_simple_workflow(self):
        """Test complete workflow from creation to execution"""
        # Create workflow
        workflow = create_simple_workflow(api_key="test-key")

        # Process task
        task = "Research quantum computing applications"
        result = workflow.process_task(task)

        # Verify complete result
        assert result["task"] == task
        assert result["status"] == "completed"
        assert len(result["conversation"]) >= 5  # Multiple agent interactions
        assert result["final_output"]
        assert task in result["final_output"]

    def test_workflow_with_different_tasks(self):
        """Test workflow with various task types"""
        workflow = create_simple_workflow(api_key="test-key")

        tasks = [
            "Research machine learning",
            "Write about blockchain",
            "Analyze data structures"
        ]

        for task in tasks:
            result = workflow.process_task(task)
            assert result["status"] == "completed"
            assert result["task"] == task

    def test_system_configuration_propagation(self):
        """Test that configuration is properly propagated"""
        config = AgentConfig(
            api_key="test-key",
            model="gpt-3.5-turbo",
            temperature=0.3
        )
        system = AutogenMultiAgentSystem(config)

        assert system.llm_config["model"] == "gpt-3.5-turbo"
        assert system.llm_config["temperature"] == 0.3


class TestErrorHandling:
    """Test error handling and edge cases"""

    def test_empty_task(self):
        """Test handling of empty task"""
        workflow = create_simple_workflow(api_key="test-key")
        result = workflow.process_task("")

        assert result["task"] == ""
        assert result["status"] == "completed"

    def test_very_long_task(self):
        """Test handling of very long task description"""
        workflow = create_simple_workflow(api_key="test-key")
        long_task = "A" * 1000
        result = workflow.process_task(long_task)

        assert result["task"] == long_task
        assert result["status"] == "completed"

    def test_special_characters_in_task(self):
        """Test handling of special characters"""
        workflow = create_simple_workflow(api_key="test-key")
        task = "Research <AI> & ML: \"quotes\" & 'apostrophes'"
        result = workflow.process_task(task)

        assert result["task"] == task
        assert result["status"] == "completed"


@pytest.mark.parametrize("max_round", [1, 5, 10, 20])
def test_different_max_rounds(max_round):
    """Parametrized test for different max_round values"""
    system = create_autogen_system(api_key="test-key")
    result = system.run("Test task", max_round=max_round)

    assert result["max_round"] == max_round
    assert result["status"] == "completed"


@pytest.mark.parametrize("temperature", [0.0, 0.5, 0.7, 1.0])
def test_different_temperatures(temperature):
    """Parametrized test for different temperature values"""
    config = AgentConfig(temperature=temperature)
    assert config.temperature == temperature

    llm_config = config.get_llm_config()
    assert llm_config["temperature"] == temperature


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
