"""
Integration tests comparing Autogen and LangGraph implementations
"""

import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autogen_multiagent import (
    SimpleAutogenWorkflow,
    AgentConfig as AutogenConfig,
    create_simple_workflow
)

from langgraph_multiagent import (
    SimpleLangGraphWorkflow,
    LangGraphConfig,
    create_simple_langgraph_workflow
)

from conversion_utils import (
    ConversionGuide,
    compare_execution_results,
    generate_conversion_report,
    CONVERSION_MAPPINGS,
    AgentRole,
    create_node_template,
    create_routing_template
)


class TestBothImplementations:
    """Test both Autogen and LangGraph implementations side by side"""

    @pytest.fixture
    def autogen_workflow(self):
        """Create Autogen workflow"""
        return create_simple_workflow(api_key="test-key")

    @pytest.fixture
    def langgraph_workflow(self):
        """Create LangGraph workflow"""
        return create_simple_langgraph_workflow(api_key="test-key")

    def test_both_complete_task(self, autogen_workflow, langgraph_workflow):
        """Test that both implementations complete tasks"""
        task = "Research AI frameworks"

        autogen_result = autogen_workflow.process_task(task)
        langgraph_result = langgraph_workflow.run(task)

        assert autogen_result["status"] == "completed"
        assert langgraph_result["status"] == "completed"

    def test_both_handle_same_task(self, autogen_workflow, langgraph_workflow):
        """Test that both handle the same task"""
        task = "Analyze machine learning"

        autogen_result = autogen_workflow.process_task(task)
        langgraph_result = langgraph_workflow.run(task)

        assert autogen_result["task"] == task
        assert langgraph_result["task"] == task

    def test_both_produce_conversation(self, autogen_workflow, langgraph_workflow):
        """Test that both produce conversation history"""
        task = "Test task"

        autogen_result = autogen_workflow.process_task(task)
        langgraph_result = langgraph_workflow.run(task)

        assert len(autogen_result["conversation"]) > 0
        assert len(langgraph_result["conversation"]) > 0

    def test_both_produce_final_output(self, autogen_workflow, langgraph_workflow):
        """Test that both produce final output"""
        task = "Test task"

        autogen_result = autogen_workflow.process_task(task)
        langgraph_result = langgraph_workflow.run(task)

        assert autogen_result["final_output"] is not None
        assert langgraph_result["final_output"] is not None

    def test_similar_conversation_structure(self, autogen_workflow, langgraph_workflow):
        """Test that conversations have similar structure"""
        task = "Test task"

        autogen_result = autogen_workflow.process_task(task)
        langgraph_result = langgraph_workflow.run(task)

        # Both should have messages with agent and message fields
        for msg in autogen_result["conversation"]:
            assert "agent" in msg
            assert "message" in msg

        for msg in langgraph_result["conversation"]:
            assert "agent" in msg
            assert "message" in msg

    def test_similar_agent_participation(self, autogen_workflow, langgraph_workflow):
        """Test that similar agents participate in both"""
        task = "Test task"

        autogen_result = autogen_workflow.process_task(task)
        langgraph_result = langgraph_workflow.run(task)

        autogen_agents = set(msg["agent"] for msg in autogen_result["conversation"])
        langgraph_agents = set(msg["agent"] for msg in langgraph_result["conversation"])

        # Should have some overlap in agent names
        common_agents = autogen_agents.intersection(langgraph_agents)
        assert len(common_agents) > 0


class TestConversionUtils:
    """Test conversion utility functions"""

    def test_conversion_guide_steps(self):
        """Test conversion guide provides steps"""
        guide = ConversionGuide()
        steps = guide.get_conversion_steps()

        assert isinstance(steps, list)
        assert len(steps) > 0
        assert all(isinstance(step, str) for step in steps)

    def test_conversion_guide_differences(self):
        """Test conversion guide provides key differences"""
        guide = ConversionGuide()
        differences = guide.get_key_differences()

        assert isinstance(differences, dict)
        assert "Architecture" in differences
        assert "Agent Communication" in differences

        for category, diff in differences.items():
            assert "Autogen" in diff
            assert "LangGraph" in diff

    def test_conversion_guide_best_practices(self):
        """Test conversion guide provides best practices"""
        guide = ConversionGuide()
        practices = guide.get_best_practices()

        assert isinstance(practices, list)
        assert len(practices) > 0
        assert all(isinstance(practice, str) for practice in practices)

    def test_conversion_mappings_exist(self):
        """Test that conversion mappings are defined"""
        assert len(CONVERSION_MAPPINGS) > 0

        for mapping in CONVERSION_MAPPINGS:
            assert mapping.autogen_concept
            assert mapping.langgraph_concept
            assert mapping.description

    def test_compare_execution_results(self):
        """Test execution result comparison"""
        autogen_result = {
            "task": "Test task",
            "status": "completed",
            "conversation": [
                {"agent": "A1", "message": "M1"},
                {"agent": "A2", "message": "M2"}
            ],
            "final_output": "Output"
        }

        langgraph_result = {
            "task": "Test task",
            "status": "completed",
            "conversation": [
                {"agent": "A1", "message": "M1"}
            ],
            "final_output": "Output"
        }

        comparison = compare_execution_results(autogen_result, langgraph_result)

        assert comparison["task_match"] is True
        assert comparison["status_match"] is True
        assert "conversation_length" in comparison
        assert "agents_involved" in comparison

    def test_generate_conversion_report(self):
        """Test conversion report generation"""
        autogen_agents = ["Coordinator", "Researcher", "Writer"]
        langgraph_nodes = ["coordinator", "researcher", "writer"]

        report = generate_conversion_report(autogen_agents, langgraph_nodes)

        assert isinstance(report, str)
        assert "Coordinator" in report
        assert "coordinator" in report
        assert "Conversion Status" in report

    def test_create_node_template(self):
        """Test node template generation"""
        template = create_node_template("Researcher", AgentRole.RESEARCHER)

        assert isinstance(template, str)
        assert "researcher_node" in template
        assert "def " in template
        assert "AgentState" in template

    def test_create_routing_template(self):
        """Test routing template generation"""
        agents = ["coordinator", "researcher", "writer"]
        template = create_routing_template(agents)

        assert isinstance(template, str)
        assert "def route_agent" in template
        assert "coordinator" in template
        assert "researcher" in template
        assert "writer" in template


class TestEndToEndConversion:
    """End-to-end tests for the conversion process"""

    def test_complete_workflow_comparison(self):
        """Test complete workflow comparison"""
        # Create both workflows
        autogen = create_simple_workflow(api_key="test-key")
        langgraph = create_simple_langgraph_workflow(api_key="test-key")

        # Run same task on both
        task = "Research quantum computing"
        autogen_result = autogen.process_task(task)
        langgraph_result = langgraph.run(task)

        # Compare results
        comparison = compare_execution_results(autogen_result, langgraph_result)

        assert comparison["task_match"] is True
        assert comparison["status_match"] is True
        assert comparison["has_final_output"]["autogen"] is True
        assert comparison["has_final_output"]["langgraph"] is True

    def test_multiple_tasks_consistency(self):
        """Test consistency across multiple tasks"""
        autogen = create_simple_workflow(api_key="test-key")
        langgraph = create_simple_langgraph_workflow(api_key="test-key")

        tasks = [
            "Research AI",
            "Write about ML",
            "Analyze data"
        ]

        for task in tasks:
            autogen_result = autogen.process_task(task)
            langgraph_result = langgraph.run(task)

            assert autogen_result["status"] == "completed"
            assert langgraph_result["status"] == "completed"
            assert autogen_result["task"] == langgraph_result["task"]


class TestAgentRoles:
    """Test agent role definitions"""

    def test_agent_role_enum(self):
        """Test AgentRole enum"""
        assert AgentRole.COORDINATOR.value == "coordinator"
        assert AgentRole.RESEARCHER.value == "researcher"
        assert AgentRole.WRITER.value == "writer"
        assert AgentRole.CRITIC.value == "critic"

    def test_all_roles_accessible(self):
        """Test all roles are accessible"""
        roles = [role for role in AgentRole]
        assert len(roles) >= 5  # At least 5 common roles


class TestConfigurationCompatibility:
    """Test configuration compatibility between implementations"""

    def test_similar_config_structure(self):
        """Test that configs have similar structure"""
        autogen_config = AutogenConfig(
            api_key="test-key",
            model="gpt-4",
            temperature=0.7
        )

        langgraph_config = LangGraphConfig(
            api_key="test-key",
            model="gpt-4",
            temperature=0.7
        )

        assert autogen_config.api_key == langgraph_config.api_key
        assert autogen_config.model == langgraph_config.model
        assert autogen_config.temperature == langgraph_config.temperature

    def test_config_portability(self):
        """Test that config values can be ported"""
        # Create autogen config
        autogen_config = AutogenConfig(
            api_key="test-key",
            model="gpt-3.5-turbo",
            temperature=0.5
        )

        # Use same values for langgraph
        langgraph_config = LangGraphConfig(
            api_key=autogen_config.api_key,
            model=autogen_config.model,
            temperature=autogen_config.temperature
        )

        assert autogen_config.model == langgraph_config.model
        assert autogen_config.temperature == langgraph_config.temperature


class TestResultStructure:
    """Test result structure compatibility"""

    def test_result_keys(self):
        """Test that results have expected keys"""
        autogen = create_simple_workflow(api_key="test-key")
        langgraph = create_simple_langgraph_workflow(api_key="test-key")

        autogen_result = autogen.process_task("Test")
        langgraph_result = langgraph.run("Test")

        # Common keys
        common_keys = ["task", "status", "conversation", "final_output"]

        for key in common_keys:
            assert key in autogen_result
            assert key in langgraph_result

    def test_result_types(self):
        """Test that result types are compatible"""
        autogen = create_simple_workflow(api_key="test-key")
        langgraph = create_simple_langgraph_workflow(api_key="test-key")

        autogen_result = autogen.process_task("Test")
        langgraph_result = langgraph.run("Test")

        # Check types match
        assert type(autogen_result["task"]) == type(langgraph_result["task"])
        assert type(autogen_result["status"]) == type(langgraph_result["status"])
        assert type(autogen_result["conversation"]) == type(langgraph_result["conversation"])


class TestConversionDocumentation:
    """Test that conversion documentation is accessible"""

    def test_conversion_guide_accessible(self):
        """Test that conversion guide is accessible"""
        guide = ConversionGuide()
        assert guide is not None

    def test_all_guide_methods_work(self):
        """Test that all guide methods work"""
        guide = ConversionGuide()

        steps = guide.get_conversion_steps()
        differences = guide.get_key_differences()
        practices = guide.get_best_practices()

        assert steps is not None
        assert differences is not None
        assert practices is not None


class TestPerformanceComparison:
    """Test performance characteristics"""

    def test_both_complete_in_reasonable_time(self):
        """Test that both implementations complete quickly"""
        import time

        autogen = create_simple_workflow(api_key="test-key")
        langgraph = create_simple_langgraph_workflow(api_key="test-key")

        task = "Quick test"

        # Time autogen
        start = time.time()
        autogen.process_task(task)
        autogen_time = time.time() - start

        # Time langgraph
        start = time.time()
        langgraph.run(task)
        langgraph_time = time.time() - start

        # Both should complete in under 5 seconds for simple workflow
        assert autogen_time < 5.0
        assert langgraph_time < 5.0


@pytest.mark.parametrize("task", [
    "Research AI",
    "Write documentation",
    "Analyze code",
    "Review architecture"
])
def test_both_handle_various_tasks(task):
    """Parametrized test for various tasks"""
    autogen = create_simple_workflow(api_key="test-key")
    langgraph = create_simple_langgraph_workflow(api_key="test-key")

    autogen_result = autogen.process_task(task)
    langgraph_result = langgraph.run(task)

    assert autogen_result["status"] == "completed"
    assert langgraph_result["status"] == "completed"
    assert autogen_result["task"] == langgraph_result["task"] == task


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
