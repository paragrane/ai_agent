"""
Test cases for conversion utility functions
"""

import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from conversion_utils import (
    ConversionMapping,
    ConversionGuide,
    AgentRole,
    autogen_to_langgraph_state,
    create_node_template,
    create_routing_template,
    compare_execution_results,
    generate_conversion_report,
    MigrationHelper,
    print_conversion_guide,
    CONVERSION_MAPPINGS
)


class TestConversionMapping:
    """Test ConversionMapping dataclass"""

    def test_create_mapping(self):
        """Test creating a conversion mapping"""
        mapping = ConversionMapping(
            "TestAutogen",
            "TestLangGraph",
            "Test description"
        )

        assert mapping.autogen_concept == "TestAutogen"
        assert mapping.langgraph_concept == "TestLangGraph"
        assert mapping.description == "Test description"

    def test_predefined_mappings(self):
        """Test that predefined mappings exist"""
        assert len(CONVERSION_MAPPINGS) > 0

        for mapping in CONVERSION_MAPPINGS:
            assert isinstance(mapping, ConversionMapping)
            assert mapping.autogen_concept
            assert mapping.langgraph_concept
            assert mapping.description


class TestAgentRole:
    """Test AgentRole enum"""

    def test_all_roles(self):
        """Test all agent roles are defined"""
        assert AgentRole.COORDINATOR.value == "coordinator"
        assert AgentRole.RESEARCHER.value == "researcher"
        assert AgentRole.WRITER.value == "writer"
        assert AgentRole.CRITIC.value == "critic"
        assert AgentRole.EXECUTOR.value == "executor"
        assert AgentRole.ANALYST.value == "analyst"

    def test_role_iteration(self):
        """Test iterating over roles"""
        roles = [role for role in AgentRole]
        assert len(roles) == 6

    def test_role_values(self):
        """Test role values are strings"""
        for role in AgentRole:
            assert isinstance(role.value, str)


class TestConversionGuide:
    """Test ConversionGuide class"""

    def test_get_conversion_steps(self):
        """Test getting conversion steps"""
        guide = ConversionGuide()
        steps = guide.get_conversion_steps()

        assert isinstance(steps, list)
        assert len(steps) == 10
        assert all(isinstance(step, str) for step in steps)

        # Check some key steps are present
        assert any("Define State Schema" in step for step in steps)
        assert any("Convert Agents" in step for step in steps)
        assert any("StateGraph" in step for step in steps)

    def test_get_key_differences(self):
        """Test getting key differences"""
        guide = ConversionGuide()
        differences = guide.get_key_differences()

        assert isinstance(differences, dict)
        assert len(differences) > 0

        # Check required categories
        assert "Architecture" in differences
        assert "Agent Communication" in differences
        assert "Workflow Control" in differences
        assert "State Management" in differences
        assert "Execution Model" in differences

        # Check each has both Autogen and LangGraph entries
        for category, diff in differences.items():
            assert "Autogen" in diff
            assert "LangGraph" in diff
            assert isinstance(diff["Autogen"], str)
            assert isinstance(diff["LangGraph"], str)

    def test_get_best_practices(self):
        """Test getting best practices"""
        guide = ConversionGuide()
        practices = guide.get_best_practices()

        assert isinstance(practices, list)
        assert len(practices) == 10
        assert all(isinstance(practice, str) for practice in practices)

        # Check some key practices are present
        assert any("state schema" in practice.lower() for practice in practices)
        assert any("node function" in practice.lower() for practice in practices)


class TestAutogenToLanggraphState:
    """Test state schema generation"""

    def test_basic_state_generation(self):
        """Test basic state schema generation"""
        state_class = autogen_to_langgraph_state(
            agents=["researcher", "writer"]
        )

        assert state_class is not None
        assert state_class.__name__ == "GeneratedState"

    def test_state_with_additional_fields(self):
        """Test state generation with additional fields"""
        state_class = autogen_to_langgraph_state(
            agents=["researcher", "writer"],
            additional_fields={"custom_field": str}
        )

        assert state_class is not None


class TestCreateNodeTemplate:
    """Test node template generation"""

    def test_create_basic_node_template(self):
        """Test creating basic node template"""
        template = create_node_template("Researcher", AgentRole.RESEARCHER)

        assert isinstance(template, str)
        assert "def researcher_node" in template
        assert "AgentState" in template
        assert "Researcher" in template

    def test_template_has_function_signature(self):
        """Test template has proper function signature"""
        template = create_node_template("Writer", AgentRole.WRITER)

        assert "def writer_node(state: AgentState)" in template
        assert "return" in template

    def test_template_has_docstring(self):
        """Test template includes docstring"""
        template = create_node_template("Critic", AgentRole.CRITIC)

        assert '"""' in template
        assert "Critic agent node" in template

    def test_different_agent_names(self):
        """Test templates for different agent names"""
        agents = [
            ("Coordinator", AgentRole.COORDINATOR),
            ("Researcher", AgentRole.RESEARCHER),
            ("Writer", AgentRole.WRITER),
            ("Critic", AgentRole.CRITIC)
        ]

        for name, role in agents:
            template = create_node_template(name, role)
            assert f"def {name.lower()}_node" in template
            assert name in template


class TestCreateRoutingTemplate:
    """Test routing template generation"""

    def test_create_basic_routing_template(self):
        """Test creating basic routing template"""
        agents = ["coordinator", "researcher", "writer"]
        template = create_routing_template(agents)

        assert isinstance(template, str)
        assert "def route_agent" in template
        assert "AgentState" in template

    def test_routing_includes_all_agents(self):
        """Test routing template includes all agents"""
        agents = ["coordinator", "researcher", "writer", "critic"]
        template = create_routing_template(agents)

        for agent in agents:
            assert agent in template

    def test_routing_has_function_signature(self):
        """Test routing template has proper signature"""
        template = create_routing_template(["agent1", "agent2"])

        assert "def route_agent(state: AgentState) -> str:" in template
        assert "return" in template

    def test_routing_has_docstring(self):
        """Test routing template has docstring"""
        template = create_routing_template(["agent1"])

        assert '"""' in template


class TestCompareExecutionResults:
    """Test execution result comparison"""

    def test_matching_results(self):
        """Test comparison of matching results"""
        result1 = {
            "task": "Test",
            "status": "completed",
            "conversation": [{"agent": "A", "message": "M"}],
            "final_output": "Output"
        }

        result2 = {
            "task": "Test",
            "status": "completed",
            "conversation": [{"agent": "A", "message": "M"}],
            "final_output": "Output"
        }

        comparison = compare_execution_results(result1, result2)

        assert comparison["task_match"] is True
        assert comparison["status_match"] is True

    def test_different_tasks(self):
        """Test comparison with different tasks"""
        result1 = {"task": "Task1", "status": "completed", "conversation": []}
        result2 = {"task": "Task2", "status": "completed", "conversation": []}

        comparison = compare_execution_results(result1, result2)

        assert comparison["task_match"] is False

    def test_different_status(self):
        """Test comparison with different status"""
        result1 = {"task": "Test", "status": "completed", "conversation": []}
        result2 = {"task": "Test", "status": "failed", "conversation": []}

        comparison = compare_execution_results(result1, result2)

        assert comparison["status_match"] is False

    def test_conversation_length_comparison(self):
        """Test conversation length comparison"""
        result1 = {
            "task": "Test",
            "status": "completed",
            "conversation": [{"agent": "A", "message": "M1"}],
            "final_output": "Out"
        }

        result2 = {
            "task": "Test",
            "status": "completed",
            "conversation": [
                {"agent": "A", "message": "M1"},
                {"agent": "B", "message": "M2"}
            ],
            "final_output": "Out"
        }

        comparison = compare_execution_results(result1, result2)

        assert comparison["conversation_length"]["autogen"] == 1
        assert comparison["conversation_length"]["langgraph"] == 2

    def test_agents_involved_comparison(self):
        """Test agents involved comparison"""
        result1 = {
            "task": "Test",
            "status": "completed",
            "conversation": [
                {"agent": "A", "message": "M1"},
                {"agent": "B", "message": "M2"}
            ],
            "final_output": "Out"
        }

        result2 = {
            "task": "Test",
            "status": "completed",
            "conversation": [
                {"agent": "A", "message": "M1"},
                {"agent": "C", "message": "M2"}
            ],
            "final_output": "Out"
        }

        comparison = compare_execution_results(result1, result2)

        assert "A" in comparison["agents_involved"]["autogen"]
        assert "B" in comparison["agents_involved"]["autogen"]
        assert "A" in comparison["agents_involved"]["langgraph"]
        assert "C" in comparison["agents_involved"]["langgraph"]

    def test_final_output_presence(self):
        """Test final output presence check"""
        result1 = {"task": "Test", "status": "completed", "conversation": [], "final_output": "Out"}
        result2 = {"task": "Test", "status": "completed", "conversation": []}

        comparison = compare_execution_results(result1, result2)

        assert comparison["has_final_output"]["autogen"] is True
        assert comparison["has_final_output"]["langgraph"] is False


class TestGenerateConversionReport:
    """Test conversion report generation"""

    def test_basic_report(self):
        """Test basic report generation"""
        autogen_agents = ["Coordinator", "Researcher"]
        langgraph_nodes = ["coordinator", "researcher"]

        report = generate_conversion_report(autogen_agents, langgraph_nodes)

        assert isinstance(report, str)
        assert "Coordinator" in report
        assert "coordinator" in report
        assert "Researcher" in report
        assert "researcher" in report

    def test_report_structure(self):
        """Test report has proper structure"""
        report = generate_conversion_report(["A1"], ["n1"])

        assert "Conversion Report" in report
        assert "Autogen Agents" in report
        assert "LangGraph Nodes" in report
        assert "Conversion Status" in report
        assert "Next Steps" in report

    def test_report_shows_counts(self):
        """Test report shows agent/node counts"""
        autogen_agents = ["A1", "A2", "A3"]
        langgraph_nodes = ["n1", "n2", "n3"]

        report = generate_conversion_report(autogen_agents, langgraph_nodes)

        assert "(3)" in report  # Count should appear


class TestMigrationHelper:
    """Test MigrationHelper class"""

    def test_initialization(self):
        """Test helper initialization"""
        helper = MigrationHelper()

        assert helper.agents_mapped == []
        assert helper.nodes_created == []

    def test_map_agent_to_node(self):
        """Test mapping agent to node"""
        helper = MigrationHelper()
        helper.map_agent_to_node("Researcher", "researcher_node")

        assert len(helper.agents_mapped) == 1
        assert helper.agents_mapped[0]["agent"] == "Researcher"
        assert helper.agents_mapped[0]["node"] == "researcher_node"

    def test_multiple_mappings(self):
        """Test multiple agent mappings"""
        helper = MigrationHelper()
        helper.map_agent_to_node("Researcher", "researcher_node")
        helper.map_agent_to_node("Writer", "writer_node")

        assert len(helper.agents_mapped) == 2

    def test_create_node(self):
        """Test recording node creation"""
        helper = MigrationHelper()

        def test_node(state):
            return state

        helper.create_node("test_node", test_node)

        assert len(helper.nodes_created) == 1
        assert helper.nodes_created[0]["name"] == "test_node"
        assert helper.nodes_created[0]["function"] == "test_node"

    def test_get_migration_summary(self):
        """Test getting migration summary"""
        helper = MigrationHelper()
        helper.map_agent_to_node("A1", "n1")

        def node1(state):
            return state

        helper.create_node("n1", node1)

        summary = helper.get_migration_summary()

        assert summary["agents_mapped"] == 1
        assert summary["nodes_created"] == 1
        assert "mappings" in summary
        assert "nodes" in summary


class TestPrintConversionGuide:
    """Test print_conversion_guide function"""

    def test_prints_without_error(self, capsys):
        """Test that print function executes without error"""
        try:
            print_conversion_guide()
            captured = capsys.readouterr()
            assert len(captured.out) > 0
        except Exception as e:
            pytest.fail(f"print_conversion_guide raised exception: {e}")

    def test_output_contains_sections(self, capsys):
        """Test output contains expected sections"""
        print_conversion_guide()
        captured = capsys.readouterr()

        assert "CONVERSION GUIDE" in captured.out
        assert "CONVERSION STEPS" in captured.out
        assert "KEY DIFFERENCES" in captured.out
        assert "BEST PRACTICES" in captured.out
        assert "CONCEPT MAPPINGS" in captured.out


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_agent_list(self):
        """Test with empty agent list"""
        report = generate_conversion_report([], [])
        assert isinstance(report, str)

    def test_none_values_in_comparison(self):
        """Test comparison with None values"""
        result1 = {"task": None, "status": None, "conversation": []}
        result2 = {"task": None, "status": None, "conversation": []}

        comparison = compare_execution_results(result1, result2)
        assert comparison["task_match"] is True

    def test_missing_fields_in_comparison(self):
        """Test comparison with missing fields"""
        result1 = {"task": "Test", "conversation": []}
        result2 = {"task": "Test", "conversation": []}

        comparison = compare_execution_results(result1, result2)
        # Should not raise an error
        assert isinstance(comparison, dict)


@pytest.mark.parametrize("role", list(AgentRole))
def test_all_roles_create_templates(role):
    """Parametrized test for all agent roles"""
    template = create_node_template(role.value.capitalize(), role)
    assert isinstance(template, str)
    assert "def " in template
    assert role.value in template


@pytest.mark.parametrize("num_agents", [1, 3, 5, 10])
def test_routing_with_different_agent_counts(num_agents):
    """Parametrized test for different numbers of agents"""
    agents = [f"agent{i}" for i in range(num_agents)]
    template = create_routing_template(agents)

    assert isinstance(template, str)
    for agent in agents:
        assert agent in template


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
