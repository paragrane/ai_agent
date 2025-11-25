"""
Test cases for generic Autogen to LangGraph converter
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autogen_to_langgraph_converter import (
    AgentDefinition,
    AgentType,
    WorkflowDefinition,
    RoutingStrategy,
    AutogenToLangGraphConverter,
    ConversionConfig,
    convert_autogen_to_langgraph
)


class TestAgentDefinition:
    """Test AgentDefinition dataclass"""

    def test_create_assistant_agent(self):
        """Test creating an assistant agent definition"""
        agent = AgentDefinition(
            name="TestAgent",
            agent_type=AgentType.ASSISTANT,
            system_message="Test message"
        )

        assert agent.name == "TestAgent"
        assert agent.agent_type == AgentType.ASSISTANT
        assert agent.system_message == "Test message"

    def test_create_user_proxy_agent(self):
        """Test creating a user proxy agent definition"""
        agent = AgentDefinition(
            name="UserProxy",
            agent_type=AgentType.USER_PROXY,
            human_input_mode="NEVER"
        )

        assert agent.name == "UserProxy"
        assert agent.agent_type == AgentType.USER_PROXY
        assert agent.human_input_mode == "NEVER"

    def test_agent_with_custom_processing(self):
        """Test agent with custom processing function"""
        def custom_func(state, agent):
            return "custom output"

        agent = AgentDefinition(
            name="Custom",
            agent_type=AgentType.CUSTOM,
            processing_function=custom_func
        )

        assert agent.processing_function is not None
        assert agent.processing_function({}, agent) == "custom output"


class TestWorkflowDefinition:
    """Test WorkflowDefinition dataclass"""

    def test_create_workflow(self):
        """Test creating a workflow definition"""
        agents = [
            AgentDefinition("Agent1", AgentType.ASSISTANT),
            AgentDefinition("Agent2", AgentType.ASSISTANT)
        ]

        workflow = WorkflowDefinition(
            agents=agents,
            routing_strategy=RoutingStrategy.SEQUENTIAL,
            max_rounds=10
        )

        assert len(workflow.agents) == 2
        assert workflow.routing_strategy == RoutingStrategy.SEQUENTIAL
        assert workflow.max_rounds == 10

    def test_workflow_with_custom_routing(self):
        """Test workflow with custom routing logic"""
        def custom_route(state):
            return "END"

        agents = [AgentDefinition("Agent1", AgentType.ASSISTANT)]
        workflow = WorkflowDefinition(
            agents=agents,
            routing_strategy=RoutingStrategy.CUSTOM,
            custom_routing_logic=custom_route
        )

        assert workflow.custom_routing_logic is not None
        assert workflow.custom_routing_logic({}) == "END"

    def test_workflow_with_exit_conditions(self):
        """Test workflow with exit conditions"""
        def exit_cond(state):
            return state.get("done", False)

        agents = [AgentDefinition("Agent1", AgentType.ASSISTANT)]
        workflow = WorkflowDefinition(
            agents=agents,
            exit_conditions=[exit_cond]
        )

        assert len(workflow.exit_conditions) == 1
        assert workflow.exit_conditions[0]({"done": True}) is True


class TestConversionConfig:
    """Test ConversionConfig"""

    def test_default_config(self):
        """Test default configuration"""
        config = ConversionConfig()

        assert config.preserve_message_history is True
        assert config.add_iteration_counter is True
        assert config.max_iterations == 50

    def test_custom_config(self):
        """Test custom configuration"""
        config = ConversionConfig(
            state_fields={"custom_field": str},
            preserve_message_history=False,
            max_iterations=20
        )

        assert "custom_field" in config.state_fields
        assert config.preserve_message_history is False
        assert config.max_iterations == 20


class TestAutogenToLangGraphConverter:
    """Test the main converter class"""

    @pytest.fixture
    def simple_workflow(self):
        """Create a simple workflow for testing"""
        agents = [
            AgentDefinition("Agent1", AgentType.ASSISTANT),
            AgentDefinition("Agent2", AgentType.ASSISTANT)
        ]
        return WorkflowDefinition(agents=agents, routing_strategy=RoutingStrategy.SEQUENTIAL)

    @pytest.fixture
    def complex_workflow(self):
        """Create a complex workflow for testing"""
        agents = [
            AgentDefinition(
                name="Coordinator",
                agent_type=AgentType.ASSISTANT,
                system_message="Coordinates workflow"
            ),
            AgentDefinition(
                name="Worker1",
                agent_type=AgentType.ASSISTANT,
                system_message="Does work"
            ),
            AgentDefinition(
                name="Worker2",
                agent_type=AgentType.ASSISTANT,
                system_message="Does more work"
            )
        ]
        return WorkflowDefinition(
            agents=agents,
            routing_strategy=RoutingStrategy.COORDINATOR_BASED
        )

    def test_converter_initialization(self, simple_workflow):
        """Test converter initialization"""
        converter = AutogenToLangGraphConverter(simple_workflow)

        assert converter.workflow == simple_workflow
        assert converter.config is not None
        assert converter.state_schema is None  # Not yet generated

    def test_convert_returns_dict(self, simple_workflow):
        """Test that convert returns a dictionary"""
        converter = AutogenToLangGraphConverter(simple_workflow)
        result = converter.convert()

        assert isinstance(result, dict)
        assert "state_schema" in result
        assert "nodes" in result
        assert "routing_function" in result
        assert "graph_structure" in result
        assert "code" in result

    def test_generate_state_schema(self, simple_workflow):
        """Test state schema generation"""
        converter = AutogenToLangGraphConverter(simple_workflow)
        state_schema = converter._generate_state_schema()

        assert state_schema is not None
        # Check that it has required fields
        annotations = getattr(state_schema, '__annotations__', {})
        assert "task" in annotations
        assert "messages" in annotations
        assert "current_agent" in annotations

    def test_convert_agents_to_nodes(self, simple_workflow):
        """Test agent to node conversion"""
        converter = AutogenToLangGraphConverter(simple_workflow)
        converter.state_schema = converter._generate_state_schema()
        nodes = converter._convert_agents_to_nodes()

        assert isinstance(nodes, dict)
        assert len(nodes) == 2
        assert "agent1_node" in nodes
        assert "agent2_node" in nodes

    def test_node_function_execution(self, simple_workflow):
        """Test that generated node functions execute"""
        converter = AutogenToLangGraphConverter(simple_workflow)
        result = converter.convert()

        # Get a node function
        node_func = result["nodes"]["agent1_node"]

        # Create test state
        test_state = {
            "task": "test task",
            "messages": [],
            "current_agent": "",
            "iteration": 0,
            "max_iterations": 10,
            "agent1_output": None,
            "agent2_output": None
        }

        # Execute node
        new_state = node_func(test_state)

        assert isinstance(new_state, dict)
        assert "agent1_output" in new_state
        assert new_state["current_agent"] == "Agent1"
        assert new_state["iteration"] == 1

    def test_sequential_routing(self, simple_workflow):
        """Test sequential routing generation"""
        converter = AutogenToLangGraphConverter(simple_workflow)
        result = converter.convert()

        route_func = result["routing_function"]

        # Test routing through agents
        state1 = {"current_agent": "Agent1", "iteration": 0, "max_iterations": 10}
        next1 = route_func(state1)
        assert next1 == "agent2_node"

        state2 = {"current_agent": "Agent2", "iteration": 1, "max_iterations": 10}
        next2 = route_func(state2)
        assert next2 == "END" or next2 == "agent1_node"

    def test_coordinator_routing(self, complex_workflow):
        """Test coordinator-based routing"""
        converter = AutogenToLangGraphConverter(complex_workflow)
        result = converter.convert()

        route_func = result["routing_function"]
        assert route_func is not None

        # Test routing
        state = {"current_agent": "Coordinator", "iteration": 0, "max_iterations": 15}
        next_agent = route_func(state)
        assert next_agent in ["worker1_node", "worker2_node", "END"]

    def test_code_generation(self, simple_workflow):
        """Test Python code generation"""
        converter = AutogenToLangGraphConverter(simple_workflow)
        result = converter.convert()

        code = result["code"]

        assert isinstance(code, str)
        assert len(code) > 0
        assert "from langgraph.graph import StateGraph" in code
        assert "class AgentState(TypedDict)" in code
        assert "def agent1_node" in code
        assert "def build_graph" in code

    def test_graph_structure_definition(self, simple_workflow):
        """Test graph structure definition"""
        converter = AutogenToLangGraphConverter(simple_workflow)
        result = converter.convert()

        structure = result["graph_structure"]

        assert "nodes" in structure
        assert "entry_point" in structure
        assert "edges" in structure
        assert len(structure["nodes"]) == 2

    def test_export_to_file(self, simple_workflow, tmp_path):
        """Test exporting to file"""
        converter = AutogenToLangGraphConverter(simple_workflow)

        output_file = tmp_path / "test_workflow.py"
        converter.export_to_file(str(output_file))

        assert output_file.exists()
        content = output_file.read_text()
        assert "from langgraph.graph import StateGraph" in content

    def test_migration_report(self, complex_workflow):
        """Test migration report generation"""
        converter = AutogenToLangGraphConverter(complex_workflow)
        converter.convert()  # Generate components first

        report = converter.get_migration_report()

        assert isinstance(report, str)
        assert "CONVERSION REPORT" in report
        assert "Total Agents: 3" in report
        assert "Coordinator" in report

    def test_custom_state_fields(self, simple_workflow):
        """Test adding custom state fields"""
        config = ConversionConfig(
            state_fields={"custom_data": str, "custom_count": int}
        )

        converter = AutogenToLangGraphConverter(simple_workflow, config)
        state_schema = converter._generate_state_schema()

        annotations = getattr(state_schema, '__annotations__', {})
        assert "custom_data" in annotations
        assert "custom_count" in annotations

    def test_exit_conditions(self, simple_workflow):
        """Test exit condition checking"""
        def exit_cond(state):
            return state.get("should_exit", False)

        simple_workflow.exit_conditions.append(exit_cond)

        converter = AutogenToLangGraphConverter(simple_workflow)

        # Test exit condition
        assert converter._should_exit({"should_exit": True}) is True
        assert converter._should_exit({"should_exit": False}) is False

    def test_max_iterations_exit(self, simple_workflow):
        """Test exit on max iterations"""
        config = ConversionConfig(max_iterations=5)
        converter = AutogenToLangGraphConverter(simple_workflow, config)

        # Should exit when at max iterations
        state = {"iteration": 5, "max_iterations": 5}
        assert converter._should_exit(state) is True

        # Should not exit when below max
        state = {"iteration": 3, "max_iterations": 5}
        assert converter._should_exit(state) is False


class TestQuickConversionFunction:
    """Test the quick conversion helper function"""

    def test_quick_conversion(self):
        """Test quick conversion function"""
        agents = [
            AgentDefinition("Agent1", AgentType.ASSISTANT),
            AgentDefinition("Agent2", AgentType.ASSISTANT)
        ]

        result = convert_autogen_to_langgraph(
            agents=agents,
            routing_strategy=RoutingStrategy.SEQUENTIAL
        )

        assert isinstance(result, dict)
        assert "state_schema" in result
        assert "nodes" in result
        assert "code" in result

    def test_quick_conversion_with_kwargs(self):
        """Test quick conversion with additional kwargs"""
        agents = [AgentDefinition("Agent1", AgentType.ASSISTANT)]

        result = convert_autogen_to_langgraph(
            agents=agents,
            routing_strategy=RoutingStrategy.SEQUENTIAL,
            max_rounds=20,
            entry_agent="Agent1"
        )

        assert result["workflow_definition"].max_rounds == 20
        assert result["workflow_definition"].entry_agent == "Agent1"


class TestRoutingStrategies:
    """Test different routing strategies"""

    def test_all_routing_strategies(self):
        """Test all built-in routing strategies"""
        agents = [
            AgentDefinition(f"Agent{i}", AgentType.ASSISTANT)
            for i in range(3)
        ]

        strategies = [
            RoutingStrategy.SEQUENTIAL,
            RoutingStrategy.CONDITIONAL,
            RoutingStrategy.ROUND_ROBIN,
            RoutingStrategy.COORDINATOR_BASED
        ]

        for strategy in strategies:
            workflow = WorkflowDefinition(agents=agents, routing_strategy=strategy)
            converter = AutogenToLangGraphConverter(workflow)
            result = converter.convert()

            assert result["routing_function"] is not None
            assert result["graph_structure"]["routing_strategy"] == strategy.value


@pytest.mark.parametrize("num_agents", [1, 2, 5, 10])
def test_different_agent_counts(num_agents):
    """Test conversion with different numbers of agents"""
    agents = [
        AgentDefinition(f"Agent{i}", AgentType.ASSISTANT)
        for i in range(num_agents)
    ]

    workflow = WorkflowDefinition(agents=agents)
    converter = AutogenToLangGraphConverter(workflow)
    result = converter.convert()

    assert len(result["nodes"]) == num_agents
    assert len(result["graph_structure"]["nodes"]) == num_agents


@pytest.mark.parametrize("strategy", [
    RoutingStrategy.SEQUENTIAL,
    RoutingStrategy.CONDITIONAL,
    RoutingStrategy.COORDINATOR_BASED
])
def test_parametrized_strategies(strategy):
    """Parametrized test for routing strategies"""
    agents = [AgentDefinition(f"Agent{i}", AgentType.ASSISTANT) for i in range(3)]
    workflow = WorkflowDefinition(agents=agents, routing_strategy=strategy)

    converter = AutogenToLangGraphConverter(workflow)
    result = converter.convert()

    assert result is not None
    assert "routing_function" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
