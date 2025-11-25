"""
Generic Autogen to LangGraph Converter

This module provides a framework-level converter that can transform any Autogen
multi-agent system into an equivalent LangGraph StateGraph implementation.

The converter is configuration-driven and works with any agent setup, not tied
to specific examples.
"""

from typing import Dict, Any, List, Callable, Optional, TypedDict, Type
from dataclasses import dataclass, field
from enum import Enum
import inspect


class AgentType(Enum):
    """Types of agents that can be converted"""
    ASSISTANT = "assistant"
    USER_PROXY = "user_proxy"
    CUSTOM = "custom"


class RoutingStrategy(Enum):
    """Strategies for routing between nodes in LangGraph"""
    SEQUENTIAL = "sequential"  # Fixed order
    CONDITIONAL = "conditional"  # Based on state
    ROUND_ROBIN = "round_robin"  # Cycle through agents
    COORDINATOR_BASED = "coordinator_based"  # Central coordinator decides
    CUSTOM = "custom"  # User-defined logic


@dataclass
class AgentDefinition:
    """
    Generic definition of an Autogen agent for conversion

    This class represents ANY Autogen agent regardless of its specific role
    """
    name: str
    agent_type: AgentType
    system_message: Optional[str] = None
    description: Optional[str] = None

    # Configuration from Autogen agent
    llm_config: Optional[Dict[str, Any]] = None
    human_input_mode: Optional[str] = None
    max_consecutive_auto_reply: Optional[int] = None
    code_execution_config: Optional[Dict[str, Any]] = None

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Custom processing function (optional)
    processing_function: Optional[Callable] = None


@dataclass
class WorkflowDefinition:
    """
    Generic definition of an Autogen workflow for conversion

    This represents the structure of ANY Autogen multi-agent system
    """
    agents: List[AgentDefinition]
    routing_strategy: RoutingStrategy = RoutingStrategy.SEQUENTIAL
    max_rounds: int = 10

    # Custom routing function for CUSTOM strategy
    custom_routing_logic: Optional[Callable] = None

    # Entry point agent
    entry_agent: Optional[str] = None

    # Exit conditions
    exit_conditions: List[Callable] = field(default_factory=list)

    # Workflow metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversionConfig:
    """Configuration for the conversion process"""

    # State field mappings
    state_fields: Dict[str, Type] = field(default_factory=dict)

    # Whether to preserve message history
    preserve_message_history: bool = True

    # Whether to add iteration counter
    add_iteration_counter: bool = True

    # Maximum iterations before forcing exit
    max_iterations: int = 50

    # Custom state schema (if provided)
    custom_state_schema: Optional[Type[TypedDict]] = None

    # Node name transformation function
    node_name_transformer: Optional[Callable[[str], str]] = None


class AutogenToLangGraphConverter:
    """
    Generic converter from Autogen to LangGraph

    This class can convert ANY Autogen multi-agent system to LangGraph,
    regardless of the specific agents or workflow structure.

    Usage:
        # Define your agents
        agents = [
            AgentDefinition(name="Agent1", agent_type=AgentType.ASSISTANT, ...),
            AgentDefinition(name="Agent2", agent_type=AgentType.ASSISTANT, ...),
        ]

        # Define workflow
        workflow = WorkflowDefinition(
            agents=agents,
            routing_strategy=RoutingStrategy.SEQUENTIAL
        )

        # Convert
        converter = AutogenToLangGraphConverter(workflow)
        langgraph_system = converter.convert()
    """

    def __init__(
        self,
        workflow_definition: WorkflowDefinition,
        conversion_config: Optional[ConversionConfig] = None
    ):
        self.workflow = workflow_definition
        self.config = conversion_config or ConversionConfig()
        self.state_schema = None
        self.nodes = {}
        self.routing_function = None

    def convert(self) -> Dict[str, Any]:
        """
        Convert Autogen workflow to LangGraph components

        Returns:
            Dictionary containing:
            - state_schema: TypedDict for state
            - nodes: Dict of node functions
            - routing_function: Function for conditional routing
            - graph_structure: Description of graph edges
            - code: Generated code string (optional)
        """
        # Step 1: Generate state schema
        self.state_schema = self._generate_state_schema()

        # Step 2: Convert agents to nodes
        self.nodes = self._convert_agents_to_nodes()

        # Step 3: Generate routing logic
        self.routing_function = self._generate_routing_logic()

        # Step 4: Define graph structure
        graph_structure = self._define_graph_structure()

        # Step 5: Generate code (optional)
        code = self._generate_code()

        return {
            "state_schema": self.state_schema,
            "nodes": self.nodes,
            "routing_function": self.routing_function,
            "graph_structure": graph_structure,
            "code": code,
            "workflow_definition": self.workflow,
            "conversion_config": self.config
        }

    def _generate_state_schema(self) -> Type[TypedDict]:
        """
        Generate a TypedDict schema for LangGraph state

        This creates a generic state schema based on the workflow definition
        """
        if self.config.custom_state_schema:
            return self.config.custom_state_schema

        # Build state fields dynamically
        state_fields = {
            "task": str,
            "messages": List[Dict[str, str]],
            "current_agent": str,
        }

        # Add iteration counter if configured
        if self.config.add_iteration_counter:
            state_fields["iteration"] = int
            state_fields["max_iterations"] = int

        # Add custom fields from config
        state_fields.update(self.config.state_fields)

        # Add agent-specific fields (generic)
        for agent in self.workflow.agents:
            # Add output field for each agent
            field_name = f"{self._normalize_name(agent.name)}_output"
            state_fields[field_name] = Optional[str]

        # Create TypedDict dynamically
        state_schema = TypedDict(
            "GenericAgentState",
            state_fields
        )

        return state_schema

    def _convert_agents_to_nodes(self) -> Dict[str, Callable]:
        """
        Convert each Autogen agent to a LangGraph node function

        Creates generic node functions that work with any agent configuration
        """
        nodes = {}

        for agent in self.workflow.agents:
            node_name = self._get_node_name(agent.name)
            node_function = self._create_node_function(agent)
            nodes[node_name] = node_function

        return nodes

    def _create_node_function(self, agent: AgentDefinition) -> Callable:
        """
        Create a generic node function for an agent

        The function signature and logic are generic and work with any agent
        """
        agent_name = agent.name
        output_field = f"{self._normalize_name(agent_name)}_output"

        # If agent has custom processing function, use it
        if agent.processing_function:
            processing_func = agent.processing_function
        else:
            # Use default processing
            processing_func = self._default_processing

        def node_function(state: Dict[str, Any]) -> Dict[str, Any]:
            """
            Generic node function for {agent_name}

            This function processes state and returns updated state
            """
            # Add message to history
            messages = state.get("messages", []).copy() if self.config.preserve_message_history else []
            messages.append({
                "agent": agent_name,
                "message": f"{agent_name} processing: {state.get('task', 'No task')}",
                "type": agent.agent_type.value
            })

            # Process using custom or default logic
            output = processing_func(state, agent)

            # Update state
            updated_state = {**state}
            updated_state["messages"] = messages
            updated_state[output_field] = output
            updated_state["current_agent"] = agent_name

            # Increment iteration if enabled
            if self.config.add_iteration_counter:
                updated_state["iteration"] = state.get("iteration", 0) + 1

            return updated_state

        # Set function name for better debugging
        node_function.__name__ = f"{self._normalize_name(agent_name)}_node"

        return node_function

    def _default_processing(self, state: Dict[str, Any], agent: AgentDefinition) -> str:
        """
        Default processing logic when no custom function is provided

        This is a placeholder that can be overridden
        """
        task = state.get("task", "")
        return f"{agent.name} processed: {task}"

    def _generate_routing_logic(self) -> Callable:
        """
        Generate routing logic based on the workflow's routing strategy

        Creates a generic routing function that works with any agent configuration
        """
        strategy = self.workflow.routing_strategy

        if strategy == RoutingStrategy.CUSTOM and self.workflow.custom_routing_logic:
            return self.workflow.custom_routing_logic

        elif strategy == RoutingStrategy.SEQUENTIAL:
            return self._create_sequential_routing()

        elif strategy == RoutingStrategy.CONDITIONAL:
            return self._create_conditional_routing()

        elif strategy == RoutingStrategy.ROUND_ROBIN:
            return self._create_round_robin_routing()

        elif strategy == RoutingStrategy.COORDINATOR_BASED:
            return self._create_coordinator_routing()

        else:
            return self._create_sequential_routing()

    def _create_sequential_routing(self) -> Callable:
        """Create sequential routing through all agents"""
        agent_names = [self._get_node_name(a.name) for a in self.workflow.agents]

        def route(state: Dict[str, Any]) -> str:
            """Route to next agent in sequence"""
            current = state.get("current_agent", "")
            current_node = self._get_node_name(current)

            # Check exit conditions
            if self._should_exit(state):
                return "END"

            # Find next agent
            try:
                current_idx = agent_names.index(current_node)
                next_idx = (current_idx + 1) % len(agent_names)

                # If we've cycled back to start, check iteration
                if next_idx == 0:
                    iteration = state.get("iteration", 0)
                    max_iter = state.get("max_iterations", self.config.max_iterations)
                    if iteration >= max_iter:
                        return "END"

                return agent_names[next_idx]
            except (ValueError, IndexError):
                # Start with first agent
                return agent_names[0] if agent_names else "END"

        return route

    def _create_conditional_routing(self) -> Callable:
        """Create conditional routing based on state"""
        def route(state: Dict[str, Any]) -> str:
            """Route based on state conditions"""
            if self._should_exit(state):
                return "END"

            # Check each agent's output to determine next
            for agent in self.workflow.agents:
                output_field = f"{self._normalize_name(agent.name)}_output"
                if state.get(output_field) is None:
                    return self._get_node_name(agent.name)

            return "END"

        return route

    def _create_round_robin_routing(self) -> Callable:
        """Create round-robin routing"""
        return self._create_sequential_routing()  # Similar to sequential

    def _create_coordinator_routing(self) -> Callable:
        """Create coordinator-based routing"""
        # Find coordinator agent (first agent by default)
        coordinator = self.workflow.agents[0] if self.workflow.agents else None
        coordinator_node = self._get_node_name(coordinator.name) if coordinator else None

        other_agents = [
            self._get_node_name(a.name)
            for a in self.workflow.agents[1:]
        ]

        def route(state: Dict[str, Any]) -> str:
            """Route through coordinator"""
            current = state.get("current_agent", "")

            if self._should_exit(state):
                return "END"

            # If coming from coordinator, go to next worker
            if current == (coordinator.name if coordinator else ""):
                # Find next worker that hasn't been processed
                for agent_node in other_agents:
                    agent_name = agent_node.replace("_node", "")
                    output_field = f"{agent_name}_output"
                    if state.get(output_field) is None:
                        return agent_node
                return "END"
            else:
                # Return to coordinator
                return coordinator_node if coordinator_node else "END"

        return route

    def _should_exit(self, state: Dict[str, Any]) -> bool:
        """Check if workflow should exit"""
        # Check iteration limit
        if self.config.add_iteration_counter:
            iteration = state.get("iteration", 0)
            max_iter = state.get("max_iterations", self.config.max_iterations)
            if iteration >= max_iter:
                return True

        # Check custom exit conditions
        for condition in self.workflow.exit_conditions:
            if condition(state):
                return True

        return False

    def _define_graph_structure(self) -> Dict[str, Any]:
        """Define the structure of the graph (nodes and edges)"""
        structure = {
            "nodes": [self._get_node_name(a.name) for a in self.workflow.agents],
            "entry_point": self._get_entry_point(),
            "edges": self._define_edges(),
            "routing_strategy": self.workflow.routing_strategy.value
        }
        return structure

    def _get_entry_point(self) -> str:
        """Determine the entry point of the graph"""
        if self.workflow.entry_agent:
            return self._get_node_name(self.workflow.entry_agent)

        # Default to first agent
        if self.workflow.agents:
            return self._get_node_name(self.workflow.agents[0].name)

        return "start"

    def _define_edges(self) -> List[Dict[str, str]]:
        """Define edges between nodes based on routing strategy"""
        edges = []

        if self.workflow.routing_strategy == RoutingStrategy.SEQUENTIAL:
            # Sequential edges
            for i in range(len(self.workflow.agents) - 1):
                edges.append({
                    "from": self._get_node_name(self.workflow.agents[i].name),
                    "to": self._get_node_name(self.workflow.agents[i + 1].name),
                    "type": "direct"
                })
            # Last agent to END
            if self.workflow.agents:
                edges.append({
                    "from": self._get_node_name(self.workflow.agents[-1].name),
                    "to": "END",
                    "type": "direct"
                })

        else:
            # Conditional edges - all nodes use routing function
            for agent in self.workflow.agents:
                edges.append({
                    "from": self._get_node_name(agent.name),
                    "to": "ROUTING",
                    "type": "conditional"
                })

        return edges

    def _generate_code(self) -> str:
        """
        Generate complete Python code for the LangGraph implementation

        Returns executable Python code as a string
        """
        code_parts = []

        # Imports
        code_parts.append('''
from typing import TypedDict, Dict, Any, List, Optional, Annotated
from langgraph.graph import StateGraph, END
import operator


''')

        # State schema
        code_parts.append(self._generate_state_schema_code())
        code_parts.append("\n\n")

        # Node functions
        for agent in self.workflow.agents:
            code_parts.append(self._generate_node_code(agent))
            code_parts.append("\n\n")

        # Routing function
        code_parts.append(self._generate_routing_code())
        code_parts.append("\n\n")

        # Graph building function
        code_parts.append(self._generate_graph_builder_code())

        return "".join(code_parts)

    def _generate_state_schema_code(self) -> str:
        """Generate code for state schema"""
        fields = []
        fields.append("    task: str")
        fields.append("    messages: Annotated[List[Dict[str, str]], operator.add]")
        fields.append("    current_agent: str")

        if self.config.add_iteration_counter:
            fields.append("    iteration: int")
            fields.append("    max_iterations: int")

        for agent in self.workflow.agents:
            field_name = f"{self._normalize_name(agent.name)}_output"
            fields.append(f"    {field_name}: Optional[str]")

        return f'''class AgentState(TypedDict):
    """State shared across all agents in the workflow"""
{chr(10).join(fields)}
'''

    def _generate_node_code(self, agent: AgentDefinition) -> str:
        """Generate code for a node function"""
        node_name = self._normalize_name(agent.name)
        output_field = f"{node_name}_output"

        return f'''def {node_name}_node(state: AgentState) -> AgentState:
    """
    Node function for {agent.name}
    {agent.description or ''}
    """
    messages = state.get("messages", [])
    messages.append({{
        "agent": "{agent.name}",
        "message": f"{{state.get('task', '')}} - Processing...",
        "type": "{agent.agent_type.value}"
    }})

    # TODO: Add your agent logic here
    output = f"{agent.name} processed: {{state.get('task', '')}}"

    return {{
        **state,
        "messages": messages,
        "{output_field}": output,
        "current_agent": "{agent.name}",
        "iteration": state.get("iteration", 0) + 1
    }}
'''

    def _generate_routing_code(self) -> str:
        """Generate code for routing function"""
        agent_nodes = [self._get_node_name(a.name) for a in self.workflow.agents]

        return f'''def route_agent(state: AgentState) -> str:
    """Route to the next agent based on workflow strategy"""
    if state.get("iteration", 0) >= state.get("max_iterations", {self.config.max_iterations}):
        return "END"

    # TODO: Implement your routing logic here
    # Current strategy: {self.workflow.routing_strategy.value}

    current = state.get("current_agent", "")
    agent_order = {agent_nodes}

    try:
        current_idx = agent_order.index(current.lower().replace(" ", "_") + "_node")
        next_idx = (current_idx + 1) % len(agent_order)
        if next_idx == 0:
            return "END"
        return agent_order[next_idx]
    except (ValueError, IndexError):
        return agent_order[0] if agent_order else "END"
'''

    def _generate_graph_builder_code(self) -> str:
        """Generate code for building the graph"""
        node_names = [self._get_node_name(a.name) for a in self.workflow.agents]
        entry_point = self._get_entry_point()

        nodes_setup = "\n    ".join([
            f'workflow.add_node("{node}", {node})'
            for node in node_names
        ])

        return f'''def build_graph() -> StateGraph:
    """Build and compile the LangGraph workflow"""
    workflow = StateGraph(AgentState)

    # Add nodes
    {nodes_setup}

    # Set entry point
    workflow.set_entry_point("{entry_point}")

    # Add edges based on routing strategy: {self.workflow.routing_strategy.value}
    # TODO: Customize edges based on your needs

    return workflow.compile()


if __name__ == "__main__":
    graph = build_graph()

    # Example usage
    initial_state = {{
        "task": "Example task",
        "messages": [],
        "current_agent": "",
        "iteration": 0,
        "max_iterations": {self.workflow.max_rounds},
        {", ".join([f'"{self._normalize_name(a.name)}_output": None' for a in self.workflow.agents])}
    }}

    result = graph.invoke(initial_state)
    print("Final state:", result)
'''

    def _get_node_name(self, agent_name: str) -> str:
        """Get the node name for an agent"""
        if self.config.node_name_transformer:
            return self.config.node_name_transformer(agent_name)
        return f"{self._normalize_name(agent_name)}_node"

    def _normalize_name(self, name: str) -> str:
        """Normalize agent name for use in code"""
        return name.lower().replace(" ", "_").replace("-", "_")

    def export_to_file(self, filepath: str) -> None:
        """Export the generated code to a Python file"""
        result = self.convert()
        code = result["code"]

        with open(filepath, 'w') as f:
            f.write(code)

    def get_migration_report(self) -> str:
        """Generate a detailed migration report"""
        report = []
        report.append("=" * 70)
        report.append("AUTOGEN TO LANGGRAPH CONVERSION REPORT")
        report.append("=" * 70)
        report.append("")

        report.append(f"Total Agents: {len(self.workflow.agents)}")
        report.append(f"Routing Strategy: {self.workflow.routing_strategy.value}")
        report.append(f"Max Rounds: {self.workflow.max_rounds}")
        report.append("")

        report.append("Agents Converted:")
        for agent in self.workflow.agents:
            report.append(f"  - {agent.name} ({agent.agent_type.value})")
            report.append(f"    → Node: {self._get_node_name(agent.name)}")

        report.append("")
        report.append("State Fields Generated:")
        if self.state_schema:
            # Get annotations if TypedDict
            annotations = getattr(self.state_schema, '__annotations__', {})
            for field_name, field_type in annotations.items():
                report.append(f"  - {field_name}: {field_type}")

        report.append("")
        report.append("Graph Structure:")
        report.append(f"  Entry Point: {self._get_entry_point()}")
        structure = self._define_graph_structure()
        report.append(f"  Nodes: {len(structure['nodes'])}")
        report.append(f"  Edges: {len(structure['edges'])}")

        report.append("")
        report.append("=" * 70)

        return "\n".join(report)


# Helper function for quick conversion
def convert_autogen_to_langgraph(
    agents: List[AgentDefinition],
    routing_strategy: RoutingStrategy = RoutingStrategy.SEQUENTIAL,
    **kwargs
) -> Dict[str, Any]:
    """
    Quick conversion function

    Args:
        agents: List of agent definitions
        routing_strategy: How to route between agents
        **kwargs: Additional workflow configuration

    Returns:
        Conversion result dictionary
    """
    workflow = WorkflowDefinition(
        agents=agents,
        routing_strategy=routing_strategy,
        **kwargs
    )

    converter = AutogenToLangGraphConverter(workflow)
    return converter.convert()
