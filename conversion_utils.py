"""
Utility functions and helpers for converting Autogen to LangGraph
"""

from typing import Dict, Any, List, Callable
from dataclasses import dataclass
from enum import Enum


class AgentRole(Enum):
    """Common agent roles in multi-agent systems"""
    COORDINATOR = "coordinator"
    RESEARCHER = "researcher"
    WRITER = "writer"
    CRITIC = "critic"
    EXECUTOR = "executor"
    ANALYST = "analyst"


@dataclass
class ConversionMapping:
    """
    Mapping between Autogen and LangGraph concepts

    Autogen Concept          -> LangGraph Concept
    ==========================================
    AssistantAgent          -> Node function in StateGraph
    UserProxyAgent          -> Entry point or external input
    GroupChat               -> StateGraph with multiple nodes
    GroupChatManager        -> Conditional edges and routing logic
    Agent.system_message    -> Node-specific logic/prompts
    Agent conversation      -> State updates through the graph
    max_consecutive_replies -> Iteration count in state
    """

    autogen_concept: str
    langgraph_concept: str
    description: str


# Conversion mappings
CONVERSION_MAPPINGS = [
    ConversionMapping(
        "AssistantAgent",
        "Node Function",
        "Each Autogen agent becomes a node function that processes and updates state"
    ),
    ConversionMapping(
        "GroupChat",
        "StateGraph",
        "The group chat becomes a state graph connecting all agent nodes"
    ),
    ConversionMapping(
        "GroupChatManager",
        "Conditional Edges",
        "Chat management becomes routing logic through conditional edges"
    ),
    ConversionMapping(
        "Agent Messages",
        "State Updates",
        "Messages are stored in state and passed between nodes"
    ),
    ConversionMapping(
        "Conversation History",
        "State.messages",
        "History is maintained in the shared state object"
    ),
]


class ConversionGuide:
    """
    Guide for converting Autogen multi-agent systems to LangGraph
    """

    @staticmethod
    def get_conversion_steps() -> List[str]:
        """
        Get step-by-step conversion guide

        Returns:
            List of conversion steps
        """
        return [
            "1. Define State Schema: Create a TypedDict representing shared state across agents",
            "2. Convert Agents to Nodes: Transform each AssistantAgent into a node function",
            "3. Extract Agent Logic: Move system_message content into node function logic",
            "4. Create StateGraph: Initialize a StateGraph with your state schema",
            "5. Add Nodes: Add each converted agent as a node in the graph",
            "6. Define Routing: Create routing logic (replaces GroupChatManager)",
            "7. Add Edges: Connect nodes with edges (conditional or direct)",
            "8. Set Entry Point: Define where the graph execution begins",
            "9. Compile Graph: Call compile() to create the executable graph",
            "10. Execute: Call invoke() with initial state to run the workflow"
        ]

    @staticmethod
    def get_key_differences() -> Dict[str, Dict[str, str]]:
        """
        Get key differences between Autogen and LangGraph

        Returns:
            Dictionary of differences
        """
        return {
            "Architecture": {
                "Autogen": "Conversational agents that communicate via messages",
                "LangGraph": "State machine with nodes that transform shared state"
            },
            "Agent Communication": {
                "Autogen": "Agents send messages to each other directly",
                "LangGraph": "Agents update shared state; routing determines next agent"
            },
            "Workflow Control": {
                "Autogen": "GroupChatManager decides next speaker",
                "LangGraph": "Conditional edges and routing functions control flow"
            },
            "State Management": {
                "Autogen": "Maintained in conversation history",
                "LangGraph": "Explicitly defined in state schema"
            },
            "Execution Model": {
                "Autogen": "Reactive - agents respond to messages",
                "LangGraph": "Proactive - graph executes through defined paths"
            }
        }

    @staticmethod
    def get_best_practices() -> List[str]:
        """
        Get best practices for conversion

        Returns:
            List of best practices
        """
        return [
            "Keep state schema simple and flat when possible",
            "Use TypedDict for state to get type checking benefits",
            "Create small, focused node functions that do one thing well",
            "Use conditional edges for dynamic routing between agents",
            "Store all important data in state, not in closures",
            "Consider using operator.add for message accumulation in state",
            "Test each node function independently before integration",
            "Use clear naming for nodes that reflects their purpose",
            "Document the graph structure and flow",
            "Consider adding iteration limits to prevent infinite loops"
        ]


def autogen_to_langgraph_state(
    agents: List[str],
    additional_fields: Dict[str, type] = None
) -> type:
    """
    Generate a basic state schema for LangGraph from Autogen agent list

    Args:
        agents: List of agent names
        additional_fields: Additional fields to include in state

    Returns:
        TypedDict class for state schema

    Example:
        >>> from typing import TypedDict
        >>> StateClass = autogen_to_langgraph_state(
        ...     agents=["researcher", "writer", "critic"],
        ...     additional_fields={"draft": str, "feedback": str}
        ... )
    """
    from typing import TypedDict, List as TList, Optional

    fields = {
        "task": str,
        "messages": TList[Dict[str, str]],
        "next_agent": str,
        "iteration": int,
    }

    if additional_fields:
        fields.update(additional_fields)

    return TypedDict("GeneratedState", fields)


def create_node_template(agent_name: str, agent_role: AgentRole) -> str:
    """
    Generate a node function template for a given agent

    Args:
        agent_name: Name of the agent
        agent_role: Role of the agent

    Returns:
        Python code string for the node function
    """
    template = f'''
def {agent_name.lower()}_node(state: AgentState) -> AgentState:
    """
    {agent_name} agent node - {agent_role.value}

    This node represents the {agent_name} agent from the Autogen system.
    """
    # Add your agent logic here
    messages = [{{
        "agent": "{agent_name}",
        "message": "Processing task: {{state['task']}}"
    }}]

    # Determine next agent
    next_agent = "coordinator"  # or routing logic

    # Update state
    return {{
        **state,
        "messages": messages,
        "next_agent": next_agent
    }}
'''
    return template


def create_routing_template(agent_names: List[str]) -> str:
    """
    Generate a routing function template

    Args:
        agent_names: List of agent names in the system

    Returns:
        Python code string for the routing function
    """
    cases = "\n        ".join([
        f'"{name.lower()}": "{name.lower()}",' for name in agent_names
    ])

    template = f'''
def route_agent(state: AgentState) -> str:
    """
    Routing function to determine next node in the graph
    """
    next_agent = state.get("next_agent", "coordinator")

    # Add your routing logic here
    if state.get("iteration", 0) >= state.get("max_iterations", 10):
        return "end"

    # Route to specific agents based on state
    routing_map = {{
        {cases}
        "end": "end"
    }}

    return routing_map.get(next_agent, "end")
'''
    return template


def compare_execution_results(
    autogen_result: Dict[str, Any],
    langgraph_result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compare results from Autogen and LangGraph implementations

    Args:
        autogen_result: Result dictionary from Autogen execution
        langgraph_result: Result dictionary from LangGraph execution

    Returns:
        Comparison dictionary with differences and similarities
    """
    comparison = {
        "task_match": autogen_result.get("task") == langgraph_result.get("task"),
        "status_match": autogen_result.get("status") == langgraph_result.get("status"),
        "conversation_length": {
            "autogen": len(autogen_result.get("conversation", [])),
            "langgraph": len(langgraph_result.get("conversation", []))
        },
        "agents_involved": {
            "autogen": set(msg.get("agent") for msg in autogen_result.get("conversation", [])),
            "langgraph": set(msg.get("agent") for msg in langgraph_result.get("conversation", []))
        },
        "has_final_output": {
            "autogen": "final_output" in autogen_result,
            "langgraph": "final_output" in langgraph_result
        }
    }

    return comparison


def generate_conversion_report(
    autogen_agents: List[str],
    langgraph_nodes: List[str]
) -> str:
    """
    Generate a conversion report

    Args:
        autogen_agents: List of Autogen agent names
        langgraph_nodes: List of LangGraph node names

    Returns:
        Formatted conversion report
    """
    report = f"""
Autogen to LangGraph Conversion Report
======================================

Original Autogen Agents ({len(autogen_agents)}):
{chr(10).join(f"  - {agent}" for agent in autogen_agents)}

Converted LangGraph Nodes ({len(langgraph_nodes)}):
{chr(10).join(f"  - {node}" for node in langgraph_nodes)}

Conversion Status:
  ✓ Agents converted to nodes
  ✓ State schema defined
  ✓ Routing logic implemented
  ✓ Graph compiled

Next Steps:
  1. Test each node function independently
  2. Verify routing logic
  3. Compare outputs with original Autogen system
  4. Optimize based on results
"""
    return report


class MigrationHelper:
    """Helper class for migrating from Autogen to LangGraph"""

    def __init__(self):
        self.agents_mapped = []
        self.nodes_created = []

    def map_agent_to_node(self, agent_name: str, node_name: str):
        """Record agent to node mapping"""
        self.agents_mapped.append({
            "agent": agent_name,
            "node": node_name
        })

    def create_node(self, node_name: str, node_function: Callable):
        """Record node creation"""
        self.nodes_created.append({
            "name": node_name,
            "function": node_function.__name__
        })

    def get_migration_summary(self) -> Dict[str, Any]:
        """Get migration summary"""
        return {
            "agents_mapped": len(self.agents_mapped),
            "nodes_created": len(self.nodes_created),
            "mappings": self.agents_mapped,
            "nodes": self.nodes_created
        }


def print_conversion_guide():
    """Print the complete conversion guide"""
    guide = ConversionGuide()

    print("=" * 60)
    print("AUTOGEN TO LANGGRAPH CONVERSION GUIDE")
    print("=" * 60)

    print("\n### CONVERSION STEPS ###\n")
    for step in guide.get_conversion_steps():
        print(step)

    print("\n### KEY DIFFERENCES ###\n")
    for category, differences in guide.get_key_differences().items():
        print(f"{category}:")
        print(f"  Autogen:   {differences['Autogen']}")
        print(f"  LangGraph: {differences['LangGraph']}")
        print()

    print("\n### BEST PRACTICES ###\n")
    for i, practice in enumerate(guide.get_best_practices(), 1):
        print(f"{i}. {practice}")

    print("\n### CONCEPT MAPPINGS ###\n")
    for mapping in CONVERSION_MAPPINGS:
        print(f"{mapping.autogen_concept} -> {mapping.langgraph_concept}")
        print(f"  {mapping.description}")
        print()


if __name__ == "__main__":
    print_conversion_guide()
