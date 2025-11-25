"""
Example: Custom Routing Strategy

This example demonstrates how to define custom routing logic when converting
Autogen to LangGraph.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autogen_to_langgraph_converter import (
    AgentDefinition,
    AgentType,
    WorkflowDefinition,
    RoutingStrategy,
    AutogenToLangGraphConverter,
    ConversionConfig
)
from typing import Dict, Any


def custom_routing_logic(state: Dict[str, Any]) -> str:
    """
    Custom routing function

    This function decides which agent to route to based on the state.
    It can implement ANY logic you need.
    """
    # Check iteration limit
    if state.get("iteration", 0) >= state.get("max_iterations", 20):
        return "END"

    current_agent = state.get("current_agent", "")

    # Custom logic: Route based on task complexity
    task = state.get("task", "")

    # If task mentions "urgent", skip validator
    if "urgent" in task.lower():
        if current_agent == "Processor":
            return "finalizer_node"
        elif current_agent == "Finalizer":
            return "END"
        else:
            return "processor_node"

    # Normal flow: Processor -> Validator -> Finalizer
    if not current_agent or current_agent == "":
        return "processor_node"
    elif current_agent == "Processor":
        return "validator_node"
    elif current_agent == "Validator":
        # Check if validation passed
        validator_output = state.get("validator_output", "")
        if "approved" in validator_output.lower():
            return "finalizer_node"
        else:
            # Reprocess
            return "processor_node"
    elif current_agent == "Finalizer":
        return "END"

    return "END"


def priority_based_routing(state: Dict[str, Any]) -> str:
    """
    Priority-based routing example

    Routes based on task priority level
    """
    if state.get("iteration", 0) >= state.get("max_iterations", 15):
        return "END"

    priority = state.get("priority", "normal")
    current = state.get("current_agent", "")

    # High priority tasks go through fast track
    if priority == "high":
        if current == "":
            return "fast_processor_node"
        elif current == "FastProcessor":
            return "quick_validator_node"
        elif current == "QuickValidator":
            return "END"

    # Normal priority uses standard workflow
    if current == "":
        return "standard_processor_node"
    elif current == "StandardProcessor":
        return "detailed_validator_node"
    elif current == "DetailedValidator":
        return "quality_checker_node"
    elif current == "QualityChecker":
        return "END"

    return "END"


def main():
    print("=" * 70)
    print("CUSTOM ROUTING STRATEGY EXAMPLES")
    print("=" * 70)
    print()

    # Example 1: Complex conditional routing
    print("Example 1: Complex Conditional Routing")
    print("-" * 70)

    agents1 = [
        AgentDefinition(
            name="Processor",
            agent_type=AgentType.ASSISTANT,
            system_message="Process incoming requests"
        ),
        AgentDefinition(
            name="Validator",
            agent_type=AgentType.ASSISTANT,
            system_message="Validate processed results"
        ),
        AgentDefinition(
            name="Finalizer",
            agent_type=AgentType.ASSISTANT,
            system_message="Finalize and output results"
        )
    ]

    workflow1 = WorkflowDefinition(
        agents=agents1,
        routing_strategy=RoutingStrategy.CUSTOM,
        custom_routing_logic=custom_routing_logic,
        max_rounds=10
    )

    converter1 = AutogenToLangGraphConverter(workflow1)
    result1 = converter1.convert()

    print(converter1.get_migration_report())
    converter1.export_to_file("custom_routing_workflow.py")
    print("Generated: custom_routing_workflow.py")
    print()

    # Example 2: Priority-based routing
    print("\nExample 2: Priority-Based Routing")
    print("-" * 70)

    agents2 = [
        AgentDefinition(
            name="FastProcessor",
            agent_type=AgentType.ASSISTANT,
            system_message="Fast processing for high-priority tasks"
        ),
        AgentDefinition(
            name="QuickValidator",
            agent_type=AgentType.ASSISTANT,
            system_message="Quick validation for high-priority"
        ),
        AgentDefinition(
            name="StandardProcessor",
            agent_type=AgentType.ASSISTANT,
            system_message="Standard processing"
        ),
        AgentDefinition(
            name="DetailedValidator",
            agent_type=AgentType.ASSISTANT,
            system_message="Detailed validation"
        ),
        AgentDefinition(
            name="QualityChecker",
            agent_type=AgentType.ASSISTANT,
            system_message="Final quality check"
        )
    ]

    # Add custom state field for priority
    config2 = ConversionConfig(
        state_fields={"priority": str},
        max_iterations=15
    )

    workflow2 = WorkflowDefinition(
        agents=agents2,
        routing_strategy=RoutingStrategy.CUSTOM,
        custom_routing_logic=priority_based_routing,
        max_rounds=15
    )

    converter2 = AutogenToLangGraphConverter(workflow2, config2)
    result2 = converter2.convert()

    print(converter2.get_migration_report())
    converter2.export_to_file("priority_routing_workflow.py")
    print("Generated: priority_routing_workflow.py")
    print()

    # Example 3: State-based routing with exit conditions
    print("\nExample 3: State-Based Routing with Exit Conditions")
    print("-" * 70)

    def exit_on_error(state: Dict[str, Any]) -> bool:
        """Exit if error detected"""
        return "error" in state.get("messages", [])[-1].get("message", "").lower()

    def exit_on_success(state: Dict[str, Any]) -> bool:
        """Exit if all outputs are successful"""
        return all(
            state.get(f"{agent.name.lower()}_output") is not None
            for agent in agents1
        )

    workflow3 = WorkflowDefinition(
        agents=agents1,
        routing_strategy=RoutingStrategy.CONDITIONAL,
        exit_conditions=[exit_on_error, exit_on_success],
        max_rounds=5
    )

    converter3 = AutogenToLangGraphConverter(workflow3)
    result3 = converter3.convert()

    print(converter3.get_migration_report())
    print()

    print("=" * 70)
    print("Custom routing examples complete!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("1. Custom routing functions give full control over workflow")
    print("2. State can include any fields you need for routing decisions")
    print("3. Exit conditions can be added for early termination")
    print("4. Multiple routing strategies can be combined")


if __name__ == "__main__":
    main()
