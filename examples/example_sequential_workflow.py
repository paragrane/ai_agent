"""
Example: Sequential Workflow Conversion

This example demonstrates converting a simple sequential Autogen workflow
to LangGraph.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autogen_to_langgraph_converter import (
    AgentDefinition,
    AgentType,
    WorkflowDefinition,
    RoutingStrategy,
    AutogenToLangGraphConverter
)


def main():
    """Example of sequential workflow conversion"""

    # Define agents in sequence
    agents = [
        AgentDefinition(
            name="InputProcessor",
            agent_type=AgentType.USER_PROXY,
            description="Processes user input"
        ),
        AgentDefinition(
            name="Analyzer",
            agent_type=AgentType.ASSISTANT,
            system_message="Analyzes the input data",
            description="Performs analysis on processed input"
        ),
        AgentDefinition(
            name="Generator",
            agent_type=AgentType.ASSISTANT,
            system_message="Generates output based on analysis",
            description="Creates final output"
        ),
        AgentDefinition(
            name="Validator",
            agent_type=AgentType.ASSISTANT,
            system_message="Validates the generated output",
            description="Ensures output quality"
        )
    ]

    # Create sequential workflow
    workflow = WorkflowDefinition(
        agents=agents,
        routing_strategy=RoutingStrategy.SEQUENTIAL,
        max_rounds=5
    )

    # Convert
    converter = AutogenToLangGraphConverter(workflow)
    result = converter.convert()

    print("Sequential Workflow Converted!")
    print()
    print(converter.get_migration_report())

    # Export
    converter.export_to_file("sequential_workflow_langgraph.py")
    print("\nGenerated code saved to: sequential_workflow_langgraph.py")


if __name__ == "__main__":
    main()
