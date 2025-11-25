"""
Example: Research Multi-Agent System

This example shows how to use the generic converter with a research-focused
multi-agent system.
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


def create_research_workflow():
    """Create a research-focused multi-agent workflow"""

    # Define agents - these represent ANY research agents, not specific to one use case
    agents = [
        AgentDefinition(
            name="Coordinator",
            agent_type=AgentType.ASSISTANT,
            system_message="Coordinates the research workflow and delegates tasks",
            description="Central coordinator that manages the research process"
        ),
        AgentDefinition(
            name="Researcher",
            agent_type=AgentType.ASSISTANT,
            system_message="Conducts research and gathers information",
            description="Research specialist that collects and analyzes data"
        ),
        AgentDefinition(
            name="Writer",
            agent_type=AgentType.ASSISTANT,
            system_message="Creates written content based on research",
            description="Content creator that synthesizes research into documents"
        ),
        AgentDefinition(
            name="Reviewer",
            agent_type=AgentType.ASSISTANT,
            system_message="Reviews content for quality and accuracy",
            description="Quality assurance reviewer"
        )
    ]

    # Define workflow with coordinator-based routing
    workflow = WorkflowDefinition(
        agents=agents,
        routing_strategy=RoutingStrategy.COORDINATOR_BASED,
        entry_agent="Coordinator",
        max_rounds=15
    )

    return workflow


def main():
    print("=" * 70)
    print("RESEARCH MULTI-AGENT SYSTEM CONVERSION")
    print("=" * 70)
    print()

    # Create workflow
    workflow = create_research_workflow()

    # Configure conversion
    config = ConversionConfig(
        state_fields={
            "research_findings": str,
            "draft_document": str,
            "review_feedback": str
        },
        preserve_message_history=True,
        max_iterations=20
    )

    # Convert to LangGraph
    converter = AutogenToLangGraphConverter(workflow, config)
    result = converter.convert()

    # Display results
    print("Conversion Complete!")
    print()
    print(converter.get_migration_report())
    print()

    # Save generated code
    output_file = "research_workflow_langgraph.py"
    converter.export_to_file(output_file)
    print(f"Generated LangGraph code saved to: {output_file}")
    print()

    # Show code preview
    print("Code Preview:")
    print("-" * 70)
    print(result["code"][:1000])
    print("...")
    print("-" * 70)


if __name__ == "__main__":
    main()
