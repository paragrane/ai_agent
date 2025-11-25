"""
Example: Automatic Conversion from Existing Autogen Code

This example shows how to automatically analyze and convert existing Autogen
code without manually defining agents.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autogen_analyzer import (
    AutogenAnalyzer,
    auto_convert_from_source,
    auto_convert_from_file
)


# Example Autogen code to convert
EXAMPLE_AUTOGEN_CODE = '''
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

# Define configuration
llm_config = {
    "model": "gpt-4",
    "temperature": 0.7
}

# Create data processing agents
data_collector = AssistantAgent(
    name="DataCollector",
    system_message="Collect and prepare data for analysis",
    llm_config=llm_config
)

data_analyzer = AssistantAgent(
    name="DataAnalyzer",
    system_message="Analyze data and extract insights",
    llm_config=llm_config
)

report_generator = AssistantAgent(
    name="ReportGenerator",
    system_message="Generate reports based on analysis",
    llm_config=llm_config
)

quality_checker = AssistantAgent(
    name="QualityChecker",
    system_message="Verify quality of reports",
    llm_config=llm_config
)

user_proxy = UserProxyAgent(
    name="User",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    code_execution_config={"use_docker": False}
)

# Create group chat
agents_list = [user_proxy, data_collector, data_analyzer, report_generator, quality_checker]
groupchat = GroupChat(agents=agents_list, messages=[], max_round=12)
manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)
'''


def example_analyze_code():
    """Example: Analyze Autogen code"""
    print("=" * 70)
    print("ANALYZING AUTOGEN CODE")
    print("=" * 70)
    print()

    analyzer = AutogenAnalyzer()
    analysis = analyzer.analyze_source_code(EXAMPLE_AUTOGEN_CODE)

    print(f"Agents Found: {len(analysis.agents)}")
    print()
    for agent in analysis.agents:
        print(f"  - {agent.name} ({agent.agent_type.value})")
        if agent.system_message:
            print(f"    Message: {agent.system_message[:60]}...")

    print()
    print(f"Suggested Routing: {analysis.suggested_routing.value}")
    print(f"Entry Point: {analysis.entry_point}")
    print(f"Max Rounds: {analysis.max_rounds}")

    if analysis.warnings:
        print("\nWarnings:")
        for warning in analysis.warnings:
            print(f"  - {warning}")


def example_auto_convert():
    """Example: Automatically convert Autogen code to LangGraph"""
    print("\n" + "=" * 70)
    print("AUTO-CONVERTING TO LANGGRAPH")
    print("=" * 70)
    print()

    result = auto_convert_from_source(
        EXAMPLE_AUTOGEN_CODE,
        output_file="data_processing_langgraph.py"
    )

    if "error" in result:
        print(f"Error: {result['error']}")
        return

    print("Conversion Successful!")
    print()
    print(f"Agents Converted: {result['analysis']['agents_found']}")
    print(f"Routing Strategy: {result['analysis']['routing_strategy']}")
    print(f"Output File: {result.get('output_file', 'N/A')}")
    print()

    # Show code preview
    print("Generated Code Preview:")
    print("-" * 70)
    lines = result['code'].split('\n')
    for i, line in enumerate(lines[:30]):
        print(f"{i+1:3d} | {line}")
    print("    | ...")
    print("-" * 70)


def example_custom_processing():
    """Example: Add custom processing logic to agents"""
    print("\n" + "=" * 70)
    print("ADDING CUSTOM PROCESSING LOGIC")
    print("=" * 70)
    print()

    from autogen_to_langgraph_converter import (
        AgentDefinition,
        AgentType,
        WorkflowDefinition,
        RoutingStrategy,
        AutogenToLangGraphConverter
    )

    # Custom processing function
    def custom_data_processor(state, agent):
        """Custom processing for data collection agent"""
        task = state.get("task", "")
        return f"{agent.name} collected data for: {task}"

    # Define agents with custom processing
    agents = [
        AgentDefinition(
            name="DataCollector",
            agent_type=AgentType.ASSISTANT,
            system_message="Collect data",
            processing_function=custom_data_processor
        ),
        AgentDefinition(
            name="DataProcessor",
            agent_type=AgentType.ASSISTANT,
            system_message="Process collected data"
        )
    ]

    workflow = WorkflowDefinition(
        agents=agents,
        routing_strategy=RoutingStrategy.SEQUENTIAL
    )

    converter = AutogenToLangGraphConverter(workflow)
    result = converter.convert()

    print("Conversion with custom processing complete!")
    print()
    print(converter.get_migration_report())


def main():
    """Run all examples"""
    # Example 1: Analyze code
    example_analyze_code()

    # Example 2: Auto-convert
    example_auto_convert()

    # Example 3: Custom processing
    example_custom_processing()

    print("\n" + "=" * 70)
    print("ALL EXAMPLES COMPLETE")
    print("=" * 70)
    print("\nGenerated Files:")
    print("  - data_processing_langgraph.py")


if __name__ == "__main__":
    main()
