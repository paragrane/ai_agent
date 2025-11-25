"""
Example usage demonstrating both Autogen and LangGraph implementations
"""

from autogen_multiagent import create_simple_workflow
from langgraph_multiagent import create_simple_langgraph_workflow
from conversion_utils import (
    compare_execution_results,
    print_conversion_guide,
    ConversionGuide
)


def example_autogen():
    """Example using Autogen multi-agent system"""
    print("=" * 60)
    print("AUTOGEN MULTI-AGENT EXAMPLE")
    print("=" * 60)

    # Create workflow (uses mock implementation for demonstration)
    workflow = create_simple_workflow(api_key="demo-key")

    # Process a task
    task = "Research and write about AI agent frameworks"
    print(f"\nTask: {task}\n")

    result = workflow.process_task(task)

    # Display results
    print(f"Status: {result['status']}")
    print(f"\nConversation ({len(result['conversation'])} messages):")
    for msg in result['conversation']:
        print(f"  [{msg['agent']}]: {msg['message']}")

    print(f"\nFinal Output:\n{result['final_output']}")
    print()


def example_langgraph():
    """Example using LangGraph multi-agent system"""
    print("=" * 60)
    print("LANGGRAPH MULTI-AGENT EXAMPLE")
    print("=" * 60)

    # Create workflow (uses mock implementation for demonstration)
    workflow = create_simple_langgraph_workflow(api_key="demo-key")

    # Run the same task
    task = "Research and write about AI agent frameworks"
    print(f"\nTask: {task}\n")

    result = workflow.run(task)

    # Display results
    print(f"Status: {result['status']}")
    print(f"\nConversation ({len(result['conversation'])} messages):")
    for msg in result['conversation']:
        print(f"  [{msg['agent']}]: {msg['message']}")

    print(f"\nFinal Output:\n{result['final_output']}")
    print()


def example_comparison():
    """Example comparing both implementations"""
    print("=" * 60)
    print("COMPARING BOTH IMPLEMENTATIONS")
    print("=" * 60)

    # Create both workflows
    autogen = create_simple_workflow(api_key="demo-key")
    langgraph = create_simple_langgraph_workflow(api_key="demo-key")

    # Run same task
    task = "Analyze quantum computing trends"
    print(f"\nTask: {task}\n")

    autogen_result = autogen.process_task(task)
    langgraph_result = langgraph.run(task)

    # Compare results
    comparison = compare_execution_results(autogen_result, langgraph_result)

    print("Comparison Results:")
    print(f"  Task Match: {comparison['task_match']}")
    print(f"  Status Match: {comparison['status_match']}")
    print(f"  Conversation Lengths:")
    print(f"    - Autogen: {comparison['conversation_length']['autogen']}")
    print(f"    - LangGraph: {comparison['conversation_length']['langgraph']}")
    print(f"  Agents Involved:")
    print(f"    - Autogen: {comparison['agents_involved']['autogen']}")
    print(f"    - LangGraph: {comparison['agents_involved']['langgraph']}")
    print(f"  Has Final Output:")
    print(f"    - Autogen: {comparison['has_final_output']['autogen']}")
    print(f"    - LangGraph: {comparison['has_final_output']['langgraph']}")
    print()


def example_conversion_guide():
    """Example showing conversion guide usage"""
    print("=" * 60)
    print("CONVERSION GUIDE")
    print("=" * 60)
    print()

    guide = ConversionGuide()

    # Show conversion steps
    print("Conversion Steps:")
    for i, step in enumerate(guide.get_conversion_steps(), 1):
        print(f"{i}. {step}")

    print("\n" + "=" * 60)
    print("KEY DIFFERENCES")
    print("=" * 60)

    differences = guide.get_key_differences()
    for category, diff in differences.items():
        print(f"\n{category}:")
        print(f"  Autogen:   {diff['Autogen']}")
        print(f"  LangGraph: {diff['LangGraph']}")

    print("\n" + "=" * 60)
    print("BEST PRACTICES")
    print("=" * 60)
    print()

    for i, practice in enumerate(guide.get_best_practices(), 1):
        print(f"{i}. {practice}")
    print()


def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("AUTOGEN TO LANGGRAPH CONVERSION EXAMPLES")
    print("=" * 60)
    print("\nThis script demonstrates:")
    print("1. Autogen multi-agent implementation")
    print("2. LangGraph multi-agent implementation")
    print("3. Comparison between both")
    print("4. Conversion guide and utilities")
    print()

    # Run examples
    example_autogen()
    input("\nPress Enter to continue...")

    example_langgraph()
    input("\nPress Enter to continue...")

    example_comparison()
    input("\nPress Enter to continue...")

    example_conversion_guide()

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Review the code in autogen_multiagent.py and langgraph_multiagent.py")
    print("2. Run the test suite: pytest -v")
    print("3. Explore conversion_utils.py for migration helpers")
    print("4. Read the README.md for detailed documentation")
    print()


if __name__ == "__main__":
    main()
