# Autogen to LangGraph Multi-Agent Conversion

A comprehensive guide and implementation for converting Autogen multi-agent systems to LangGraph's state machine approach, complete with examples and extensive test coverage.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture Comparison](#architecture-comparison)
- [Conversion Guide](#conversion-guide)
- [Examples](#examples)
- [Testing](#testing)
- [Documentation](#documentation)

## 🎯 Overview

This project demonstrates how to convert multi-agent systems from Autogen's conversational paradigm to LangGraph's state machine approach. It includes:

- **Working implementations** of both Autogen and LangGraph multi-agent systems
- **Conversion utilities** to help migrate your code
- **Comprehensive test suite** with 100+ test cases
- **Detailed documentation** and best practices

## ✨ Features

- ✅ **Complete Autogen implementation** with coordinator, researcher, writer, and critic agents
- ✅ **Equivalent LangGraph implementation** using StateGraph
- ✅ **Conversion utilities** for mapping agents to nodes
- ✅ **Side-by-side comparison** tools
- ✅ **100+ test cases** covering both implementations
- ✅ **Integration tests** comparing outputs
- ✅ **Detailed documentation** and migration guide

## 📦 Installation

### Prerequisites

- Python 3.8+
- pip or conda

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd ai_agent

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
- pyautogen>=0.2.0
- langgraph>=0.2.0
- langchain>=0.1.0
- pytest>=7.4.0
- python-dotenv>=1.0.0
```

## 🚀 Quick Start

### Autogen Implementation

```python
from autogen_multiagent import create_simple_workflow

# Create workflow
workflow = create_simple_workflow(api_key="your-api-key")

# Process a task
result = workflow.process_task("Research and write about AI agent frameworks")

print(f"Status: {result['status']}")
print(f"Output: {result['final_output']}")
```

### LangGraph Implementation

```python
from langgraph_multiagent import create_simple_langgraph_workflow

# Create workflow
workflow = create_simple_langgraph_workflow(api_key="your-api-key")

# Run the same task
result = workflow.run("Research and write about AI agent frameworks")

print(f"Status: {result['status']}")
print(f"Output: {result['final_output']}")
```

### Compare Both Implementations

```python
from conversion_utils import compare_execution_results
from autogen_multiagent import create_simple_workflow
from langgraph_multiagent import create_simple_langgraph_workflow

# Create both workflows
autogen = create_simple_workflow(api_key="your-api-key")
langgraph = create_simple_langgraph_workflow(api_key="your-api-key")

# Run same task
task = "Research quantum computing"
autogen_result = autogen.process_task(task)
langgraph_result = langgraph.run(task)

# Compare results
comparison = compare_execution_results(autogen_result, langgraph_result)
print(comparison)
```

## 🏗️ Architecture Comparison

### Autogen Architecture

```
┌─────────────────┐
│  UserProxyAgent │
└────────┬────────┘
         │
    ┌────▼─────┐
    │GroupChat │
    │ Manager  │
    └────┬─────┘
         │
    ┌────▼──────────────────────┐
    │   Conversational Agents   │
    ├──────────┬────────┬───────┤
    │Coordinator│Researcher│Writer│
    └──────────┴────────┴───────┘
          Messages Flow
```

### LangGraph Architecture

```
        START
          │
          ▼
    ┌──────────┐
    │Coordinator│◄─────┐
    └─────┬────┘      │
          │           │
     ┌────▼──────┐    │
     │  Routing  │    │
     └─┬───┬───┬─┘    │
       │   │   │      │
   ┌───▼┐ ┌▼─┐ ┌▼──┐ │
   │Rsrch││Wrt││Crtc│─┘
   └────┘ └──┘ └───┘
          │
          ▼
       [Finalizer]
          │
          ▼
         END
```

### Key Differences

| Aspect | Autogen | LangGraph |
|--------|---------|-----------|
| **Architecture** | Conversational agents | State machine |
| **Communication** | Direct message passing | Shared state updates |
| **Workflow Control** | GroupChatManager | Conditional edges + routing |
| **State Management** | Conversation history | Explicit state schema |
| **Execution** | Reactive (message-driven) | Proactive (graph traversal) |

## 📚 Conversion Guide

### Step-by-Step Process

1. **Define State Schema**
   ```python
   from typing import TypedDict

   class AgentState(TypedDict):
       task: str
       messages: List[Dict[str, str]]
       # Add your state fields
   ```

2. **Convert Agents to Nodes**
   ```python
   def researcher_node(state: AgentState) -> AgentState:
       # Agent logic here
       return {**state, "research_data": "..."}
   ```

3. **Create StateGraph**
   ```python
   from langgraph.graph import StateGraph

   workflow = StateGraph(AgentState)
   workflow.add_node("researcher", researcher_node)
   ```

4. **Add Routing Logic**
   ```python
   def route_agent(state: AgentState) -> str:
       # Routing logic
       return "next_agent"

   workflow.add_conditional_edges("coordinator", route_agent, {...})
   ```

5. **Compile and Execute**
   ```python
   graph = workflow.compile()
   result = graph.invoke(initial_state)
   ```

### Mapping Guide

| Autogen Concept | LangGraph Concept | Notes |
|----------------|-------------------|-------|
| `AssistantAgent` | Node function | Each agent becomes a state transformation function |
| `GroupChat` | `StateGraph` | Conversation becomes a state machine |
| `GroupChatManager` | Conditional edges | Chat management becomes routing logic |
| `system_message` | Node logic | Agent instructions embedded in node functions |
| Conversation history | `state.messages` | History stored in shared state |

### Best Practices

1. ✅ Keep state schema simple and flat
2. ✅ Use TypedDict for type checking
3. ✅ Create small, focused node functions
4. ✅ Test nodes independently
5. ✅ Add iteration limits to prevent infinite loops
6. ✅ Document the graph structure
7. ✅ Use meaningful node names
8. ✅ Handle edge cases in routing logic

## 💡 Examples

### Example 1: Simple Sequential Workflow

```python
from langgraph_multiagent import SimpleLangGraphWorkflow, LangGraphConfig

config = LangGraphConfig(api_key="your-key")
workflow = SimpleLangGraphWorkflow(config)

result = workflow.run("Analyze machine learning trends")
print(result["final_output"])
```

### Example 2: Complex Multi-Agent System

```python
from langgraph_multiagent import LangGraphMultiAgentSystem, LangGraphConfig

config = LangGraphConfig(api_key="your-key", max_iterations=20)
system = LangGraphMultiAgentSystem(config)

result = system.run("Research and write a comprehensive report on AI ethics")

# View conversation
for msg in result["conversation"]:
    print(f"[{msg['agent']}]: {msg['message']}")
```

### Example 3: Using Conversion Utilities

```python
from conversion_utils import (
    ConversionGuide,
    create_node_template,
    AgentRole,
    print_conversion_guide
)

# Print complete conversion guide
print_conversion_guide()

# Generate node template
template = create_node_template("Analyzer", AgentRole.ANALYST)
print(template)

# Get conversion steps
guide = ConversionGuide()
for step in guide.get_conversion_steps():
    print(step)
```

## 🧪 Testing

### Run All Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_autogen_multiagent.py

# Run with verbose output
pytest -v
```

### Test Structure

```
tests/
├── __init__.py
├── test_autogen_multiagent.py       # Autogen implementation tests
├── test_langgraph_multiagent.py     # LangGraph implementation tests
├── test_integration.py              # Integration & comparison tests
└── test_conversion_utils.py         # Utility function tests
```

### Test Coverage

- ✅ **100+ test cases** across all modules
- ✅ **Unit tests** for individual components
- ✅ **Integration tests** comparing implementations
- ✅ **Parametrized tests** for various scenarios
- ✅ **Edge case handling**

### Example Test Output

```bash
$ pytest -v

tests/test_autogen_multiagent.py::TestAgentConfig::test_default_config PASSED
tests/test_autogen_multiagent.py::TestSimpleAutogenWorkflow::test_process_task PASSED
tests/test_langgraph_multiagent.py::TestLangGraphMultiAgentSystem::test_run PASSED
tests/test_integration.py::TestBothImplementations::test_both_complete_task PASSED
...

==================== 100 passed in 2.34s ====================
```

## 📖 Documentation

### Module Documentation

#### `autogen_multiagent.py`

Main Autogen implementation with:
- `AutogenMultiAgentSystem`: Full multi-agent system
- `SimpleAutogenWorkflow`: Simplified workflow for testing
- `AgentConfig`: Configuration dataclass
- Factory functions for easy instantiation

#### `langgraph_multiagent.py`

LangGraph conversion with:
- `LangGraphMultiAgentSystem`: Full state machine implementation
- `SimpleLangGraphWorkflow`: Simplified linear workflow
- `LangGraphConfig`: Configuration dataclass
- `AgentState`: TypedDict for state schema

#### `conversion_utils.py`

Utilities including:
- `ConversionGuide`: Step-by-step migration guide
- `compare_execution_results()`: Compare outputs
- `create_node_template()`: Generate node code
- `create_routing_template()`: Generate routing code
- `MigrationHelper`: Track conversion progress

### API Reference

See inline documentation in each module for detailed API information.

### Additional Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Autogen Documentation](https://microsoft.github.io/autogen/)
- [Conversion Guide (detailed)](docs/CONVERSION_GUIDE.md)

## 🔧 Configuration

### Environment Variables

Create a `.env` file:

```env
OPENAI_API_KEY=your-api-key-here
```

### Configuration Options

```python
# Autogen Config
config = AgentConfig(
    api_key="your-key",
    model="gpt-4",           # or "gpt-3.5-turbo"
    temperature=0.7          # 0.0 to 1.0
)

# LangGraph Config
config = LangGraphConfig(
    api_key="your-key",
    model="gpt-4",
    temperature=0.7,
    max_iterations=10        # Prevent infinite loops
)
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📝 License

[MIT License](LICENSE)

## 🙏 Acknowledgments

- [Microsoft Autogen](https://github.com/microsoft/autogen)
- [LangChain & LangGraph](https://github.com/langchain-ai/langgraph)
- Community contributors

## 📧 Contact

For questions or issues, please open a GitHub issue.

---

**Happy Converting! 🚀**
