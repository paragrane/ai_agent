# Generic Autogen to LangGraph Converter

A powerful, framework-level tool for converting **ANY** Autogen multi-agent system to LangGraph's state machine approach. Configuration-driven, automatic, and extensible.

## 🎯 Overview

This project provides a **generic conversion framework** that can transform any Autogen multi-agent system into an equivalent LangGraph implementation, regardless of the specific agents or workflow structure.

### Key Principles

- ✅ **Generic, not specific**: Works with ANY agent configuration, not tied to examples
- ✅ **Configuration-driven**: Define agents and routing, get LangGraph code
- ✅ **Automatic analysis**: Can analyze existing Autogen code and convert automatically
- ✅ **Extensible**: Support custom routing logic and processing functions
- ✅ **Production-ready**: Full test suite with 100+ tests

## ✨ Features

### Core Converter
- **`AutogenToLangGraphConverter`**: Main conversion engine
- **`AgentDefinition`**: Generic agent representation
- **`WorkflowDefinition`**: Generic workflow specification
- **Multiple routing strategies**: Sequential, Conditional, Coordinator-based, Custom
- **Code generation**: Produces executable Python code
- **State schema generation**: Automatic TypedDict creation

### Automatic Analysis
- **`AutogenAnalyzer`**: Analyzes existing Autogen code
- **AST parsing**: Extracts agent definitions from Python code
- **Pattern detection**: Identifies GroupChat, Manager patterns
- **Auto-conversion**: `auto_convert_from_file()` function

### Examples & Documentation
- Multiple examples showing different use cases
- Comprehensive test suite (100+ tests)
- Detailed conversion reports
- Best practices guide

## 📦 Installation

```bash
# Clone repository
git clone <repository-url>
cd ai_agent

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Quick Start

### Method 1: Manual Configuration

Define your agents and convert:

```python
from autogen_to_langgraph_converter import (
    AgentDefinition,
    AgentType,
    WorkflowDefinition,
    RoutingStrategy,
    AutogenToLangGraphConverter
)

# Define your agents (any agents you want!)
agents = [
    AgentDefinition(
        name="Coordinator",
        agent_type=AgentType.ASSISTANT,
        system_message="Coordinates the workflow"
    ),
    AgentDefinition(
        name="Specialist1",
        agent_type=AgentType.ASSISTANT,
        system_message="Handles specific tasks"
    ),
    AgentDefinition(
        name="Specialist2",
        agent_type=AgentType.ASSISTANT,
        system_message="Handles other tasks"
    )
]

# Define workflow structure
workflow = WorkflowDefinition(
    agents=agents,
    routing_strategy=RoutingStrategy.COORDINATOR_BASED,
    max_rounds=15
)

# Convert to LangGraph
converter = AutogenToLangGraphConverter(workflow)
result = converter.convert()

# Export generated code
converter.export_to_file("my_langgraph_system.py")

# View conversion report
print(converter.get_migration_report())
```

### Method 2: Automatic Conversion

Analyze existing Autogen code and convert automatically:

```python
from autogen_analyzer import auto_convert_from_file

# Automatically analyze and convert your Autogen code
result = auto_convert_from_file(
    input_file="my_autogen_system.py",
    output_file="my_langgraph_system.py"
)

print(f"Converted {result['analysis']['agents_found']} agents")
print(f"Routing strategy: {result['analysis']['routing_strategy']}")
```

### Method 3: Analyze Then Customize

```python
from autogen_analyzer import AutogenAnalyzer
from autogen_to_langgraph_converter import AutogenToLangGraphConverter, WorkflowDefinition

# Step 1: Analyze existing code
analyzer = AutogenAnalyzer()
analysis = analyzer.analyze_file("my_autogen_system.py")

print(f"Found {len(analysis.agents)} agents")
print(f"Suggested routing: {analysis.suggested_routing}")

# Step 2: Customize if needed
workflow = WorkflowDefinition(
    agents=analysis.agents,
    routing_strategy=analysis.suggested_routing,
    max_rounds=20  # Override default
)

# Step 3: Convert
converter = AutogenToLangGraphConverter(workflow)
converter.export_to_file("customized_output.py")
```

## 🏗️ Architecture

### Generic Conversion Flow

```
┌─────────────────────┐
│  Autogen System     │
│  (Any Structure)    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  AgentDefinition    │  ◄── Generic agent representation
│  (Name, Type, etc)  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ WorkflowDefinition  │  ◄── Generic workflow structure
│ (Agents, Routing)   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Autogen To LangGraph│  ◄── Generic converter
│      Converter      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  LangGraph System   │  ◄── Generated StateGraph
│    (StateGraph)     │
└─────────────────────┘
```

### Key Components

| Component | Purpose | Generic? |
|-----------|---------|----------|
| `AgentDefinition` | Represents any Autogen agent | ✅ Yes |
| `WorkflowDefinition` | Describes any workflow structure | ✅ Yes |
| `AutogenToLangGraphConverter` | Core conversion engine | ✅ Yes |
| `RoutingStrategy` | Enum of routing patterns | ✅ Yes |
| `AutogenAnalyzer` | Analyzes any Autogen code | ✅ Yes |

## 📚 Routing Strategies

The converter supports multiple routing strategies:

### 1. Sequential
Agents execute in fixed order:
```python
RoutingStrategy.SEQUENTIAL
# A → B → C → END
```

### 2. Conditional
Route based on state conditions:
```python
RoutingStrategy.CONDITIONAL
# Coordinator checks state → routes to appropriate agent
```

### 3. Coordinator-Based
Central coordinator delegates to workers:
```python
RoutingStrategy.COORDINATOR_BASED
# Coordinator ↔ Worker1
#            ↔ Worker2
#            ↔ Worker3
```

### 4. Custom
Define your own routing logic:
```python
def my_routing_logic(state):
    if state.get("priority") == "high":
        return "express_handler_node"
    return "standard_handler_node"

workflow = WorkflowDefinition(
    agents=agents,
    routing_strategy=RoutingStrategy.CUSTOM,
    custom_routing_logic=my_routing_logic
)
```

## 💡 Examples

### Example 1: Data Processing Pipeline

```python
agents = [
    AgentDefinition("DataCollector", AgentType.ASSISTANT),
    AgentDefinition("DataProcessor", AgentType.ASSISTANT),
    AgentDefinition("DataValidator", AgentType.ASSISTANT)
]

workflow = WorkflowDefinition(agents=agents, routing_strategy=RoutingStrategy.SEQUENTIAL)
converter = AutogenToLangGraphConverter(workflow)
converter.export_to_file("data_pipeline.py")
```

### Example 2: Research Team

```python
agents = [
    AgentDefinition("Coordinator", AgentType.ASSISTANT,
                   system_message="Manages research workflow"),
    AgentDefinition("Researcher", AgentType.ASSISTANT,
                   system_message="Conducts research"),
    AgentDefinition("Analyst", AgentType.ASSISTANT,
                   system_message="Analyzes findings"),
    AgentDefinition("Writer", AgentType.ASSISTANT,
                   system_message="Writes reports")
]

workflow = WorkflowDefinition(
    agents=agents,
    routing_strategy=RoutingStrategy.COORDINATOR_BASED
)
```

### Example 3: Custom Processing

```python
def custom_processor(state, agent):
    """Custom processing logic for specific agent"""
    data = state.get("input_data", "")
    return f"{agent.name} processed: {data.upper()}"

agents = [
    AgentDefinition(
        "CustomAgent",
        AgentType.ASSISTANT,
        processing_function=custom_processor
    )
]
```

See `examples/` directory for complete examples:
- `example_research_agents.py` - Research workflow
- `example_sequential_workflow.py` - Sequential pipeline
- `example_auto_conversion.py` - Automatic conversion
- `example_custom_routing.py` - Custom routing strategies

## 🧪 Testing

Run the comprehensive test suite:

```bash
# All tests
pytest

# Specific test file
pytest tests/test_generic_converter.py -v

# With coverage
pytest --cov=. --cov-report=html
```

### Test Coverage

- ✅ `test_generic_converter.py` - Core converter tests (50+ tests)
- ✅ `test_autogen_analyzer.py` - Analyzer tests (40+ tests)
- ✅ `test_autogen_multiagent.py` - Example implementations
- ✅ `test_langgraph_multiagent.py` - Example conversions
- ✅ `test_integration.py` - Integration tests
- ✅ `test_conversion_utils.py` - Utility tests

## 📖 API Reference

### Core Classes

#### `AgentDefinition`
```python
AgentDefinition(
    name: str,                              # Agent name
    agent_type: AgentType,                  # ASSISTANT, USER_PROXY, or CUSTOM
    system_message: Optional[str] = None,   # System message/prompt
    description: Optional[str] = None,      # Description
    llm_config: Optional[Dict] = None,      # LLM configuration
    processing_function: Optional[Callable] = None  # Custom processing
)
```

#### `WorkflowDefinition`
```python
WorkflowDefinition(
    agents: List[AgentDefinition],          # List of agents
    routing_strategy: RoutingStrategy,       # How to route between agents
    max_rounds: int = 10,                   # Maximum iterations
    custom_routing_logic: Optional[Callable] = None,  # Custom routing function
    entry_agent: Optional[str] = None,      # Starting agent
    exit_conditions: List[Callable] = []    # Early exit conditions
)
```

#### `AutogenToLangGraphConverter`
```python
converter = AutogenToLangGraphConverter(
    workflow_definition: WorkflowDefinition,
    conversion_config: Optional[ConversionConfig] = None
)

# Main methods
result = converter.convert()                 # Returns conversion result dict
converter.export_to_file(filepath)          # Export to Python file
report = converter.get_migration_report()    # Get detailed report
```

#### `ConversionConfig`
```python
ConversionConfig(
    state_fields: Dict[str, Type] = {},     # Custom state fields
    preserve_message_history: bool = True,   # Keep message history
    add_iteration_counter: bool = True,      # Add iteration tracking
    max_iterations: int = 50                 # Max iterations before exit
)
```

### Helper Functions

```python
# Quick conversion
result = convert_autogen_to_langgraph(agents, routing_strategy)

# Automatic conversion from file
result = auto_convert_from_file("input.py", "output.py")

# Automatic conversion from code string
result = auto_convert_from_source(code_string, "output.py")
```

## 🔧 Advanced Usage

### Custom State Fields

```python
config = ConversionConfig(
    state_fields={
        "user_id": str,
        "session_data": Dict,
        "priority_level": int
    }
)

converter = AutogenToLangGraphConverter(workflow, config)
```

### Exit Conditions

```python
def exit_on_error(state):
    return "error" in state.get("status", "")

def exit_on_completion(state):
    return state.get("all_tasks_done", False)

workflow = WorkflowDefinition(
    agents=agents,
    exit_conditions=[exit_on_error, exit_on_completion]
)
```

### Node Name Transformation

```python
def transform_name(agent_name):
    return f"custom_{agent_name.lower()}_processor"

config = ConversionConfig(
    node_name_transformer=transform_name
)
```

## 📊 Conversion Report

The converter generates detailed migration reports:

```
======================================================================
AUTOGEN TO LANGGRAPH CONVERSION REPORT
======================================================================

Total Agents: 4
Routing Strategy: coordinator_based
Max Rounds: 15

Agents Converted:
  - Coordinator (assistant)
    → Node: coordinator_node
  - Worker1 (assistant)
    → Node: worker1_node
  - Worker2 (assistant)
    → Node: worker2_node
  - Validator (assistant)
    → Node: validator_node

State Fields Generated:
  - task: <class 'str'>
  - messages: typing.List[typing.Dict[str, str]]
  - current_agent: <class 'str'>
  - iteration: <class 'int'>
  - coordinator_output: typing.Union[str, NoneType]
  - worker1_output: typing.Union[str, NoneType]
  - worker2_output: typing.Union[str, NoneType]
  - validator_output: typing.Union[str, NoneType]

Graph Structure:
  Entry Point: coordinator_node
  Nodes: 4
  Edges: 4

======================================================================
```

## 🤝 Contributing

We welcome contributions! The framework is designed to be extensible:

1. Add new routing strategies in `autogen_to_langgraph_converter.py`
2. Enhance the analyzer in `autogen_analyzer.py`
3. Add examples in `examples/`
4. Add tests for new features

## 📝 License

MIT License

## 🙏 Acknowledgments

- [Microsoft Autogen](https://github.com/microsoft/autogen) - Original framework
- [LangGraph](https://github.com/langchain-ai/langgraph) - Target framework
- Community contributors

---

## 🎓 Tutorial

### Step-by-Step: Converting Your First System

1. **Define your agents:**
```python
from autogen_to_langgraph_converter import AgentDefinition, AgentType

my_agents = [
    AgentDefinition("Agent1", AgentType.ASSISTANT, system_message="First agent"),
    AgentDefinition("Agent2", AgentType.ASSISTANT, system_message="Second agent")
]
```

2. **Create a workflow:**
```python
from autogen_to_langgraph_converter import WorkflowDefinition, RoutingStrategy

my_workflow = WorkflowDefinition(
    agents=my_agents,
    routing_strategy=RoutingStrategy.SEQUENTIAL
)
```

3. **Convert:**
```python
from autogen_to_langgraph_converter import AutogenToLangGraphConverter

converter = AutogenToLangGraphConverter(my_workflow)
result = converter.convert()
```

4. **Export and use:**
```python
converter.export_to_file("my_converted_system.py")
print(converter.get_migration_report())
```

That's it! Your Autogen system is now converted to LangGraph.

---

**Ready to convert? Start with the examples in `examples/` or run `python examples/example_auto_conversion.py` to see it in action!**
