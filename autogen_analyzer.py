"""
Autogen System Analyzer

This module provides utilities to analyze existing Autogen code and extract
agent definitions, workflow patterns, and conversation structures automatically.

This enables automatic conversion of any Autogen system without manual configuration.
"""

from typing import Dict, Any, List, Optional, Tuple
import ast
import inspect
import re
from dataclasses import dataclass
from autogen_to_langgraph_converter import (
    AgentDefinition,
    AgentType,
    WorkflowDefinition,
    RoutingStrategy
)


@dataclass
class AnalysisResult:
    """Result of analyzing an Autogen system"""
    agents: List[AgentDefinition]
    workflow_patterns: Dict[str, Any]
    suggested_routing: RoutingStrategy
    entry_point: Optional[str] = None
    max_rounds: int = 10
    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class AutogenAnalyzer:
    """
    Analyzes Autogen code to extract agent definitions and workflow patterns

    This class can inspect:
    - Python source code files
    - Python objects/instances
    - Running Autogen systems
    """

    def __init__(self):
        self.agents_found = []
        self.patterns_detected = {}
        self.warnings = []

    def analyze_source_code(self, source_code: str) -> AnalysisResult:
        """
        Analyze Python source code containing Autogen agents

        Args:
            source_code: String containing Python code

        Returns:
            AnalysisResult with extracted information
        """
        self.agents_found = []
        self.patterns_detected = {}
        self.warnings = []

        try:
            tree = ast.parse(source_code)
            self._analyze_ast(tree)
        except SyntaxError as e:
            self.warnings.append(f"Syntax error in source code: {e}")

        # Determine routing strategy
        routing = self._infer_routing_strategy()

        return AnalysisResult(
            agents=self.agents_found,
            workflow_patterns=self.patterns_detected,
            suggested_routing=routing,
            entry_point=self._infer_entry_point(),
            max_rounds=self.patterns_detected.get("max_rounds", 10),
            warnings=self.warnings
        )

    def analyze_file(self, filepath: str) -> AnalysisResult:
        """
        Analyze a Python file containing Autogen code

        Args:
            filepath: Path to Python file

        Returns:
            AnalysisResult with extracted information
        """
        try:
            with open(filepath, 'r') as f:
                source_code = f.read()
            return self.analyze_source_code(source_code)
        except FileNotFoundError:
            self.warnings.append(f"File not found: {filepath}")
            return AnalysisResult([], {}, RoutingStrategy.SEQUENTIAL, warnings=self.warnings)

    def analyze_objects(self, objects: List[Any]) -> AnalysisResult:
        """
        Analyze Autogen agent objects directly

        Args:
            objects: List of Autogen agent instances

        Returns:
            AnalysisResult with extracted information
        """
        self.agents_found = []
        self.patterns_detected = {}
        self.warnings = []

        for obj in objects:
            agent_def = self._extract_agent_from_object(obj)
            if agent_def:
                self.agents_found.append(agent_def)

        routing = self._infer_routing_strategy()

        return AnalysisResult(
            agents=self.agents_found,
            workflow_patterns=self.patterns_detected,
            suggested_routing=routing,
            entry_point=self._infer_entry_point(),
            warnings=self.warnings
        )

    def _analyze_ast(self, tree: ast.AST) -> None:
        """Analyze AST to find agent definitions"""
        for node in ast.walk(tree):
            # Look for AssistantAgent or UserProxyAgent instantiations
            if isinstance(node, ast.Call):
                self._analyze_agent_call(node)

            # Look for GroupChat or GroupChatManager
            elif isinstance(node, ast.Assign):
                self._analyze_assignment(node)

    def _analyze_agent_call(self, node: ast.Call) -> None:
        """Analyze a function call that might create an agent"""
        func_name = self._get_function_name(node.func)

        if not func_name:
            return

        # Check if it's an agent creation
        if "AssistantAgent" in func_name:
            agent = self._extract_assistant_agent(node)
            if agent:
                self.agents_found.append(agent)

        elif "UserProxyAgent" in func_name:
            agent = self._extract_user_proxy_agent(node)
            if agent:
                self.agents_found.append(agent)

        elif "GroupChat" in func_name and "Manager" not in func_name:
            self._extract_groupchat_config(node)

        elif "GroupChatManager" in func_name:
            self._extract_manager_config(node)

    def _extract_assistant_agent(self, node: ast.Call) -> Optional[AgentDefinition]:
        """Extract AssistantAgent definition from AST node"""
        name = None
        system_message = None
        llm_config = None

        # Extract keyword arguments
        for keyword in node.keywords:
            if keyword.arg == "name":
                name = self._extract_string_value(keyword.value)
            elif keyword.arg == "system_message":
                system_message = self._extract_string_value(keyword.value)
            elif keyword.arg == "llm_config":
                llm_config = self._extract_dict_value(keyword.value)

        if name:
            return AgentDefinition(
                name=name,
                agent_type=AgentType.ASSISTANT,
                system_message=system_message,
                llm_config=llm_config
            )

        return None

    def _extract_user_proxy_agent(self, node: ast.Call) -> Optional[AgentDefinition]:
        """Extract UserProxyAgent definition from AST node"""
        name = None
        human_input_mode = None
        max_consecutive_auto_reply = None
        code_execution_config = None

        for keyword in node.keywords:
            if keyword.arg == "name":
                name = self._extract_string_value(keyword.value)
            elif keyword.arg == "human_input_mode":
                human_input_mode = self._extract_string_value(keyword.value)
            elif keyword.arg == "max_consecutive_auto_reply":
                max_consecutive_auto_reply = self._extract_number_value(keyword.value)
            elif keyword.arg == "code_execution_config":
                code_execution_config = self._extract_dict_value(keyword.value)

        if name:
            return AgentDefinition(
                name=name,
                agent_type=AgentType.USER_PROXY,
                human_input_mode=human_input_mode,
                max_consecutive_auto_reply=max_consecutive_auto_reply,
                code_execution_config=code_execution_config
            )

        return None

    def _extract_groupchat_config(self, node: ast.Call) -> None:
        """Extract GroupChat configuration"""
        for keyword in node.keywords:
            if keyword.arg == "max_round":
                max_rounds = self._extract_number_value(keyword.value)
                if max_rounds:
                    self.patterns_detected["max_rounds"] = max_rounds
            elif keyword.arg == "agents":
                # Note: This would be a list of agent references
                self.patterns_detected["uses_groupchat"] = True

    def _extract_manager_config(self, node: ast.Call) -> None:
        """Extract GroupChatManager configuration"""
        self.patterns_detected["uses_manager"] = True
        self.patterns_detected["has_coordinator"] = True

    def _analyze_assignment(self, node: ast.Assign) -> None:
        """Analyze variable assignments"""
        if isinstance(node.value, ast.Call):
            self._analyze_agent_call(node.value)

    def _extract_agent_from_object(self, obj: Any) -> Optional[AgentDefinition]:
        """Extract agent definition from an object instance"""
        try:
            # Try to get agent properties
            name = getattr(obj, 'name', None)
            if not name:
                return None

            # Determine agent type
            class_name = obj.__class__.__name__
            if "UserProxy" in class_name:
                agent_type = AgentType.USER_PROXY
            elif "Assistant" in class_name:
                agent_type = AgentType.ASSISTANT
            else:
                agent_type = AgentType.CUSTOM

            # Extract configuration
            system_message = getattr(obj, 'system_message', None)
            llm_config = getattr(obj, 'llm_config', None)
            human_input_mode = getattr(obj, 'human_input_mode', None)
            max_consecutive_auto_reply = getattr(obj, 'max_consecutive_auto_reply', None)

            return AgentDefinition(
                name=name,
                agent_type=agent_type,
                system_message=system_message,
                llm_config=llm_config,
                human_input_mode=human_input_mode,
                max_consecutive_auto_reply=max_consecutive_auto_reply
            )

        except Exception as e:
            self.warnings.append(f"Error extracting agent from object: {e}")
            return None

    def _infer_routing_strategy(self) -> RoutingStrategy:
        """Infer the routing strategy from detected patterns"""
        if self.patterns_detected.get("uses_manager"):
            return RoutingStrategy.COORDINATOR_BASED

        if self.patterns_detected.get("uses_groupchat"):
            return RoutingStrategy.CONDITIONAL

        # Default to sequential
        return RoutingStrategy.SEQUENTIAL

    def _infer_entry_point(self) -> Optional[str]:
        """Infer the entry point agent"""
        if not self.agents_found:
            return None

        # Look for UserProxy as entry point
        for agent in self.agents_found:
            if agent.agent_type == AgentType.USER_PROXY:
                return agent.name

        # Default to first agent
        return self.agents_found[0].name if self.agents_found else None

    def _get_function_name(self, node: ast.AST) -> Optional[str]:
        """Extract function name from AST node"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return node.attr
        return None

    def _extract_string_value(self, node: ast.AST) -> Optional[str]:
        """Extract string value from AST node"""
        if isinstance(node, ast.Constant):
            return str(node.value) if isinstance(node.value, str) else None
        elif isinstance(node, ast.Str):  # Python 3.7 compatibility
            return node.s
        return None

    def _extract_number_value(self, node: ast.AST) -> Optional[int]:
        """Extract number value from AST node"""
        if isinstance(node, ast.Constant):
            return int(node.value) if isinstance(node.value, (int, float)) else None
        elif isinstance(node, ast.Num):  # Python 3.7 compatibility
            return int(node.n)
        return None

    def _extract_dict_value(self, node: ast.AST) -> Optional[Dict[str, Any]]:
        """Extract dictionary value from AST node (simplified)"""
        if isinstance(node, ast.Dict):
            # Basic dictionary extraction
            result = {}
            for key, value in zip(node.keys, node.values):
                key_str = self._extract_string_value(key)
                if key_str:
                    # Try to extract value
                    if isinstance(value, ast.Constant):
                        result[key_str] = value.value
            return result if result else None
        return None


def auto_convert_from_source(source_code: str, output_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Automatically analyze Autogen source code and generate LangGraph conversion

    Args:
        source_code: Source code containing Autogen agents
        output_file: Optional file path to save generated code

    Returns:
        Conversion result dictionary
    """
    from autogen_to_langgraph_converter import AutogenToLangGraphConverter

    # Step 1: Analyze the source code
    analyzer = AutogenAnalyzer()
    analysis = analyzer.analyze_source_code(source_code)

    if not analysis.agents:
        print("Warning: No agents found in source code")
        return {"error": "No agents found", "warnings": analysis.warnings}

    # Step 2: Create workflow definition
    workflow = WorkflowDefinition(
        agents=analysis.agents,
        routing_strategy=analysis.suggested_routing,
        entry_agent=analysis.entry_point,
        max_rounds=analysis.max_rounds
    )

    # Step 3: Convert to LangGraph
    converter = AutogenToLangGraphConverter(workflow)
    result = converter.convert()

    # Step 4: Save to file if requested
    if output_file:
        converter.export_to_file(output_file)
        result["output_file"] = output_file

    # Add analysis info
    result["analysis"] = {
        "agents_found": len(analysis.agents),
        "routing_strategy": analysis.suggested_routing.value,
        "warnings": analysis.warnings
    }

    return result


def auto_convert_from_file(input_file: str, output_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Automatically analyze Autogen file and generate LangGraph conversion

    Args:
        input_file: Path to file containing Autogen code
        output_file: Optional path to save generated LangGraph code

    Returns:
        Conversion result dictionary
    """
    try:
        with open(input_file, 'r') as f:
            source_code = f.read()

        if not output_file:
            # Generate output filename
            output_file = input_file.replace('.py', '_langgraph.py')

        return auto_convert_from_source(source_code, output_file)

    except FileNotFoundError:
        return {"error": f"File not found: {input_file}"}


if __name__ == "__main__":
    # Example usage
    example_code = '''
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

# Create agents
coordinator = AssistantAgent(
    name="Coordinator",
    system_message="You coordinate the workflow",
    llm_config={"model": "gpt-4"}
)

researcher = AssistantAgent(
    name="Researcher",
    system_message="You research topics",
    llm_config={"model": "gpt-4"}
)

user_proxy = UserProxyAgent(
    name="User",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=5
)

# Create group chat
groupchat = GroupChat(
    agents=[user_proxy, coordinator, researcher],
    messages=[],
    max_round=10
)

manager = GroupChatManager(groupchat=groupchat)
'''

    print("Analyzing example Autogen code...\n")
    result = auto_convert_from_source(example_code)

    if "error" not in result:
        print("Analysis successful!")
        print(f"Agents found: {result['analysis']['agents_found']}")
        print(f"Routing strategy: {result['analysis']['routing_strategy']}")
        print("\nGenerated LangGraph code preview:")
        print(result['code'][:500] + "...\n")
    else:
        print(f"Error: {result['error']}")
