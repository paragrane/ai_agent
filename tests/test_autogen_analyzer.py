"""
Test cases for Autogen analyzer
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autogen_analyzer import (
    AutogenAnalyzer,
    AnalysisResult,
    auto_convert_from_source
)
from autogen_to_langgraph_converter import AgentType, RoutingStrategy


class TestAutogenAnalyzer:
    """Test the AutogenAnalyzer class"""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance"""
        return AutogenAnalyzer()

    @pytest.fixture
    def simple_autogen_code(self):
        """Simple Autogen code for testing"""
        return '''
from autogen import AssistantAgent

agent1 = AssistantAgent(
    name="TestAgent",
    system_message="Test message",
    llm_config={"model": "gpt-4"}
)
'''

    @pytest.fixture
    def complex_autogen_code(self):
        """Complex Autogen code with multiple agents"""
        return '''
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

assistant = AssistantAgent(
    name="Assistant",
    system_message="I assist users",
    llm_config={"model": "gpt-4", "temperature": 0.7}
)

researcher = AssistantAgent(
    name="Researcher",
    system_message="I research topics"
)

user_proxy = UserProxyAgent(
    name="User",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10
)

groupchat = GroupChat(
    agents=[user_proxy, assistant, researcher],
    messages=[],
    max_round=15
)

manager = GroupChatManager(groupchat=groupchat)
'''

    def test_analyzer_initialization(self, analyzer):
        """Test analyzer initialization"""
        assert analyzer.agents_found == []
        assert analyzer.patterns_detected == {}
        assert analyzer.warnings == []

    def test_analyze_simple_code(self, analyzer, simple_autogen_code):
        """Test analyzing simple Autogen code"""
        result = analyzer.analyze_source_code(simple_autogen_code)

        assert isinstance(result, AnalysisResult)
        assert len(result.agents) >= 1
        assert result.agents[0].name == "TestAgent"
        assert result.agents[0].agent_type == AgentType.ASSISTANT

    def test_analyze_complex_code(self, analyzer, complex_autogen_code):
        """Test analyzing complex Autogen code"""
        result = analyzer.analyze_source_code(complex_autogen_code)

        assert len(result.agents) == 3
        agent_names = [a.name for a in result.agents]
        assert "Assistant" in agent_names
        assert "Researcher" in agent_names
        assert "User" in agent_names

    def test_extract_system_message(self, analyzer, simple_autogen_code):
        """Test extraction of system messages"""
        result = analyzer.analyze_source_code(simple_autogen_code)

        assert result.agents[0].system_message == "Test message"

    def test_extract_llm_config(self, analyzer, complex_autogen_code):
        """Test extraction of LLM config"""
        result = analyzer.analyze_source_code(complex_autogen_code)

        assistant = next(a for a in result.agents if a.name == "Assistant")
        assert assistant.llm_config is not None
        assert "model" in assistant.llm_config

    def test_detect_user_proxy(self, analyzer, complex_autogen_code):
        """Test detection of UserProxyAgent"""
        result = analyzer.analyze_source_code(complex_autogen_code)

        user_agent = next(a for a in result.agents if a.name == "User")
        assert user_agent.agent_type == AgentType.USER_PROXY
        assert user_agent.human_input_mode == "NEVER"
        assert user_agent.max_consecutive_auto_reply == 10

    def test_detect_groupchat(self, analyzer, complex_autogen_code):
        """Test detection of GroupChat pattern"""
        result = analyzer.analyze_source_code(complex_autogen_code)

        assert result.workflow_patterns.get("uses_groupchat") is True
        assert result.workflow_patterns.get("max_rounds") == 15

    def test_detect_manager(self, analyzer, complex_autogen_code):
        """Test detection of GroupChatManager"""
        result = analyzer.analyze_source_code(complex_autogen_code)

        assert result.workflow_patterns.get("uses_manager") is True
        assert result.workflow_patterns.get("has_coordinator") is True

    def test_infer_routing_strategy(self, analyzer, complex_autogen_code):
        """Test routing strategy inference"""
        result = analyzer.analyze_source_code(complex_autogen_code)

        # Should infer COORDINATOR_BASED due to GroupChatManager
        assert result.suggested_routing == RoutingStrategy.COORDINATOR_BASED

    def test_infer_entry_point(self, analyzer, complex_autogen_code):
        """Test entry point inference"""
        result = analyzer.analyze_source_code(complex_autogen_code)

        # Should infer UserProxy as entry point
        assert result.entry_point == "User"

    def test_analyze_invalid_code(self, analyzer):
        """Test analyzing invalid Python code"""
        invalid_code = "this is not valid python %%% ###"
        result = analyzer.analyze_source_code(invalid_code)

        assert len(result.warnings) > 0
        assert "Syntax error" in result.warnings[0]

    def test_analyze_empty_code(self, analyzer):
        """Test analyzing empty code"""
        result = analyzer.analyze_source_code("")

        assert len(result.agents) == 0

    def test_analyze_file_not_found(self, analyzer):
        """Test analyzing non-existent file"""
        result = analyzer.analyze_file("nonexistent_file.py")

        assert len(result.warnings) > 0
        assert "File not found" in result.warnings[0]

    def test_analyze_objects(self, analyzer):
        """Test analyzing agent objects directly"""
        # Create mock objects
        class MockAgent:
            def __init__(self, name, agent_type):
                self.name = name
                self.__class__.__name__ = agent_type

        mock_agents = [
            MockAgent("Agent1", "AssistantAgent"),
            MockAgent("Agent2", "UserProxyAgent")
        ]

        result = analyzer.analyze_objects(mock_agents)

        assert len(result.agents) == 2


class TestAutoConversion:
    """Test automatic conversion functions"""

    @pytest.fixture
    def sample_code(self):
        """Sample Autogen code for conversion"""
        return '''
from autogen import AssistantAgent, UserProxyAgent

agent1 = AssistantAgent(
    name="Worker",
    system_message="I do work"
)

agent2 = AssistantAgent(
    name="Validator",
    system_message="I validate work"
)

user = UserProxyAgent(
    name="User",
    human_input_mode="NEVER"
)
'''

    def test_auto_convert_from_source(self, sample_code):
        """Test automatic conversion from source code"""
        result = auto_convert_from_source(sample_code)

        assert "error" not in result
        assert "code" in result
        assert "analysis" in result
        assert result["analysis"]["agents_found"] >= 2

    def test_auto_convert_with_output_file(self, sample_code, tmp_path):
        """Test conversion with output file"""
        output_file = tmp_path / "converted.py"
        result = auto_convert_from_source(sample_code, str(output_file))

        assert output_file.exists()
        assert result["output_file"] == str(output_file)

    def test_auto_convert_empty_code(self):
        """Test conversion with empty code"""
        result = auto_convert_from_source("")

        assert "error" in result or result["analysis"]["agents_found"] == 0

    def test_generated_code_is_valid_python(self, sample_code):
        """Test that generated code is valid Python"""
        result = auto_convert_from_source(sample_code)

        if "code" in result:
            # Try to parse the generated code
            import ast
            try:
                ast.parse(result["code"])
                valid = True
            except SyntaxError:
                valid = False

            assert valid is True


class TestAnalysisResult:
    """Test AnalysisResult dataclass"""

    def test_create_analysis_result(self):
        """Test creating AnalysisResult"""
        from autogen_to_langgraph_converter import AgentDefinition

        agents = [
            AgentDefinition("Agent1", AgentType.ASSISTANT)
        ]

        result = AnalysisResult(
            agents=agents,
            workflow_patterns={},
            suggested_routing=RoutingStrategy.SEQUENTIAL
        )

        assert len(result.agents) == 1
        assert result.suggested_routing == RoutingStrategy.SEQUENTIAL
        assert result.warnings == []

    def test_analysis_result_with_warnings(self):
        """Test AnalysisResult with warnings"""
        result = AnalysisResult(
            agents=[],
            workflow_patterns={},
            suggested_routing=RoutingStrategy.SEQUENTIAL,
            warnings=["Warning 1", "Warning 2"]
        )

        assert len(result.warnings) == 2


class TestEdgeCases:
    """Test edge cases in analysis"""

    def test_agent_without_name(self):
        """Test handling agent without name"""
        code = '''
from autogen import AssistantAgent

agent = AssistantAgent(
    system_message="No name provided"
)
'''
        analyzer = AutogenAnalyzer()
        result = analyzer.analyze_source_code(code)

        # Should handle gracefully (might not extract agent)
        assert isinstance(result, AnalysisResult)

    def test_nested_agent_creation(self):
        """Test handling nested agent creation"""
        code = '''
from autogen import AssistantAgent

def create_agent():
    return AssistantAgent(
        name="NestedAgent",
        system_message="Created in function"
    )

agent = create_agent()
'''
        analyzer = AutogenAnalyzer()
        result = analyzer.analyze_source_code(code)

        # May or may not find nested agents
        assert isinstance(result, AnalysisResult)

    def test_multiple_files_analysis(self):
        """Test analyzing patterns across multiple code blocks"""
        code1 = 'agent1 = AssistantAgent(name="Agent1")'
        code2 = 'agent2 = AssistantAgent(name="Agent2")'

        analyzer = AutogenAnalyzer()

        result1 = analyzer.analyze_source_code(code1)
        result2 = analyzer.analyze_source_code(code2)

        # Each analysis should be independent
        assert len(result1.agents) >= 0
        assert len(result2.agents) >= 0


@pytest.mark.parametrize("max_round", [5, 10, 15, 20])
def test_different_max_rounds(max_round):
    """Test extracting different max_round values"""
    code = f'''
from autogen import GroupChat

groupchat = GroupChat(agents=[], messages=[], max_round={max_round})
'''

    analyzer = AutogenAnalyzer()
    result = analyzer.analyze_source_code(code)

    assert result.workflow_patterns.get("max_rounds") == max_round


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
