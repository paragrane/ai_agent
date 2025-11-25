"""
LangGraph Multi-Agent System - Converted from Autogen
A research assistant system using LangGraph's state machine approach.
"""

from typing import List, Dict, Any, Optional, TypedDict, Annotated
from dataclasses import dataclass
import operator


try:
    from langgraph.graph import StateGraph, END
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
    from langchain_openai import ChatOpenAI
except ImportError:
    # Mock classes for demonstration if langgraph is not installed
    class StateGraph:
        def __init__(self, state_schema):
            self.state_schema = state_schema
            self.nodes = {}
            self.edges = []

        def add_node(self, name: str, func):
            self.nodes[name] = func

        def add_edge(self, from_node: str, to_node: str):
            self.edges.append((from_node, to_node))

        def add_conditional_edges(self, from_node: str, condition_func, mapping: Dict[str, str]):
            pass

        def set_entry_point(self, node: str):
            self.entry_point = node

        def compile(self):
            return self

        def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
            return state

    END = "END"

    class BaseMessage:
        def __init__(self, content: str):
            self.content = content

    class HumanMessage(BaseMessage):
        pass

    class AIMessage(BaseMessage):
        pass

    class ChatOpenAI:
        def __init__(self, model: str, temperature: float):
            self.model = model
            self.temperature = temperature


# State definition for the agent workflow
class AgentState(TypedDict):
    """State shared across all agents in the workflow"""
    task: str
    messages: Annotated[List[Dict[str, str]], operator.add]
    research_data: Optional[str]
    draft_content: Optional[str]
    feedback: Optional[str]
    final_output: Optional[str]
    next_agent: str
    iteration: int
    max_iterations: int


@dataclass
class LangGraphConfig:
    """Configuration for LangGraph multi-agent system"""
    api_key: Optional[str] = None
    model: str = "gpt-4"
    temperature: float = 0.7
    max_iterations: int = 10

    def get_llm(self):
        """Get configured LLM instance"""
        return ChatOpenAI(
            model=self.model,
            temperature=self.temperature
        )


class LangGraphMultiAgentSystem:
    """
    Multi-agent research assistant system using LangGraph.

    This is a conversion of the Autogen system to use LangGraph's
    state machine approach instead of conversational agents.

    Agents are now nodes in a state graph:
    - Coordinator: Entry point, manages workflow
    - Researcher: Gathers information
    - Writer: Creates content
    - Critic: Reviews and provides feedback
    - Finalizer: Completes the workflow
    """

    def __init__(self, config: LangGraphConfig):
        self.config = config
        self.llm = config.get_llm()
        self.graph = self._build_graph()

    def _coordinator_node(self, state: AgentState) -> AgentState:
        """
        Coordinator agent node - analyzes task and routes to appropriate agent
        """
        messages = [{
            "agent": "Coordinator",
            "message": f"Analyzing task: {state['task']}"
        }]

        # Determine next step based on current state
        if state.get("research_data") is None:
            next_agent = "researcher"
            messages.append({
                "agent": "Coordinator",
                "message": "Routing to Researcher for information gathering"
            })
        elif state.get("draft_content") is None:
            next_agent = "writer"
            messages.append({
                "agent": "Coordinator",
                "message": "Routing to Writer for content creation"
            })
        elif state.get("feedback") is None:
            next_agent = "critic"
            messages.append({
                "agent": "Coordinator",
                "message": "Routing to Critic for review"
            })
        else:
            next_agent = "finalizer"
            messages.append({
                "agent": "Coordinator",
                "message": "All steps complete, finalizing output"
            })

        return {
            **state,
            "messages": messages,
            "next_agent": next_agent,
            "iteration": state.get("iteration", 0) + 1
        }

    def _researcher_node(self, state: AgentState) -> AgentState:
        """
        Researcher agent node - gathers and analyzes information
        """
        messages = [
            {
                "agent": "Researcher",
                "message": f"Researching topic: {state['task']}"
            },
            {
                "agent": "Researcher",
                "message": "Gathering relevant information from knowledge base..."
            },
            {
                "agent": "Researcher",
                "message": "Analysis complete. Key findings compiled."
            }
        ]

        research_data = f"Research findings for: {state['task']}\n" \
                       f"- Key concept analysis\n" \
                       f"- Current trends and developments\n" \
                       f"- Important considerations"

        return {
            **state,
            "messages": messages,
            "research_data": research_data,
            "next_agent": "coordinator"
        }

    def _writer_node(self, state: AgentState) -> AgentState:
        """
        Writer agent node - creates content based on research
        """
        messages = [
            {
                "agent": "Writer",
                "message": "Creating content based on research findings..."
            },
            {
                "agent": "Writer",
                "message": "Structuring information logically..."
            },
            {
                "agent": "Writer",
                "message": "Draft complete and ready for review."
            }
        ]

        draft_content = f"Content for: {state['task']}\n\n" \
                       f"Based on research: {state.get('research_data', 'N/A')}\n\n" \
                       f"[Well-structured content incorporating research findings]"

        return {
            **state,
            "messages": messages,
            "draft_content": draft_content,
            "next_agent": "coordinator"
        }

    def _critic_node(self, state: AgentState) -> AgentState:
        """
        Critic agent node - reviews content and provides feedback
        """
        messages = [
            {
                "agent": "Critic",
                "message": "Reviewing draft content..."
            },
            {
                "agent": "Critic",
                "message": "Checking for accuracy, completeness, and clarity..."
            },
            {
                "agent": "Critic",
                "message": "Review complete. Feedback provided."
            }
        ]

        feedback = "Content review feedback:\n" \
                  "- Content is well-structured\n" \
                  "- Research is properly incorporated\n" \
                  "- Clarity is good\n" \
                  "- Ready for finalization"

        return {
            **state,
            "messages": messages,
            "feedback": feedback,
            "next_agent": "coordinator"
        }

    def _finalizer_node(self, state: AgentState) -> AgentState:
        """
        Finalizer node - completes the workflow and prepares final output
        """
        messages = [{
            "agent": "Finalizer",
            "message": "Finalizing output..."
        }, {
            "agent": "Finalizer",
            "message": "Task completed successfully."
        }]

        final_output = f"Final Output for: {state['task']}\n\n" \
                      f"{state.get('draft_content', '')}\n\n" \
                      f"Review: {state.get('feedback', '')}"

        return {
            **state,
            "messages": messages,
            "final_output": final_output,
            "next_agent": "end"
        }

    def _route_agent(self, state: AgentState) -> str:
        """
        Routing function to determine next node in the graph
        """
        next_agent = state.get("next_agent", "coordinator")

        # Check if we've hit max iterations
        if state.get("iteration", 0) >= state.get("max_iterations", 10):
            return "end"

        # Route based on next_agent
        if next_agent == "end":
            return "end"
        return next_agent

    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph state machine

        Graph structure:
        coordinator -> researcher -> coordinator -> writer -> coordinator -> critic -> coordinator -> finalizer -> END
        """
        # Create the graph
        workflow = StateGraph(AgentState)

        # Add nodes for each agent
        workflow.add_node("coordinator", self._coordinator_node)
        workflow.add_node("researcher", self._researcher_node)
        workflow.add_node("writer", self._writer_node)
        workflow.add_node("critic", self._critic_node)
        workflow.add_node("finalizer", self._finalizer_node)

        # Set entry point
        workflow.set_entry_point("coordinator")

        # Add conditional edges based on routing logic
        workflow.add_conditional_edges(
            "coordinator",
            self._route_agent,
            {
                "researcher": "researcher",
                "writer": "writer",
                "critic": "critic",
                "finalizer": "finalizer",
                "end": END
            }
        )

        # Each agent returns to coordinator for routing
        workflow.add_edge("researcher", "coordinator")
        workflow.add_edge("writer", "coordinator")
        workflow.add_edge("critic", "coordinator")
        workflow.add_edge("finalizer", END)

        # Compile the graph
        return workflow.compile()

    def run(self, task: str, max_iterations: int = 10) -> Dict[str, Any]:
        """
        Run the multi-agent system on a task

        Args:
            task: The task description
            max_iterations: Maximum number of iterations

        Returns:
            Dictionary containing the final state and results
        """
        # Initialize state
        initial_state: AgentState = {
            "task": task,
            "messages": [],
            "research_data": None,
            "draft_content": None,
            "feedback": None,
            "final_output": None,
            "next_agent": "coordinator",
            "iteration": 0,
            "max_iterations": max_iterations
        }

        # Execute the graph
        final_state = self.graph.invoke(initial_state)

        # Format results
        return {
            "task": final_state["task"],
            "status": "completed",
            "conversation": final_state["messages"],
            "final_output": final_state.get("final_output", "Task incomplete"),
            "iterations": final_state["iteration"]
        }

    def get_graph_visualization(self) -> str:
        """
        Get a text representation of the graph structure

        Returns:
            String representation of the graph
        """
        return """
LangGraph Multi-Agent Workflow:

    START
      |
      v
[Coordinator] <----+
      |            |
      v            |
  [Routing]        |
      |            |
      +---> [Researcher] ---+
      |                     |
      +---> [Writer] -------+
      |                     |
      +---> [Critic] -------+
      |
      v
 [Finalizer]
      |
      v
     END
"""


class SimpleLangGraphWorkflow:
    """
    Simplified LangGraph workflow for easier comparison with Autogen
    """

    def __init__(self, config: LangGraphConfig):
        self.config = config
        self.workflow = self._create_workflow()

    def _create_workflow(self) -> StateGraph:
        """Create simplified workflow graph"""
        workflow = StateGraph(AgentState)

        # Add sequential nodes
        workflow.add_node("start", self._start_node)
        workflow.add_node("research", self._research_node)
        workflow.add_node("write", self._write_node)
        workflow.add_node("review", self._review_node)
        workflow.add_node("finalize", self._finalize_node)

        # Set entry and create linear flow
        workflow.set_entry_point("start")
        workflow.add_edge("start", "research")
        workflow.add_edge("research", "write")
        workflow.add_edge("write", "review")
        workflow.add_edge("review", "finalize")
        workflow.add_edge("finalize", END)

        return workflow.compile()

    def _start_node(self, state: AgentState) -> AgentState:
        """Start node"""
        return {
            **state,
            "messages": [{"agent": "Coordinator", "message": f"Starting task: {state['task']}"}]
        }

    def _research_node(self, state: AgentState) -> AgentState:
        """Research node"""
        return {
            **state,
            "messages": [{"agent": "Researcher", "message": "Research completed"}],
            "research_data": "Research findings"
        }

    def _write_node(self, state: AgentState) -> AgentState:
        """Write node"""
        return {
            **state,
            "messages": [{"agent": "Writer", "message": "Draft created"}],
            "draft_content": "Draft content"
        }

    def _review_node(self, state: AgentState) -> AgentState:
        """Review node"""
        return {
            **state,
            "messages": [{"agent": "Critic", "message": "Review completed"}],
            "feedback": "Positive feedback"
        }

    def _finalize_node(self, state: AgentState) -> AgentState:
        """Finalize node"""
        return {
            **state,
            "messages": [{"agent": "Finalizer", "message": "Task completed"}],
            "final_output": f"Final output for: {state['task']}"
        }

    def run(self, task: str) -> Dict[str, Any]:
        """Run the workflow"""
        initial_state: AgentState = {
            "task": task,
            "messages": [],
            "research_data": None,
            "draft_content": None,
            "feedback": None,
            "final_output": None,
            "next_agent": "start",
            "iteration": 0,
            "max_iterations": 10
        }

        final_state = self.workflow.invoke(initial_state)

        return {
            "task": final_state["task"],
            "status": "completed",
            "conversation": final_state["messages"],
            "final_output": final_state.get("final_output", "Task incomplete")
        }


def create_langgraph_system(api_key: Optional[str] = None) -> LangGraphMultiAgentSystem:
    """
    Factory function to create a LangGraph multi-agent system

    Args:
        api_key: Optional API key for LLM

    Returns:
        Configured LangGraphMultiAgentSystem
    """
    import os
    config = LangGraphConfig(api_key=api_key or os.getenv("OPENAI_API_KEY"))
    return LangGraphMultiAgentSystem(config)


def create_simple_langgraph_workflow(api_key: Optional[str] = None) -> SimpleLangGraphWorkflow:
    """
    Factory function to create a simple LangGraph workflow

    Args:
        api_key: Optional API key for LLM

    Returns:
        Configured SimpleLangGraphWorkflow
    """
    import os
    config = LangGraphConfig(api_key=api_key or os.getenv("OPENAI_API_KEY"))
    return SimpleLangGraphWorkflow(config)


if __name__ == "__main__":
    # Example usage
    system = create_simple_langgraph_workflow()
    result = system.run("Research and write about AI agent frameworks")

    print("Task:", result["task"])
    print("Status:", result["status"])
    print("\nConversation:")
    for msg in result["conversation"]:
        print(f"[{msg['agent']}]: {msg['message']}")
    print("\nFinal Output:", result["final_output"])

    # Show graph visualization
    full_system = create_langgraph_system()
    print("\n" + full_system.get_graph_visualization())
