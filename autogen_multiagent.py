"""
Autogen Multi-Agent System Example
A research assistant system with multiple specialized agents.
"""

from typing import List, Dict, Any, Optional
import os
from dataclasses import dataclass


try:
    from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
except ImportError:
    # Mock classes for demonstration if autogen is not installed
    class AssistantAgent:
        def __init__(self, name: str, llm_config: Dict[str, Any], system_message: str):
            self.name = name
            self.llm_config = llm_config
            self.system_message = system_message

    class UserProxyAgent:
        def __init__(self, name: str, human_input_mode: str, max_consecutive_auto_reply: int, code_execution_config: Dict[str, Any]):
            self.name = name
            self.human_input_mode = human_input_mode
            self.max_consecutive_auto_reply = max_consecutive_auto_reply
            self.code_execution_config = code_execution_config

    class GroupChat:
        def __init__(self, agents: List[Any], messages: List[str], max_round: int):
            self.agents = agents
            self.messages = messages
            self.max_round = max_round

    class GroupChatManager:
        def __init__(self, groupchat: GroupChat, llm_config: Dict[str, Any]):
            self.groupchat = groupchat
            self.llm_config = llm_config


@dataclass
class AgentConfig:
    """Configuration for agent initialization"""
    api_key: Optional[str] = None
    model: str = "gpt-4"
    temperature: float = 0.7

    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration dictionary"""
        config = {
            "model": self.model,
            "temperature": self.temperature,
        }
        if self.api_key:
            config["api_key"] = self.api_key
        return config


class AutogenMultiAgentSystem:
    """
    A multi-agent research assistant system using Autogen.

    Agents:
    - Coordinator: Manages the workflow and delegates tasks
    - Researcher: Gathers and analyzes information
    - Writer: Creates content based on research
    - Critic: Reviews and provides feedback
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self.llm_config = config.get_llm_config()

        # Initialize agents
        self.coordinator = self._create_coordinator()
        self.researcher = self._create_researcher()
        self.writer = self._create_writer()
        self.critic = self._create_critic()
        self.user_proxy = self._create_user_proxy()

        # Create group chat
        self.agents = [
            self.user_proxy,
            self.coordinator,
            self.researcher,
            self.writer,
            self.critic
        ]

    def _create_coordinator(self) -> AssistantAgent:
        """Create the coordinator agent"""
        system_message = """You are a Coordinator agent. Your role is to:
1. Understand the user's request
2. Break it down into tasks
3. Delegate to appropriate agents (Researcher, Writer, Critic)
4. Ensure the workflow progresses smoothly
5. Synthesize final results

Always start by analyzing the request and creating a plan."""

        return AssistantAgent(
            name="Coordinator",
            llm_config=self.llm_config,
            system_message=system_message
        )

    def _create_researcher(self) -> AssistantAgent:
        """Create the researcher agent"""
        system_message = """You are a Researcher agent. Your role is to:
1. Gather relevant information on topics
2. Analyze and synthesize findings
3. Provide well-structured research summaries
4. Cite sources when available

Focus on accuracy and thoroughness."""

        return AssistantAgent(
            name="Researcher",
            llm_config=self.llm_config,
            system_message=system_message
        )

    def _create_writer(self) -> AssistantAgent:
        """Create the writer agent"""
        system_message = """You are a Writer agent. Your role is to:
1. Create clear, engaging content
2. Structure information logically
3. Adapt tone to the audience
4. Incorporate research findings

Focus on clarity and readability."""

        return AssistantAgent(
            name="Writer",
            llm_config=self.llm_config,
            system_message=system_message
        )

    def _create_critic(self) -> AssistantAgent:
        """Create the critic agent"""
        system_message = """You are a Critic agent. Your role is to:
1. Review content for quality
2. Check accuracy and completeness
3. Suggest improvements
4. Provide constructive feedback

Be thorough but constructive."""

        return AssistantAgent(
            name="Critic",
            llm_config=self.llm_config,
            system_message=system_message
        )

    def _create_user_proxy(self) -> UserProxyAgent:
        """Create the user proxy agent"""
        return UserProxyAgent(
            name="User",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=0,
            code_execution_config={"use_docker": False}
        )

    def create_group_chat(self, max_round: int = 10) -> GroupChat:
        """Create a group chat with all agents"""
        return GroupChat(
            agents=self.agents,
            messages=[],
            max_round=max_round
        )

    def run(self, task: str, max_round: int = 10) -> Dict[str, Any]:
        """
        Run the multi-agent system on a task

        Args:
            task: The task description
            max_round: Maximum number of conversation rounds

        Returns:
            Dictionary containing the conversation history and results
        """
        groupchat = self.create_group_chat(max_round=max_round)
        manager = GroupChatManager(
            groupchat=groupchat,
            llm_config=self.llm_config
        )

        # Simulate the conversation flow
        # In real implementation, this would use autogen's chat mechanism
        results = {
            "task": task,
            "agents": [agent.name for agent in self.agents],
            "max_round": max_round,
            "status": "completed",
            "conversation": []
        }

        return results


class SimpleAutogenWorkflow:
    """
    Simplified autogen workflow for easier testing and conversion
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self.llm_config = config.get_llm_config()
        self.conversation_history: List[Dict[str, str]] = []

    def add_message(self, agent: str, message: str):
        """Add a message to conversation history"""
        self.conversation_history.append({
            "agent": agent,
            "message": message
        })

    def process_task(self, task: str) -> Dict[str, Any]:
        """
        Process a task through the agent workflow

        Workflow:
        1. Coordinator receives task
        2. Researcher gathers information
        3. Writer creates content
        4. Critic reviews
        5. Coordinator finalizes
        """
        self.conversation_history = []

        # Step 1: Coordinator analyzes task
        self.add_message("Coordinator", f"Analyzing task: {task}")
        self.add_message("Coordinator", "Breaking down into subtasks...")

        # Step 2: Researcher gathers info
        self.add_message("Researcher", "Gathering relevant information...")
        self.add_message("Researcher", "Research complete. Key findings compiled.")

        # Step 3: Writer creates content
        self.add_message("Writer", "Creating content based on research...")
        self.add_message("Writer", "Draft complete.")

        # Step 4: Critic reviews
        self.add_message("Critic", "Reviewing content...")
        self.add_message("Critic", "Feedback provided. Suggesting improvements.")

        # Step 5: Writer revises
        self.add_message("Writer", "Incorporating feedback...")
        self.add_message("Writer", "Final version ready.")

        # Step 6: Coordinator finalizes
        self.add_message("Coordinator", "Task completed successfully.")

        return {
            "task": task,
            "status": "completed",
            "conversation": self.conversation_history,
            "final_output": f"Completed research and content creation for: {task}"
        }

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the conversation history"""
        return self.conversation_history


def create_autogen_system(api_key: Optional[str] = None) -> AutogenMultiAgentSystem:
    """
    Factory function to create an autogen multi-agent system

    Args:
        api_key: Optional API key for LLM

    Returns:
        Configured AutogenMultiAgentSystem
    """
    config = AgentConfig(api_key=api_key or os.getenv("OPENAI_API_KEY"))
    return AutogenMultiAgentSystem(config)


def create_simple_workflow(api_key: Optional[str] = None) -> SimpleAutogenWorkflow:
    """
    Factory function to create a simple autogen workflow

    Args:
        api_key: Optional API key for LLM

    Returns:
        Configured SimpleAutogenWorkflow
    """
    config = AgentConfig(api_key=api_key or os.getenv("OPENAI_API_KEY"))
    return SimpleAutogenWorkflow(config)


if __name__ == "__main__":
    # Example usage
    workflow = create_simple_workflow()
    result = workflow.process_task("Research and write about AI agent frameworks")

    print("Task:", result["task"])
    print("Status:", result["status"])
    print("\nConversation:")
    for msg in result["conversation"]:
        print(f"[{msg['agent']}]: {msg['message']}")
    print("\nFinal Output:", result["final_output"])
