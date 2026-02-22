"""
Subagent Manager - Spawn and manage subagents for complex tasks.

Like Claude Code's Task tool, subagents can:
- Handle specific tasks autonomously
- Run in parallel
- Have isolated context
- Use restricted tool sets
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable
from enum import Enum
from pathlib import Path
import asyncio
import uuid
import os
import yaml


class SubagentType(Enum):
    """Built-in subagent types."""

    EXPLORE = "Explore"  # Fast, read-only codebase exploration
    PLAN = "Plan"  # Research for planning
    GENERAL = "General"  # Full capabilities
    BASH = "Bash"  # Shell command specialist
    CUSTOM = "Custom"  # User-defined


@dataclass
class SubagentConfig:
    """Configuration for a subagent."""

    name: str
    description: str
    type: SubagentType = SubagentType.CUSTOM

    # Capabilities
    tools: List[str] = field(default_factory=list)
    disallowed_tools: List[str] = field(default_factory=list)

    # Model
    model: str = "inherit"  # "inherit" uses parent's model

    # System prompt
    prompt: str = ""

    # Limits
    max_turns: int = 10
    timeout: int = 300  # seconds


@dataclass
class SubagentExecution:
    """A running or completed subagent."""

    id: str
    config: SubagentConfig
    task: str
    status: str = "pending"  # pending, running, completed, failed
    result: Optional[str] = None
    error: Optional[str] = None
    turns: int = 0
    messages: List[Dict] = field(default_factory=list)


class SubagentManager:
    """
    Manages subagent spawning and execution.

    Features:
    - Built-in agents (Explore, Plan, General, Bash)
    - Custom agents from .neuro/agents/*.md
    - Parallel execution
    - Context isolation
    - Tool restrictions
    """

    AGENTS_DIR = ".neuro/agents"
    USER_AGENTS_DIR = "~/.neuro/agents"

    # Built-in agent configurations
    BUILTINS = {
        "Explore": SubagentConfig(
            name="Explore",
            description="Fast, read-only agent for searching and exploring codebases. Use for finding files, searching code, understanding structure.",
            type=SubagentType.EXPLORE,
            tools=["read_file", "list_files", "web_search"],
            disallowed_tools=["write_file", "edit_file", "run_command"],
            max_turns=5,
            prompt="""You are an exploration agent. Your job is to quickly find information in the codebase.

RULES:
- Only read files, never modify
- Be fast and focused
- Return concise findings
- If you can't find something in 3 attempts, report what you did find""",
        ),
        "Plan": SubagentConfig(
            name="Plan",
            description="Research agent for gathering context before planning implementation. Analyzes codebase structure and patterns.",
            type=SubagentType.PLAN,
            tools=["read_file", "list_files", "web_search"],
            disallowed_tools=["write_file", "edit_file"],
            max_turns=8,
            prompt="""You are a planning agent. Your job is to research and understand the codebase to help plan implementations.

TASKS:
- Find relevant files and patterns
- Understand existing architecture
- Identify dependencies
- Note potential issues

Return structured findings that will help plan the implementation.""",
        ),
        "General": SubagentConfig(
            name="General",
            description="General-purpose agent with full capabilities. Use for complex multi-step tasks.",
            type=SubagentType.GENERAL,
            tools=[],  # Inherit all
            max_turns=10,
            prompt="""You are a general-purpose agent. Complete the assigned task thoroughly.

PRINCIPLES:
- Work step by step
- Verify your work
- Report progress
- Ask for clarification if needed""",
        ),
        "Bash": SubagentConfig(
            name="Bash",
            description="Shell command specialist. Use for running commands, installations, builds.",
            type=SubagentType.BASH,
            tools=["run_command", "read_file", "list_files"],
            disallowed_tools=["write_file", "edit_file"],
            max_turns=5,
            prompt="""You are a shell command specialist. Execute commands safely and report results.

RULES:
- Always explain what a command does before running
- Check for errors in output
- Never run destructive commands without confirmation
- Use safe defaults""",
        ),
    }

    def __init__(
        self,
        project_dir: str = ".",
        chat_fn: Optional[Callable] = None,
        tool_executor: Optional[Any] = None,
    ):
        self.project_dir = os.path.abspath(project_dir)
        self.chat_fn = chat_fn
        self.tool_executor = tool_executor

        self._configs: Dict[str, SubagentConfig] = dict(self.BUILTINS)
        self._running: Dict[str, SubagentExecution] = {}
        self._completed: Dict[str, SubagentExecution] = {}

        self._load_custom_agents()

    def _load_custom_agents(self):
        """Load custom agents from agent directories."""
        dirs = [
            os.path.join(self.project_dir, self.AGENTS_DIR),
            os.path.expanduser(self.USER_AGENTS_DIR),
        ]

        for agents_dir in dirs:
            if os.path.exists(agents_dir):
                for file in Path(agents_dir).glob("*.md"):
                    try:
                        config = self._load_agent_file(str(file))
                        if config:
                            self._configs[config.name] = config
                    except Exception as e:
                        print(f"Error loading agent {file}: {e}")

    def _load_agent_file(self, path: str) -> Optional[SubagentConfig]:
        """Load agent config from markdown file with YAML frontmatter."""
        with open(path) as f:
            content = f.read()

        # Parse frontmatter
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                frontmatter = yaml.safe_load(parts[1])
                prompt = parts[2].strip()
            else:
                return None
        else:
            return None

        return SubagentConfig(
            name=frontmatter.get("name", Path(path).stem),
            description=frontmatter.get("description", ""),
            type=SubagentType.CUSTOM,
            tools=frontmatter.get("tools", []),
            disallowed_tools=frontmatter.get("disallowedTools", []),
            model=frontmatter.get("model", "inherit"),
            max_turns=frontmatter.get("maxTurns", 10),
            prompt=prompt,
        )

    def get_available(self) -> List[SubagentConfig]:
        """Get all available subagent configurations."""
        return list(self._configs.values())

    def get_config(self, name: str) -> Optional[SubagentConfig]:
        """Get a specific subagent configuration."""
        return self._configs.get(name)

    async def spawn(
        self,
        agent_type: str,
        task: str,
        background: bool = False,
    ) -> SubagentExecution:
        """
        Spawn a subagent to handle a task.

        Args:
            agent_type: Name of agent (e.g., "Explore", "Plan", "General")
            task: The task description
            background: Run in background without blocking

        Returns:
            SubagentExecution instance
        """
        config = self._configs.get(agent_type)
        if not config:
            raise ValueError(f"Agent type not found: {agent_type}")

        agent_id = str(uuid.uuid4())[:8]

        execution = SubagentExecution(
            id=agent_id,
            config=config,
            task=task,
            status="pending",
        )

        self._running[agent_id] = execution

        if background:
            asyncio.create_task(self._run_agent(execution))
        else:
            await self._run_agent(execution)

        return execution

    async def _run_agent(self, execution: SubagentExecution):
        """Run a subagent to completion."""
        execution.status = "running"
        config = execution.config

        # Build system prompt
        system_prompt = f"""You are {config.name}, a specialized agent.

{config.prompt}

YOUR TASK:
{execution.task}

CONSTRAINTS:
- Maximum {config.max_turns} turns
- Stay focused on the task
- Report when complete or blocked"""

        # Initialize conversation
        execution.messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": execution.task},
        ]

        try:
            for turn in range(config.max_turns):
                execution.turns = turn + 1

                if not self.chat_fn:
                    execution.error = "No chat function available"
                    execution.status = "failed"
                    break

                # Get response from LLM
                response = await self.chat_fn(
                    execution.messages,
                    model=config.model if config.model != "inherit" else None,
                )

                execution.messages.append(
                    {
                        "role": "assistant",
                        "content": response,
                    }
                )

                # Check for tool calls
                tool_calls = self._extract_tool_calls(response)

                if tool_calls and self.tool_executor:
                    for tool_name, tool_args in tool_calls:
                        # Check if tool is allowed
                        if config.disallowed_tools and tool_name in config.disallowed_tools:
                            execution.messages.append(
                                {
                                    "role": "tool",
                                    "content": f"Error: Tool {tool_name} is not allowed for this agent",
                                }
                            )
                            continue

                        if config.tools and tool_name not in config.tools:
                            execution.messages.append(
                                {
                                    "role": "tool",
                                    "content": f"Error: Tool {tool_name} is not in allowed tools list",
                                }
                            )
                            continue

                        # Execute tool
                        result = await self.tool_executor.execute(
                            tool_name,
                            tool_args,
                            show_spinner=False,
                        )

                        execution.messages.append(
                            {
                                "role": "tool",
                                "content": f"Tool {tool_name} result: {result.output or result.error}",
                            }
                        )
                else:
                    # No tool calls, check if done
                    if self._is_complete(response):
                        break

            # Extract final result
            execution.result = self._extract_result(execution.messages)
            execution.status = "completed"

        except asyncio.TimeoutError:
            execution.error = "Timeout"
            execution.status = "failed"
        except Exception as e:
            execution.error = str(e)
            execution.status = "failed"

        # Move to completed
        self._completed[execution.id] = execution
        if execution.id in self._running:
            del self._running[execution.id]

        return execution

    def _extract_tool_calls(self, response: str) -> List[tuple]:
        """Extract tool calls from response."""
        import re

        calls = []
        tool_matches = re.findall(r"<tool>(\w+)</tool>", response)
        args_matches = re.findall(r"<args>(.*?)</args>", response, re.DOTALL)

        for i, tool_name in enumerate(tool_matches):
            args = {}
            if i < len(args_matches):
                try:
                    args = json.loads(args_matches[i])
                except:
                    pass
            calls.append((tool_name, args))

        return calls

    def _is_complete(self, response: str) -> bool:
        """Check if the agent indicates completion."""
        completion_phrases = [
            "task complete",
            "task completed",
            "i have completed",
            "finished",
            "done",
            "that's all",
        ]
        response_lower = response.lower()
        return any(phrase in response_lower for phrase in completion_phrases)

    def _extract_result(self, messages: List[Dict]) -> str:
        """Extract the final result from conversation."""
        # Get last assistant message
        for msg in reversed(messages):
            if msg["role"] == "assistant":
                return msg["content"]
        return ""

    def get_running(self) -> List[SubagentExecution]:
        """Get all running subagents."""
        return list(self._running.values())

    def get_execution(self, agent_id: str) -> Optional[SubagentExecution]:
        """Get an execution by ID."""
        return self._running.get(agent_id) or self._completed.get(agent_id)

    async def wait_for(self, agent_id: str, timeout: int = 300) -> SubagentExecution:
        """Wait for a subagent to complete."""
        start = asyncio.get_event_loop().time()

        while True:
            execution = self.get_execution(agent_id)
            if execution and execution.status in ("completed", "failed"):
                return execution

            if asyncio.get_event_loop().time() - start > timeout:
                raise asyncio.TimeoutError(f"Agent {agent_id} timed out")

            await asyncio.sleep(0.5)


# Import json for tool call parsing
import json
