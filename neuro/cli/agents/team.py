"""
Agent Team System - Coordinate multiple agents working together.

Inspired by Claude Code's multi-agent architecture:
- AgentTeam: Manages a group of teammates
- Teammate: An agent with its own context and optional worktree
- TeammateMailbox: Message passing between agents
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable
from enum import Enum
import asyncio
import json
import uuid
import os
import subprocess
import time


AGENT_COLORS = ["purple", "magenta", "orchid", "medium_purple3", "blue_violet", "plum2"]


class TeammateStatus(Enum):
    IDLE = "idle"
    WORKING = "working"
    WAITING = "waiting"
    DONE = "done"
    FAILED = "failed"


@dataclass
class Message:
    from_agent: str
    to_agent: str
    content: str
    timestamp: float = 0.0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()


class TeammateMailbox:
    """Message queue for inter-agent communication."""

    def __init__(self):
        self._queues: Dict[str, asyncio.Queue] = {}

    def register(self, agent_id: str):
        self._queues[agent_id] = asyncio.Queue()

    async def send(self, msg: Message):
        if msg.to_agent in self._queues:
            await self._queues[msg.to_agent].put(msg)

    async def receive(self, agent_id: str, timeout: float = 0.1) -> Optional[Message]:
        if agent_id not in self._queues:
            return None
        try:
            return await asyncio.wait_for(self._queues[agent_id].get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    def has_messages(self, agent_id: str) -> bool:
        return agent_id in self._queues and not self._queues[agent_id].empty()


@dataclass
class Teammate:
    id: str
    name: str
    role: str
    color: str
    status: TeammateStatus = TeammateStatus.IDLE
    task: str = ""
    worktree_path: Optional[str] = None
    messages: List[Dict] = field(default_factory=list)
    result: Optional[str] = None
    turns: int = 0
    max_turns: int = 15
    tools: List[str] = field(default_factory=list)
    disallowed_tools: List[str] = field(default_factory=list)


@dataclass
class AgentTeam:
    id: str
    name: str
    task: str
    teammates: Dict[str, Teammate] = field(default_factory=dict)
    mailbox: TeammateMailbox = field(default_factory=TeammateMailbox)
    status: str = "idle"

    def add_teammate(
        self,
        name: str,
        role: str,
        task: str,
        tools: List[str] = None,
        disallowed_tools: List[str] = None,
        max_turns: int = 15,
    ) -> Teammate:
        teammate_id = str(uuid.uuid4())[:8]
        color_idx = len(self.teammates) % len(AGENT_COLORS)

        teammate = Teammate(
            id=teammate_id,
            name=name,
            role=role,
            color=AGENT_COLORS[color_idx],
            task=task,
            tools=tools or [],
            disallowed_tools=disallowed_tools or [],
            max_turns=max_turns,
        )
        self.teammates[teammate_id] = teammate
        self.mailbox.register(teammate_id)
        return teammate


class TeamManager:
    """Manages agent teams."""

    def __init__(
        self,
        project_dir: str,
        chat_fn: Optional[Callable] = None,
        tool_executor: Optional[Any] = None,
        ui: Optional[Any] = None,
    ):
        self.project_dir = os.path.abspath(project_dir)
        self.chat_fn = chat_fn
        self.tool_executor = tool_executor
        self.ui = ui
        self._teams: Dict[str, AgentTeam] = {}
        self._active_team: Optional[str] = None

    def create_team(self, name: str, task: str) -> AgentTeam:
        team_id = str(uuid.uuid4())[:8]
        team = AgentTeam(id=team_id, name=name, task=task)
        self._teams[team_id] = team
        self._active_team = team_id
        return team

    async def run_team(self, team: AgentTeam) -> List:
        """Run all teammates in parallel."""
        team.status = "working"

        if self.ui:
            self.ui.print()
            self.ui.print(
                f"[bold]Running team '{team.name}'[/bold] "
                f"[dim]({len(team.teammates)} agents)[/dim]"
            )
            self.ui.print_divider()

        tasks = []
        for teammate in team.teammates.values():
            tasks.append(self._run_teammate(team, teammate))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        team.status = "done"

        if self.ui:
            self.ui.print()
            self.ui.print_divider()
            self.ui.print("[bold]Team Results[/bold]")
            self.ui.print()
            for t in team.teammates.values():
                status_icon = (
                    "[green]done[/green]"
                    if t.status == TeammateStatus.DONE
                    else "[red]failed[/red]"
                )
                self.ui.print(
                    f"  [{t.color}]●[/{t.color}] "
                    f"[bold]{t.name}[/bold] ({t.role}) — {status_icon} "
                    f"[dim]({t.turns} turns)[/dim]"
                )
                if t.result:
                    preview = t.result[:200].replace("\n", " ")
                    self.ui.print(f"    [dim]{preview}...[/dim]")
            self.ui.print()

        return results

    async def _run_teammate(self, team: AgentTeam, teammate: Teammate):
        """Run a single teammate agent."""
        teammate.status = TeammateStatus.WORKING

        if self.ui:
            self.ui.print(
                f"  [{teammate.color}]▶[/{teammate.color}] "
                f"[bold]{teammate.name}[/bold] [dim]({teammate.role})[/dim] starting..."
            )

        system_prompt = (
            f'You are {teammate.name}, a team member with role: {teammate.role}.\n'
            f'You are part of team "{team.name}" working on: {team.task}\n'
            f'\n'
            f'YOUR SPECIFIC TASK: {teammate.task}\n'
            f'\n'
            f'RULES:\n'
            f'- Focus only on your assigned task\n'
            f'- Work step by step\n'
            f'- Report completion clearly when done\n'
            f'- Maximum {teammate.max_turns} turns'
        )

        teammate.messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": teammate.task},
        ]

        try:
            for turn in range(teammate.max_turns):
                teammate.turns = turn + 1

                # Check for messages from other teammates
                msg = await team.mailbox.receive(teammate.id, timeout=0.1)
                if msg:
                    teammate.messages.append({
                        "role": "user",
                        "content": f"[Message from teammate {msg.from_agent}]: {msg.content}",
                    })

                if not self.chat_fn:
                    teammate.result = "Error: No chat function available"
                    teammate.status = TeammateStatus.FAILED
                    break

                response = await self.chat_fn(teammate.messages)
                teammate.messages.append({"role": "assistant", "content": response})

                if self.ui:
                    preview = response[:60].replace("\n", " ")
                    self.ui.print(
                        f"    [{teammate.color}]│[/{teammate.color}] "
                        f"[dim]Turn {turn + 1}: {preview}...[/dim]"
                    )

                # Check for tool calls
                tool_calls = self._extract_tool_calls(response)
                if tool_calls and self.tool_executor:
                    for tool_name, tool_args in tool_calls:
                        if teammate.disallowed_tools and tool_name in teammate.disallowed_tools:
                            teammate.messages.append({
                                "role": "tool",
                                "content": f"Error: Tool {tool_name} not allowed for this agent",
                            })
                            continue

                        if teammate.tools and tool_name not in teammate.tools:
                            teammate.messages.append({
                                "role": "tool",
                                "content": f"Error: Tool {tool_name} not in allowed list",
                            })
                            continue

                        result = await self.tool_executor.execute(
                            tool_name, tool_args, show_spinner=False
                        )
                        teammate.messages.append({
                            "role": "tool",
                            "content": f"Tool {tool_name}: {result.output or result.error}",
                        })
                else:
                    if self._is_complete(response):
                        break

            teammate.result = ""
            for msg_item in reversed(teammate.messages):
                if msg_item["role"] == "assistant":
                    teammate.result = msg_item["content"]
                    break
            teammate.status = TeammateStatus.DONE

        except Exception as e:
            teammate.status = TeammateStatus.FAILED
            teammate.result = f"Error: {e}"

        if self.ui:
            icon = (
                "[green]✓[/green]"
                if teammate.status == TeammateStatus.DONE
                else "[red]✗[/red]"
            )
            self.ui.print(
                f"  [{teammate.color}]◀[/{teammate.color}] "
                f"[bold]{teammate.name}[/bold] {icon} "
                f"[dim]({teammate.turns} turns)[/dim]"
            )

    def _extract_tool_calls(self, response: str) -> List[tuple]:
        import re
        import json as _json

        calls = []
        tool_matches = re.findall(r"<tool>(\w+)</tool>", response)
        args_matches = re.findall(r"<args>(.*?)</args>", response, re.DOTALL)
        for i, tool_name in enumerate(tool_matches):
            args = {}
            if i < len(args_matches):
                try:
                    args = _json.loads(args_matches[i])
                except Exception:
                    pass
            calls.append((tool_name, args))
        return calls

    def _is_complete(self, response: str) -> bool:
        phrases = ["task complete", "finished", "done", "completed", "that's all"]
        return any(p in response.lower() for p in phrases)

    def _setup_worktree(self, teammate: Teammate) -> Optional[str]:
        """Create isolated git worktree for a teammate with symlink support."""
        try:
            worktree_dir = os.path.join(self.project_dir, ".neuro", "worktrees", teammate.id)
            branch_name = f"team/{teammate.id}"
            subprocess.run(
                ["git", "worktree", "add", worktree_dir, "-b", branch_name],
                cwd=self.project_dir,
                capture_output=True,
            )
            teammate.worktree_path = worktree_dir

            # Symlink shared directories from main repo into worktree
            symlink_dirs = self._get_symlink_directories()
            for dir_name in symlink_dirs:
                src = os.path.join(self.project_dir, dir_name)
                dst = os.path.join(worktree_dir, dir_name)
                if os.path.isdir(src) and not os.path.exists(dst):
                    try:
                        os.symlink(src, dst)
                    except OSError:
                        pass  # Skip on failure (permissions, etc.)

            return worktree_dir
        except Exception:
            return None

    def _get_symlink_directories(self) -> List[str]:
        """Read symlink directories config from .neuro/settings.json."""
        defaults = ["node_modules", ".venv", "vendor", "__pycache__"]
        settings_path = os.path.join(self.project_dir, ".neuro", "settings.json")
        try:
            if os.path.exists(settings_path):
                with open(settings_path) as f:
                    settings = json.load(f)
                return settings.get("worktree", {}).get("symlinkDirectories", defaults)
        except Exception:
            pass
        return defaults

    def _cleanup_worktree(self, teammate: Teammate):
        """Remove a teammate's worktree."""
        if teammate.worktree_path:
            try:
                subprocess.run(
                    ["git", "worktree", "remove", teammate.worktree_path, "--force"],
                    cwd=self.project_dir,
                    capture_output=True,
                )
            except Exception:
                pass

    def get_active_team(self) -> Optional[AgentTeam]:
        if self._active_team:
            return self._teams.get(self._active_team)
        return None

    def list_teams(self) -> List[AgentTeam]:
        return list(self._teams.values())
