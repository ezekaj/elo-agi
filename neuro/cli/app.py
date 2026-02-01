"""
NEURO CLI Application

The main application class that orchestrates all components.
"""

import asyncio
import sys
import os
import json
import signal
from typing import Optional, Dict, Any, List
from pathlib import Path

from .core.stream import StreamHandler
from .core.session import SessionManager
from .core.permissions import PermissionManager, PermissionMode
from .core.hooks import HooksManager
from .core.mcp import MCPManager
from .tools.executor import ToolExecutor
from .tools.registry import ToolRegistry
from .agents.manager import SubagentManager
from .skills.loader import SkillsLoader
from .ui.renderer import UIRenderer
from .ui.status_bar import StatusBar
from .ide.integration import create_integration, detect_ide, IDEType


class NeuroApp:
    """
    Main NEURO CLI Application.

    Orchestrates:
    - Streaming LLM responses
    - Tool execution with permissions
    - Session management
    - Hooks lifecycle
    - UI rendering
    """

    def __init__(
        self,
        model: str = "ministral-3:8b",
        verbose: bool = False,
        permission_mode: str = "default",
        system_prompt: Optional[str] = None,
        no_session_persistence: bool = False,
        project_dir: str = ".",
    ):
        self.model = model
        self.verbose = verbose
        self.project_dir = os.path.abspath(project_dir)
        self.no_session_persistence = no_session_persistence

        # Initialize components
        self.ui = UIRenderer()
        self.status_bar = StatusBar()

        self.stream_handler = StreamHandler(
            model=model,
            on_token=self._on_token,
        )

        self.session_manager = SessionManager(
            project_dir=self.project_dir,
            persist=not no_session_persistence,
        )

        self.permission_manager = PermissionManager(
            mode=PermissionMode(permission_mode),
            project_dir=self.project_dir,
        )

        self.hooks_manager = HooksManager(
            project_dir=self.project_dir,
        )

        self.tool_registry = ToolRegistry()

        self.tool_executor = ToolExecutor(
            registry=self.tool_registry,
            permissions=self.permission_manager,
            hooks=self.hooks_manager,
            ui=self.ui,
        )

        # MCP Manager
        self.mcp_manager = MCPManager(
            project_dir=self.project_dir,
        )

        # Subagent Manager
        self.subagent_manager = SubagentManager(
            project_dir=self.project_dir,
            chat_fn=self._chat_for_agent,
            tool_executor=self.tool_executor,
        )

        # Skills Loader
        self.skills_loader = SkillsLoader(
            project_dir=self.project_dir,
        )

        # IDE Integration
        self.ide_type = detect_ide()
        self.ide_integration = create_integration(
            workspace_root=self.project_dir,
            ide_type=self.ide_type,
        )

        # System prompt
        self.system_prompt = system_prompt or self._default_system_prompt()

        # Runtime state
        self._current_session = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def _default_system_prompt(self) -> str:
        """Get default system prompt."""
        return """You are NEURO, an advanced AI assistant with neuroscience-inspired cognition.

CAPABILITIES:
- Read, write, and edit files
- Execute shell commands
- Search the web
- Git operations
- Learn from conversations

PRINCIPLES:
1. Be direct and concise
2. Admit uncertainty
3. Use tools when needed
4. Never fabricate information

TOOL FORMAT:
To use a tool, respond with:
<tool>tool_name</tool>
<args>{"param": "value"}</args>

Available tools: read_file, write_file, edit_file, run_command, web_search, git_status, git_commit"""

    def run_interactive(
        self,
        initial_prompt: Optional[str] = None,
        resume_session: bool = False,
        session_id: Optional[str] = None,
    ) -> int:
        """Run interactive REPL mode."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        try:
            return self._loop.run_until_complete(
                self._interactive_loop(
                    initial_prompt=initial_prompt,
                    resume_session=resume_session,
                    session_id=session_id,
                )
            )
        except KeyboardInterrupt:
            self.ui.print()
            self._loop.run_until_complete(self._cleanup())
            return 0
        finally:
            self._loop.close()

    async def _interactive_loop(
        self,
        initial_prompt: Optional[str] = None,
        resume_session: bool = False,
        session_id: Optional[str] = None,
    ) -> int:
        """Main interactive loop."""
        # Print header
        self.ui.print_header()

        # Check Ollama connection
        available = await self._check_ollama()
        if not available:
            self.ui.print_error("Ollama not available. Run: ollama serve")
            return 1

        self.ui.print_status("ollama", f"Connected ({self.model})")

        # Connect IDE integration if available
        if self.ide_integration:
            connected = await self.ide_integration.connect()
            if connected:
                self.ui.print_status("ide", f"{self.ide_type.value} integration active")

        # Load or create session
        if resume_session or session_id:
            if session_id:
                # Resume specific session
                self._current_session = self.session_manager.resume(session_id)
            else:
                # Show session picker
                self._current_session = await self._pick_session()

            if self._current_session:
                msg_count = len(self._current_session.messages)
                self.ui.print_success(f"Resumed session: {self._current_session.id[:8]} ({msg_count} messages)")
                # Show recent history
                if msg_count > 0:
                    self.ui.print_dim("Recent history:")
                    for msg in self._current_session.messages[-4:]:
                        role = "[cyan]You:[/cyan]" if msg.role == "user" else "[green]NEURO:[/green]"
                        content = msg.content[:60] + "..." if len(msg.content) > 60 else msg.content
                        self.ui.print(f"  {role} {content}")
                    self.ui.print()
            else:
                self._current_session = self.session_manager.new_session()
        else:
            self._current_session = self.session_manager.new_session()

        # Run SessionStart hook
        await self.hooks_manager.run_hook("SessionStart", session_id=self._current_session.id)

        # Print commands help
        self.ui.print_dim("Commands: /help /status /model /tools /compact /clear /exit")
        self.ui.print_divider()

        # Track model info (no persistent status bar - causes rendering issues)
        self.status_bar.update(
            model=self.model,
            mode=self.permission_manager.mode.value,
        )

        # Process initial prompt if provided
        if initial_prompt:
            await self._process_input(initial_prompt)

        # Main loop
        while True:
            try:
                user_input = await self._get_input()

                if not user_input:
                    continue

                # Handle commands
                if user_input.startswith("/"):
                    should_exit = await self._handle_command(user_input)
                    if should_exit:
                        break
                    continue

                # Handle shell commands
                if user_input.startswith("!"):
                    await self._handle_shell_command(user_input[1:])
                    continue

                # Process normal input
                await self._process_input(user_input)

            except (EOFError, KeyboardInterrupt):
                self.ui.print()
                break

        # Cleanup
        await self._cleanup()
        return 0

    async def _get_input(self) -> str:
        """Get input from user."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.ui.print_user_prompt
        )

    async def _process_input(self, user_input: str):
        """Process user input and generate response."""
        # Detect ultrathink mode
        ultrathink = False
        if user_input.lower().startswith("ultrathink") or user_input.lower().startswith("/ultrathink"):
            ultrathink = True
            # Remove the ultrathink prefix from the prompt
            if user_input.lower().startswith("/ultrathink"):
                user_input = user_input[11:].strip()
            else:
                user_input = user_input[10:].strip()

            if not user_input:
                self.ui.print_error("Please provide a prompt after ultrathink")
                return

            self.ui.print()
            self.ui.print("[bold magenta]ULTRATHINK MODE[/bold magenta] [dim]Deep reasoning enabled[/dim]")

        # Expand file references (@./file)
        user_input = await self._expand_file_refs(user_input)

        # Run UserPromptSubmit hook
        hook_result = await self.hooks_manager.run_hook(
            "UserPromptSubmit",
            prompt=user_input,
        )
        if hook_result.get("decision") == "block":
            self.ui.print_error(f"Blocked: {hook_result.get('reason', 'Hook rejected')}")
            return

        # Add to session
        self._current_session.add_message("user", user_input)

        # Get response with streaming
        self.ui.print_assistant_label()
        self.ui.start_live()

        full_response = ""
        tool_calls = []

        async for event in self.stream_handler.stream(
            messages=self._current_session.get_history(),
            system_prompt=self.system_prompt,
            ultrathink=ultrathink,
        ):
            if event.type.value == "token":
                full_response += event.content
                self.ui.append_live(event.content)
            elif event.type.value == "tool_use_start":
                # Parse and execute tool
                tool_result = await self._handle_tool_call(event.content)
                if tool_result:
                    tool_calls.append(tool_result)
            elif event.type.value == "done":
                self.status_bar.update(
                    tokens=self._current_session.token_count,
                )

        self.ui.stop_live()

        # Add response to session
        self._current_session.add_message("assistant", full_response)

        # If tools were called, continue the conversation
        if tool_calls:
            for tool_name, tool_output in tool_calls:
                self._current_session.add_message(
                    "tool",
                    f"Tool {tool_name} result: {tool_output}",
                    tool_name=tool_name,
                )
            # Continue with tool results
            await self._process_input(f"Continue with the tool results above.")

        # Save session
        if not self.no_session_persistence:
            self.session_manager.save(self._current_session)

    def _on_token(self, token: str):
        """Callback for each streamed token."""
        # Now handled by UI live streaming
        pass

    async def _handle_tool_call(self, content: str) -> Optional[tuple]:
        """Parse and execute a tool call."""
        import re

        # Parse tool name and args
        tool_match = re.search(r'<tool>(\w+)</tool>', content)
        args_match = re.search(r'<args>(.*?)</args>', content, re.DOTALL)

        if not tool_match:
            return None

        tool_name = tool_match.group(1)
        tool_args = {}

        if args_match:
            try:
                tool_args = json.loads(args_match.group(1))
            except json.JSONDecodeError:
                pass

        # Execute tool
        result = await self.tool_executor.execute(tool_name, tool_args)

        if result.status.value == "success":
            return (tool_name, result.output)
        else:
            return (tool_name, f"Error: {result.error}")

    async def _handle_command(self, command: str) -> bool:
        """Handle slash commands. Returns True if should exit."""
        cmd = command.lower().split()[0]
        args = command[len(cmd):].strip()

        if cmd in ("/exit", "/quit"):
            return True

        elif cmd == "/clear":
            self._current_session = self.session_manager.new_session()
            self.ui.print_dim("Conversation cleared")

        elif cmd == "/help":
            self._print_help()

        elif cmd == "/status":
            await self._print_status()

        elif cmd == "/model":
            await self._handle_model_command(args)

        elif cmd == "/tools":
            self._print_tools()

        elif cmd == "/compact":
            await self._compact_context()

        elif cmd == "/cost":
            self._print_cost()

        elif cmd == "/context":
            self._print_context()

        elif cmd == "/permissions":
            self._print_permissions()

        elif cmd == "/skills":
            self._print_skills()

        elif cmd == "/agents":
            self._print_agents()

        elif cmd == "/mcp":
            await self._handle_mcp_command(args)

        elif cmd == "/hooks":
            self._print_hooks()

        elif cmd == "/ide":
            await self._handle_ide_command(args)

        elif cmd == "/ultrathink":
            if args:
                await self._process_input(f"ultrathink {args}")
            else:
                self.ui.print_dim("Usage: /ultrathink <your question>")
                self.ui.print_dim("Enables deep reasoning mode with max tokens")

        elif cmd == "/agent" or cmd == "/task":
            await self._handle_agent_command(args)

        # Check if it's a skill command
        elif cmd.startswith("/"):
            skill_name = cmd[1:]  # Remove leading /
            skill = self.skills_loader.get_skill(skill_name)
            if skill:
                await self._execute_skill(skill_name, args)
            else:
                self.ui.print_dim(f"Unknown command: {cmd}. Try /help")

        else:
            self.ui.print_dim(f"Unknown command: {cmd}. Try /help")

        return False

    async def _handle_shell_command(self, cmd: str):
        """Execute shell command."""
        self.ui.print_dim(f"$ {cmd}")
        try:
            import subprocess
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=30
            )
            if result.stdout:
                print(f"  {result.stdout.rstrip()}")
            if result.stderr:
                self.ui.print_error(result.stderr.rstrip())
        except Exception as e:
            self.ui.print_error(str(e))

    async def _expand_file_refs(self, text: str) -> str:
        """Expand @./file references in input."""
        import re

        refs = re.findall(r'@(\.?/[^\s]+)', text)
        for ref in refs:
            path = os.path.expanduser(ref)
            if os.path.exists(path):
                try:
                    if os.path.isdir(path):
                        files = os.listdir(path)[:20]
                        content = f"[Directory {ref}]: {', '.join(files)}"
                    else:
                        with open(path) as f:
                            content = f.read()[:5000]
                        content = f"[File {ref}]:\n```\n{content}\n```"
                    text = text.replace(f"@{ref}", content)
                    self.ui.print_dim(f"Included: {ref}")
                except Exception as e:
                    self.ui.print_error(f"Could not read {ref}: {e}")

        return text

    def _print_help(self):
        """Print help message."""
        commands = [
            ("/help", "Show this help"),
            ("/status", "System status"),
            ("/ultrathink", "Deep reasoning mode (max tokens)"),
            ("/model [name]", "View/switch model"),
            ("/tools", "List available tools"),
            ("/skills", "List available skills"),
            ("/agents", "List available agents"),
            ("/mcp [cmd]", "MCP server management"),
            ("/hooks", "View configured hooks"),
            ("/ide [cmd]", "IDE integration"),
            ("/compact", "Compress context"),
            ("/cost", "Token usage"),
            ("/context", "Context usage bar"),
            ("/permissions", "View permissions"),
            ("/clear", "Clear conversation"),
            ("/exit", "Exit NEURO"),
        ]

        self.ui.print_help(commands, title="Commands")

        # Skills
        skills = self.skills_loader.list_skills()
        if skills:
            skill_list = [(f"/{s.name}", s.description[:40]) for s in skills[:6]]
            self.ui.print_help(skill_list, title="Skills")
            if len(skills) > 6:
                self.ui.print_dim("...use /skills to see all")

        # Agents & Syntax
        self.ui.print("[bold]Agents:[/bold]")
        self.ui.print("  [cyan]/agent <type> <task>[/cyan] Spawn a subagent")
        self.ui.print("  [dim]Types: Explore, Plan, General, Bash[/dim]")
        self.ui.print()
        self.ui.print("[bold]Syntax:[/bold]")
        self.ui.print("  [cyan]@./file[/cyan]              Include file content")
        self.ui.print("  [cyan]!command[/cyan]            Execute shell command")
        self.ui.print()

    async def _print_status(self):
        """Print system status."""
        self.ui.print()
        self.ui.print("[bold]NEURO Status[/bold]")
        self.ui.print_divider()

        data = {
            "Model": self.model,
            "Permission mode": self.permission_manager.mode.value,
        }

        if self._current_session:
            data["Session"] = self._current_session.id[:8]
            data["Messages"] = str(len(self._current_session.messages))
            data["Tokens"] = f"~{self._current_session.token_count:,}"

        tools = self.tool_registry.list_tools()
        data["Tools"] = f"{len(tools)} available"

        self.ui.print_key_value(data)
        self.ui.print()

    async def _handle_model_command(self, args: str):
        """Handle /model command."""
        if args:
            self.model = args
            self.stream_handler.model = args
            self.status_bar.update(model=args)
            self.ui.print_success(f"Switched to: {args}")
        else:
            self.ui.print(f"  Current model: [cyan]{self.model}[/cyan]")
            # List available models
            try:
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    async with session.get("http://localhost:11434/api/tags") as r:
                        if r.status == 200:
                            data = await r.json()
                            models = [m["name"] for m in data.get("models", [])]
                            self.ui.print(f"  Available: [dim]{', '.join(models[:5])}[/dim]")
            except Exception:
                pass

    def _print_tools(self):
        """Print available tools."""
        self.ui.print()
        self.ui.print("[bold]Available Tools[/bold]")
        self.ui.print_divider()

        tools_list = [(name, tool.description) for name, tool in self.tool_registry.tools.items()]
        self.ui.print_help(tools_list, title="")
        self.ui.print()

    async def _compact_context(self):
        """Compact conversation context."""
        if self._current_session:
            success = await self.session_manager.compact(self._current_session)
            if success:
                self.ui.print_dim("Context compacted")
            else:
                self.ui.print_dim("Not enough context to compact")

    def _print_cost(self):
        """Print token usage."""
        self.ui.print()
        self.ui.print("[bold]Token Usage[/bold]")
        self.ui.print_divider()

        if self._current_session:
            self.ui.print(f"  Messages: {len(self._current_session.messages)}")
            self.ui.print(f"  Est. tokens: ~{self._current_session.token_count:,}")
        self.ui.print(f"  Model: {self.model}")
        self.ui.print("  [dim](Local model - no API cost)[/dim]")
        self.ui.print()

    def _print_context(self):
        """Print context usage visualization."""
        if not self._current_session:
            return

        total = self._current_session.token_count
        max_context = 32000

        self.ui.print()
        self.ui.print_progress_bar(total, max_context, label="Context")
        self.ui.print(f"  [dim]~{total:,} / ~{max_context:,} tokens[/dim]")
        self.ui.print()

    def _print_permissions(self):
        """Print current permissions."""
        self.ui.print()
        self.ui.print("[bold]Permissions[/bold]")
        self.ui.print_divider()
        self.ui.print_key_value({
            "Mode": self.permission_manager.mode.value,
            "Session allows": str(len(self.permission_manager._session_allows)),
            "Session denies": str(len(self.permission_manager._session_denies)),
        })
        self.ui.print()

    def _print_hooks(self):
        """Print configured hooks."""
        self.ui.print()
        self.ui.print("[bold]Hooks[/bold]")
        self.ui.print_divider()

        if not self.hooks_manager.has_hooks():
            self.ui.print_dim("No hooks configured")
            self.ui.print()
            self.ui.print_dim("Add hooks to ~/.neuro/settings.json:")
            self.ui.print('[dim]{"hooks": {"PreToolUse": [{"type": "command", "command": "./script.sh"}]}}[/dim]')
            self.ui.print()
            return

        all_hooks = self.hooks_manager.list_all_hooks()
        for event, handlers in all_hooks.items():
            self.ui.print(f"\n  [cyan]{event}[/cyan]")
            for handler in handlers:
                if handler.get("command"):
                    self.ui.print(f"    [dim]command:[/dim] {handler['command']}")
                if handler.get("url"):
                    self.ui.print(f"    [dim]url:[/dim] {handler['url']}")
                if handler.get("matcher"):
                    self.ui.print(f"    [dim]matcher:[/dim] {handler['matcher']}")
        self.ui.print()

    async def _pick_session(self):
        """Show session picker and return selected session."""
        sessions = self.session_manager.list_sessions(limit=10)

        if not sessions:
            self.ui.print_dim("No previous sessions found")
            return None

        self.ui.print()
        self.ui.print("[bold]Recent Sessions[/bold]")
        self.ui.print_divider()

        from datetime import datetime

        for i, sess in enumerate(sessions, 1):
            created = datetime.fromisoformat(sess["created_at"])
            age = datetime.now() - created
            if age.days > 0:
                time_str = f"{age.days}d ago"
            elif age.seconds > 3600:
                time_str = f"{age.seconds // 3600}h ago"
            else:
                time_str = f"{age.seconds // 60}m ago"

            self.ui.print(f"  [cyan]{i}.[/cyan] {sess['id'][:8]} [dim]({time_str})[/dim]")

        self.ui.print(f"  [cyan]n.[/cyan] [dim]New session[/dim]")
        self.ui.print()

        try:
            choice = self.ui.console.input("[dim]Select session:[/dim] ").strip().lower()

            if choice == 'n' or choice == '':
                return None

            idx = int(choice) - 1
            if 0 <= idx < len(sessions):
                return self.session_manager.resume(sessions[idx]["id"])
        except (ValueError, EOFError, KeyboardInterrupt):
            pass

        return None

    async def _check_ollama(self) -> bool:
        """Check if Ollama is available."""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "http://localhost:11434/api/tags",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as r:
                    return r.status == 200
        except Exception:
            return False

    async def _cleanup(self):
        """Cleanup on exit."""
        try:
            # Run SessionEnd hook
            if self._current_session:
                await self.hooks_manager.run_hook(
                    "SessionEnd",
                    session_id=self._current_session.id,
                )

            # Save session
            if self._current_session and not self.no_session_persistence:
                self.session_manager.save(self._current_session)
                self.ui.print_dim(f"Session saved: {self._current_session.id[:8]}")

            # Close stream handler
            await self.stream_handler.close()

            # Disconnect IDE integration
            if self.ide_integration and self.ide_integration.connected:
                await self.ide_integration.disconnect()

            self.ui.print_dim("Goodbye!")
        except Exception:
            # Silently handle cleanup errors on interrupt
            pass

    def run_print_mode(
        self,
        prompt: str,
        output_format: str = "text",
        stream: bool = True,
    ) -> int:
        """Run in print mode (non-interactive)."""
        if not prompt:
            print("Error: No prompt provided", file=sys.stderr)
            return 1

        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        try:
            return self._loop.run_until_complete(
                self._print_mode(prompt, output_format, stream)
            )
        finally:
            self._loop.close()

    async def _print_mode(
        self,
        prompt: str,
        output_format: str,
        stream: bool,
    ) -> int:
        """Print mode implementation."""
        # Check Ollama
        if not await self._check_ollama():
            print("Error: Ollama not available", file=sys.stderr)
            return 1

        response = ""

        async for event in self.stream_handler.stream(
            messages=[{"role": "user", "content": prompt}],
            system_prompt=self.system_prompt,
        ):
            if event.type.value == "token":
                response += event.content
                if stream and output_format == "text":
                    print(event.content, end="", flush=True)

        # Close stream handler
        await self.stream_handler.close()

        if output_format == "json":
            print(json.dumps({"response": response}))
        elif output_format == "stream-json":
            print(json.dumps({"type": "message", "content": response}))
        elif not stream:
            print(response)
        else:
            print()  # Final newline

        return 0

    # =========================================================================
    # Skills, Agents, and MCP
    # =========================================================================

    def _print_skills(self):
        """Print available skills."""
        self.ui.print()
        self.ui.print("[bold]Available Skills[/bold]")
        self.ui.print_divider()

        for skill in self.skills_loader.list_skills():
            aliases = f" [dim]({', '.join(skill.aliases)})[/dim]" if skill.aliases else ""
            self.ui.print(f"  [cyan]/{skill.name:12}[/cyan] {skill.description}{aliases}")
        self.ui.print()
        self.ui.print_dim("Use /<skill> [args] to run a skill")
        self.ui.print()

    def _print_agents(self):
        """Print available agents."""
        self.ui.print()
        self.ui.print("[bold]Available Agents[/bold]")
        self.ui.print_divider()

        for config in self.subagent_manager.get_available():
            tools_str = f" [dim][{len(config.tools)} tools][/dim]" if config.tools else ""
            self.ui.print(f"  [cyan]{config.name:12}[/cyan] {config.description}{tools_str}")
        self.ui.print()
        self.ui.print_dim("Use /agent <type> <task> to spawn an agent")
        self.ui.print()

        # Show running agents
        running = self.subagent_manager.get_running()
        if running:
            self.ui.print("[bold]Running Agents:[/bold]")
            for exec in running:
                self.ui.print(f"  [yellow]●[/yellow] {exec.id}: {exec.config.name} - {exec.task[:40]}...")
            self.ui.print()

    async def _handle_mcp_command(self, args: str):
        """Handle /mcp command."""
        parts = args.split() if args else []
        subcmd = parts[0] if parts else "list"

        if subcmd == "list":
            self.ui.print()
            self.ui.print("[bold]MCP Servers[/bold]")
            self.ui.print_divider()

            servers = self.mcp_manager.get_servers()
            if not servers:
                self.ui.print_dim("No MCP servers configured")
                self.ui.print_dim("Add servers to ~/.neuro/mcp.json")
            else:
                for name in servers:
                    connected = self.mcp_manager.is_connected(name)
                    status = "[green]●[/green]" if connected else "[dim]○[/dim]"
                    self.ui.print(f"  {status} {name}")
            self.ui.print()

        elif subcmd == "connect":
            if len(parts) < 2:
                self.ui.print_dim("Usage: /mcp connect <server>")
                return
            server = parts[1]
            self.ui.print(f"  Connecting to {server}...")
            success = await self.mcp_manager.connect(server)
            if success:
                self.ui.print_success(f"Connected to {server}")
                tools = [t for t in self.mcp_manager.get_tools() if t.server == server]
                if tools:
                    self.ui.print_dim(f"Discovered {len(tools)} tools")
            else:
                self.ui.print_error(f"Failed to connect to {server}")

        elif subcmd == "tools":
            tools = self.mcp_manager.get_tools()
            if tools:
                self.ui.print()
                self.ui.print("[bold]MCP Tools[/bold]")
                self.ui.print_divider()
                for tool in tools:
                    self.ui.print(f"  [cyan]{tool.name}[/cyan]")
                    self.ui.print(f"    [dim]{tool.description}[/dim]")
            else:
                self.ui.print_dim("No MCP tools available")
            self.ui.print()

        else:
            self.ui.print_dim("MCP commands: list, connect <server>, tools")

    async def _handle_ide_command(self, args: str):
        """Handle /ide command."""
        parts = args.split() if args else []
        subcmd = parts[0] if parts else "status"

        if subcmd == "status":
            self.ui.print()
            self.ui.print("[bold]IDE Integration[/bold]")
            self.ui.print_divider()

            self.ui.print(f"  Detected: {self.ide_type.value}")
            if self.ide_integration:
                status = "[green]connected[/green]" if self.ide_integration.connected else "[dim]disconnected[/dim]"
                self.ui.print(f"  Status: {status}")

                if self.ide_integration.connected:
                    ctx = await self.ide_integration.get_context()
                    if ctx.file_path:
                        self.ui.print(f"  Current file: {ctx.file_path}")
                    if ctx.line_number:
                        self.ui.print(f"  Line: {ctx.line_number}")
                    if ctx.open_files:
                        self.ui.print(f"  Open files: {len(ctx.open_files)}")
            else:
                self.ui.print_dim("No integration available")
            self.ui.print()

        elif subcmd == "open":
            if len(parts) < 2:
                self.ui.print_dim("Usage: /ide open <file> [line]")
                return

            file_path = parts[1]
            line = int(parts[2]) if len(parts) > 2 else None

            if self.ide_integration and self.ide_integration.connected:
                success = await self.ide_integration.open_file(file_path, line)
                if success:
                    self.ui.print_success(f"Opened {file_path}")
                else:
                    self.ui.print_error("Failed to open file")
            else:
                # Fallback: just try the code command
                import subprocess
                cmd = ["code", "--goto", f"{file_path}:{line}" if line else file_path]
                try:
                    subprocess.run(cmd, check=True)
                    self.ui.print_success(f"Opened {file_path}")
                except Exception as e:
                    self.ui.print_error(str(e))

        elif subcmd == "context":
            if self.ide_integration and self.ide_integration.connected:
                ctx = await self.ide_integration.get_context()
                self.ui.print()
                self.ui.print("[bold]Editor Context[/bold]")
                self.ui.print_divider()
                self.ui.print_key_value({
                    "File": ctx.file_path or "None",
                    "Line": str(ctx.line_number or "-"),
                    "Column": str(ctx.column or "-"),
                    "Language": ctx.language or "Unknown",
                })
                if ctx.selection:
                    sel = ctx.selection[:50] + "..." if len(ctx.selection) > 50 else ctx.selection
                    self.ui.print(f"  Selection: {sel}")
                self.ui.print()
            else:
                self.ui.print_dim("IDE not connected")

        else:
            self.ui.print_dim("IDE commands: status, open <file> [line], context")

    async def _handle_agent_command(self, args: str):
        """Handle /agent command."""
        if not args:
            self.ui.print_dim("Usage: /agent <type> <task>")
            self.ui.print_dim("Types: Explore, Plan, General, Bash")
            return

        parts = args.split(maxsplit=1)
        agent_type = parts[0]
        task = parts[1] if len(parts) > 1 else ""

        if not task:
            self.ui.print_dim("Please provide a task for the agent")
            return

        config = self.subagent_manager.get_config(agent_type)
        if not config:
            self.ui.print_error(f"Unknown agent type: {agent_type}")
            return

        self.ui.print()
        self.ui.print(f"[cyan]Spawning {agent_type} agent...[/cyan]")
        self.ui.print_dim(f"Task: {task}")

        try:
            with self.ui.spinner(f"Running {agent_type} agent"):
                execution = await self.subagent_manager.spawn(agent_type, task)

            if execution.status == "completed":
                self.ui.print_success("Agent completed")
                self.ui.print_dim(f"Turns: {execution.turns}")
                if execution.result:
                    self.ui.print()
                    self.ui.print("[bold]Result:[/bold]")
                    self.ui.print_divider()
                    # Print result with markdown rendering
                    self.ui.print_markdown(execution.result[:2000])
            else:
                self.ui.print_error(f"Agent failed: {execution.error}")

        except Exception as e:
            self.ui.print_error(str(e))

        self.ui.print()

    async def _execute_skill(self, skill_name: str, args: str):
        """Execute a skill."""
        skill = self.skills_loader.get_skill(skill_name)
        if not skill:
            self.ui.print_error(f"Skill not found: {skill_name}")
            return

        self.ui.print()
        self.ui.print(f"[cyan]Running /{skill_name}...[/cyan]")

        # Build the prompt
        prompt = self.skills_loader.execute_skill(skill_name, args)

        # Add to history and process
        self._current_session.add_message("user", f"/{skill_name} {args}")

        # Stream the response
        self.ui.print_assistant_label()
        self.ui.start_live()

        full_response = ""
        async for event in self.stream_handler.stream(
            messages=[{"role": "user", "content": prompt}],
            system_prompt=self.system_prompt,
        ):
            if event.type.value == "token":
                full_response += event.content
                self.ui.append_live(event.content)

        self.ui.stop_live()
        self._current_session.add_message("assistant", full_response)

    async def _chat_for_agent(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
    ) -> str:
        """Chat function for subagents."""
        use_model = model or self.model

        # Create a temporary stream handler if model differs
        if use_model != self.model:
            handler = StreamHandler(model=use_model)
        else:
            handler = self.stream_handler

        response = ""
        async for event in handler.stream(messages):
            if event.type.value == "token":
                response += event.content

        if use_model != self.model:
            await handler.close()

        return response
