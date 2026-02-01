"""
NEURO CLI Application

The main application class that orchestrates all components.
"""

import asyncio
import sys
import os
import json
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
            self._current_session = self.session_manager.resume(session_id)
            if self._current_session:
                self.ui.print_dim(f"Resumed session: {self._current_session.id[:8]}")
            else:
                self._current_session = self.session_manager.new_session()
        else:
            self._current_session = self.session_manager.new_session()

        # Run SessionStart hook
        await self.hooks_manager.run_hook("SessionStart", session_id=self._current_session.id)

        # Print commands help
        self.ui.print_dim("Commands: /help /status /model /tools /compact /clear /exit")
        self.ui.print_divider()

        # Start status bar
        self.status_bar.start()
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
            except (EOFError, KeyboardInterrupt):
                break

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

        # Cleanup
        await self._cleanup()
        return 0

    async def _get_input(self) -> str:
        """Get input from user."""
        # Use asyncio-compatible input
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: input(f"\n  {self.ui.CYAN}You:{self.ui.RESET} ").strip()
        )

    async def _process_input(self, user_input: str):
        """Process user input and generate response."""
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

        # Update status
        self.status_bar.update(message="Thinking...")

        # Get response with streaming
        print(f"\n  {self.ui.GREEN}NEURO:{self.ui.RESET} ", end="", flush=True)

        full_response = ""
        tool_calls = []

        async for event in self.stream_handler.stream(
            messages=self._current_session.get_history(),
            system_prompt=self.system_prompt,
        ):
            if event.type.value == "token":
                full_response += event.content
            elif event.type.value == "tool_use_start":
                # Parse and execute tool
                tool_result = await self._handle_tool_call(event.content)
                if tool_result:
                    tool_calls.append(tool_result)
            elif event.type.value == "done":
                self.status_bar.update(
                    tokens=self._current_session.token_count,
                    message="",
                )

        print()  # Newline after response

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
        print(token, end="", flush=True)

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

        print(f"\n  {self.ui.BOLD}Commands:{self.ui.RESET}")
        for cmd, desc in commands:
            print(f"  {self.ui.CYAN}{cmd:20}{self.ui.RESET} {desc}")

        # Skills
        skills = self.skills_loader.list_skills()
        if skills:
            print(f"\n  {self.ui.BOLD}Skills:{self.ui.RESET}")
            for skill in skills[:6]:  # Show first 6
                print(f"  {self.ui.CYAN}/{skill.name:19}{self.ui.RESET} {skill.description[:35]}")
            if len(skills) > 6:
                print(f"  {self.ui.DIM}...use /skills to see all{self.ui.RESET}")

        # Agents
        print(f"\n  {self.ui.BOLD}Agents:{self.ui.RESET}")
        print(f"  {self.ui.CYAN}/agent <type> <task>{self.ui.RESET} Spawn a subagent")
        print(f"  {self.ui.DIM}  Types: Explore, Plan, General, Bash{self.ui.RESET}")

        print(f"\n  {self.ui.BOLD}Syntax:{self.ui.RESET}")
        print(f"  {self.ui.CYAN}@./file{self.ui.RESET}             Include file content")
        print(f"  {self.ui.CYAN}!command{self.ui.RESET}           Execute shell command")
        print()

    async def _print_status(self):
        """Print system status."""
        print(f"\n  {self.ui.BOLD}NEURO Status{self.ui.RESET}")
        self.ui.print_divider()

        # Model
        print(f"  Model: {self.model}")

        # Permission mode
        print(f"  Permission mode: {self.permission_manager.mode.value}")

        # Session
        if self._current_session:
            print(f"  Session: {self._current_session.id[:8]}")
            print(f"  Messages: {len(self._current_session.messages)}")
            print(f"  Tokens: ~{self._current_session.token_count:,}")

        # Tools
        tools = self.tool_registry.list_tools()
        print(f"  Tools: {len(tools)} available")

        print()

    async def _handle_model_command(self, args: str):
        """Handle /model command."""
        if args:
            self.model = args
            self.stream_handler.model = args
            self.status_bar.update(model=args)
            self.ui.print_dim(f"Switched to: {args}")
        else:
            print(f"  Current model: {self.model}")
            # List available models
            try:
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    async with session.get("http://localhost:11434/api/tags") as r:
                        if r.status == 200:
                            data = await r.json()
                            models = [m["name"] for m in data.get("models", [])]
                            print(f"  Available: {', '.join(models[:5])}")
            except Exception:
                pass

    def _print_tools(self):
        """Print available tools."""
        print(f"\n  {self.ui.BOLD}Available Tools:{self.ui.RESET}")
        self.ui.print_divider()

        for name, tool in self.tool_registry.tools.items():
            print(f"  {self.ui.CYAN}{name:15}{self.ui.RESET} {tool.description}")
        print()

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
        print(f"\n  {self.ui.BOLD}Token Usage{self.ui.RESET}")
        self.ui.print_divider()

        if self._current_session:
            print(f"  Messages: {len(self._current_session.messages)}")
            print(f"  Est. tokens: ~{self._current_session.token_count:,}")
        print(f"  Model: {self.model}")
        print(f"  {self.ui.DIM}(Local model - no API cost){self.ui.RESET}")
        print()

    def _print_context(self):
        """Print context usage visualization."""
        if not self._current_session:
            return

        total = self._current_session.token_count
        max_context = 32000
        pct = min(1.0, total / max_context)

        bar_width = 40
        filled = int(bar_width * pct)

        if pct < 0.5:
            color = self.ui.GREEN
        elif pct < 0.8:
            color = self.ui.YELLOW
        else:
            color = self.ui.RED

        bar = f"{color}{'█' * filled}{self.ui.DIM}{'░' * (bar_width - filled)}{self.ui.RESET}"
        print(f"\n  [{bar}] {pct:.0%}")
        print(f"  {self.ui.DIM}~{total:,} / ~{max_context:,} tokens{self.ui.RESET}")
        print()

    def _print_permissions(self):
        """Print current permissions."""
        print(f"\n  {self.ui.BOLD}Permissions{self.ui.RESET}")
        self.ui.print_divider()
        print(f"  Mode: {self.permission_manager.mode.value}")
        print(f"  Session allows: {len(self.permission_manager._session_allows)}")
        print(f"  Session denies: {len(self.permission_manager._session_denies)}")
        print()

    def _print_hooks(self):
        """Print configured hooks."""
        print(f"\n  {self.ui.BOLD}Hooks{self.ui.RESET}")
        self.ui.print_divider()

        if not self.hooks_manager.has_hooks():
            print(f"  {self.ui.DIM}No hooks configured{self.ui.RESET}")
            print(f"\n  {self.ui.DIM}Add hooks to ~/.neuro/settings.json:{self.ui.RESET}")
            print(f'  {self.ui.DIM}{{"hooks": {{"PreToolUse": [{{"type": "command", "command": "./script.sh"}}]}}}}{self.ui.RESET}')
            print()
            return

        all_hooks = self.hooks_manager.list_all_hooks()
        for event, handlers in all_hooks.items():
            print(f"\n  {self.ui.CYAN}{event}{self.ui.RESET}")
            for handler in handlers:
                if handler.get("command"):
                    print(f"    {self.ui.DIM}command:{self.ui.RESET} {handler['command']}")
                if handler.get("url"):
                    print(f"    {self.ui.DIM}url:{self.ui.RESET} {handler['url']}")
                if handler.get("matcher"):
                    print(f"    {self.ui.DIM}matcher:{self.ui.RESET} {handler['matcher']}")
        print()

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
        # Run SessionEnd hook
        if self._current_session:
            await self.hooks_manager.run_hook(
                "SessionEnd",
                session_id=self._current_session.id,
            )

        # Stop status bar
        self.status_bar.stop()

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
        print(f"\n  {self.ui.BOLD}Available Skills:{self.ui.RESET}")
        self.ui.print_divider()

        for skill in self.skills_loader.list_skills():
            aliases = f" ({', '.join(skill.aliases)})" if skill.aliases else ""
            print(f"  {self.ui.CYAN}/{skill.name:12}{self.ui.RESET} {skill.description}{self.ui.DIM}{aliases}{self.ui.RESET}")
        print()
        print(f"  {self.ui.DIM}Use /<skill> [args] to run a skill{self.ui.RESET}")
        print()

    def _print_agents(self):
        """Print available agents."""
        print(f"\n  {self.ui.BOLD}Available Agents:{self.ui.RESET}")
        self.ui.print_divider()

        for config in self.subagent_manager.get_available():
            tools_str = f" [{len(config.tools)} tools]" if config.tools else ""
            print(f"  {self.ui.CYAN}{config.name:12}{self.ui.RESET} {config.description}{self.ui.DIM}{tools_str}{self.ui.RESET}")
        print()
        print(f"  {self.ui.DIM}Use /agent <type> <task> to spawn an agent{self.ui.RESET}")
        print()

        # Show running agents
        running = self.subagent_manager.get_running()
        if running:
            print(f"  {self.ui.BOLD}Running Agents:{self.ui.RESET}")
            for exec in running:
                print(f"  {self.ui.YELLOW}●{self.ui.RESET} {exec.id}: {exec.config.name} - {exec.task[:40]}...")
            print()

    async def _handle_mcp_command(self, args: str):
        """Handle /mcp command."""
        parts = args.split() if args else []
        subcmd = parts[0] if parts else "list"

        if subcmd == "list":
            print(f"\n  {self.ui.BOLD}MCP Servers:{self.ui.RESET}")
            self.ui.print_divider()

            servers = self.mcp_manager.get_servers()
            if not servers:
                print(f"  {self.ui.DIM}No MCP servers configured{self.ui.RESET}")
                print(f"  {self.ui.DIM}Add servers to ~/.neuro/mcp.json{self.ui.RESET}")
            else:
                for name in servers:
                    connected = self.mcp_manager.is_connected(name)
                    status = f"{self.ui.GREEN}●{self.ui.RESET}" if connected else f"{self.ui.DIM}○{self.ui.RESET}"
                    print(f"  {status} {name}")
            print()

        elif subcmd == "connect":
            if len(parts) < 2:
                print(f"  {self.ui.DIM}Usage: /mcp connect <server>{self.ui.RESET}")
                return
            server = parts[1]
            print(f"  Connecting to {server}...")
            success = await self.mcp_manager.connect(server)
            if success:
                self.ui.print_success(f"Connected to {server}")
                tools = [t for t in self.mcp_manager.get_tools() if t.server == server]
                if tools:
                    print(f"  {self.ui.DIM}Discovered {len(tools)} tools{self.ui.RESET}")
            else:
                self.ui.print_error(f"Failed to connect to {server}")

        elif subcmd == "tools":
            tools = self.mcp_manager.get_tools()
            if tools:
                print(f"\n  {self.ui.BOLD}MCP Tools:{self.ui.RESET}")
                self.ui.print_divider()
                for tool in tools:
                    print(f"  {self.ui.CYAN}{tool.name}{self.ui.RESET}")
                    print(f"    {self.ui.DIM}{tool.description}{self.ui.RESET}")
            else:
                print(f"  {self.ui.DIM}No MCP tools available{self.ui.RESET}")
            print()

        else:
            print(f"  {self.ui.DIM}MCP commands: list, connect <server>, tools{self.ui.RESET}")

    async def _handle_ide_command(self, args: str):
        """Handle /ide command."""
        parts = args.split() if args else []
        subcmd = parts[0] if parts else "status"

        if subcmd == "status":
            print(f"\n  {self.ui.BOLD}IDE Integration{self.ui.RESET}")
            self.ui.print_divider()

            print(f"  Detected: {self.ide_type.value}")
            if self.ide_integration:
                status = f"{self.ui.GREEN}connected{self.ui.RESET}" if self.ide_integration.connected else f"{self.ui.DIM}disconnected{self.ui.RESET}"
                print(f"  Status: {status}")

                if self.ide_integration.connected:
                    ctx = await self.ide_integration.get_context()
                    if ctx.file_path:
                        print(f"  Current file: {ctx.file_path}")
                    if ctx.line_number:
                        print(f"  Line: {ctx.line_number}")
                    if ctx.open_files:
                        print(f"  Open files: {len(ctx.open_files)}")
            else:
                print(f"  {self.ui.DIM}No integration available{self.ui.RESET}")
            print()

        elif subcmd == "open":
            if len(parts) < 2:
                print(f"  {self.ui.DIM}Usage: /ide open <file> [line]{self.ui.RESET}")
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
                print(f"\n  {self.ui.BOLD}Editor Context{self.ui.RESET}")
                self.ui.print_divider()
                print(f"  File: {ctx.file_path or 'None'}")
                print(f"  Line: {ctx.line_number or '-'}")
                print(f"  Column: {ctx.column or '-'}")
                print(f"  Language: {ctx.language or 'Unknown'}")
                if ctx.selection:
                    sel = ctx.selection[:50] + "..." if len(ctx.selection) > 50 else ctx.selection
                    print(f"  Selection: {sel}")
                print()
            else:
                print(f"  {self.ui.DIM}IDE not connected{self.ui.RESET}")

        else:
            print(f"  {self.ui.DIM}IDE commands: status, open <file> [line], context{self.ui.RESET}")

    async def _handle_agent_command(self, args: str):
        """Handle /agent command."""
        if not args:
            print(f"  {self.ui.DIM}Usage: /agent <type> <task>{self.ui.RESET}")
            print(f"  {self.ui.DIM}Types: Explore, Plan, General, Bash{self.ui.RESET}")
            return

        parts = args.split(maxsplit=1)
        agent_type = parts[0]
        task = parts[1] if len(parts) > 1 else ""

        if not task:
            print(f"  {self.ui.DIM}Please provide a task for the agent{self.ui.RESET}")
            return

        config = self.subagent_manager.get_config(agent_type)
        if not config:
            print(f"  {self.ui.RED}Unknown agent type: {agent_type}{self.ui.RESET}")
            return

        print(f"\n  {self.ui.CYAN}Spawning {agent_type} agent...{self.ui.RESET}")
        print(f"  {self.ui.DIM}Task: {task}{self.ui.RESET}")

        try:
            execution = await self.subagent_manager.spawn(agent_type, task)

            if execution.status == "completed":
                print(f"\n  {self.ui.GREEN}Agent completed{self.ui.RESET}")
                print(f"  {self.ui.DIM}Turns: {execution.turns}{self.ui.RESET}")
                if execution.result:
                    print(f"\n  {self.ui.BOLD}Result:{self.ui.RESET}")
                    self.ui.print_divider()
                    for line in execution.result.split('\n')[:20]:
                        print(f"  {line}")
            else:
                print(f"\n  {self.ui.RED}Agent failed: {execution.error}{self.ui.RESET}")

        except Exception as e:
            self.ui.print_error(str(e))

        print()

    async def _execute_skill(self, skill_name: str, args: str):
        """Execute a skill."""
        skill = self.skills_loader.get_skill(skill_name)
        if not skill:
            self.ui.print_error(f"Skill not found: {skill_name}")
            return

        print(f"\n  {self.ui.CYAN}Running /{skill_name}...{self.ui.RESET}")

        # Build the prompt
        prompt = self.skills_loader.execute_skill(skill_name, args)

        # Add to history and process
        self._current_session.add_message("user", f"/{skill_name} {args}")

        # Stream the response
        print(f"\n  {self.ui.GREEN}NEURO:{self.ui.RESET} ", end="", flush=True)

        full_response = ""
        async for event in self.stream_handler.stream(
            messages=[{"role": "user", "content": prompt}],
            system_prompt=self.system_prompt,
        ):
            if event.type.value == "token":
                full_response += event.content

        print()  # Newline after response

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
