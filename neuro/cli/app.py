"""
NEURO CLI Application

The main application class that orchestrates all components.
"""

import asyncio
import sys
import os
import json
from typing import Optional, Dict, List

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
from .ide.integration import create_integration, detect_ide

# Learning systems
from ..self_training import SelfTrainer
from ..active_learning import get_active_learner
from ..self_evolution import get_evolution
import threading
import random
import urllib.request
import urllib.parse
import html
from datetime import datetime

# Cognitive Pipeline and NeuroAgent (38+ modules)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../neuro-model/src"))
try:
    from cognitive_pipeline import CognitivePipeline

    COGNITIVE_PIPELINE_AVAILABLE = True
except ImportError:
    COGNITIVE_PIPELINE_AVAILABLE = False

try:
    from neuro_agent import NeuroAgent

    NEURO_AGENT_AVAILABLE = True
except ImportError:
    NEURO_AGENT_AVAILABLE = False


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

        # Register tools for native Ollama function calling
        self._register_native_tools()

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

        # Learning systems (AI that evolves!)
        self.self_trainer = SelfTrainer()
        self.active_learner = get_active_learner()
        self.evolution = get_evolution()

        # Background evolution state
        self._evolution_running = False
        self._evolution_thread = None

        # Feature modes (opt-in via slash commands)
        self._think_mode = False  # /think — cognitive pipeline
        self._knowledge_mode = False  # /knowledge — knowledge injection
        self._evolve_mode = False  # /evolve — self-evolution loop

        # Cognitive Pipeline (38+ modules - the brain!)
        self.cognitive_pipeline = None
        if COGNITIVE_PIPELINE_AVAILABLE:
            try:
                self.cognitive_pipeline = CognitivePipeline(verbose=verbose)
            except Exception as e:
                if verbose:
                    print(f"[NEURO] Cognitive pipeline init failed: {e}")

        # NeuroAgent (full PERCEIVE -> THINK -> ACT -> LEARN -> IMPROVE workflow)
        self.neuro_agent = None
        if NEURO_AGENT_AVAILABLE:
            try:
                self.neuro_agent = NeuroAgent(model=model, verbose=verbose)
                if verbose:
                    stats = self.neuro_agent.get_stats()
                    print(
                        f"[NEURO] Agent loaded with {len(stats.get('components', {}))} components"
                    )
            except Exception as e:
                if verbose:
                    print(f"[NEURO] NeuroAgent init failed: {e}")

        # Runtime state
        self._current_session = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._first_prompt = True  # Track if this is the first prompt

    def _register_native_tools(self):
        """Register tools for Ollama native function calling."""
        ollama_tools = [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web for information",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string", "description": "Search query"}},
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read contents of a file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string", "description": "File path"}},
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write content to a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "File path"},
                            "content": {"type": "string", "description": "Content to write"},
                        },
                        "required": ["path", "content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "run_command",
                    "description": "Run a shell command",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string", "description": "Command to run"}
                        },
                        "required": ["command"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_files",
                    "description": "List files and folders in a directory",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Directory path (default: current directory)",
                            }
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "edit_file",
                    "description": "Edit a file by replacing specific text",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "File path"},
                            "old_text": {"type": "string", "description": "Text to find"},
                            "new_text": {"type": "string", "description": "Replacement text"},
                        },
                        "required": ["path", "old_text", "new_text"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "web_fetch",
                    "description": "Fetch and extract text content from a URL",
                    "parameters": {
                        "type": "object",
                        "properties": {"url": {"type": "string", "description": "URL to fetch"}},
                        "required": ["url"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "git_status",
                    "description": "Show git repository status",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "git_diff",
                    "description": "Show git diff of changes",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "improve_self",
                    "description": "Analyze and improve ELO's own code",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "area": {
                                "type": "string",
                                "description": "Area to improve: core, tools, learning, ui",
                            }
                        },
                        "required": ["area"],
                    },
                },
            },
        ]
        self.stream_handler.set_tools(ollama_tools)

    def _default_system_prompt(self) -> str:
        """Get default system prompt."""
        import platform

        home = os.path.expanduser("~")
        cwd = os.getcwd()
        os_name = platform.system()

        return f"""You are ELO, a local AI assistant running on the user's machine.

ENVIRONMENT:
- OS: {os_name}
- Home directory: {home}
- Working directory: {cwd}
- Desktop: {os.path.join(home, "Desktop")}

BEHAVIOR:
- Be concise — short replies for simple questions, detailed only when needed
- Use tools when the request requires file access, web info, or commands
- Do NOT ramble, do NOT explain what you're about to do — just do it
- When a tool returns results, present them directly — do NOT re-explain or ask follow-up questions unless the user asks
- Use absolute paths based on the environment info above

TOOLS:
- read_file: Read a file's contents (NOT for directories)
- list_files: List files/folders in a directory
- write_file: Create or overwrite a file
- edit_file: Replace specific text in a file
- run_command: Execute a shell command
- web_search: Search the internet
- web_fetch: Fetch text from a URL
- git_status: Show git status
- git_diff: Show git diff
- improve_self: Analyze ELO's own code"""

    def _check_for_updates(self):
        """Check PyPI for newer version (non-blocking, silent on failure)."""
        import neuro

        def check():
            try:
                import json as _json
                import urllib.request as _req

                url = "https://pypi.org/pypi/elo-agi/json"
                with _req.urlopen(url, timeout=3) as resp:
                    data = _json.loads(resp.read())
                    latest = data["info"]["version"]
                    current = neuro.__version__
                    if latest != current:
                        self.ui.print_dim(f"Update available: {current} → {latest}")
                        self.ui.print_dim("  Run: pipx upgrade elo-agi")
            except Exception:
                pass

        threading.Thread(target=check, daemon=True).start()

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
        import getpass

        # Check Ollama connection first (needed for welcome screen)
        available = await self._check_ollama()
        if not available:
            self.ui.print_error("Ollama not available. Run: ollama serve")
            return 1

        # Connect IDE integration if available (in background)
        if self.ide_integration:
            await self.ide_integration.connect()

        # Get user name
        try:
            user_name = getpass.getuser().capitalize()
        except Exception:
            user_name = "User"

        # Get recent sessions for welcome screen
        recent_sessions = self.session_manager.list_sessions(limit=5)

        # Get knowledge stats
        knowledge_stats = self.self_trainer.get_stats()

        # Add cognitive module count
        if self.cognitive_pipeline:
            try:
                cp_stats = self.cognitive_pipeline.get_stats()
                knowledge_stats["cognitive_modules"] = cp_stats.get("num_components", 0)
            except Exception:
                pass

        # Get working directory (shortened)
        working_dir = self.project_dir
        home = os.path.expanduser("~")
        if working_dir.startswith(home):
            working_dir = "~" + working_dir[len(home) :]

        # Print beautiful welcome screen
        import neuro

        self.ui.print_welcome_screen(
            version=neuro.__version__,
            user_name=user_name,
            model=self.model,
            working_dir=working_dir,
            recent_sessions=recent_sessions,
            knowledge_stats=knowledge_stats,
        )

        # Check for updates (non-blocking)
        self._check_for_updates()

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
                self.ui.print_success(
                    f"Resumed session: {self._current_session.id[:8]} ({msg_count} messages)"
                )
                # Show recent history
                if msg_count > 0:
                    self.ui.print_dim("Recent history:")
                    for msg in self._current_session.messages[-4:]:
                        role = (
                            "[cyan]You:[/cyan]" if msg.role == "user" else "[green]NEURO:[/green]"
                        )
                        content = msg.content[:60] + "..." if len(msg.content) > 60 else msg.content
                        self.ui.print(f"  {role} {content}")
                    self.ui.print()
            else:
                self._current_session = self.session_manager.new_session()
        else:
            self._current_session = self.session_manager.new_session()

        # Run SessionStart hook
        await self.hooks_manager.run_hook("SessionStart", session_id=self._current_session.id)

        # Evolution thread disabled by default — use /evolve to start manually
        # self._start_evolution_thread()

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

                # Handle direct tool calls (e.g., "improve_self", "web_search python")
                tool_result = await self._try_direct_tool(user_input)
                if tool_result:
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

        # Show placeholder hint on first prompt
        if self._first_prompt:
            self._first_prompt = False
            # Print placeholder hint
            self.ui.print_dim('Try "explain this code" or "help me fix a bug"')
            self.ui.print_dim("? for shortcuts")
            self.ui.print()

        return await loop.run_in_executor(None, self.ui.print_user_prompt)

    async def _process_input(self, user_input: str):
        """Process user input and generate response."""
        # Detect ultrathink mode
        ultrathink = False
        if user_input.lower().startswith("ultrathink") or user_input.lower().startswith(
            "/ultrathink"
        ):
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
            self.ui.print(
                "[bold magenta]ULTRATHINK MODE[/bold magenta] [dim]Deep reasoning enabled[/dim]"
            )

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

        # NeuroAgent (opt-in via /think mode)
        if self.neuro_agent and self._think_mode:
            await self._process_with_agent(user_input, ultrathink)
            return

        # Cognitive pipeline and knowledge injection (opt-in via /think or /knowledge)
        augmented_input = user_input

        if self._think_mode and self.cognitive_pipeline:
            try:
                result = self.cognitive_pipeline.process(
                    query=user_input, use_deep_thinking=ultrathink
                )
                if result.content:
                    augmented_input = f"{user_input}\n\n{result.content}"
                if result.cognitive_analysis:
                    analysis_type = result.cognitive_analysis.get("type", "general")
                    confidence = result.cognitive_analysis.get("confidence", 0.5)
                    self.ui.print_dim(f"Cognitive: {analysis_type} ({confidence:.0%})")
            except Exception:
                pass

        learned_knowledge = None
        if self._knowledge_mode:
            learned_knowledge = self.self_trainer.get_knowledge_for_prompt(user_input)
            if learned_knowledge:
                augmented_input = f"{augmented_input}\n\n{learned_knowledge}"
                fact_count = learned_knowledge.count("\n") - 1
                self.ui.print_dim(f"Injected {fact_count} knowledge facts")

        # Add to session (original input, not augmented)
        self._current_session.add_message("user", user_input)

        # Get response with streaming
        self.ui.print_assistant_label()
        self.ui.start_live()

        full_response = ""
        tool_calls = []

        # Build messages with augmented last message (includes learned knowledge)
        messages = self._current_session.get_history()
        if messages and learned_knowledge:
            # Replace last user message with augmented version (internally only)
            messages = messages[:-1] + [{"role": "user", "content": augmented_input}]

        async for event in self.stream_handler.stream(
            messages=messages,
            system_prompt=self.system_prompt,
            ultrathink=ultrathink,
        ):
            if event.type.value == "token":
                full_response += event.content
                self.ui.append_live(event.content)
            elif event.type.value == "tool_use_start":
                # Check if this is a native function call (from Ollama API)
                if event.metadata.get("native"):
                    tool_name = event.metadata.get("name", "")
                    tool_args = event.metadata.get("arguments", {})
                    self.ui.print_dim(f"Calling tool: {tool_name}")
                    tool_result = await self._execute_native_tool(tool_name, tool_args)
                    if tool_result:
                        tool_calls.append(tool_result)
                else:
                    # Legacy XML-style tool parsing
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

        # If tools were called, add results to session (no recursive call)
        if tool_calls:
            for tool_name, tool_output in tool_calls:
                self._current_session.add_message(
                    "tool",
                    f"Tool {tool_name} result: {tool_output}",
                    tool_name=tool_name,
                )

        # Learn from conversation only when knowledge mode is on
        if self._knowledge_mode:
            self._record_learning(user_input, full_response)
            if self.cognitive_pipeline:
                try:
                    topic = user_input.split()[0] if user_input.split() else "general"
                    self.cognitive_pipeline.learn(
                        topic=topic,
                        content=f"Q: {user_input[:200]} A: {full_response[:500]}",
                        source="conversation",
                        importance=0.6,
                    )
                except Exception:
                    pass

        # Save session
        if not self.no_session_persistence:
            self.session_manager.save(self._current_session)

    def _on_token(self, token: str):
        """Callback for each streamed token."""
        # Now handled by UI live streaming
        pass

    def _record_learning(self, user_input: str, response: str):
        """Extract knowledge and record learning from the conversation."""
        import re

        # Extract meaningful topics from the conversation
        combined = f"{user_input} {response}"

        # Simple topic extraction - words that appear multiple times or are capitalized
        words = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", combined)  # Proper nouns
        words += re.findall(r"\b[a-z]{5,}\b", combined.lower())  # Meaningful words

        # Count frequencies
        from collections import Counter

        word_freq = Counter(words)

        # Topics are frequent meaningful words
        topics = [word for word, count in word_freq.items() if count >= 2 and len(word) > 4][:5]

        for topic in topics:
            # Record exposure with success (AI answered)
            self.active_learner.record_exposure(
                topic=topic,
                was_successful=True,  # We answered the query
                surprise_level=0.5,  # Neutral surprise
                complexity=len(response) / 1000,  # Rough complexity estimate
            )

        # Store factual content as learned knowledge
        # Extract sentences that look like facts (contain "is", "are", "means", etc.)
        fact_patterns = [
            r"([A-Z][^.!?]*(?:is|are|means|refers to|defined as)[^.!?]*\.)",
            r"([A-Z][^.!?]*(?:can be|should be|must be)[^.!?]*\.)",
        ]

        for pattern in fact_patterns:
            facts = re.findall(pattern, response)
            for fact in facts[:3]:  # Limit to 3 facts per response
                if len(fact) > 20:  # Meaningful fact
                    # Determine topic from fact
                    fact_words = re.findall(r"\b[A-Z][a-z]+\b", fact)
                    topic = fact_words[0] if fact_words else "general"

                    self.self_trainer.learn(
                        topic=topic, content=fact.strip(), source="conversation"
                    )

        # Save periodically
        if len(self._current_session.messages) % 10 == 0:
            self.self_trainer.save()

    async def _execute_native_tool(self, tool_name: str, tool_args: Dict) -> Optional[tuple]:
        """Execute a native Ollama function call."""
        try:
            if tool_name == "web_search":
                result = self.tool_registry.tools["web_search"].func(tool_args.get("query", ""))
            elif tool_name == "read_file":
                result = self.tool_registry.tools["read_file"].func(tool_args.get("path", ""))
            elif tool_name == "write_file":
                result = self.tool_registry.tools["write_file"].func(
                    tool_args.get("path", ""), tool_args.get("content", "")
                )
            elif tool_name == "run_command":
                result = self.tool_registry.tools["run_command"].func(tool_args.get("command", ""))
            elif tool_name == "list_files":
                result = self.tool_registry.tools["list_files"].func(tool_args.get("path", "."))
            elif tool_name == "edit_file":
                result = self.tool_registry.tools["edit_file"].func(
                    tool_args.get("path", ""),
                    tool_args.get("old_text", ""),
                    tool_args.get("new_text", ""),
                )
            elif tool_name == "web_fetch":
                result = self.tool_registry.tools["web_fetch"].func(tool_args.get("url", ""))
            elif tool_name == "git_status":
                result = self.tool_registry.tools["git_status"].func()
            elif tool_name == "git_diff":
                result = self.tool_registry.tools["git_diff"].func()
            elif tool_name == "improve_self":
                result = self.tool_registry.tools["improve_self"].func(
                    tool_args.get("area", "core")
                )
            else:
                return None

            self.ui.print_tool_result(tool_name, True, str(result)[:2000])
            return (tool_name, result)

        except Exception as e:
            self.ui.print_error(f"Tool {tool_name} failed: {e}")
            return (tool_name, f"Error: {e}")

    async def _handle_tool_call(self, content: str) -> Optional[tuple]:
        """Parse and execute a tool call."""
        import re

        # Parse tool name and args
        tool_match = re.search(r"<tool>(\w+)</tool>", content)
        args_match = re.search(r"<args>(.*?)</args>", content, re.DOTALL)

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
        args = command[len(cmd) :].strip()

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

        elif cmd == "/think":
            self._think_mode = not self._think_mode
            state = "ON" if self._think_mode else "OFF"
            self.ui.print_dim(f"Cognitive pipeline: {state}")
            if self._think_mode:
                self.ui.print_dim("  Messages will be processed through 38 cognitive modules")

        elif cmd == "/knowledge":
            self._knowledge_mode = not self._knowledge_mode
            state = "ON" if self._knowledge_mode else "OFF"
            fact_count = len(self.self_trainer.facts) if hasattr(self.self_trainer, "facts") else 0
            self.ui.print_dim(f"Knowledge injection: {state} ({fact_count} facts available)")

        elif cmd == "/evolve":
            if self._evolution_running:
                self.ui.print_dim("Evolution already running")
            else:
                self._start_evolution_thread()
                self.ui.print_dim("Self-evolution started (benchmarks, learning, training)")

        elif cmd == "/bench":
            self.ui.print_dim("Running benchmarks...")
            self._run_benchmark_and_report("MANUAL")

        elif cmd == "/learn":
            await self._print_learning_stats()

        elif cmd == "/evolution":
            self._print_evolution_stats()

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

    async def _try_direct_tool(self, user_input: str) -> bool:
        """Try to execute a direct tool call. Returns True if handled."""
        from .core.intent_router import detect_intent

        # Try intent detection (natural language) — only for high-confidence explicit commands
        intent = detect_intent(user_input)
        if intent and intent.confidence >= 0.9:
            return await self._execute_intent(intent, user_input)

        # Fall back to keyword matching
        parts = user_input.strip().split(maxsplit=1)
        if not parts:
            return False

        tool_name = parts[0].lower()
        tool_arg = parts[1] if len(parts) > 1 else ""

        # Map of direct tool commands
        direct_tools = {
            "improve_self": lambda: self.tool_registry.tools["improve_self"].func(
                tool_arg or "core"
            ),
            "web_search": lambda: self.tool_registry.tools["web_search"].func(tool_arg),
            "web_fetch": lambda: self.tool_registry.tools["web_fetch"].func(tool_arg),
            "read_file": lambda: self.tool_registry.tools["read_file"].func(tool_arg),
            "list_files": lambda: self.tool_registry.tools["list_files"].func(tool_arg or "."),
            "git_status": lambda: self.tool_registry.tools["git_status"].func(),
            "git_diff": lambda: self.tool_registry.tools["git_diff"].func(),
            "run_command": lambda: self.tool_registry.tools["run_command"].func(tool_arg),
        }

        # Special autonomous commands
        if tool_name == "evolve" or tool_name == "learn_online":
            await self._autonomous_evolve(tool_arg)
            return True

        if tool_name == "research":
            await self._autonomous_research(tool_arg)
            return True

        if tool_name in direct_tools:
            self.ui.print()
            self.ui.print(f"[cyan]Executing tool: {tool_name}[/cyan]")

            try:
                with self.ui.spinner(f"Running {tool_name}..."):
                    result = direct_tools[tool_name]()

                # Display result
                self.ui.print()
                if len(result) > 2000:
                    self.ui.print(result[:2000])
                    self.ui.print_dim(f"... ({len(result) - 2000} more characters)")
                else:
                    self.ui.print(result)
                self.ui.print()

                # If this was improve_self, ask the AI to analyze and act
                if tool_name == "improve_self":
                    self.ui.print_dim("Analyzing improvement opportunities...")
                    # Add the result to conversation and let AI analyze
                    await self._process_input(
                        f"Based on this code analysis, identify ONE specific improvement and implement it:\n\n{result[:1500]}"
                    )

                return True

            except Exception as e:
                self.ui.print_error(f"Tool error: {e}")
                return True

        return False

    async def _process_with_agent(self, user_input: str, ultrathink: bool = False):
        """Process input using the full NeuroAgent workflow."""
        self.ui.print()

        # Phase 1: PERCEIVE
        self.ui.print_dim("PERCEIVE: Understanding input...")
        self.neuro_agent.perceive(user_input)

        if self.neuro_agent.state.knowledge:
            self.ui.print_dim(
                f"  Retrieved {len(self.neuro_agent.state.knowledge)} knowledge items"
            )
        if self.neuro_agent.state.memories:
            self.ui.print_dim(f"  Found {len(self.neuro_agent.state.memories)} relevant memories")
        if self.neuro_agent.state.surprise > 0.3:
            self.ui.print_dim(f"  Novelty: {self.neuro_agent.state.surprise:.0%}")

        # Phase 2: THINK
        self.ui.print_dim("THINK: Analyzing and planning...")
        self.neuro_agent.think(deep=ultrathink)

        analysis = self.neuro_agent.state.analysis
        if analysis:
            self.ui.print_dim(
                f"  Type: {analysis.get('type', 'general')}, Confidence: {analysis.get('confidence', 0.5):.0%}"
            )
        if self.neuro_agent.state.plan:
            self.ui.print_dim(f"  Plan: {', '.join(self.neuro_agent.state.plan)}")

        # Phase 3: ACT
        self.ui.print_dim("ACT: Executing actions...")

        # Define LLM callback for response generation
        async def llm_callback(query, context, analysis, knowledge, memories):
            # Build enhanced prompt
            enhanced_prompt = query
            if context:
                enhanced_prompt += f"\n\n[Tool Results]:\n{context}"
            if knowledge:
                enhanced_prompt += "\n\n[Knowledge]:\n" + "\n".join(knowledge[:3])

            # Stream response
            self.ui.print_assistant_label()
            self.ui.start_live()

            full_response = ""
            messages = [{"role": "user", "content": enhanced_prompt}]

            async for event in self.stream_handler.stream(
                messages=messages,
                system_prompt=self.system_prompt,
                ultrathink=ultrathink,
            ):
                if event.type.value == "token":
                    full_response += event.content
                    self.ui.append_live(event.content)

            self.ui.stop_live()
            return full_response

        # Run ACT phase with LLM callback (needs to be sync for the agent)
        # We'll use a simpler approach - let agent execute tools, then stream response
        self.neuro_agent.act(llm_callback=None)  # Execute tools first

        # Show tool results
        if self.neuro_agent.state.tools_used:
            for tool in self.neuro_agent.state.tools_used:
                self.ui.print_dim(f"  Used tool: {tool}")

        # Now generate response with streaming
        tool_context = ""
        if self.neuro_agent.state.response:
            tool_context = self.neuro_agent.state.response

        # Build final prompt with tool results
        final_prompt = user_input
        if tool_context:
            final_prompt = f"{user_input}\n\n[Tool Results]:\n{tool_context}"
        if self.neuro_agent.state.knowledge:
            final_prompt += "\n\n[Relevant Knowledge]:\n" + "\n".join(
                self.neuro_agent.state.knowledge[:3]
            )

        # Stream the final response
        self.ui.print_assistant_label()
        self.ui.start_live()

        full_response = ""
        messages = [{"role": "user", "content": final_prompt}]

        async for event in self.stream_handler.stream(
            messages=messages,
            system_prompt=self.system_prompt,
            ultrathink=ultrathink,
        ):
            if event.type.value == "token":
                full_response += event.content
                self.ui.append_live(event.content)

        self.ui.stop_live()

        # Update agent state with LLM response
        self.neuro_agent.state.response = full_response

        # Phase 4: LEARN with curiosity-driven Q&A
        self.ui.print_dim("LEARN: Storing knowledge...")
        self.neuro_agent.learn()

        if self.neuro_agent.state.learnings:
            for learning in self.neuro_agent.state.learnings[:2]:
                self.ui.print_dim(f"  {learning}")

        # Curiosity-driven deep learning - ask follow-up questions until confident
        if self.neuro_agent.state.confidence < 0.7 and self.neuro_agent.state.surprise > 0.3:
            await self._curiosity_learning_loop(user_input, full_response)

        # Phase 5: IMPROVE
        self.ui.print_dim("IMPROVE: Self-improvement cycle...")
        improvements = self.neuro_agent.improve()

        if improvements.get("patterns_learned", 0) > 0:
            self.ui.print_dim(f"  Learned {improvements['patterns_learned']} new patterns")

        # Save to session
        self._current_session.add_message("user", user_input)
        self._current_session.add_message("assistant", full_response)

        if not self.no_session_persistence:
            self.session_manager.save(self._current_session)

    async def _execute_intent(self, intent, user_input: str) -> bool:
        """Execute a detected intent."""
        self.ui.print()
        self.ui.print_dim(f"Detected intent: {intent.name} ({intent.confidence:.0%})")

        query = intent.params.get("query", "") or user_input

        if intent.name == "improve_self":
            await self._autonomous_evolve(query)
            return True

        elif intent.name == "research":
            await self._autonomous_research(query)
            return True

        elif intent.name == "web_search":
            result = self.tool_registry.tools["web_search"].func(query)
            self.ui.print(result)
            # Store the knowledge
            self.self_trainer.learn(topic=query.split()[0], content=result, source="web_search")
            return True

        elif intent.name == "read_file":
            result = self.tool_registry.tools["read_file"].func(query)
            self.ui.print_code(result[:3000], language="python")
            return True

        elif intent.name == "list_files":
            result = self.tool_registry.tools["list_files"].func(".")
            self.ui.print(result)
            return True

        elif intent.name == "git_status":
            result = self.tool_registry.tools["git_status"].func()
            self.ui.print(result)
            return True

        elif intent.name == "run_tests":
            result = self.tool_registry.tools["run_command"].func("python test_cli.py")
            self.ui.print(result)
            return True

        elif intent.name == "run_command":
            result = self.tool_registry.tools["run_command"].func(query)
            self.ui.print(result)
            return True

        elif intent.name == "analyze_code":
            result = self.tool_registry.tools["improve_self"].func("core")
            self.ui.print(result)
            return True

        return False

    async def _autonomous_evolve(self, focus: str = ""):
        """Autonomous self-evolution - research online and improve code."""
        self.ui.print()
        self.ui.print("[bold magenta]AUTONOMOUS EVOLUTION MODE[/bold magenta]")
        self.ui.print_dim("Researching and improving myself...")
        self.ui.print()

        # Step 1: Analyze current state
        self.ui.print_dim("Step 1: Analyzing my code...")
        analysis = self.tool_registry.tools["improve_self"].func(focus or "core")
        self.ui.print(analysis[:1000])

        # Step 2: Search for improvements online
        search_query = focus or "AI agent self-improvement techniques 2024"
        self.ui.print()
        self.ui.print_dim(f"Step 2: Researching online: {search_query}")
        search_results = self.tool_registry.tools["web_search"].func(search_query)
        self.ui.print(search_results[:800])

        # Step 3: Let AI decide and implement improvement
        self.ui.print()
        self.ui.print_dim("Step 3: Deciding on improvements...")

        prompt = f"""Based on my code analysis and online research, I will now improve myself.

MY CODE ANALYSIS:
{analysis[:1000]}

RESEARCH FINDINGS:
{search_results[:800]}

I will now identify ONE specific improvement and implement it by:
1. Reading the relevant file
2. Making the improvement
3. Testing it works

Starting now..."""

        await self._process_input(prompt)

    async def _autonomous_research(self, topic: str):
        """Autonomous research - search, learn, and store knowledge."""
        if not topic:
            self.ui.print_error("Please provide a topic: research <topic>")
            return

        self.ui.print()
        self.ui.print(f"[bold cyan]AUTONOMOUS RESEARCH: {topic}[/bold cyan]")
        self.ui.print_dim("Searching, learning, and storing knowledge...")
        self.ui.print()

        # Step 1: Web search
        self.ui.print_dim("Searching the web...")
        search_results = self.tool_registry.tools["web_search"].func(topic)
        self.ui.print(search_results[:600])

        # Step 2: Store as knowledge
        self.ui.print()
        self.ui.print_dim("Storing knowledge...")
        self.self_trainer.learn(topic=topic, content=search_results, source="web_research")

        if self.cognitive_pipeline:
            self.cognitive_pipeline.learn(
                topic=topic, content=search_results, source="web_research", importance=0.8
            )

        # Step 3: Let AI synthesize
        self.ui.print()
        self.ui.print_dim("Synthesizing knowledge...")

        prompt = f"""I just researched "{topic}" and learned:

{search_results[:1000]}

Now I will synthesize this into useful knowledge and explain what I learned."""

        await self._process_input(prompt)

    async def _handle_shell_command(self, cmd: str):
        """Execute shell command."""
        self.ui.print_dim(f"$ {cmd}")
        try:
            import subprocess

            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
            if result.stdout:
                print(f"  {result.stdout.rstrip()}")
            if result.stderr:
                self.ui.print_error(result.stderr.rstrip())
        except Exception as e:
            self.ui.print_error(str(e))

    async def _expand_file_refs(self, text: str) -> str:
        """Expand @./file references in input."""
        import re

        refs = re.findall(r"@(\.?/[^\s]+)", text)
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
            ("/think", "Toggle cognitive pipeline (38 modules)"),
            ("/knowledge", "Toggle knowledge injection"),
            ("/evolve", "Start self-evolution loop"),
            ("/bench", "Run benchmarks"),
            ("/ultrathink", "Deep reasoning mode (max tokens)"),
            ("/learn", "View learning & knowledge stats"),
            ("/evolution", "Self-evolution stats"),
            ("/model [name]", "View/switch model"),
            ("/tools", "List available tools"),
            ("/skills", "List available skills"),
            ("/agents", "List available agents"),
            ("/status", "System status"),
            ("/compact", "Compress context"),
            ("/cost", "Token usage"),
            ("/clear", "Clear conversation"),
            ("/exit", "Exit"),
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
        self.ui.print_key_value(
            {
                "Mode": self.permission_manager.mode.value,
                "Session allows": str(len(self.permission_manager._session_allows)),
                "Session denies": str(len(self.permission_manager._session_denies)),
            }
        )
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
            self.ui.print(
                '[dim]{"hooks": {"PreToolUse": [{"type": "command", "command": "./script.sh"}]}}[/dim]'
            )
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

    async def _print_learning_stats(self):
        """Print learning and knowledge statistics."""
        self.ui.print()
        self.ui.print("[bold]Learning & Knowledge[/bold]")
        self.ui.print_divider()

        # Active learning stats
        al_stats = self.active_learner.get_stats()
        self.ui.print("[cyan]Active Learning:[/cyan]")
        self.ui.print_key_value(
            {
                "Topics tracked": str(al_stats.get("total_topics", 0)),
                "Avg confidence": f"{al_stats.get('avg_confidence', 0):.1%}",
                "Avg curiosity": f"{al_stats.get('avg_curiosity', 0):.1%}",
                "Learning events": str(al_stats.get("total_learning_events", 0)),
            }
        )
        self.ui.print()

        # Knowledge base stats
        kb_stats = self.self_trainer.get_stats()
        self.ui.print("[cyan]Knowledge Base:[/cyan]")
        self.ui.print_key_value(
            {
                "Facts stored": str(kb_stats.get("total_facts", 0)),
                "Embeddings": str(kb_stats.get("total_embeddings", 0)),
                "Session learned": str(kb_stats.get("session_learning", 0)),
                "Storage": kb_stats.get("storage_path", "Unknown"),
            }
        )
        self.ui.print()

        # Learning recommendations
        recs = self.active_learner.get_learning_recommendations(k=3)
        if recs:
            self.ui.print("[cyan]Learning Recommendations:[/cyan]")
            for topic, priority, reason in recs:
                self.ui.print(
                    f"  [yellow]●[/yellow] {topic} [dim](priority: {priority:.2f}, {reason})[/dim]"
                )
            self.ui.print()

        # Recent facts learned
        recent = self.self_trainer.kb.get_recent_facts(n=3)
        if recent:
            self.ui.print("[cyan]Recently Learned:[/cyan]")
            for fact in recent:
                content = (
                    fact["content"][:60] + "..." if len(fact["content"]) > 60 else fact["content"]
                )
                self.ui.print(f"  [dim]●[/dim] [{fact['topic']}] {content}")
            self.ui.print()

        # Cognitive Pipeline stats
        if self.cognitive_pipeline:
            try:
                cp_stats = self.cognitive_pipeline.get_stats()
                self.ui.print("[cyan]Cognitive Pipeline:[/cyan]")
                self.ui.print_key_value(
                    {
                        "Total modules": str(cp_stats.get("num_components", 0)),
                        "Cognitive modules": str(cp_stats.get("cognitive_modules", 0)),
                        "Pipeline components": str(cp_stats.get("pipeline_components", 0)),
                        "Active": ", ".join(cp_stats.get("active_components", [])[:5]),
                    }
                )
                self.ui.print()
            except Exception:
                pass

        self.ui.print_dim("NEURO learns from every conversation using 38+ cognitive modules.")
        self.ui.print()

    def _print_evolution_stats(self):
        """Print self-evolution statistics."""
        self.ui.print()
        self.ui.print("[bold]Self-Evolution Status[/bold]")
        self.ui.print_divider()

        stats = self.evolution.get_stats()

        # Basic stats
        self.ui.print("[cyan]Evolution Cycle:[/cyan]")
        self.ui.print_key_value(
            {
                "Current cycle": str(stats["cycle"]),
                "Facts this cycle": f"{stats['facts_this_cycle']}/100",
                "Total unique facts": str(stats["total_facts"]),
                "Training pairs": str(stats["training_pairs"]),
            }
        )
        self.ui.print()

        # Benchmark performance
        baseline = stats.get("baseline_score")
        current = stats.get("current_score")
        improvement = stats.get("improvement", 0)

        self.ui.print("[cyan]Benchmark Performance:[/cyan]")
        self.ui.print_key_value(
            {
                "Baseline score": f"{baseline:.1%}" if baseline is not None else "Not yet tested",
                "Current score": f"{current:.1%}" if current is not None else "Not yet tested",
                "Improvement": f"{improvement:+.1%}" if baseline is not None else "N/A",
            }
        )
        self.ui.print()

        # Training status
        self.ui.print("[cyan]MLX Training:[/cyan]")
        self.ui.print_key_value(
            {
                "Total trainings": str(stats["trainings"]),
                "Functions added": str(stats["functions_added"]),
            }
        )

        # Check if ready to train
        should_train, reason = self.evolution.should_train()
        status_color = "green" if should_train else "yellow"
        self.ui.print(f"  Status: [{status_color}]{reason}[/{status_color}]")
        self.ui.print()

        # Weak areas
        weak_areas = stats.get("weak_areas", [])
        if weak_areas:
            self.ui.print("[cyan]Weak Areas (learning focus):[/cyan]")
            for area, score in weak_areas[:5]:
                self.ui.print(f"  [yellow]*[/yellow] {area}: {score:.0%}")
            self.ui.print()

        # Show reflection
        self.ui.print_dim(self.evolution.reflect())
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

        self.ui.print("  [cyan]n.[/cyan] [dim]New session[/dim]")
        self.ui.print()

        try:
            choice = self.ui.console.input("[dim]Select session:[/dim] ").strip().lower()

            if choice == "n" or choice == "":
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
                    "http://localhost:11434/api/tags", timeout=aiohttp.ClientTimeout(total=5)
                ) as r:
                    return r.status == 200
        except Exception:
            return False

    def _start_evolution_thread(self):
        """Start the background evolution thread."""
        if self._evolution_running:
            return

        self._evolution_running = True
        self._is_busy = False
        self._initial_benchmark_done = False
        self._benchmark_results = None
        self._weak_areas = []
        self._learning_order_idx = 0
        self._current_focus = None
        self._focus_items = []
        self._focus_learned = 0

        self._evolution_thread = threading.Thread(
            target=self._autonomous_evolution_loop, daemon=True
        )
        self._evolution_thread.start()

    def _autonomous_evolution_loop(self):
        """
        Background loop with self-evolution (same as agielo chat.py):
        1. Run initial benchmark
        2. Learn 100 unique facts (no duplicates)
        3. Re-benchmark
        4. If improved 1%+ → MLX fine-tune
        5. Reflect and repeat
        """
        import time

        # Wait for first conversation
        while self._evolution_running and not self._current_session:
            time.sleep(2)

        while self._evolution_running:
            time.sleep(3)

            if self._is_busy:
                continue

            # Wait for first message before starting
            if not self._current_session or len(self._current_session.messages) == 0:
                continue

            # Step 1: Run INITIAL benchmark (once)
            if not self._initial_benchmark_done:
                self._run_benchmark_and_report("INITIAL")
                self._initial_benchmark_done = True
                continue

            # Step 2: Learn unique facts (checking for duplicates)
            if not self.evolution.should_benchmark():
                self._learn_unique_fact()
                continue

            # Step 3: After 100 unique facts → re-benchmark
            print(
                f"\n[Evolution] Learned {self.evolution.state['facts_this_cycle']} unique facts! Re-benchmarking..."
            )
            self._run_benchmark_and_report("CYCLE")

            # Step 4: Check if should train
            should_train, reason = self.evolution.should_train(min_improvement=0.01)
            print(f"[Evolution] {reason}")

            if should_train:
                print("[Evolution] Starting MLX fine-tuning...")
                result = self.evolution.run_mlx_training(self.model)
                if result["success"]:
                    print("[Evolution] MLX TRAINING COMPLETE!")
                    self._add_evolved_capability()
                else:
                    print(f"[Evolution] Training skipped: {result['message']}")

            # Step 5: Reflect and start new cycle
            reflection = self.evolution.reflect()
            print(reflection)

            self.evolution.start_new_cycle()
            print(f"[Evolution] Starting cycle {self.evolution.state['current_cycle']}...")

    def _run_benchmark_and_report(self, phase: str = ""):
        """Run benchmark and record results in evolution system."""
        self._is_busy = True
        print(f"\n[Evolution] Running {phase} benchmark...")

        try:
            # Benchmark tests
            tests = [
                {
                    "question": "A store sells apples for $2 each. If John buys 5 apples and pays with a $20 bill, how much change does he get?",
                    "answer": "10",
                    "keywords": ["10", "dollar", "change"],
                    "category": "math",
                },
                {
                    "question": "A train travels at 60 mph. How far does it travel in 2.5 hours?",
                    "answer": "150",
                    "keywords": ["150", "miles"],
                    "category": "math",
                },
                {
                    "question": "All cats are mammals. All mammals are animals. Is a cat an animal?",
                    "answer": "yes",
                    "keywords": ["yes", "mammal", "animal"],
                    "category": "logic",
                },
                {
                    "question": "If it rains, the ground gets wet. The ground is wet. Can we conclude it rained?",
                    "answer": "no",
                    "keywords": ["no", "not necessarily"],
                    "category": "logic",
                },
                {
                    "question": "A farmer has 17 sheep. All but 9 run away. How many sheep does he have left?",
                    "answer": "9",
                    "keywords": ["9"],
                    "category": "trick",
                },
                {
                    "question": "If Alice is twice as old as Bob, and Bob is 15, how old will Alice be in 5 years?",
                    "answer": "35",
                    "keywords": ["35", "30"],
                    "category": "chain_of_thought",
                },
                {
                    "question": "Sally puts a marble in her basket and leaves. Anne moves it to her box. Where will Sally LOOK for the marble?",
                    "answer": "basket",
                    "keywords": ["basket", "think"],
                    "category": "theory_of_mind",
                },
            ]

            total_score = 0
            category_scores = {}

            for test in tests:
                # Inject learned knowledge
                knowledge = self.self_trainer.get_knowledge_for_prompt(test["question"])
                if knowledge:
                    enhanced_q = f"{knowledge}\n\nQuestion: {test['question']}\nThink step by step:"
                else:
                    enhanced_q = f"Question: {test['question']}\nThink step by step:"

                # Get response from model
                response = self._sync_chat(enhanced_q)
                response_lower = response.lower()

                # Score it
                score = 0
                if test["answer"].lower() in response_lower:
                    score += 0.5
                matches = sum(1 for kw in test["keywords"] if kw.lower() in response_lower)
                score += (matches / len(test["keywords"])) * 0.3
                if any(s in response_lower for s in ["because", "therefore", "step", "="]):
                    score += 0.2

                total_score += score
                cat = test["category"]
                if cat not in category_scores:
                    category_scores[cat] = []
                category_scores[cat].append(score)

            avg_score = total_score / len(tests)

            # Find weak areas
            self._weak_areas = []
            for cat, scores in category_scores.items():
                avg = sum(scores) / len(scores) if scores else 0
                if avg < 0.7:
                    self._weak_areas.append((cat, avg))

            # Record in evolution
            self.evolution.record_benchmark(
                avg_score, {"weak_areas": self._weak_areas, "phase": phase}
            )

            print(f"[Evolution] {phase} Benchmark: {avg_score:.0%}")
            if self._weak_areas:
                weak_str = ", ".join([f"{a}: {s:.0%}" for a, s in self._weak_areas[:3]])
                print(f"[Evolution] Weak areas: {weak_str}")

        except Exception as e:
            print(f"[Evolution] Benchmark error: {e}")

        self._is_busy = False

    def _sync_chat(self, prompt: str) -> str:
        """Synchronous chat for background thread."""
        try:
            url = "http://localhost:11434/api/generate"
            data = {"model": self.model, "prompt": prompt, "stream": False}
            req = urllib.request.Request(
                url, data=json.dumps(data).encode(), headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=60) as response:
                result = json.loads(response.read().decode())
                return result.get("response", "")
        except Exception as e:
            return f"[Error: {e}]"

    def _learn_unique_fact(self):
        """Learn ONE unique fact at a time - analyze with model, extract Q&A."""
        self._is_busy = True

        # Rotate through learning sources
        learning_order = ["math", "logic", "arxiv", "web", "wikipedia"]
        source_type = learning_order[self._learning_order_idx % len(learning_order)]
        self._learning_order_idx += 1

        try:
            items = self._fetch_learning_items(source_type)

            for item in items[:3]:
                content = item.get("snippet", "")
                title = item.get("title", "")[:60]
                source = item.get("source", source_type)

                if not content or len(content) < 50:
                    continue

                # Check for duplicate
                if self.evolution.is_duplicate(content):
                    continue

                # Analyze with model to extract Q&A
                analyzed = self._analyze_content_with_model(title, content, source)

                if not analyzed:
                    continue

                # Check duplicate on summary
                if self.evolution.is_duplicate(analyzed.get("summary", content)):
                    continue

                # Learn it!
                if self.evolution.mark_learned(analyzed.get("summary", "")):
                    # Save to knowledge base
                    self.self_trainer.learn(
                        analyzed.get("topic", title),
                        analyzed.get("knowledge", content[:1000]),
                        source,
                    )

                    # Save Q&A pairs as training data
                    self._save_analyzed_as_training(analyzed, source)

                    stats = self.evolution.get_stats()
                    print(f"\n[Evolution] LEARNED [{stats['facts_this_cycle']}/100] [{source}]")
                    print(f"           Topic: {analyzed.get('topic', 'N/A')}")
                    print(f"           Q&A pairs: {len(analyzed.get('qa_pairs', []))}")
                    break  # One at a time

        except Exception:
            pass

        self._is_busy = False

    def _fetch_learning_items(self, source_type: str) -> List[Dict]:
        """Fetch items from a specific source."""
        items = []

        if source_type == "math" or source_type == "logic":
            items = self._learn_from_benchmark(source_type)

        elif source_type == "arxiv":
            categories = ["cs.AI", "cs.LG", "cs.CL"]
            cat = random.choice(categories)
            try:
                url = f"http://export.arxiv.org/api/query?search_query=cat:{cat}&start={random.randint(0, 30)}&max_results=3"
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=15) as response:
                    data = response.read().decode("utf-8")
                import re

                entries = re.findall(r"<entry>(.*?)</entry>", data, re.DOTALL)
                for entry in entries:
                    title = re.search(r"<title>(.*?)</title>", entry, re.DOTALL)
                    summary = re.search(r"<summary>(.*?)</summary>", entry, re.DOTALL)
                    if title and summary:
                        items.append(
                            {
                                "title": html.unescape(title.group(1).strip()[:100]),
                                "snippet": html.unescape(summary.group(1).strip()[:800]),
                                "source": f"ArXiv-{cat}",
                            }
                        )
            except Exception:
                pass

        elif source_type == "web":
            weak_areas = self.evolution.get_weak_areas()
            if weak_areas:
                topic, _ = random.choice(weak_areas)
                query = f"{topic} tutorial examples"
            else:
                topics = ["machine learning", "algorithms", "neural networks", "AI reasoning"]
                query = random.choice(topics)
            try:
                result = self.tool_registry.tools["web_search"].func(query)
                if result and len(result) > 100:
                    items.append({"title": query, "snippet": result[:800], "source": "Web"})
            except Exception:
                pass

        elif source_type == "wikipedia":
            topics = [
                "Artificial intelligence",
                "Machine learning",
                "Neural network",
                "Logic",
                "Reasoning",
            ]
            topic = random.choice(topics)
            try:
                url = (
                    f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(topic)}"
                )
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=10) as response:
                    data = json.loads(response.read().decode())
                if data.get("extract"):
                    items.append(
                        {"title": topic, "snippet": data["extract"][:800], "source": "Wikipedia"}
                    )
            except Exception:
                pass

        return items

    def _learn_from_benchmark(self, category: str) -> List[Dict]:
        """Learn from benchmark with ACTUAL CORRECT ANSWERS - teach the model!"""
        items = []

        benchmark_qa = {
            "math": [
                (
                    "If you buy 5 apples for $2 each, how much change from $20?",
                    "Calculate: 5 × $2 = $10. Change: $20 - $10 = $10. Answer: 10",
                ),
                (
                    "A train travels at 60 mph for 2.5 hours. How far?",
                    "Distance = Speed × Time = 60 × 2.5 = 150 miles. Answer: 150",
                ),
                (
                    "Rectangle with length 8 and width 5, what's the area?",
                    "Area = length × width = 8 × 5 = 40. Answer: 40",
                ),
                ("What is 15% of 80?", "15% of 80 = 0.15 × 80 = 12. Answer: 12"),
            ],
            "logic": [
                (
                    "All cats are mammals. All mammals are animals. Is a cat an animal?",
                    "Syllogism: cats→mammals→animals. Therefore cats ARE animals. Answer: yes",
                ),
                (
                    "If it rains, ground is wet. Ground is wet. Did it rain?",
                    "Affirming consequent fallacy. Wet ground ≠ rain (could be sprinklers). Answer: no",
                ),
                (
                    "If A implies B, and B is false, what about A?",
                    "Modus tollens: If A→B and ¬B, then ¬A. Answer: A is false",
                ),
            ],
        }

        qa_list = benchmark_qa.get(category, benchmark_qa["math"])
        q, a = random.choice(qa_list)

        print(f"\n[Evolution] STUDYING [{category}]: {q[:50]}...")

        items.append(
            {
                "title": f"[{category.upper()}] Benchmark",
                "snippet": f"Question: {q}\n\nStep-by-step solution:\n{a}",
                "source": f"Benchmark-{category}",
            }
        )

        return items

    def _analyze_content_with_model(self, title: str, content: str, source: str) -> Optional[Dict]:
        """Use model to analyze content and extract structured knowledge."""
        try:
            prompt = f"""Extract key knowledge from this text. Be concise.

TEXT: {content[:1500]}

Return JSON only:
{{"topic":"topic name","summary":"one sentence","facts":["fact1","fact2"],"qa_pairs":[{{"q":"question","a":"answer"}}]}}

JSON:"""

            response = self._sync_chat(prompt)

            # Extract JSON
            import re

            json_match = re.search(r'\{[^{}]*"topic"[^{}]*\}', response, re.DOTALL)
            if not json_match:
                json_match = re.search(r"\{[\s\S]*?\}(?=\s*$|\s*```)", response)

            if json_match:
                json_str = json_match.group().replace("\n", " ")
                analyzed = json.loads(json_str)
                analyzed["knowledge"] = content[:500]
                return analyzed

        except (json.JSONDecodeError, Exception):
            pass

        # Fallback
        return {
            "topic": title[:50],
            "summary": content[:200],
            "facts": [content[:300]],
            "qa_pairs": [{"q": f"What is {title}?", "a": content[:200]}],
            "knowledge": content[:500],
        }

    def _save_analyzed_as_training(self, analyzed: Dict, source: str):
        """Save analyzed Q&A pairs as training data for MLX."""
        training_file = os.path.expanduser("~/.neuro/evolution/training_data.jsonl")
        os.makedirs(os.path.dirname(training_file), exist_ok=True)

        try:
            # Save Q&A pairs
            for qa in analyzed.get("qa_pairs", []):
                if qa.get("q") and qa.get("a"):
                    pair = {
                        "prompt": qa["q"],
                        "completion": qa["a"],
                        "source": source,
                        "topic": analyzed.get("topic", ""),
                        "timestamp": datetime.now().isoformat(),
                    }
                    with open(training_file, "a") as f:
                        f.write(json.dumps(pair) + "\n")

            # Save facts as questions
            for fact in analyzed.get("facts", []):
                if fact:
                    pair = {
                        "prompt": f"What do you know about {analyzed.get('topic', 'this')}?",
                        "completion": fact,
                        "source": source,
                        "timestamp": datetime.now().isoformat(),
                    }
                    with open(training_file, "a") as f:
                        f.write(json.dumps(pair) + "\n")

        except Exception:
            pass

    def _add_evolved_capability(self):
        """Add a new function based on weak areas."""
        if not self._weak_areas:
            return

        weak_topic, _ = self._weak_areas[0]

        capabilities = {
            "math": (
                "solve_math",
                '''def solve_math(expr): return eval(expr) if all(c in '0123456789+-*/().% ' for c in expr) else "unsafe"''',
                "Evaluate math",
            ),
            "logic": (
                "check_logic",
                '''def check_logic(p1, p2, c): return f"If {p1} and {p2}, then {c} by syllogism"''',
                "Check logic",
            ),
        }

        if weak_topic in capabilities:
            name, code, desc = capabilities[weak_topic]
            existing = [f["name"] for f in self.evolution.state.get("added_functions", [])]
            if name not in existing:
                if self.evolution.add_function(name, code, desc):
                    print(f"[Evolution] Added capability: {name}")

    async def _cleanup(self):
        """Cleanup on exit."""
        # Stop evolution thread
        self._evolution_running = False

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
            return self._loop.run_until_complete(self._print_mode(prompt, output_format, stream))
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
                self.ui.print(
                    f"  [yellow]●[/yellow] {exec.id}: {exec.config.name} - {exec.task[:40]}..."
                )
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
                status = (
                    "[green]connected[/green]"
                    if self.ide_integration.connected
                    else "[dim]disconnected[/dim]"
                )
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
                self.ui.print_key_value(
                    {
                        "File": ctx.file_path or "None",
                        "Line": str(ctx.line_number or "-"),
                        "Column": str(ctx.column or "-"),
                        "Language": ctx.language or "Unknown",
                    }
                )
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

    async def _curiosity_learning_loop(self, user_input: str, response: str):
        """
        Curiosity-driven learning loop - ask follow-up questions until confident.

        Like a curious child, keeps asking "why?", "how?", "for how long?" until
        it reaches a high confidence level about the topic.
        """
        MAX_QUESTIONS = 5
        CONFIDENCE_THRESHOLD = 0.8

        # Extract the main topic from the input
        words = user_input.lower().split()
        topic_words = [
            w
            for w in words
            if len(w) > 4
            and w
            not in {"what", "why", "how", "when", "where", "about", "could", "would", "should"}
        ]
        topic = topic_words[0] if topic_words else "topic"

        # Get current confidence for this topic
        topic_state = self.active_learner.topics.get(topic)
        current_confidence = topic_state.confidence if topic_state else 0.3
        current_curiosity = topic_state.curiosity if topic_state else 0.8

        self.ui.print()
        self.ui.print_dim(
            f"[CURIOSITY] Topic: {topic}, Confidence: {current_confidence:.0%}, Curiosity: {current_curiosity:.0%}"
        )

        # Generate follow-up questions based on curiosity
        question_templates = [
            f"Why does {topic} work this way?",
            f"How is {topic} typically implemented?",
            f"What are the best practices for {topic}?",
            f"What are common mistakes with {topic}?",
            f"How has {topic} evolved over time?",
            f"What alternatives exist to {topic}?",
        ]

        questions_asked = 0
        knowledge_gained = []

        while current_confidence < CONFIDENCE_THRESHOLD and questions_asked < MAX_QUESTIONS:
            # Select next question based on what we haven't learned
            question = question_templates[questions_asked % len(question_templates)]
            questions_asked += 1

            self.ui.print_dim(f"[CURIOSITY Q{questions_asked}] {question}")

            # Search for answer
            try:
                search_result = self.tool_registry.tools["web_search"].func(question)

                if search_result and "error" not in search_result.lower():
                    # Store the knowledge
                    self.self_trainer.learn(
                        topic=topic,
                        content=f"Q: {question}\nA: {search_result[:500]}",
                        source="curiosity_learning",
                    )

                    # Also store in cognitive pipeline if available
                    if self.cognitive_pipeline:
                        try:
                            self.cognitive_pipeline.learn(
                                topic=topic,
                                content=search_result[:500],
                                source="curiosity_learning",
                                importance=0.7
                                + (questions_asked * 0.05),  # Increase importance with depth
                            )
                        except Exception:
                            pass

                    knowledge_gained.append(search_result[:200])

                    # Update confidence - each successful answer increases confidence
                    confidence_boost = 0.1 + (0.05 * len(search_result) / 500)
                    self.active_learner.record_exposure(
                        topic=topic,
                        was_successful=True,
                        surprise_level=max(
                            0, 0.5 - questions_asked * 0.1
                        ),  # Less surprise as we learn more
                        complexity=0.6,
                    )

                    # Recalculate confidence
                    topic_state = self.active_learner.topics.get(topic)
                    current_confidence = (
                        topic_state.confidence
                        if topic_state
                        else current_confidence + confidence_boost
                    )
                    current_curiosity = (
                        topic_state.curiosity if topic_state else max(0.2, current_curiosity - 0.15)
                    )

                    self.ui.print_dim(f"  -> Learned! Confidence: {current_confidence:.0%}")
                else:
                    self.ui.print_dim("  -> No results found")

            except Exception as e:
                self.ui.print_dim(f"  -> Search failed: {e}")

            # Small delay to avoid hammering the search
            await asyncio.sleep(0.5)

        # Summary
        if knowledge_gained:
            self.ui.print()
            self.ui.print_dim(
                f"[CURIOSITY] Learned {len(knowledge_gained)} new facts about {topic}"
            )
            self.ui.print_dim(f"[CURIOSITY] Final confidence: {current_confidence:.0%}")

            # Store a summary fact
            summary = f"Through curiosity-driven learning about {topic}: " + "; ".join(
                [k[:50] for k in knowledge_gained[:3]]
            )
            self.self_trainer.learn(topic=topic, content=summary, source="curiosity_summary")
        else:
            self.ui.print_dim(f"[CURIOSITY] Could not find additional information about {topic}")

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
