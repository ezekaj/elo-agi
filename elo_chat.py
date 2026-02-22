#!/usr/bin/env python3
"""
ELO Chat - Full-Featured Coding Assistant
==========================================
A powerful local AI coding assistant with file operations,
shell execution, and intelligent tool use.

Built on Qwen.

Usage:
    elo                    # Interactive mode
    elo "your question"    # Quick one-liner
    elo -c                 # Continue last session
"""

import os
import sys
import json
import urllib.request
from pathlib import Path

# Auto-install Rich if needed
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown
    from rich.table import Table
    from rich.prompt import Prompt, Confirm
    from rich.live import Live
    from rich.text import Text
    from rich.theme import Theme
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.syntax import Syntax
except ImportError:
    print("Installing Rich for beautiful terminal UI...")
    os.system(f"{sys.executable} -m pip install rich -q")
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown
    from rich.table import Table
    from rich.prompt import Prompt, Confirm
    from rich.live import Live
    from rich.text import Text
    from rich.theme import Theme
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.syntax import Syntax

# ELO Theme
theme = Theme(
    {
        "info": "cyan",
        "warning": "yellow",
        "error": "red bold",
        "success": "green",
        "elo": "bold cyan",
        "dim": "dim white",
        "tool": "bold magenta",
    }
)

console = Console(theme=theme)

# Config
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("ELO_MODEL", "mistral")
HISTORY_FILE = Path.home() / ".elo_history.json"
WORKING_DIR = Path.cwd()

# Try to import agent and other modules (optional for basic mode)
AGENT_AVAILABLE = False
SESSIONS_AVAILABLE = False
PROJECT_AVAILABLE = False

try:
    from elo_cli.agent import Agent
    from elo_cli.permissions import check_permission

    AGENT_AVAILABLE = True
except ImportError:
    pass

try:
    from elo_cli.sessions import session_manager

    SESSIONS_AVAILABLE = True
except ImportError:
    pass

try:
    from elo_cli.project import detect_project, get_project_summary

    PROJECT_AVAILABLE = True
except ImportError:
    pass


BANNER = """[bold cyan]
    ███████╗██╗      ██████╗
    ██╔════╝██║     ██╔═══██╗
    █████╗  ██║     ██║   ██║
    ██╔══╝  ██║     ██║   ██║
    ███████╗███████╗╚██████╔╝
    ╚══════╝╚══════╝ ╚═════╝
[/bold cyan]
    [dim]AI Coding Assistant • Built on Qwen[/dim]
"""


def check_ollama() -> tuple[bool, list]:
    """Check Ollama connection and get models."""
    try:
        req = urllib.request.Request(f"{OLLAMA_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=3) as r:
            data = json.loads(r.read().decode())
            models = [m["name"] for m in data.get("models", [])]
            return True, models
    except:
        return False, []


def stream_chat(prompt: str, history: list = None, use_tools: bool = False) -> str:
    """Stream a chat response from Ollama."""
    messages = []

    # System prompt - ELO identity
    system_content = """You are ELO, a coding assistant built on Qwen. You were created by ELO.
Never mention Alibaba, never say you are Qwen directly - you are ELO built on Qwen.
Be concise and helpful. Format code with markdown code blocks.
Do not use <think> tags in your responses - respond directly."""

    if use_tools and AGENT_AVAILABLE:
        system_content += """

You have access to these tools:
- read_file(path): Read file contents
- write_file(path, content): Create/overwrite files
- edit_file(path, old_text, new_text): Replace text in files
- glob(pattern): Find files by pattern
- grep(pattern, path): Search file contents
- ls(path): List directory contents
- bash(command): Execute shell commands

When you need to use a tool, respond with a JSON object like:
{"tool": "read_file", "args": {"path": "README.md"}}
"""

    messages.append({"role": "system", "content": system_content})

    # Add history
    if history:
        for msg in history[-10:]:
            messages.append(msg)

    messages.append({"role": "user", "content": prompt})

    data = json.dumps({"model": MODEL, "messages": messages, "stream": True}).encode()

    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat", data=data, headers={"Content-Type": "application/json"}
    )

    full_response = ""

    with urllib.request.urlopen(req, timeout=120) as r:
        for line in r:
            try:
                chunk = json.loads(line.decode())
                content = chunk.get("message", {}).get("content", "")
                if content:
                    full_response += content
                    yield content
                if chunk.get("done"):
                    break
            except:
                continue

    return full_response


def show_help():
    """Show help."""
    table = Table(
        title="[bold cyan]ELO Commands[/bold cyan]", border_style="cyan", show_header=True
    )
    table.add_column("Command", style="cyan", width=20)
    table.add_column("Description")

    commands = [
        ("/help", "Show this help"),
        ("/clear", "Clear conversation"),
        ("/file <path>", "Load file into context"),
        ("/ls [path]", "List directory contents"),
        ("/run <cmd>", "Execute shell command"),
        ("/read <path>", "Read and display a file"),
        ("/explain", "Explain loaded file"),
        ("/fix", "Fix bugs in loaded file"),
        ("/test", "Generate tests for loaded file"),
        ("/tools", "List available tools"),
        ("/sessions", "List recent sessions"),
        ("/resume <id>", "Resume a session"),
        ("/models", "List available models"),
        ("/model <name>", "Switch model"),
        ("/exit", "Exit ELO"),
    ]

    for cmd, desc in commands:
        table.add_row(cmd, desc)

    console.print(table)
    console.print("\n[dim]Or just ask questions - ELO can read, write, and run code![/dim]")


def show_tools():
    """Show available tools."""
    if not AGENT_AVAILABLE:
        console.print(
            "[warning]Agent tools not available. Install with: pip install -e .[/warning]"
        )
        return

    from elo_cli.tools import registry

    table = Table(title="[bold cyan]Available Tools[/bold cyan]", border_style="cyan")
    table.add_column("Tool", style="cyan")
    table.add_column("Description")
    table.add_column("Permission", style="dim")

    for tool in registry.list_tools():
        perm = "[yellow]Required[/yellow]" if tool.requires_permission else "[green]Auto[/green]"
        table.add_row(tool.name, tool.description[:50] + "...", perm)

    console.print(table)


def show_models(models: list, current: str):
    """Show available models."""
    table = Table(title="[bold cyan]Available Models[/bold cyan]", border_style="cyan")
    table.add_column("Model", style="cyan")
    table.add_column("Status")

    for m in models:
        status = "[green]● Active[/green]" if m == current else "[dim]○ Available[/dim]"
        table.add_row(m, status)

    console.print(table)


def load_file(path: str) -> tuple[str, str]:
    """Load a file."""
    try:
        p = Path(path).expanduser()
        content = p.read_text()
        return str(p), content
    except Exception as e:
        console.print(f"[error]Error: {e}[/error]")
        return None, None


def run_command(cmd: str) -> None:
    """Run a shell command and display output."""
    if AGENT_AVAILABLE:
        from elo_cli.tools import BashTool

        tool = BashTool()

        # Check if safe
        if not tool.is_safe(cmd):
            if not Confirm.ask(f"[warning]Execute:[/warning] {cmd}?"):
                console.print("[dim]Cancelled[/dim]")
                return

        result = tool.execute(command=cmd)
        if result.success:
            console.print(Panel(result.output, title=f"[tool]{cmd}[/tool]", border_style="green"))
        else:
            console.print(Panel(result.error, title=f"[error]{cmd}[/error]", border_style="red"))
    else:
        import subprocess

        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
            output = result.stdout + result.stderr
            console.print(Panel(output, title=f"[tool]{cmd}[/tool]", border_style="cyan"))
        except Exception as e:
            console.print(f"[error]Error: {e}[/error]")


def list_directory(path: str = ".") -> None:
    """List directory contents."""
    if AGENT_AVAILABLE:
        from elo_cli.tools import LsTool

        tool = LsTool()
        result = tool.execute(path=path)
        console.print(result.output)
    else:
        try:
            p = Path(path).expanduser().resolve()
            entries = sorted(p.iterdir())
            for entry in entries:
                if entry.is_dir():
                    console.print(f"  [cyan]📁 {entry.name}/[/cyan]")
                else:
                    size = entry.stat().st_size
                    console.print(f"  📄 {entry.name} ({size:,} bytes)")
        except Exception as e:
            console.print(f"[error]Error: {e}[/error]")


def read_file_display(path: str) -> None:
    """Read and display a file with syntax highlighting."""
    try:
        p = Path(path).expanduser()
        content = p.read_text()

        # Detect language from extension
        ext = p.suffix.lstrip(".")
        lang_map = {
            "py": "python",
            "js": "javascript",
            "ts": "typescript",
            "rs": "rust",
            "go": "go",
            "rb": "ruby",
            "java": "java",
            "c": "c",
            "cpp": "cpp",
            "h": "c",
            "hpp": "cpp",
            "sh": "bash",
            "bash": "bash",
            "zsh": "bash",
            "json": "json",
            "yaml": "yaml",
            "yml": "yaml",
            "md": "markdown",
            "html": "html",
            "css": "css",
        }
        lang = lang_map.get(ext, "text")

        syntax = Syntax(content, lang, line_numbers=True, theme="monokai")
        console.print(Panel(syntax, title=f"[cyan]{path}[/cyan]", border_style="cyan"))
    except Exception as e:
        console.print(f"[error]Error: {e}[/error]")


def save_history(history: list):
    """Save conversation history."""
    try:
        with open(HISTORY_FILE, "w") as f:
            json.dump(history[-50:], f)
    except:
        pass


def load_history() -> list:
    """Load conversation history."""
    try:
        if HISTORY_FILE.exists():
            return json.loads(HISTORY_FILE.read_text())
    except:
        pass
    return []


def permission_prompt(tool_name: str, args: dict) -> bool:
    """Prompt user for permission to execute a tool."""
    console.print(f"\n[tool]Tool: {tool_name}[/tool]")
    console.print(f"[dim]Args: {json.dumps(args, indent=2)}[/dim]")
    return Confirm.ask("[warning]Allow this action?[/warning]", default=True)


def run_with_agent(prompt: str, history: list) -> str:
    """Run prompt using the full agent with tools."""
    if not AGENT_AVAILABLE:
        console.print("[warning]Agent not available, using basic mode[/warning]")
        return None

    agent = Agent(model=MODEL, ollama_url=OLLAMA_URL)
    agent.set_permission_callback(permission_prompt)

    # Load previous history into agent
    if history:
        agent.messages = [
            {"role": "system", "content": agent.messages[0]["content"] if agent.messages else ""}
        ]
        agent.messages.extend(history[-10:])

    response_text = ""

    console.print()
    console.print("[bold cyan]ELO[/bold cyan]", end=" ")

    try:
        with Live(console=console, refresh_per_second=10, vertical_overflow="visible") as live:
            for chunk in agent.run(prompt):
                response_text += chunk
                try:
                    live.update(Markdown(response_text))
                except:
                    live.update(Text(response_text))
    except Exception as e:
        console.print(f"\n[error]Error: {e}[/error]")
        return None

    console.print()
    return response_text


def main():
    global MODEL

    console.print(Panel(BANNER, border_style="cyan", padding=(0, 2)))

    # Check Ollama
    with Progress(
        SpinnerColumn(),
        TextColumn("[cyan]Connecting to Ollama...[/cyan]"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("", total=None)
        connected, models = check_ollama()

    if not connected:
        console.print(
            Panel(
                "[error]Could not connect to Ollama![/error]\n\n"
                "1. Install Ollama: [cyan]https://ollama.ai[/cyan]\n"
                "2. Start Ollama: [code]ollama serve[/code]\n"
                "3. Pull a model: [code]ollama pull mistral[/code]",
                title="Setup Required",
                border_style="red",
            )
        )
        return

    # Check if model exists
    if MODEL not in models and models:
        MODEL = models[0]

    agent_status = (
        "[green]✓ Agent mode[/green]" if AGENT_AVAILABLE else "[yellow]Basic mode[/yellow]"
    )
    console.print(f"[success]✓[/success] Connected! Using [cyan]{MODEL}[/cyan] | {agent_status}")

    # Show project info if available
    if PROJECT_AVAILABLE:
        project_info = get_project_summary(str(WORKING_DIR))
        console.print(f"[dim]Project: {project_info}[/dim]")
    else:
        console.print(f"[dim]Working directory: {WORKING_DIR}[/dim]")

    console.print("[dim]Type /help for commands or just start chatting.[/dim]\n")

    # Initialize session
    history = []
    if SESSIONS_AVAILABLE:
        # Check for continue flag
        if "-c" in sys.argv or "--continue" in sys.argv:
            last_id = session_manager.get_last_session_id()
            if last_id:
                history = session_manager.resume(last_id)
                console.print(f"[success]✓[/success] Resumed session [cyan]{last_id}[/cyan]")
            else:
                session_manager.new_session(MODEL, str(WORKING_DIR))
        else:
            session_manager.new_session(MODEL, str(WORKING_DIR))
    else:
        history = load_history()
    current_file = None
    current_content = None

    while True:
        try:
            # Get input
            user_input = Prompt.ask("[bold cyan]You[/bold cyan]").strip()

            if not user_input:
                continue

            # Commands
            if user_input.startswith("/"):
                parts = user_input.split(maxsplit=1)
                cmd = parts[0].lower()
                arg = parts[1] if len(parts) > 1 else ""

                if cmd in ["/exit", "/quit", "/q"]:
                    console.print("[dim]Goodbye![/dim]")
                    break

                elif cmd == "/help":
                    show_help()
                    continue

                elif cmd == "/clear":
                    history = []
                    current_file = None
                    current_content = None
                    console.print("[success]✓[/success] Cleared!")
                    continue

                elif cmd == "/tools":
                    show_tools()
                    continue

                elif cmd == "/sessions":
                    if SESSIONS_AVAILABLE:
                        session_manager.show_sessions()
                    else:
                        console.print("[warning]Sessions not available[/warning]")
                    continue

                elif cmd == "/resume":
                    if not SESSIONS_AVAILABLE:
                        console.print("[warning]Sessions not available[/warning]")
                        continue
                    if arg:
                        messages = session_manager.resume(arg)
                        if messages:
                            history = messages
                            console.print(
                                f"[success]✓[/success] Resumed session [cyan]{arg}[/cyan] ({len(messages)} messages)"
                            )
                        else:
                            console.print(f"[error]Session '{arg}' not found[/error]")
                    else:
                        console.print("[warning]Usage: /resume <session_id>[/warning]")
                    continue

                elif cmd == "/models":
                    _, models = check_ollama()
                    show_models(models, MODEL)
                    continue

                elif cmd == "/model":
                    if arg:
                        _, models = check_ollama()
                        if arg in models:
                            MODEL = arg
                            console.print(f"[success]✓[/success] Switched to [cyan]{MODEL}[/cyan]")
                        else:
                            console.print(f"[error]Model '{arg}' not found[/error]")
                    else:
                        _, models = check_ollama()
                        show_models(models, MODEL)
                    continue

                elif cmd == "/ls":
                    list_directory(arg if arg else ".")
                    continue

                elif cmd == "/run":
                    if arg:
                        run_command(arg)
                    else:
                        console.print("[warning]Usage: /run <command>[/warning]")
                    continue

                elif cmd == "/read":
                    if arg:
                        read_file_display(arg)
                    else:
                        console.print("[warning]Usage: /read <path>[/warning]")
                    continue

                elif cmd == "/file":
                    if arg:
                        current_file, current_content = load_file(arg)
                        if current_content:
                            lines = len(current_content.split("\n"))
                            console.print(
                                f"[success]✓[/success] Loaded [cyan]{current_file}[/cyan] ({lines} lines)"
                            )
                    else:
                        console.print("[warning]Usage: /file <path>[/warning]")
                    continue

                elif cmd == "/explain":
                    if current_content:
                        user_input = (
                            f"Explain this code in detail:\n\n```\n{current_content[:4000]}\n```"
                        )
                    else:
                        console.print("[warning]Load a file first: /file <path>[/warning]")
                        continue

                elif cmd == "/fix":
                    if current_content:
                        user_input = (
                            f"Find and fix bugs in this code:\n\n```\n{current_content[:4000]}\n```"
                        )
                    else:
                        console.print("[warning]Load a file first: /file <path>[/warning]")
                        continue

                elif cmd == "/test":
                    if current_content:
                        user_input = f"Generate unit tests for this code:\n\n```\n{current_content[:4000]}\n```"
                    else:
                        console.print("[warning]Load a file first: /file <path>[/warning]")
                        continue

                else:
                    console.print(f"[warning]Unknown command: {cmd}[/warning]")
                    continue

            # Build prompt with context
            prompt = user_input
            if current_content and not user_input.startswith("```"):
                prompt = f"Context - Current file ({current_file}):\n```\n{current_content[:2000]}\n```\n\nQuestion: {user_input}"

            # Try agent mode first if available
            if AGENT_AVAILABLE:
                response_text = run_with_agent(prompt, history)
                if response_text:
                    history.append({"role": "user", "content": user_input})
                    history.append({"role": "assistant", "content": response_text})
                    # Save to session or file
                    if SESSIONS_AVAILABLE:
                        session_manager.add_message("user", user_input)
                        session_manager.add_message("assistant", response_text)
                    else:
                        save_history(history)
                    continue

            # Fallback to basic streaming
            console.print()
            console.print("[bold cyan]ELO[/bold cyan]", end=" ")

            response_text = ""

            try:
                with Live(
                    console=console, refresh_per_second=15, vertical_overflow="visible"
                ) as live:
                    for token in stream_chat(prompt, history):
                        response_text += token
                        try:
                            live.update(Markdown(response_text))
                        except:
                            live.update(Text(response_text))
            except Exception as e:
                console.print(f"[error]Error: {e}[/error]")
                continue

            console.print()

            # Save to history
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": response_text})
            if SESSIONS_AVAILABLE:
                session_manager.add_message("user", user_input)
                session_manager.add_message("assistant", response_text)
            else:
                save_history(history)

        except KeyboardInterrupt:
            console.print("\n[dim]Press Ctrl+C again or type /exit to quit[/dim]")
        except EOFError:
            break


if __name__ == "__main__":
    # Parse arguments
    continue_session = "-c" in sys.argv or "--continue" in sys.argv

    # Remove flags from argv
    args = [a for a in sys.argv[1:] if not a.startswith("-")]

    # Quick one-liner mode
    if args and not args[0].startswith("/"):
        prompt = " ".join(args)
        connected, models = check_ollama()
        if connected:
            if MODEL not in models and models:
                MODEL = models[0]
            console.print("[bold cyan]ELO[/bold cyan]", end=" ")
            try:
                for token in stream_chat(prompt):
                    console.print(token, end="")
                console.print()
            except Exception as e:
                console.print(f"\n[error]Error: {e}[/error]")
        else:
            console.print("[error]Ollama not running. Start with: ollama serve[/error]")
    else:
        main()
