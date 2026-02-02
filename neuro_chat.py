#!/usr/bin/env python3
"""
NEURO Chat - Beautiful Terminal Interface
==========================================
The easiest way to use NEURO. Just run: python neuro_chat.py

Features:
- Beautiful Rich terminal UI
- Streaming responses
- File context
- Conversation history
- One-command setup
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
    from rich.prompt import Prompt
    from rich.live import Live
    from rich.text import Text
    from rich.theme import Theme
    from rich.progress import Progress, SpinnerColumn, TextColumn
except ImportError:
    print("Installing Rich for beautiful terminal UI...")
    os.system(f"{sys.executable} -m pip install rich -q")
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown
    from rich.table import Table
    from rich.prompt import Prompt
    from rich.live import Live
    from rich.text import Text
    from rich.theme import Theme
    from rich.progress import Progress, SpinnerColumn, TextColumn

# NEURO Theme
theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "red bold",
    "success": "green",
    "neuro": "bold cyan",
    "dim": "dim white",
})

console = Console(theme=theme)

# Config
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("NEURO_MODEL", "mistral")
HISTORY_FILE = Path.home() / ".neuro_history.json"


BANNER = """[bold cyan]
    ███╗   ██╗███████╗██╗   ██╗██████╗  ██████╗
    ████╗  ██║██╔════╝██║   ██║██╔══██╗██╔═══██╗
    ██╔██╗ ██║█████╗  ██║   ██║██████╔╝██║   ██║
    ██║╚██╗██║██╔══╝  ██║   ██║██╔══██╗██║   ██║
    ██║ ╚████║███████╗╚██████╔╝██║  ██║╚██████╔╝
    ╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝
[/bold cyan]
    [dim]Local AI That Learns From Your Code[/dim]
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


def stream_chat(prompt: str, history: list = None) -> str:
    """Stream a chat response from Ollama."""
    messages = []

    # System prompt
    messages.append({
        "role": "system",
        "content": "You are NEURO, a helpful AI coding assistant. Be concise and helpful. Format code with markdown code blocks."
    })

    # Add history
    if history:
        for msg in history[-10:]:  # Last 10 messages
            messages.append(msg)

    # Add current message
    messages.append({"role": "user", "content": prompt})

    data = json.dumps({
        "model": MODEL,
        "messages": messages,
        "stream": True
    }).encode()

    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"}
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
    table = Table(title="[bold cyan]NEURO Commands[/bold cyan]", border_style="cyan", show_header=True)
    table.add_column("Command", style="cyan", width=20)
    table.add_column("Description")

    commands = [
        ("/help", "Show this help"),
        ("/clear", "Clear conversation"),
        ("/file <path>", "Load file into context"),
        ("/explain", "Explain loaded file"),
        ("/fix", "Fix bugs in loaded file"),
        ("/test", "Generate tests for loaded file"),
        ("/models", "List available models"),
        ("/model <name>", "Switch model"),
        ("/exit", "Exit NEURO"),
    ]

    for cmd, desc in commands:
        table.add_row(cmd, desc)

    console.print(table)
    console.print("\n[dim]Or just type your question to chat![/dim]")


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


def save_history(history: list):
    """Save conversation history."""
    try:
        with open(HISTORY_FILE, 'w') as f:
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


def main():
    global MODEL

    console.print(Panel(BANNER, border_style="cyan", padding=(0, 2)))

    # Check Ollama
    with Progress(
        SpinnerColumn(),
        TextColumn("[cyan]Connecting to Ollama...[/cyan]"),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task("", total=None)
        connected, models = check_ollama()

    if not connected:
        console.print(Panel(
            "[error]Could not connect to Ollama![/error]\n\n"
            "1. Install Ollama: [cyan]https://ollama.ai[/cyan]\n"
            "2. Start Ollama: [code]ollama serve[/code]\n"
            "3. Pull a model: [code]ollama pull mistral[/code]",
            title="Setup Required",
            border_style="red"
        ))
        return

    # Check if model exists
    if MODEL not in models and models:
        MODEL = models[0]

    console.print(f"[success]✓[/success] Connected! Using [cyan]{MODEL}[/cyan]")
    console.print("[dim]Type /help for commands or just start chatting.[/dim]\n")

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
                    console.print("[dim]Goodbye! 👋[/dim]")
                    break

                elif cmd == "/help":
                    show_help()

                elif cmd == "/clear":
                    history = []
                    current_file = None
                    current_content = None
                    console.print("[success]✓[/success] Cleared!")

                elif cmd == "/models":
                    _, models = check_ollama()
                    show_models(models, MODEL)

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

                elif cmd == "/file":
                    if arg:
                        current_file, current_content = load_file(arg)
                        if current_content:
                            lines = len(current_content.split('\n'))
                            console.print(f"[success]✓[/success] Loaded [cyan]{current_file}[/cyan] ({lines} lines)")
                    else:
                        console.print("[warning]Usage: /file <path>[/warning]")

                elif cmd == "/explain":
                    if current_content:
                        user_input = f"Explain this code in detail:\n\n```\n{current_content[:4000]}\n```"
                    else:
                        console.print("[warning]Load a file first: /file <path>[/warning]")
                        continue

                elif cmd == "/fix":
                    if current_content:
                        user_input = f"Find and fix bugs in this code:\n\n```\n{current_content[:4000]}\n```"
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

                # If we handled a command that doesn't need chat, continue
                if cmd in ["/help", "/clear", "/models", "/model", "/file"]:
                    continue

            # Add context if file loaded
            prompt = user_input
            if current_content and not user_input.startswith("```"):
                prompt = f"Context - Current file ({current_file}):\n```\n{current_content[:2000]}\n```\n\nQuestion: {user_input}"

            # Stream response
            console.print()
            console.print("[bold cyan]NEURO[/bold cyan]", end=" ")

            response_text = ""

            try:
                with Live(console=console, refresh_per_second=15, vertical_overflow="visible") as live:
                    for token in stream_chat(prompt, history):
                        response_text += token
                        # Try markdown rendering
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
            save_history(history)

        except KeyboardInterrupt:
            console.print("\n[dim]Press Ctrl+C again or type /exit to quit[/dim]")
        except EOFError:
            break


if __name__ == "__main__":
    # Quick one-liner mode
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        prompt = " ".join(sys.argv[1:])
        connected, models = check_ollama()
        if connected:
            # Auto-select model if default not available
            if MODEL not in models and models:
                MODEL = models[0]
            console.print("[bold cyan]NEURO[/bold cyan]", end=" ")
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
