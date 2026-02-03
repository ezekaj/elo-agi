#!/usr/bin/env python3
"""
ELO v1.0 - AI Coding Assistant with Tools
Built on Qwen, created by ELO
"""

import sys
import os
import re
import json
import glob as glob_module
import subprocess
from pathlib import Path

# Auto-install dependencies
try:
    import mlx_lm
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown
    from rich.prompt import Prompt, Confirm
    from rich.syntax import Syntax
except ImportError:
    print("Installing dependencies...")
    os.system(f"{sys.executable} -m pip install mlx-lm rich -q")
    import mlx_lm
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown
    from rich.prompt import Prompt, Confirm
    from rich.syntax import Syntax

console = Console()

MODEL_PATH = os.path.expanduser("~/Desktop/neuro-model-trained/elo-v1-final")

BANNER = """[bold cyan]
    ███████╗██╗      ██████╗
    ██╔════╝██║     ██╔═══██╗
    █████╗  ██║     ██║   ██║
    ██╔══╝  ██║     ██║   ██║
    ███████╗███████╗╚██████╔╝
    ╚══════╝╚══════╝ ╚═════╝
[/bold cyan]
    [dim]v1.0 • AI Coding Assistant • Tools Enabled[/dim]
"""

SYSTEM_PROMPT = """You are ELO, a coding assistant built on Qwen. Created by ELO.
Do not use <think> tags. Respond directly.

TOOLS AVAILABLE:
- glob(pattern): Find files matching pattern. Example: <tool>{"name": "glob", "args": {"pattern": "**/*.py"}}</tool>
- read_file(path): Read a file. Example: <tool>{"name": "read_file", "args": {"path": "README.md"}}</tool>
- ls(path): List directory. Example: <tool>{"name": "ls", "args": {"path": "."}}</tool>
- bash(command): Run shell command. Example: <tool>{"name": "bash", "args": {"command": "git status"}}</tool>
- grep(pattern, path): Search in files. Example: <tool>{"name": "grep", "args": {"pattern": "TODO", "path": "."}}</tool>
- web_search(query): Search the web. Example: <tool>{"name": "web_search", "args": {"query": "latest AI news"}}</tool>

When you need to use a tool, output: <tool>{"name": "...", "args": {...}}</tool>
After seeing results, provide a helpful summary."""


def parse_tool_calls(text):
    """Extract tool calls from response."""
    pattern = r'<tool>(.*?)</tool>'
    matches = re.findall(pattern, text, re.DOTALL)
    tools = []
    for m in matches:
        try:
            tools.append(json.loads(m.strip()))
        except:
            pass
    return tools


def execute_tool(name, args):
    """Execute a tool and return result."""
    try:
        if name == "glob":
            pattern = args.get("pattern", "*")
            files = glob_module.glob(pattern, recursive=True)
            return "\n".join(files[:50]) if files else "No files found"

        elif name == "read_file":
            path = args.get("path", "")
            p = Path(path).expanduser()
            if p.exists():
                content = p.read_text()
                return content[:3000] + ("\n...(truncated)" if len(content) > 3000 else "")
            return f"File not found: {path}"

        elif name == "ls":
            path = args.get("path", ".")
            p = Path(path).expanduser().resolve()
            if p.exists():
                entries = sorted(p.iterdir())
                result = []
                for e in entries[:30]:
                    if e.is_dir():
                        result.append(f"📁 {e.name}/")
                    else:
                        result.append(f"📄 {e.name}")
                return "\n".join(result)
            return f"Directory not found: {path}"

        elif name == "bash":
            cmd = args.get("command", "")
            # Safety check
            dangerous = ["rm -rf", "sudo", "> /dev", "mkfs", "dd if="]
            if any(d in cmd for d in dangerous):
                return "⚠️ Dangerous command blocked"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
            output = result.stdout + result.stderr
            return output[:2000] if output else "(no output)"

        elif name == "grep":
            pattern = args.get("pattern", "")
            path = args.get("path", ".")
            cmd = f"grep -r '{pattern}' {path} 2>/dev/null | head -20"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
            return result.stdout if result.stdout else "No matches found"

        elif name == "os":
            # Handle os.listdir style calls
            method = args.get("method", "")
            if method == "listdir":
                path = args.get("path", ".")
                return "\n".join(os.listdir(path)[:30])
            return "Unknown os method"

        elif name == "web_search":
            query = args.get("query", "")
            try:
                import urllib.request
                import urllib.parse
                # Use DuckDuckGo HTML
                url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=10) as r:
                    html = r.read().decode('utf-8')
                # Extract results
                results = []
                import re
                links = re.findall(r'<a rel="nofollow" class="result__a" href="([^"]+)"[^>]*>([^<]+)</a>', html)
                snippets = re.findall(r'<a class="result__snippet"[^>]*>([^<]+)', html)
                for i, (href, title) in enumerate(links[:5]):
                    snippet = snippets[i] if i < len(snippets) else ""
                    results.append(f"{i+1}. {title}\n   {snippet[:100]}...")
                return "\n\n".join(results) if results else "No results found"
            except Exception as e:
                return f"Search error: {e}"

        else:
            return f"Unknown tool: {name}"

    except Exception as e:
        return f"Error: {e}"


def clean_response(text):
    """Remove think tags from response."""
    # Remove <think>...</think>
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Remove standalone <think> or </think>
    text = text.replace('<think>', '').replace('</think>', '')
    return text.strip()


def main():
    console.print(Panel(BANNER, border_style="cyan"))
    console.print("[cyan]Loading ELO v1.0...[/cyan]")

    try:
        model, tokenizer = mlx_lm.load(MODEL_PATH)
    except Exception as e:
        console.print(f"[red]Error loading model: {e}[/red]")
        return

    console.print("[green]✓[/green] ELO v1.0 loaded with tools!")
    console.print(f"[dim]Working directory: {os.getcwd()}[/dim]")
    console.print("[dim]Commands: /exit, /clear, /ls, /run <cmd>[/dim]\n")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        try:
            user_input = Prompt.ask("[bold cyan]You[/bold cyan]").strip()

            if not user_input:
                continue

            # Built-in commands
            if user_input.lower() in ["/exit", "/quit", "/q"]:
                console.print("[dim]Goodbye![/dim]")
                break

            if user_input.lower() == "/clear":
                messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                console.print("[green]✓[/green] Cleared!")
                continue

            if user_input.lower().startswith("/ls"):
                path = user_input[3:].strip() or "."
                console.print(execute_tool("ls", {"path": path}))
                continue

            if user_input.lower().startswith("/run "):
                cmd = user_input[5:].strip()
                console.print(Panel(execute_tool("bash", {"command": cmd}), title=cmd, border_style="cyan"))
                continue

            messages.append({"role": "user", "content": user_input})

            # Agent loop - let model use tools
            for iteration in range(5):  # Max 5 tool iterations
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                response = mlx_lm.generate(
                    model, tokenizer,
                    prompt=prompt,
                    max_tokens=500,
                    verbose=False
                )

                response = clean_response(response)

                # Check for tool calls
                tool_calls = parse_tool_calls(response)

                if not tool_calls:
                    # No tools, show final response
                    console.print()
                    console.print("[bold cyan]ELO[/bold cyan]")
                    console.print(Markdown(response))
                    console.print()
                    messages.append({"role": "assistant", "content": response})
                    break

                # Execute tools
                messages.append({"role": "assistant", "content": response})

                tool_results = []
                for tc in tool_calls:
                    name = tc.get("name", "")
                    args = tc.get("args", {})

                    console.print(f"\n[magenta]🔧 {name}[/magenta] {args}")

                    result = execute_tool(name, args)
                    tool_results.append(f"[{name}] Result:\n{result}")

                    # Show result
                    if len(result) < 500:
                        console.print(Panel(result, border_style="green"))
                    else:
                        console.print(f"[dim]{result[:500]}...[/dim]")

                # Feed results back to model
                results_msg = "\n\n".join(tool_results)
                messages.append({"role": "user", "content": f"Tool results:\n{results_msg}\n\nNow summarize what you found."})

        except KeyboardInterrupt:
            console.print("\n[dim]Ctrl+C again to exit[/dim]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


if __name__ == "__main__":
    if len(sys.argv) > 1 and not sys.argv[1].startswith("/"):
        # One-liner mode
        prompt = " ".join(sys.argv[1:])
        try:
            model, tokenizer = mlx_lm.load(MODEL_PATH)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
            full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            response = mlx_lm.generate(model, tokenizer, prompt=full_prompt, max_tokens=500, verbose=False)
            response = clean_response(response)
            print(response)
        except Exception as e:
            print(f"Error: {e}")
    else:
        main()
