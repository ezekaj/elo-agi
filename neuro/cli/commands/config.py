"""Config command."""

import os
import json

from rich.console import Console
from rich.syntax import Syntax

console = Console()


def cmd_config(args):
    """Open or display configuration."""
    config_path = os.path.expanduser("~/.neuro/settings.json")

    console.print()
    console.print("[bold]Configuration[/bold]")
    console.print("─" * 50)
    console.print(f"Config file: [purple]{config_path}[/purple]")

    if os.path.exists(config_path):
        try:
            with open(config_path) as f:
                cfg = json.load(f)
            console.print()
            console.print("[bold]Current settings:[/bold]")
            syntax = Syntax(json.dumps(cfg, indent=2), "json", theme="monokai")
            console.print(syntax)
        except Exception as e:
            console.print(f"[red]Error reading config:[/red] {e}")
    else:
        console.print("[dim](Config file not found - using defaults)[/dim]")
        console.print()
        console.print("[bold]Create one with:[/bold]")
        console.print("  [purple]mkdir -p ~/.neuro[/purple]")

        example = {"model": "ministral-3:8b", "permission_mode": "default", "hooks": {}}
        console.print(f"  [purple]cat > {config_path} << 'EOF'[/purple]")
        syntax = Syntax(json.dumps(example, indent=2), "json", theme="monokai")
        console.print(syntax)
        console.print("  [purple]EOF[/purple]")

    console.print()
    return 0
