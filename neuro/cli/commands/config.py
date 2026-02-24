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
    console.print(f"Config file: [#9333EA]{config_path}[/#9333EA]")

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
        console.print("  [#9333EA]mkdir -p ~/.neuro[/#9333EA]")

        example = {"model": "ministral-3:8b", "permission_mode": "default", "hooks": {}}
        console.print(f"  [#9333EA]cat > {config_path} << 'EOF'[/#9333EA]")
        syntax = Syntax(json.dumps(example, indent=2), "json", theme="monokai")
        console.print(syntax)
        console.print("  [#9333EA]EOF[/#9333EA]")

    console.print()
    return 0
