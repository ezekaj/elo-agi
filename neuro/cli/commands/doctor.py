"""Doctor command - check installation health."""

import os
import sys
import asyncio

from rich.console import Console
from rich.table import Table

console = Console()


def cmd_doctor(args):
    """Check NEURO installation health."""
    return asyncio.run(_check_health())


async def _check_health():
    """Run health checks."""
    console.print()
    console.print("[bold]NEURO Doctor[/bold]")
    console.print("─" * 50)

    checks = []

    # Check Python version
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    checks.append(("Python", py_version, True, ""))

    # Check Ollama
    ollama_ok = False
    ollama_info = ""
    try:
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.get(
                "http://localhost:11434/api/tags", timeout=aiohttp.ClientTimeout(total=5)
            ) as r:
                if r.status == 200:
                    data = await r.json()
                    models = [m["name"] for m in data.get("models", [])]
                    ollama_ok = True
                    ollama_info = f"{len(models)} models available"
                else:
                    ollama_info = f"Status {r.status}"
    except Exception as e:
        ollama_info = "Not running - run: ollama serve"
    checks.append(("Ollama", "Connected" if ollama_ok else "Not connected", ollama_ok, ollama_info))

    # Check aiohttp
    aiohttp_ok = False
    try:
        import aiohttp

        aiohttp_ok = True
        checks.append(("aiohttp", "Installed", True, ""))
    except ImportError:
        checks.append(("aiohttp", "Missing", False, "pip install aiohttp"))

    # Check rich
    rich_ok = False
    try:
        import rich

        rich_ok = True
        checks.append(("rich", "Installed", True, ""))
    except ImportError:
        checks.append(("rich", "Missing", False, "pip install rich"))

    # Check pyyaml
    yaml_ok = False
    try:
        import yaml

        yaml_ok = True
        checks.append(("pyyaml", "Installed", True, ""))
    except ImportError:
        checks.append(("pyyaml", "Missing", False, "pip install pyyaml"))

    # Check config directory
    config_dir = os.path.expanduser("~/.neuro")
    config_ok = os.path.exists(config_dir)
    checks.append(
        (
            "Config dir",
            config_dir if config_ok else "Not found",
            config_ok,
            "" if config_ok else f"mkdir -p {config_dir}",
        )
    )

    # Check sessions directory
    sessions_found = 0
    for root, dirs, files in os.walk(os.path.expanduser("~/.neuro/projects")):
        for f in files:
            if f.endswith(".jsonl"):
                sessions_found += 1
    checks.append(("Sessions", f"{sessions_found} saved", True, ""))

    # Print results
    table = Table(show_header=True, header_style="bold")
    table.add_column("Component")
    table.add_column("Status")
    table.add_column("Info")

    issues = []
    for name, status, ok, info in checks:
        icon = "[green]✓[/green]" if ok else "[red]✗[/red]"
        table.add_row(f"{icon} {name}", status, f"[dim]{info}[/dim]" if info else "")
        if not ok and info:
            issues.append(info)

    console.print(table)
    console.print()

    if issues:
        console.print(f"[yellow]Found {len(issues)} issue(s):[/yellow]")
        for issue in issues:
            console.print(f"  [dim]→[/dim] {issue}")
        console.print()
        return 1

    console.print("[green]✓ All checks passed![/green]")
    console.print()
    return 0
