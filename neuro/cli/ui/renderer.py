"""
UI Renderer - Beautiful terminal UI using Rich.

Provides a production-grade CLI experience with:
- Markdown rendering with syntax highlighting
- Animated spinners and progress bars
- Panels, tables, and structured output
- Live streaming updates
- Diff rendering for file changes
"""

from typing import Optional, List, Dict, Any
from contextlib import contextmanager

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.rule import Rule
from rich.tree import Tree
from rich.theme import Theme

# Custom theme matching modern CLI aesthetics
NEURO_THEME = Theme(
    {
        "info": "cyan",
        "success": "green",
        "warning": "yellow",
        "error": "red",
        "dim": "dim white",
        "highlight": "bold cyan",
        "user": "bold cyan",
        "assistant": "bold green",
        "tool": "bold yellow",
        "code": "bright_white on grey15",
    }
)


class UIRenderer:
    """
    Production-grade terminal UI renderer.

    Uses Rich for beautiful, responsive terminal output.
    """

    def __init__(self, theme: Optional[Theme] = None):
        self.console = Console(
            theme=theme or NEURO_THEME,
            highlight=True,
            markup=True,
        )
        self._live: Optional[Live] = None
        self._current_response = ""

    # =========================================================================
    # Header & Branding
    # =========================================================================

    def print_header(self):
        """Print the branded header."""
        header = Text()
        header.append("\n")
        header.append("  NEURO", style="bold cyan")
        header.append(" v3.0\n", style="dim")
        header.append("  Local AI That Learns From Your Code\n", style="dim italic")
        self.console.print(header)
        self.console.print()

    def print_minimal_header(self):
        """Print a minimal one-line header."""
        self.console.print(
            Text.assemble(
                ("NEURO", "bold cyan"),
                (" ready", "dim"),
            )
        )

    def print_welcome_screen(
        self,
        version: str,
        user_name: str,
        model: str,
        working_dir: str,
        recent_sessions: list = None,
        knowledge_stats: dict = None,
    ):
        """Print a beautiful welcome screen like Claude Code."""
        from rich.columns import Columns
        from rich.align import Align

        # ASCII art logo for NEURO (using Text with styles)
        logo = Text()
        logo.append("    ╔╗╔╔═╗╦ ╦╦═╗╔═╗\n", style="bold cyan")
        logo.append("    ║║║║╣ ║ ║╠╦╝║ ║\n", style="bold cyan")
        logo.append("    ╝╚╝╚═╝╚═╝╩╚═╚═╝", style="bold cyan")

        # Create left panel content
        left_content = Text()
        left_content.append(f"\n     Welcome back {user_name}!\n\n", style="bold white")
        left_content.append_text(logo)
        left_content.append(f"\n\n    {model}", style="dim")
        left_content.append(f"\n    {working_dir}\n", style="dim cyan")

        # Create right panel - Tips
        tips_content = Text()
        tips_content.append("Tips for getting started\n", style="bold yellow")
        tips_content.append("Run ", style="dim")
        tips_content.append("/help", style="cyan")
        tips_content.append(" to see all commands\n", style="dim")
        tips_content.append("Use ", style="dim")
        tips_content.append("@./file", style="cyan")
        tips_content.append(" to include file content\n", style="dim")
        tips_content.append("Try ", style="dim")
        tips_content.append("/ultrathink", style="cyan")
        tips_content.append(" for deep reasoning\n", style="dim")
        tips_content.append("Run ", style="dim")
        tips_content.append("/learn", style="cyan")
        tips_content.append(" to see what I've learned\n", style="dim")

        # Recent activity section
        activity_content = Text()
        activity_content.append("\nRecent activity\n", style="bold yellow")

        if recent_sessions and len(recent_sessions) > 0:
            from datetime import datetime

            for sess in recent_sessions[:3]:
                created = datetime.fromisoformat(sess.get("created_at", datetime.now().isoformat()))
                age = datetime.now() - created
                if age.days > 0:
                    time_str = f"{age.days}d ago"
                elif age.seconds > 3600:
                    time_str = f"{age.seconds // 3600}h ago"
                else:
                    time_str = f"{age.seconds // 60}m ago"

                preview = sess.get("preview", "")[:25] + "..." if sess.get("preview") else ""
                activity_content.append(f"  {sess['id'][:8]}", style="cyan")
                activity_content.append(f" ({time_str})", style="dim")
                if preview:
                    activity_content.append(f" {preview}", style="dim")
                activity_content.append("\n")
        else:
            activity_content.append("  No recent activity\n", style="dim")

        # Knowledge stats if available
        if knowledge_stats:
            facts = knowledge_stats.get("total_facts", 0)
            modules = knowledge_stats.get("cognitive_modules", 0)
            if facts > 0 or modules > 0:
                activity_content.append(
                    f"\n  Brain: {modules} modules, {facts} facts\n", style="dim green"
                )

        # Combine right panel
        right_text = Text()
        right_text.append_text(tips_content)
        right_text.append_text(activity_content)

        # Create panels
        left_panel = Panel(
            Align.center(left_content),
            border_style="cyan",
            title="NEURO v" + version,
            title_align="left",
            padding=(0, 1),
        )

        right_panel = Panel(
            right_text,
            border_style="yellow",
            padding=(0, 1),
        )

        # Print side by side
        self.console.print()

        # Use columns for side-by-side layout
        columns = Columns([left_panel, right_panel], expand=True, equal=True)
        self.console.print(columns)
        self.console.print()

    def print_input_prompt(self, placeholder: str = ""):
        """Print the input prompt with placeholder hint."""
        self.console.print()
        if placeholder:
            prompt_text = Text()
            prompt_text.append("> ", style="bold cyan")
            prompt_text.append(placeholder, style="dim italic")
            # Move cursor back to start of placeholder
            self.console.print(prompt_text, end="")
            # Clear the line and show just the prompt
            print("\r", end="")
        return self.console.input("[bold cyan]>[/bold cyan] ")

    # =========================================================================
    # Messages & Status
    # =========================================================================

    def print_user_prompt(self):
        """Print the user input prompt."""
        self.console.print()
        return self.console.input("[bold cyan]>[/bold cyan] ")

    def print_assistant_label(self):
        """Print the assistant label before response."""
        self.console.print()
        self.console.print("[bold green]NEURO[/bold green]", end=" ")

    def print_status(self, label: str, value: str, style: str = "success"):
        """Print a status indicator."""
        icon = {
            "success": "[green]●[/green]",
            "warning": "[yellow]●[/yellow]",
            "error": "[red]●[/red]",
            "info": "[cyan]●[/cyan]",
        }.get(style, "[dim]●[/dim]")
        self.console.print(f"  {icon} {label}: [dim]{value}[/dim]")

    def print_dim(self, text: str):
        """Print dimmed text."""
        self.console.print(f"  [dim]{text}[/dim]")

    def print_error(self, text: str):
        """Print error message."""
        self.console.print(f"  [red]✗[/red] {text}")

    def print_success(self, text: str):
        """Print success message."""
        self.console.print(f"  [green]✓[/green] {text}")

    def print_warning(self, text: str):
        """Print warning message."""
        self.console.print(f"  [yellow]![/yellow] {text}")

    def print_info(self, text: str):
        """Print info message."""
        self.console.print(f"  [cyan]i[/cyan] {text}")

    # =========================================================================
    # Dividers & Structure
    # =========================================================================

    def print_divider(self, title: str = ""):
        """Print a divider line."""
        if title:
            self.console.print(Rule(title, style="dim"))
        else:
            self.console.print(Rule(style="dim"))

    def print_rule(self, title: str = "", style: str = "dim"):
        """Print a styled rule."""
        self.console.print(Rule(title, style=style))

    # =========================================================================
    # Markdown & Code
    # =========================================================================

    def print_markdown(self, text: str):
        """Render and print markdown."""
        md = Markdown(text, code_theme="monokai")
        self.console.print(md)

    def print_code(self, code: str, language: str = "python", line_numbers: bool = True):
        """Print syntax-highlighted code."""
        syntax = Syntax(
            code,
            language,
            theme="monokai",
            line_numbers=line_numbers,
            word_wrap=True,
        )
        self.console.print(syntax)

    def print_diff(self, diff_text: str):
        """Print a colored diff."""
        for line in diff_text.split("\n"):
            if line.startswith("+") and not line.startswith("+++"):
                self.console.print(f"[green]{line}[/green]")
            elif line.startswith("-") and not line.startswith("---"):
                self.console.print(f"[red]{line}[/red]")
            elif line.startswith("@@"):
                self.console.print(f"[cyan]{line}[/cyan]")
            else:
                self.console.print(f"[dim]{line}[/dim]")

    # =========================================================================
    # Panels & Boxes
    # =========================================================================

    def print_panel(self, content: str, title: str = "", style: str = "cyan"):
        """Print content in a bordered panel."""
        panel = Panel(
            content,
            title=title if title else None,
            border_style=style,
            padding=(0, 1),
        )
        self.console.print(panel)

    def print_box(self, title: str, lines: List[str], style: str = "cyan"):
        """Print a box with title and content lines."""
        content = "\n".join(lines)
        self.print_panel(content, title=title, style=style)

    # =========================================================================
    # Tables
    # =========================================================================

    def print_table(
        self,
        headers: List[str],
        rows: List[List[str]],
        title: str = "",
        show_header: bool = True,
    ):
        """Print a formatted table."""
        table = Table(
            title=title if title else None,
            show_header=show_header,
            header_style="bold cyan",
            border_style="dim",
        )

        for header in headers:
            table.add_column(header)

        for row in rows:
            table.add_row(*row)

        self.console.print(table)

    def print_key_value(self, data: Dict[str, str], title: str = ""):
        """Print key-value pairs in a table."""
        table = Table(
            title=title if title else None,
            show_header=False,
            box=None,
            padding=(0, 2),
        )
        table.add_column(style="cyan")
        table.add_column(style="")

        for key, value in data.items():
            table.add_row(key, str(value))

        self.console.print(table)

    # =========================================================================
    # Trees
    # =========================================================================

    def print_tree(self, title: str, items: Dict[str, Any]):
        """Print a tree structure."""
        tree = Tree(f"[bold]{title}[/bold]")
        self._build_tree(tree, items)
        self.console.print(tree)

    def _build_tree(self, tree: Tree, items: Dict[str, Any]):
        """Recursively build tree nodes."""
        for key, value in items.items():
            if isinstance(value, dict):
                branch = tree.add(f"[cyan]{key}[/cyan]")
                self._build_tree(branch, value)
            elif isinstance(value, list):
                branch = tree.add(f"[cyan]{key}[/cyan]")
                for item in value:
                    branch.add(f"[dim]{item}[/dim]")
            else:
                tree.add(f"[cyan]{key}:[/cyan] [dim]{value}[/dim]")

    # =========================================================================
    # Progress & Spinners
    # =========================================================================

    @contextmanager
    def spinner(self, message: str = "Processing"):
        """Context manager for showing a spinner."""
        with self.console.status(f"[cyan]{message}[/cyan]", spinner="dots"):
            yield

    @contextmanager
    def progress(self, description: str = "Working"):
        """Context manager for a progress bar."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console,
        ) as progress:
            yield progress

    def print_progress_bar(self, current: int, total: int, label: str = ""):
        """Print a static progress bar."""
        if total == 0:
            pct = 0
        else:
            pct = current / total

        width = 30
        filled = int(width * pct)

        if pct < 0.5:
            color = "green"
        elif pct < 0.8:
            color = "yellow"
        else:
            color = "red"

        bar = f"[{color}]{'█' * filled}[/{color}][dim]{'░' * (width - filled)}[/dim]"
        self.console.print(f"  {label} [{bar}] {pct:.0%}")

    # =========================================================================
    # Live Streaming
    # =========================================================================

    def start_live(self):
        """Start live updating mode for streaming."""
        self._current_response = ""
        self._live = Live(
            Text(""),
            console=self.console,
            refresh_per_second=15,
            vertical_overflow="visible",
        )
        self._live.start()

    def update_live(self, text: str):
        """Update the live display with new text."""
        self._current_response = text
        if self._live:
            # Render as markdown for nice formatting
            try:
                md = Markdown(self._current_response)
                self._live.update(md)
            except Exception:
                self._live.update(Text(self._current_response))

    def append_live(self, token: str):
        """Append a token to the live display."""
        self._current_response += token
        self.update_live(self._current_response)

    def stop_live(self):
        """Stop live updating mode."""
        if self._live:
            self._live.stop()
            self._live = None
        # Print final rendered version
        if self._current_response:
            self.console.print()

    # =========================================================================
    # Prompts & Input
    # =========================================================================

    def confirm(self, message: str, default: bool = False) -> bool:
        """Ask for confirmation."""
        suffix = " [Y/n]" if default else " [y/N]"
        response = self.console.input(f"  [cyan]?[/cyan] {message}{suffix} ").strip().lower()

        if not response:
            return default
        return response in ("y", "yes")

    def select(self, message: str, options: List[str]) -> int:
        """Present options and get selection."""
        self.console.print(f"\n  [cyan]?[/cyan] {message}")
        for i, option in enumerate(options, 1):
            self.console.print(f"    [dim]{i}.[/dim] {option}")

        while True:
            try:
                choice = int(self.console.input("  [dim]Select:[/dim] "))
                if 1 <= choice <= len(options):
                    return choice - 1
            except (ValueError, EOFError):
                pass
            self.console.print("  [red]Invalid selection[/red]")

    # =========================================================================
    # Tool Execution Display
    # =========================================================================

    def print_tool_start(self, tool_name: str, description: str = ""):
        """Print tool execution start."""
        desc = f" [dim]{description}[/dim]" if description else ""
        self.console.print(f"  [yellow]▶[/yellow] [bold]{tool_name}[/bold]{desc}")

    def print_tool_result(self, tool_name: str, success: bool, output: str = ""):
        """Print tool execution result."""
        icon = "[green]✓[/green]" if success else "[red]✗[/red]"
        self.console.print(f"  {icon} [bold]{tool_name}[/bold]")
        if output:
            # Truncate long output
            lines = output.split("\n")
            if len(lines) > 5:
                for line in lines[:5]:
                    self.console.print(f"    [dim]{line}[/dim]")
                self.console.print(f"    [dim]... ({len(lines) - 5} more lines)[/dim]")
            else:
                for line in lines:
                    self.console.print(f"    [dim]{line}[/dim]")

    # =========================================================================
    # Help & Commands
    # =========================================================================

    def print_help(self, commands: List[tuple], title: str = "Commands"):
        """Print formatted help."""
        table = Table(
            title=title,
            show_header=False,
            box=None,
            padding=(0, 2),
            title_style="bold",
        )
        table.add_column(style="cyan", min_width=18)
        table.add_column(style="dim")

        for cmd, desc in commands:
            table.add_row(cmd, desc)

        self.console.print()
        self.console.print(table)
        self.console.print()

    # =========================================================================
    # Utility
    # =========================================================================

    def clear(self):
        """Clear the terminal."""
        self.console.clear()

    def print(self, *args, **kwargs):
        """Pass-through to console.print."""
        self.console.print(*args, **kwargs)

    @property
    def width(self) -> int:
        """Get terminal width."""
        return self.console.width

    @property
    def height(self) -> int:
        """Get terminal height."""
        return self.console.height
