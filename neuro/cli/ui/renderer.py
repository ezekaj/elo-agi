"""
UI Renderer - Beautiful terminal UI using Rich.

Provides a production-grade CLI experience with:
- Markdown rendering with syntax highlighting
- Animated spinners and progress bars
- Panels, tables, and structured output
- Live streaming updates
- Diff rendering for file changes

Color palette extracted from Claude Code source (claude_readable_v2.js lines 113971-114399),
mapped to ELO's purple/cyan brand identity.
"""

import os
import random
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
from enum import Enum

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
from rich import box


class ToolDisplayState(Enum):
    """Tool execution display states (matches Claude Code pattern)."""

    PROGRESS = "progress"
    SUCCESS = "success"
    ERROR = "error"
    REJECTED = "rejected"


# Maps tool names to active verbs for spinner display
TOOL_VERBS: Dict[str, str] = {
    "read_file": "Reading",
    "write_file": "Writing",
    "edit_file": "Editing",
    "run_command": "Running",
    "bash": "Running",
    "list_files": "Listing",
    "search_files": "Searching",
    "grep_files": "Searching",
    "web_search": "Searching web",
    "web_fetch": "Fetching",
    "ask_user": "Waiting for input",
    "create_task": "Creating task",
    "update_task": "Updating task",
    "plan_create": "Planning",
    "plan_update": "Updating plan",
}

# Tips shown when tools run longer than 30s
TOOL_TIPS = [
    "Long-running commands can be cancelled with Ctrl+C",
    "Use /compact to reduce context when conversations get long",
    "Use /cost to see token usage",
    "Use /status to see current session info",
    "Try /think for deeper reasoning on complex problems",
    "Use Tab for input completion",
]

# Welcome screen tips (shown randomly on startup)
WELCOME_TIPS = [
    "Start with small features or bug fixes, tell ELO to propose a plan",
    "Use /plan to prepare for complex requests before making changes",
    "Use /think for deeper reasoning on complex problems",
    "Use /compact to reduce context when conversations get long",
    "Use @./file to include file content in your message",
    "Use !command to execute shell commands inline",
    "Use /cost to see token usage and costs",
    "Use /agent to spawn specialized subagents for complex tasks",
    "Use /team to coordinate multiple agents working in parallel",
    "Use /knowledge to inject learned facts into conversations",
]

# Icon constants (matching Claude Code's icon set from source)
ICONS = {
    "success": "\u2714",     # ✔ HEAVY CHECK MARK
    "error": "\u2718",       # ✘ HEAVY BALLOT X
    "warning": "\u26a0",     # ⚠ WARNING SIGN
    "info": "\u25cf",        # ● BLACK CIRCLE
    "rejected": "\u2298",    # ⊘ CIRCLED DIVISION SLASH
    "prompt": "\u276f",      # ❯ HEAVY RIGHT-POINTING ANGLE QUOTATION MARK
    "selected": "\u25c9",    # ◉ FISHEYE
    "unselected": "\u25ef",  # ◯ LARGE CIRCLE
    "progress": "\u280b",    # ⠋ BRAILLE
}

# Tree connectors (matching Claude Code's tool tree from source)
TREE = {
    "first": "\u252c\u2500",    # ┬─
    "middle": "\u251c\u2500",   # ├─
    "last": "\u2514\u2500",     # └─
    "single": "\u2500\u2500",   # ──
    "pipe": "\u2502",           # │
}

# ELO color palette — hex values extracted from Claude Code source,
# mapped to ELO brand: Claude orange (#D77757) → ELO purple (#9333EA),
# Claude blue (#5769F7) → ELO cyan (#06B6D4)
ELO_THEME = Theme(
    {
        # Brand
        "brand": "bold #9333EA",
        "brand.dim": "#7C3AED",
        # Semantic states (matched from Claude Code source)
        "info": "#06B6D4",
        "success": "#2C7A39",
        "warning": "#966C1E",
        "error": "#AB2B3F",
        # Text hierarchy
        "text.dim": "#AFAFAF",
        "text.inactive": "#666666",
        # Roles
        "user": "bold #9333EA",
        "assistant": "bold #7C3AED",
        "tool": "bold #06B6D4",
        "code": "bright_white on grey11",
        # Modes
        "plan": "bold #006666",
        "team": "bold #9333EA",
        "task": "bold #06B6D4",
        "fast": "bold #FF6A00",
        # Diff (exact Claude Code values)
        "diff.added": "#69DB7C",
        "diff.removed": "#FFA8B4",
        "diff.hunk": "#9333EA",
        # Permission
        "permission": "#06B6D4",
        "permission.danger": "#AB2B3F",
        # Borders
        "border": "#999999",
        "border.brand": "#9333EA",
        # Backwards compat
        "highlight": "bold #9333EA",
        "dim": "#AFAFAF",
    }
)


class UIRenderer:
    """
    Production-grade terminal UI renderer.

    Uses Rich for beautiful, responsive terminal output.
    """

    def __init__(self, theme: Optional[Theme] = None):
        self.console = Console(
            theme=theme or ELO_THEME,
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
        import neuro

        header.append("  ELO-AGI", style="bold #9333EA")
        header.append(f" v{neuro.__version__}\n", style="#AFAFAF")
        header.append("  Local AI That Learns From Your Code\n", style="#AFAFAF italic")
        self.console.print(header)
        self.console.print()

    def print_minimal_header(self):
        """Print a minimal one-line header."""
        self.console.print(
            Text.assemble(
                ("ELO", "bold #9333EA"),
                (" ready", "#AFAFAF"),
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
        project_dir: str = None,
        session_id: str = None,
        git_info: dict = None,
    ):
        """Print full welcome screen (Claude Code dot-separator style)."""
        self.console.print()

        # Title: ELO + version
        header = Text()
        header.append("  ELO", style="bold #9333EA")
        header.append(f" v{version}", style="#AFAFAF")
        self.console.print(header)

        # Ellipsis separator (58 chars, matching Claude Code's vkR=58)
        ellipsis = "\u2026" * 58
        self.console.print(f"  [#666666]{ellipsis}[/#666666]")
        self.console.print()

        # Project info: working dir · model
        info = Text()
        info.append("  ")
        info.append(working_dir, style="#AFAFAF")
        info.append(" \u00b7 ", style="#666666")
        info.append(model, style="#666666")
        self.console.print(info)

        # Stats line: git branch · modules · facts · CLAUDE.md
        stats_parts = []

        # Git info
        if git_info:
            branch = git_info.get("branch", "")
            dirty = git_info.get("dirty", False)
            if branch:
                label = branch + (" *" if dirty else "")
                stats_parts.append(("git", label))

        if knowledge_stats:
            modules = knowledge_stats.get("cognitive_modules", 0)
            facts = knowledge_stats.get("total_facts", 0)
            if modules > 0:
                stats_parts.append(("text", f"{modules} modules"))
            if facts > 0:
                stats_parts.append(("text", f"{facts:,} facts"))

        # Check CLAUDE.md
        if project_dir:
            claude_md = os.path.exists(os.path.join(project_dir, "CLAUDE.md"))
            if claude_md:
                stats_parts.append(("success", "CLAUDE.md"))
            else:
                stats_parts.append(("hint", "run /init to create CLAUDE.md"))

        if stats_parts:
            stats = Text()
            stats.append("  ")
            for i, (kind, part) in enumerate(stats_parts):
                if i > 0:
                    stats.append(" \u00b7 ", style="#666666")
                if kind == "success":
                    stats.append(part, style="#AFAFAF")
                    stats.append(" \u2714", style="#2C7A39")
                elif kind == "hint":
                    stats.append(part, style="#666666")
                elif kind == "git":
                    stats.append(part, style="#9333EA")
                else:
                    stats.append(part, style="#AFAFAF")
            self.console.print(stats)

        # Session ID
        if session_id:
            self.console.print(f"  [#AFAFAF]Session: {session_id[:8]}[/#AFAFAF]")

        self.console.print()

        # Random tip
        tip = random.choice(WELCOME_TIPS)
        self.console.print(f"  [#AFAFAF]Tip: {tip}[/#AFAFAF]")
        self.console.print()

    def print_thinking(self):
        """Print thinking/requesting indicator (same line as assistant label)."""
        self.console.print("[#666666]\u2026[/#666666]")

    def print_response_meta(self, duration: float = 0, tokens: int = 0, cost: float = 0):
        """Print response metadata (duration, tokens, cost)."""
        parts = []
        if duration > 0.5:
            parts.append(f"{duration:.1f}s")
        if tokens > 0:
            parts.append(f"{tokens:,} tokens")
        if cost > 0:
            parts.append(f"${cost:.4f}")
        if parts:
            meta = " \u00b7 ".join(parts)
            self.console.print(f"  [#AFAFAF]{meta}[/#AFAFAF]")

    def print_input_prompt(self, placeholder: str = ""):
        """Print the input prompt with placeholder hint."""
        self.console.print()
        if placeholder:
            prompt_text = Text()
            prompt_text.append(f"{ICONS['prompt']} ", style="bold #9333EA")
            prompt_text.append(placeholder, style="#AFAFAF italic")
            self.console.print(prompt_text, end="")
            print("\r", end="")
        return self.console.input(f"[#9333EA]{ICONS['prompt']}[/#9333EA] ")

    # =========================================================================
    # Messages & Status
    # =========================================================================

    def print_user_prompt(self):
        """Print the user input prompt."""
        self.console.print()
        return self.console.input(f"[#9333EA]{ICONS['prompt']}[/#9333EA] ")

    def print_assistant_label(self):
        """Print the assistant label before response."""
        self.console.print()
        self.console.print("[bold #7C3AED]ELO[/bold #7C3AED]", end=" ")

    def print_status(self, label: str, value: str, style: str = "success"):
        """Print a status indicator."""
        icon = {
            "success": f"[#2C7A39]{ICONS['success']}[/#2C7A39]",
            "warning": f"[#966C1E]{ICONS['warning']}[/#966C1E]",
            "error": f"[#AB2B3F]{ICONS['error']}[/#AB2B3F]",
            "info": f"[#06B6D4]{ICONS['info']}[/#06B6D4]",
        }.get(style, f"[#AFAFAF]{ICONS['info']}[/#AFAFAF]")
        self.console.print(f"  {icon} {label}: [#AFAFAF]{value}[/#AFAFAF]")

    def print_dim(self, text: str):
        """Print dimmed text."""
        self.console.print(f"  [#AFAFAF]{text}[/#AFAFAF]")

    def print_error(self, text: str):
        """Print error message."""
        self.console.print(f"  [#AB2B3F]{ICONS['error']}[/#AB2B3F] {text}")

    def print_success(self, text: str):
        """Print success message."""
        self.console.print(f"  [#2C7A39]{ICONS['success']}[/#2C7A39] {text}")

    def print_warning(self, text: str):
        """Print warning message."""
        self.console.print(f"  [#966C1E]{ICONS['warning']}[/#966C1E] {text}")

    def print_info(self, text: str):
        """Print info message."""
        self.console.print(f"  [#06B6D4]{ICONS['info']}[/#06B6D4] {text}")

    # =========================================================================
    # Dividers & Structure
    # =========================================================================

    def print_divider(self, title: str = ""):
        """Print a divider line."""
        if title:
            self.console.print(Rule(title, style="#666666"))
        else:
            self.console.print(Rule(style="#666666"))

    def print_rule(self, title: str = "", style: str = "#666666"):
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
        """Print a colored diff (Claude Code palette)."""
        for line in diff_text.split("\n"):
            if line.startswith("+++") or line.startswith("---"):
                self.console.print(f"[bold #AFAFAF]{line}[/bold #AFAFAF]")
            elif line.startswith("+"):
                self.console.print(f"[#69DB7C]{line}[/#69DB7C]")
            elif line.startswith("-"):
                self.console.print(f"[#FFA8B4]{line}[/#FFA8B4]")
            elif line.startswith("@@"):
                self.console.print(f"[#9333EA]{line}[/#9333EA]")
            else:
                self.console.print(f"[#AFAFAF]{line}[/#AFAFAF]")

    # =========================================================================
    # Panels & Boxes
    # =========================================================================

    def print_panel(self, content: str, title: str = "", style: str = "#06B6D4"):
        """Print content in a bordered panel with rounded corners."""
        panel = Panel(
            content,
            title=title if title else None,
            box=box.ROUNDED,
            border_style=style,
            padding=(0, 1),
        )
        self.console.print(panel)

    def print_box(self, title: str, lines: List[str], style: str = "#06B6D4"):
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
            header_style="bold #9333EA",
            border_style="#666666",
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
        table.add_column(style="#9333EA")
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
                branch = tree.add(f"[#9333EA]{key}[/#9333EA]")
                self._build_tree(branch, value)
            elif isinstance(value, list):
                branch = tree.add(f"[#9333EA]{key}[/#9333EA]")
                for item in value:
                    branch.add(f"[#AFAFAF]{item}[/#AFAFAF]")
            else:
                tree.add(f"[#9333EA]{key}:[/#9333EA] [#AFAFAF]{value}[/#AFAFAF]")

    # =========================================================================
    # Progress & Spinners
    # =========================================================================

    @contextmanager
    def spinner(self, message: str = "Processing"):
        """Context manager for showing a spinner."""
        with self.console.status(f"[#9333EA]{message}[/#9333EA]", spinner="dots"):
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
            color = "#2C7A39"
        elif pct < 0.8:
            color = "#966C1E"
        else:
            color = "#AB2B3F"

        filled_char = "\u2588" * filled
        empty_char = "\u2591" * (width - filled)
        bar = f"[{color}]{filled_char}[/{color}][#666666]{empty_char}[/#666666]"
        self.console.print(f"  {label} [{bar}] {pct:.0%}")

    # =========================================================================
    # Live Streaming
    # =========================================================================

    def start_live(self):
        """Start live updating mode for streaming."""
        self._current_response = ""
        self._last_render_len = 0
        self._live = Live(
            Text(""),
            console=self.console,
            refresh_per_second=10,
            vertical_overflow="visible",
        )
        self._live.start()

    def update_live(self, text: str):
        """Update the live display with new text."""
        self._current_response = text
        if self._live:
            try:
                md = Markdown(self._current_response)
                self._live.update(md)
            except Exception:
                self._live.update(Text(self._current_response))
            self._last_render_len = len(self._current_response)

    def append_live(self, token: str):
        """Append a token to the live display."""
        self._current_response += token
        # Throttle markdown re-parsing: only re-render every 30 chars or on newlines
        delta = len(self._current_response) - self._last_render_len
        if delta >= 30 or "\n" in token:
            self.update_live(self._current_response)

    def stop_live(self):
        """Stop live updating mode."""
        if self._live:
            # Final render with complete content
            if self._current_response:
                try:
                    md = Markdown(self._current_response)
                    self._live.update(md)
                except Exception:
                    self._live.update(Text(self._current_response))
            self._live.stop()
            self._live = None
        if self._current_response:
            self.console.print()

    # =========================================================================
    # Prompts & Input
    # =========================================================================

    def confirm(self, message: str, default: bool = False) -> bool:
        """Ask for confirmation."""
        suffix = " [Y/n]" if default else " [y/N]"
        response = self.console.input(
            f"  [#06B6D4]?[/#06B6D4] {message}{suffix} "
        ).strip().lower()

        if not response:
            return default
        return response in ("y", "yes")

    def select(self, message: str, options: List[str]) -> int:
        """Present options and get selection."""
        self.console.print(f"\n  [#06B6D4]?[/#06B6D4] {message}")
        for i, option in enumerate(options, 1):
            self.console.print(f"    [#666666]{i}.[/#666666] {option}")

        while True:
            try:
                choice = int(self.console.input("  [#AFAFAF]Select:[/#AFAFAF] "))
                if 1 <= choice <= len(options):
                    return choice - 1
            except (ValueError, EOFError):
                pass
            self.console.print("  [#AB2B3F]Invalid selection[/#AB2B3F]")

    # =========================================================================
    # Tool Execution Display
    # =========================================================================

    def print_permission_card(self, tool_name: str, args: Dict[str, Any], risk_level: str = "normal"):
        """Print a permission request card with rounded border."""
        content = Text()
        content.append(f"{tool_name}\n", style="bold")

        for k, v in list(args.items())[:3]:
            val_str = str(v)[:80]
            if len(str(v)) > 80:
                val_str += "..."
            content.append(f"  {k}: ", style="#AFAFAF")
            content.append(f"{val_str}\n")

        border = "#AB2B3F" if risk_level == "high" else "#06B6D4"
        panel = Panel(
            content,
            title="[bold]Permission Required[/bold]",
            box=box.ROUNDED,
            border_style=border,
            padding=(0, 1),
        )
        self.console.print()
        self.console.print(panel)

    def print_tool_start(self, tool_name: str, description: str = ""):
        """Print tool execution start with active verb."""
        verb = TOOL_VERBS.get(tool_name, "Running")
        desc = f" [#AFAFAF]{description}[/#AFAFAF]" if description else ""
        self.console.print(
            f"  [#9333EA]{ICONS['progress']}[/#9333EA] {verb} [bold]{tool_name}[/bold]{desc}"
        )

    def print_tool_result(
        self,
        tool_name: str,
        success: bool,
        output: str = "",
        duration: float = 0.0,
        state: ToolDisplayState = None,
    ):
        """Print tool execution result with state icon and duration."""
        if state is None:
            state = ToolDisplayState.SUCCESS if success else ToolDisplayState.ERROR

        icons = {
            ToolDisplayState.SUCCESS: f"[#2C7A39]{ICONS['success']}[/#2C7A39]",
            ToolDisplayState.ERROR: f"[#AB2B3F]{ICONS['error']}[/#AB2B3F]",
            ToolDisplayState.REJECTED: f"[#966C1E]{ICONS['rejected']}[/#966C1E]",
            ToolDisplayState.PROGRESS: f"[#9333EA]{ICONS['progress']}[/#9333EA]",
        }
        icon = icons.get(state, icons[ToolDisplayState.SUCCESS])

        dur_str = f" [#AFAFAF]({duration:.1f}s)[/#AFAFAF]" if duration > 0 else ""
        self.console.print(f"  {icon} [bold]{tool_name}[/bold]{dur_str}")

        if output:
            lines = output.split("\n")
            if len(lines) > 20:
                for line in lines[:20]:
                    self.console.print(f"    [#AFAFAF]{line}[/#AFAFAF]")
                self.console.print(f"    [#AFAFAF]... ({len(lines) - 20} more lines)[/#AFAFAF]")
            else:
                for line in lines:
                    self.console.print(f"    [#AFAFAF]{line}[/#AFAFAF]")

    def print_tool_tree_start(self, tool_name: str, description: str = "", position: str = "single"):
        """Print tool execution start with tree connector."""
        verb = TOOL_VERBS.get(tool_name, "Running")
        desc = f" [#AFAFAF]{description}[/#AFAFAF]" if description else ""
        connector = TREE.get(position, TREE["single"])
        self.console.print(
            f"  [#666666]{connector}[/#666666] [#9333EA]{ICONS['progress']}[/#9333EA]"
            f" {verb} [bold]{tool_name}[/bold]{desc}"
        )

    def print_tool_tree_result(
        self,
        tool_name: str,
        success: bool,
        output: str = "",
        duration: float = 0.0,
        position: str = "single",
        state: ToolDisplayState = None,
    ):
        """Print tool execution result with tree connector."""
        if state is None:
            state = ToolDisplayState.SUCCESS if success else ToolDisplayState.ERROR

        icons_map = {
            ToolDisplayState.SUCCESS: f"[#2C7A39]{ICONS['success']}[/#2C7A39]",
            ToolDisplayState.ERROR: f"[#AB2B3F]{ICONS['error']}[/#AB2B3F]",
            ToolDisplayState.REJECTED: f"[#966C1E]{ICONS['rejected']}[/#966C1E]",
            ToolDisplayState.PROGRESS: f"[#9333EA]{ICONS['progress']}[/#9333EA]",
        }
        icon = icons_map.get(state, icons_map[ToolDisplayState.SUCCESS])
        connector = TREE.get(position, TREE["single"])

        dur_str = f" [#AFAFAF]({duration:.1f}s)[/#AFAFAF]" if duration > 0 else ""
        self.console.print(f"  [#666666]{connector}[/#666666] {icon} [bold]{tool_name}[/bold]{dur_str}")

        if output:
            indent = f"  {TREE['pipe']}   " if position in ("first", "middle") else "      "
            lines = output.split("\n")
            show_lines = lines[:20]
            for line in show_lines:
                self.console.print(f"{indent}[#AFAFAF]{line}[/#AFAFAF]")
            if len(lines) > 20:
                self.console.print(f"{indent}[#AFAFAF]... ({len(lines) - 20} more lines)[/#AFAFAF]")

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
        table.add_column(style="#9333EA", min_width=18)
        table.add_column(style="#AFAFAF")

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
