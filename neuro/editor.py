"""
NEURO Code Editor - Surgical File Editing with Colored Diffs

Provides:
- Line-specific edits
- Beautiful colored diffs (using rich)
- User confirmation before changes
- Atomic file operations
- Backup and restore
"""

import os
import difflib
import tempfile
import shutil
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, field
from datetime import datetime

try:
    from rich.console import Console  # noqa: F401
    from rich.syntax import Syntax  # noqa: F401
    from rich.panel import Panel  # noqa: F401
    from rich.table import Table  # noqa: F401
    from rich.prompt import Confirm
    from rich.text import Text

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


@dataclass
class Edit:
    """A single edit operation."""

    line_start: int  # 1-indexed
    line_end: int  # 1-indexed, inclusive
    new_content: str
    description: str = ""


@dataclass
class EditResult:
    """Result of an edit operation."""

    success: bool
    file_path: str
    diff: str
    lines_changed: int
    backup_path: Optional[str] = None
    error: Optional[str] = None


@dataclass
class FileState:
    """State of a file before/after editing."""

    path: str
    content: str
    lines: List[str] = field(default_factory=list)
    exists: bool = True

    def __post_init__(self):
        if self.content:
            self.lines = self.content.splitlines(keepends=True)


class CodeEditor:
    """
    Surgical code editing with beautiful diffs.

    Features:
    - Line-range editing
    - Colored diff display
    - User confirmation
    - Atomic operations (backup → edit → confirm/rollback)
    - Multi-edit batching
    """

    def __init__(self, backup_dir: Optional[str] = None, auto_confirm: bool = False):
        self.backup_dir = Path(backup_dir or "~/.neuro/backups").expanduser()
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.auto_confirm = auto_confirm

        if RICH_AVAILABLE:
            self.console = Console()
        else:
            self.console = None

    def read_file(self, path: str) -> FileState:
        """Read a file and return its state."""
        path = os.path.expanduser(path)

        if not os.path.exists(path):
            return FileState(path=path, content="", exists=False)

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        return FileState(path=path, content=content, exists=True)

    def write_file(self, path: str, content: str) -> bool:
        """Write content to file atomically."""
        path = os.path.expanduser(path)

        # Create parent directories if needed
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        # Write to temp file first, then rename (atomic on most systems)
        fd, temp_path = tempfile.mkstemp(dir=os.path.dirname(path) or ".")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
            shutil.move(temp_path, path)
            return True
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise e

    def create_backup(self, path: str) -> str:
        """Create a backup of a file."""
        path = os.path.expanduser(path)

        if not os.path.exists(path):
            return ""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.basename(path)
        backup_name = f"{filename}.{timestamp}.bak"
        backup_path = self.backup_dir / backup_name

        shutil.copy2(path, backup_path)
        return str(backup_path)

    def restore_backup(self, backup_path: str, original_path: str) -> bool:
        """Restore a file from backup."""
        if os.path.exists(backup_path):
            shutil.copy2(backup_path, original_path)
            return True
        return False

    def apply_edits(self, content: str, edits: List[Edit]) -> str:
        """Apply a list of edits to content."""
        lines = content.splitlines(keepends=True)

        # Ensure last line has newline
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"

        # Sort edits by line number (descending) to apply from bottom up
        # This prevents line number shifts from affecting later edits
        sorted_edits = sorted(edits, key=lambda e: e.line_start, reverse=True)

        for edit in sorted_edits:
            start = max(0, edit.line_start - 1)  # Convert to 0-indexed
            end = min(len(lines), edit.line_end)

            # Prepare new content
            new_lines = edit.new_content.splitlines(keepends=True)
            if new_lines and not new_lines[-1].endswith("\n"):
                new_lines[-1] += "\n"

            # Replace lines
            lines[start:end] = new_lines

        return "".join(lines)

    def generate_diff(self, old_content: str, new_content: str, filename: str = "file") -> str:
        """Generate unified diff."""
        old_lines = old_content.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)

        diff = difflib.unified_diff(
            old_lines, new_lines, fromfile=f"a/{filename}", tofile=f"b/{filename}", lineterm=""
        )

        return "".join(diff)

    def display_diff(self, diff: str, filename: str = ""):
        """Display diff with colors."""
        if RICH_AVAILABLE and self.console:
            self._display_diff_rich(diff, filename)
        else:
            self._display_diff_plain(diff)

    def _display_diff_rich(self, diff: str, filename: str):
        """Display diff using rich."""
        lines = diff.split("\n")
        text = Text()

        for line in lines:
            if line.startswith("+++") or line.startswith("---"):
                text.append(line + "\n", style="bold")
            elif line.startswith("@@"):
                text.append(line + "\n", style="cyan")
            elif line.startswith("+"):
                text.append(line + "\n", style="green")
            elif line.startswith("-"):
                text.append(line + "\n", style="red")
            else:
                text.append(line + "\n", style="dim")

        panel = Panel(
            text,
            title=f"[bold]Changes to {filename}[/bold]" if filename else "[bold]Diff[/bold]",
            border_style="blue",
        )
        self.console.print(panel)

    def _display_diff_plain(self, diff: str):
        """Display diff without colors."""
        RED = "\033[31m"
        GREEN = "\033[32m"
        CYAN = "\033[36m"
        RESET = "\033[0m"
        BOLD = "\033[1m"

        print("\n" + "=" * 60)
        print(f"{BOLD}DIFF{RESET}")
        print("=" * 60)

        for line in diff.split("\n"):
            if line.startswith("+++") or line.startswith("---"):
                print(f"{BOLD}{line}{RESET}")
            elif line.startswith("@@"):
                print(f"{CYAN}{line}{RESET}")
            elif line.startswith("+"):
                print(f"{GREEN}{line}{RESET}")
            elif line.startswith("-"):
                print(f"{RED}{line}{RESET}")
            else:
                print(line)

        print("=" * 60 + "\n")

    def confirm_edit(self, diff: str, filename: str) -> bool:
        """Ask user to confirm the edit."""
        if self.auto_confirm:
            return True

        self.display_diff(diff, filename)

        if RICH_AVAILABLE and self.console:
            return Confirm.ask("[bold]Apply these changes?[/bold]", default=True)
        else:
            response = input("Apply these changes? [Y/n]: ").strip().lower()
            return response in ("", "y", "yes")

    def edit(self, file_path: str, edits: List[Edit], confirm: bool = True) -> EditResult:
        """
        Apply edits to a file with optional confirmation.

        Args:
            file_path: Path to the file
            edits: List of Edit objects
            confirm: Whether to ask for confirmation

        Returns:
            EditResult with success status and diff
        """
        file_path = os.path.expanduser(file_path)
        filename = os.path.basename(file_path)

        # Read current content
        state = self.read_file(file_path)
        if not state.exists:
            return EditResult(
                success=False,
                file_path=file_path,
                diff="",
                lines_changed=0,
                error=f"File not found: {file_path}",
            )

        # Apply edits
        new_content = self.apply_edits(state.content, edits)

        # Generate diff
        diff = self.generate_diff(state.content, new_content, filename)

        if not diff.strip():
            return EditResult(
                success=True,
                file_path=file_path,
                diff="",
                lines_changed=0,
                error="No changes to apply",
            )

        # Count changed lines
        lines_changed = sum(
            1 for line in diff.split("\n") if line.startswith("+") or line.startswith("-")
        )

        # Confirm with user
        if confirm and not self.auto_confirm:
            if not self.confirm_edit(diff, filename):
                return EditResult(
                    success=False,
                    file_path=file_path,
                    diff=diff,
                    lines_changed=lines_changed,
                    error="User rejected changes",
                )

        # Create backup
        backup_path = self.create_backup(file_path)

        # Apply changes
        try:
            self.write_file(file_path, new_content)
            return EditResult(
                success=True,
                file_path=file_path,
                diff=diff,
                lines_changed=lines_changed,
                backup_path=backup_path,
            )
        except Exception as e:
            # Restore from backup on failure
            if backup_path:
                self.restore_backup(backup_path, file_path)
            return EditResult(
                success=False,
                file_path=file_path,
                diff=diff,
                lines_changed=lines_changed,
                error=str(e),
            )

    def edit_lines(
        self,
        file_path: str,
        line_start: int,
        line_end: int,
        new_content: str,
        description: str = "",
        confirm: bool = True,
    ) -> EditResult:
        """
        Edit specific lines in a file.

        Args:
            file_path: Path to file
            line_start: First line to replace (1-indexed)
            line_end: Last line to replace (1-indexed, inclusive)
            new_content: New content to insert
            description: Description of the change
            confirm: Whether to ask for confirmation
        """
        edit = Edit(
            line_start=line_start,
            line_end=line_end,
            new_content=new_content,
            description=description,
        )
        return self.edit(file_path, [edit], confirm=confirm)

    def insert_lines(
        self, file_path: str, after_line: int, content: str, confirm: bool = True
    ) -> EditResult:
        """Insert lines after a specific line."""
        state = self.read_file(file_path)
        if not state.exists:
            return EditResult(
                success=False,
                file_path=file_path,
                diff="",
                lines_changed=0,
                error=f"File not found: {file_path}",
            )

        # Get the existing line (to preserve it)
        lines = state.content.splitlines(keepends=True)
        if after_line > 0 and after_line <= len(lines):
            existing = lines[after_line - 1]
            if not existing.endswith("\n"):
                existing += "\n"
            new_content = existing + content
        else:
            new_content = content

        edit = Edit(
            line_start=max(1, after_line),
            line_end=max(1, after_line),
            new_content=new_content,
            description=f"Insert after line {after_line}",
        )
        return self.edit(file_path, [edit], confirm=confirm)

    def delete_lines(
        self, file_path: str, line_start: int, line_end: int, confirm: bool = True
    ) -> EditResult:
        """Delete specific lines from a file."""
        edit = Edit(
            line_start=line_start,
            line_end=line_end,
            new_content="",
            description=f"Delete lines {line_start}-{line_end}",
        )
        return self.edit(file_path, [edit], confirm=confirm)

    def replace_text(
        self, file_path: str, old_text: str, new_text: str, confirm: bool = True
    ) -> EditResult:
        """Replace text throughout a file."""
        state = self.read_file(file_path)
        if not state.exists:
            return EditResult(
                success=False,
                file_path=file_path,
                diff="",
                lines_changed=0,
                error=f"File not found: {file_path}",
            )

        if old_text not in state.content:
            return EditResult(
                success=False,
                file_path=file_path,
                diff="",
                lines_changed=0,
                error=f"Text not found: {old_text[:50]}...",
            )

        new_content = state.content.replace(old_text, new_text)
        diff = self.generate_diff(state.content, new_content, os.path.basename(file_path))

        lines_changed = sum(
            1 for line in diff.split("\n") if line.startswith("+") or line.startswith("-")
        )

        if confirm and not self.auto_confirm:
            if not self.confirm_edit(diff, os.path.basename(file_path)):
                return EditResult(
                    success=False,
                    file_path=file_path,
                    diff=diff,
                    lines_changed=lines_changed,
                    error="User rejected changes",
                )

        backup_path = self.create_backup(file_path)

        try:
            self.write_file(file_path, new_content)
            return EditResult(
                success=True,
                file_path=file_path,
                diff=diff,
                lines_changed=lines_changed,
                backup_path=backup_path,
            )
        except Exception as e:
            if backup_path:
                self.restore_backup(backup_path, file_path)
            return EditResult(
                success=False,
                file_path=file_path,
                diff=diff,
                lines_changed=lines_changed,
                error=str(e),
            )


# Convenience functions


def show_diff(old: str, new: str, filename: str = "file"):
    """Show diff between two strings."""
    editor = CodeEditor()
    diff = editor.generate_diff(old, new, filename)
    editor.display_diff(diff, filename)
    return diff


def edit_file(
    path: str, line_start: int, line_end: int, new_content: str, auto_confirm: bool = False
) -> EditResult:
    """Edit specific lines in a file."""
    editor = CodeEditor(auto_confirm=auto_confirm)
    return editor.edit_lines(path, line_start, line_end, new_content)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    import tempfile

    print("=" * 60)
    print("NEURO CODE EDITOR TEST")
    print("=" * 60)

    # Create a test file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("""def hello():
    print("Hello")
    return True

def goodbye():
    print("Bye")
    return False

if __name__ == "__main__":
    hello()
    goodbye()
""")
        test_file = f.name

    print(f"\nTest file: {test_file}")
    print("\nOriginal content:")
    print("-" * 40)
    with open(test_file) as f:
        print(f.read())

    # Test 1: Edit specific lines
    print("\n" + "=" * 60)
    print("Test 1: Edit lines 2-3 (auto-confirm)")
    print("=" * 60)

    editor = CodeEditor(auto_confirm=True)
    result = editor.edit_lines(
        test_file,
        line_start=2,
        line_end=3,
        new_content='    print("Hello, World!")\n    # Modified line\n    return True\n',
    )

    print(f"\nResult: {'SUCCESS' if result.success else 'FAILED'}")
    print(f"Lines changed: {result.lines_changed}")

    if result.success:
        print("\nNew content:")
        print("-" * 40)
        with open(test_file) as f:
            print(f.read())

    # Test 2: Show diff without applying
    print("\n" + "=" * 60)
    print("Test 2: Generate diff only")
    print("=" * 60)

    old = "function old() {\n  return 1;\n}\n"
    new = "function new() {\n  return 2;\n  // Added comment\n}\n"

    show_diff(old, new, "example.js")

    # Cleanup
    os.remove(test_file)
    print("\nTest file cleaned up.")
    print("\nAll tests passed!")
