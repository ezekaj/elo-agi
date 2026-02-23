"""
Spinner - Rich-based animated loading indicators.
"""

from typing import Optional
from contextlib import contextmanager

from rich.console import Console
from rich.status import Status


class Spinner:
    """
    Animated spinner using Rich.

    Provides a clean loading indicator during async operations.
    """

    def __init__(
        self,
        message: str = "Loading",
        spinner: str = "dots",
        console: Optional[Console] = None,
    ):
        self.message = message
        self.spinner_type = spinner
        self.console = console or Console()
        self._status: Optional[Status] = None

    def start(self, message: Optional[str] = None):
        """Start the spinner."""
        if message:
            self.message = message
        self._status = self.console.status(
            f"[purple]{self.message}[/purple]",
            spinner=self.spinner_type,
        )
        self._status.start()

    def update(self, message: str):
        """Update the spinner message."""
        self.message = message
        if self._status:
            self._status.update(f"[purple]{message}[/purple]")

    def stop(self, success: bool = True, message: Optional[str] = None):
        """Stop the spinner with final status."""
        if self._status:
            self._status.stop()
            self._status = None

        final_msg = message or self.message
        if success:
            self.console.print(f"  [green]✓[/green] {final_msg}")
        else:
            self.console.print(f"  [red]✗[/red] {final_msg}")

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        success = exc_type is None
        self.stop(success=success)
        return False


class ToolSpinner(Spinner):
    """Spinner specifically for tool execution."""

    def __init__(self, tool_name: str, console: Optional[Console] = None):
        super().__init__(
            message=f"Running {tool_name}",
            spinner="dots",
            console=console,
        )
        self.tool_name = tool_name

    def stop(self, success: bool = True, output: str = ""):
        """Stop with tool-specific output."""
        if self._status:
            self._status.stop()
            self._status = None

        if success:
            self.console.print(f"  [green]✓[/green] [bold]{self.tool_name}[/bold]")
        else:
            self.console.print(f"  [red]✗[/red] [bold]{self.tool_name}[/bold]")

        if output:
            lines = output.strip().split("\n")
            for line in lines[:3]:
                self.console.print(f"    [dim]{line}[/dim]")
            if len(lines) > 3:
                self.console.print(f"    [dim]... ({len(lines) - 3} more lines)[/dim]")


@contextmanager
def spinner(message: str = "Loading", console: Optional[Console] = None):
    """Context manager for a quick spinner."""
    s = Spinner(message, console=console)
    s.start()
    try:
        yield s
        s.stop(success=True)
    except Exception:
        s.stop(success=False)
        raise


@contextmanager
def tool_spinner(tool_name: str, console: Optional[Console] = None):
    """Context manager for tool execution spinner."""
    s = ToolSpinner(tool_name, console=console)
    s.start()
    try:
        yield s
        s.stop(success=True)
    except Exception as e:
        s.stop(success=False, output=str(e))
        raise
