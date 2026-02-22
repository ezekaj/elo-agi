"""
Status Bar - Simple status display (non-persistent).

The persistent bottom bar approach causes issues with terminal rendering.
Instead, we provide status updates inline when relevant.
"""

from typing import Optional
from dataclasses import dataclass
from rich.console import Console


@dataclass
class StatusBarState:
    """Status bar state."""

    mode: str = "default"
    model: str = ""
    tokens: int = 0
    context_pct: float = 0.0
    tool_running: Optional[str] = None
    message: str = ""


class StatusBar:
    """
    Simple status tracking (no persistent display).

    Instead of trying to maintain a persistent bottom bar (which causes
    rendering issues), we track state and provide it on demand.
    """

    def __init__(self, console: Optional[Console] = None):
        self._state = StatusBarState()
        self.console = console or Console()
        self._running = False

    def start(self):
        """Start tracking (no-op for compatibility)."""
        self._running = True

    def stop(self):
        """Stop tracking."""
        self._running = False

    def update(self, **kwargs):
        """Update status state."""
        for k, v in kwargs.items():
            if hasattr(self._state, k):
                setattr(self._state, k, v)

    def get_state(self) -> StatusBarState:
        """Get current state."""
        return self._state

    def render_inline(self) -> str:
        """Render status as inline text."""
        parts = []

        if self._state.mode != "default":
            parts.append(f"[{self._state.mode}]")

        if self._state.model:
            parts.append(self._state.model)

        if self._state.tokens > 0:
            parts.append(f"{self._state.tokens:,} tokens")

        if self._state.tool_running:
            parts.append(f"running: {self._state.tool_running}")

        if self._state.message:
            parts.append(self._state.message)

        return " | ".join(parts)

    def print_status(self):
        """Print current status inline."""
        status = self.render_inline()
        if status:
            self.console.print(f"  [dim]{status}[/dim]")
