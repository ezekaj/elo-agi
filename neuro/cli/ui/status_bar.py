"""
Status Bar - Persistent status display at bottom of terminal.
"""

import sys
import time
import threading
from typing import Optional
from dataclasses import dataclass


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
    Persistent status bar at bottom of terminal.

    Shows:
    - Permission mode
    - Model name
    - Token count
    - Current operation
    """

    # ANSI codes
    SAVE_CURSOR = "\033[s"
    RESTORE_CURSOR = "\033[u"
    MOVE_TO_BOTTOM = "\033[999;1H"
    CLEAR_LINE = "\033[K"

    DIM = "\033[2m"
    RESET = "\033[0m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"

    def __init__(self):
        self._state = StatusBarState()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_render = ""

    def start(self):
        """Start the status bar."""
        self._running = True
        self._thread = threading.Thread(target=self._update_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the status bar."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1)
        # Clear status bar
        sys.stdout.write(f"{self.SAVE_CURSOR}{self.MOVE_TO_BOTTOM}{self.CLEAR_LINE}{self.RESTORE_CURSOR}")
        sys.stdout.flush()

    def update(self, **kwargs):
        """Update status bar state."""
        for k, v in kwargs.items():
            if hasattr(self._state, k):
                setattr(self._state, k, v)

    def _render(self) -> str:
        """Render the status bar content."""
        parts = []

        # Mode indicator
        mode_colors = {
            "default": self.CYAN,
            "acceptEdits": self.GREEN,
            "plan": self.YELLOW,
            "bypassPermissions": self.YELLOW,
        }
        mode_color = mode_colors.get(self._state.mode, self.DIM)
        parts.append(f"{mode_color}[{self._state.mode}]{self.RESET}")

        # Model
        if self._state.model:
            parts.append(f"{self.DIM}{self._state.model}{self.RESET}")

        # Token count
        if self._state.tokens > 0:
            parts.append(f"{self.DIM}{self._state.tokens:,} tokens{self.RESET}")

        # Current operation
        if self._state.tool_running:
            parts.append(f"{self.CYAN}⚙ {self._state.tool_running}{self.RESET}")

        # Custom message
        if self._state.message:
            parts.append(self._state.message)

        return " │ ".join(parts)

    def _update_loop(self):
        """Background update loop."""
        while self._running:
            try:
                content = self._render()

                if content != self._last_render:
                    line = f"{self.SAVE_CURSOR}{self.MOVE_TO_BOTTOM}{self.CLEAR_LINE}  {content}{self.RESTORE_CURSOR}"
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    self._last_render = content
            except Exception:
                pass

            time.sleep(0.5)
