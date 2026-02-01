"""
Spinner - Animated loading spinner.
"""

import sys
import time
import threading
from typing import Optional, List


class Spinner:
    """Animated spinner for loading states."""

    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    RESET = "\033[0m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    RED = "\033[31m"
    DIM = "\033[2m"

    def __init__(
        self,
        message: str = "Loading",
        frames: Optional[List[str]] = None,
    ):
        self.message = message
        self.frames = frames or self.FRAMES
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._frame = 0

    def start(self, message: Optional[str] = None):
        """Start the spinner."""
        if message:
            self.message = message
        self._running = True
        self._thread = threading.Thread(target=self._animate, daemon=True)
        self._thread.start()

    def update(self, message: str):
        """Update the spinner message."""
        self.message = message

    def stop(self, success: bool = True, message: Optional[str] = None):
        """Stop the spinner with final status."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=0.5)

        # Clear line and show final status
        icon = f"{self.GREEN}✓{self.RESET}" if success else f"{self.RED}✗{self.RESET}"
        final_msg = message or self.message
        sys.stdout.write(f"\r\033[K  {icon} {final_msg}\n")
        sys.stdout.flush()

    def _animate(self):
        """Animation loop."""
        while self._running:
            frame = self.frames[self._frame % len(self.frames)]
            sys.stdout.write(f"\r\033[K  {self.CYAN}{frame}{self.RESET} {self.message}")
            sys.stdout.flush()
            self._frame += 1
            time.sleep(0.08)

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        success = exc_type is None
        self.stop(success=success)
        return False


# Convenience function
def with_spinner(message: str = "Loading"):
    """Decorator to show spinner during function execution."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            spinner = Spinner(message)
            spinner.start()
            try:
                result = func(*args, **kwargs)
                spinner.stop(success=True)
                return result
            except Exception as e:
                spinner.stop(success=False, message=str(e))
                raise
        return wrapper
    return decorator
