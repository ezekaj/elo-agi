"""
UI Renderer - Terminal UI helpers.
"""

import sys


class UIRenderer:
    """Terminal UI rendering helpers."""

    # ANSI codes
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Colors
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    MAGENTA = "\033[35m"
    BLUE = "\033[34m"

    # Bright colors
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"

    # Box drawing
    TOP_LEFT = "╭"
    TOP_RIGHT = "╮"
    BOTTOM_LEFT = "╰"
    BOTTOM_RIGHT = "╯"
    HORIZONTAL = "─"
    VERTICAL = "│"

    WIDTH = 60

    def print_header(self):
        """Print the main header."""
        print()
        print(f"  {self.CYAN}{self.BOLD}╭{'─' * 50}╮{self.RESET}")
        print(f"  {self.CYAN}{self.BOLD}│{self.RESET}{'NEURO v3.0':^50}{self.CYAN}{self.BOLD}│{self.RESET}")
        print(f"  {self.CYAN}{self.BOLD}│{self.RESET}{self.DIM}{'Local AI That Learns From Your Code':^50}{self.RESET}{self.CYAN}{self.BOLD}│{self.RESET}")
        print(f"  {self.CYAN}{self.BOLD}╰{'─' * 50}╯{self.RESET}")
        print()

    def print_divider(self, width: int = 55):
        """Print a divider line."""
        print(f"  {self.DIM}{'─' * width}{self.RESET}")

    def print_status(self, label: str, value: str, color: str = None):
        """Print a status line."""
        color = color or self.GREEN
        print(f"  {color}●{self.RESET} {label}: {value}")

    def print_dim(self, text: str):
        """Print dimmed text."""
        print(f"  {self.DIM}{text}{self.RESET}")

    def print_error(self, text: str):
        """Print error text."""
        print(f"  {self.RED}✗ {text}{self.RESET}")

    def print_success(self, text: str):
        """Print success text."""
        print(f"  {self.GREEN}✓ {text}{self.RESET}")

    def print_warning(self, text: str):
        """Print warning text."""
        print(f"  {self.YELLOW}⚠ {text}{self.RESET}")

    def print_box(self, title: str, lines: list):
        """Print a bordered box."""
        width = self.WIDTH

        print(f"  {self.TOP_LEFT}{self.HORIZONTAL * 2} {self.BOLD}{self.CYAN}{title}{self.RESET} {self.HORIZONTAL * (width - len(title) - 5)}{self.TOP_RIGHT}")

        for line in lines:
            visible_len = len(line.replace(self.RESET, "").replace(self.DIM, "").replace(self.CYAN, "").replace(self.GREEN, ""))
            padding = width - visible_len - 4
            print(f"  {self.VERTICAL} {line}{' ' * max(0, padding)} {self.VERTICAL}")

        print(f"  {self.BOTTOM_LEFT}{self.HORIZONTAL * (width - 2)}{self.BOTTOM_RIGHT}")

    def clear_line(self):
        """Clear the current line."""
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()

    def move_cursor_up(self, n: int = 1):
        """Move cursor up n lines."""
        sys.stdout.write(f"\033[{n}A")
        sys.stdout.flush()

    def progress_bar(
        self,
        current: int,
        total: int,
        width: int = 30,
        prefix: str = "",
    ) -> str:
        """Generate a progress bar string."""
        if total == 0:
            pct = 0
        else:
            pct = current / total

        filled = int(width * pct)
        empty = width - filled

        if pct < 0.5:
            color = self.GREEN
        elif pct < 0.8:
            color = self.YELLOW
        else:
            color = self.RED

        bar = f"{color}{'█' * filled}{self.DIM}{'░' * empty}{self.RESET}"

        if prefix:
            return f"{prefix} [{bar}] {pct:.0%}"
        return f"[{bar}] {pct:.0%}"
