"""
IDE Integration Base - Abstract interface for IDE integrations.

NEURO can integrate with IDEs to:
- Open files at specific lines
- Get current editor context (file, selection, cursor)
- Sync with file watchers
- Receive commands from IDE extensions
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
import os
import json
import asyncio


class IDEType(Enum):
    """Supported IDE types."""
    VSCODE = "vscode"
    CURSOR = "cursor"
    NEOVIM = "neovim"
    EMACS = "emacs"
    JETBRAINS = "jetbrains"
    UNKNOWN = "unknown"


@dataclass
class EditorContext:
    """Current editor context from IDE."""
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    column: Optional[int] = None
    selection: Optional[str] = None
    selection_start: Optional[tuple] = None  # (line, col)
    selection_end: Optional[tuple] = None  # (line, col)
    language: Optional[str] = None
    workspace_root: Optional[str] = None
    open_files: List[str] = field(default_factory=list)
    dirty_files: List[str] = field(default_factory=list)  # Unsaved files


@dataclass
class IDECommand:
    """Command to execute in IDE."""
    action: str  # "open", "goto", "edit", "save", "close"
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    column: Optional[int] = None
    content: Optional[str] = None
    range_start: Optional[tuple] = None
    range_end: Optional[tuple] = None


class IDEIntegration(ABC):
    """
    Base class for IDE integrations.

    Subclasses implement communication with specific IDEs.
    """

    def __init__(self, workspace_root: str = "."):
        self.workspace_root = os.path.abspath(workspace_root)
        self._connected = False
        self._listeners: List[callable] = []

    @property
    @abstractmethod
    def ide_type(self) -> IDEType:
        """Return the IDE type."""
        pass

    @property
    def connected(self) -> bool:
        """Check if connected to IDE."""
        return self._connected

    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the IDE."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the IDE."""
        pass

    @abstractmethod
    async def get_context(self) -> EditorContext:
        """Get current editor context."""
        pass

    @abstractmethod
    async def execute_command(self, command: IDECommand) -> bool:
        """Execute a command in the IDE."""
        pass

    # Convenience methods

    async def open_file(self, file_path: str, line: Optional[int] = None) -> bool:
        """Open a file in the IDE."""
        cmd = IDECommand(
            action="open",
            file_path=os.path.abspath(file_path),
            line_number=line,
        )
        return await self.execute_command(cmd)

    async def goto_line(self, file_path: str, line: int, column: int = 1) -> bool:
        """Go to a specific position in a file."""
        cmd = IDECommand(
            action="goto",
            file_path=os.path.abspath(file_path),
            line_number=line,
            column=column,
        )
        return await self.execute_command(cmd)

    async def get_selection(self) -> Optional[str]:
        """Get currently selected text."""
        ctx = await self.get_context()
        return ctx.selection

    async def get_current_file(self) -> Optional[str]:
        """Get path of currently open file."""
        ctx = await self.get_context()
        return ctx.file_path

    # Event handling

    def on_event(self, callback: callable):
        """Register event callback."""
        self._listeners.append(callback)

    def _emit_event(self, event_type: str, data: Dict[str, Any]):
        """Emit event to listeners."""
        for listener in self._listeners:
            try:
                listener(event_type, data)
            except Exception:
                pass


def detect_ide() -> IDEType:
    """Detect which IDE is running."""
    # Check for VSCode terminal
    if os.environ.get("TERM_PROGRAM") == "vscode":
        return IDEType.VSCODE

    # Check for Cursor
    if "cursor" in os.environ.get("TERM_PROGRAM", "").lower():
        return IDEType.CURSOR

    # Check for NVIM
    if os.environ.get("NVIM"):
        return IDEType.NEOVIM

    # Check for Emacs
    if os.environ.get("INSIDE_EMACS"):
        return IDEType.EMACS

    # Check for JetBrains terminal
    if os.environ.get("TERMINAL_EMULATOR") == "JetBrains-JediTerm":
        return IDEType.JETBRAINS

    return IDEType.UNKNOWN


def create_integration(
    workspace_root: str = ".",
    ide_type: Optional[IDEType] = None,
) -> Optional[IDEIntegration]:
    """Create an IDE integration based on detected or specified type."""
    if ide_type is None:
        ide_type = detect_ide()

    if ide_type == IDEType.VSCODE:
        from .vscode import VSCodeIntegration
        return VSCodeIntegration(workspace_root)

    elif ide_type == IDEType.CURSOR:
        from .cursor import CursorIntegration
        return CursorIntegration(workspace_root)

    elif ide_type == IDEType.NEOVIM:
        # Could add NeoVim integration later
        return None

    return None
