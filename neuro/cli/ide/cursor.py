"""
Cursor IDE Integration for NEURO CLI.

Cursor is a fork of VSCode, so it inherits most functionality.
The main difference is the CLI binary name and socket path.
"""

import os
import subprocess
from typing import Optional

from .vscode import VSCodeIntegration
from .integration import IDEType


class CursorIntegration(VSCodeIntegration):
    """
    Cursor IDE integration.

    Inherits from VSCode integration since Cursor is VSCode-based.
    """

    SOCKET_PATH = "/tmp/neuro-cursor.sock"
    PORT = 19433

    @property
    def ide_type(self) -> IDEType:
        return IDEType.CURSOR

    def _find_code_binary(self) -> Optional[str]:
        """Find the Cursor CLI binary."""
        # Cursor-specific locations
        candidates = [
            "/usr/local/bin/cursor",
            "/usr/bin/cursor",
            os.path.expanduser("~/.local/bin/cursor"),
            # macOS app location
            "/Applications/Cursor.app/Contents/Resources/app/bin/cursor",
            "/Applications/Cursor.app/Contents/MacOS/Cursor",
        ]

        for path in candidates:
            if os.path.exists(path):
                return path

        # Try PATH
        try:
            result = subprocess.run(
                ["which", "cursor"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass

        # Fallback to VSCode binary
        return super()._find_code_binary()
