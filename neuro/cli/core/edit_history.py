"""
Edit History - Track file edits for undo support.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict
from datetime import datetime
import os


@dataclass
class FileEdit:
    path: str
    original_content: str
    new_content: str
    tool_name: str = ""
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class EditHistory:
    """Track file edits for undo support."""

    MAX_HISTORY = 50

    def __init__(self, project_dir: str = "."):
        self.project_dir = os.path.abspath(project_dir)
        self._history: List[FileEdit] = []

    def record(self, path: str, original: str, new: str, tool_name: str = ""):
        """Record a file edit."""
        self._history.append(FileEdit(
            path=path,
            original_content=original,
            new_content=new,
            tool_name=tool_name,
        ))
        if len(self._history) > self.MAX_HISTORY:
            self._history.pop(0)

    def undo_last(self) -> Optional[FileEdit]:
        """Undo the last edit."""
        if not self._history:
            return None

        edit = self._history.pop()
        try:
            with open(edit.path, "w") as f:
                f.write(edit.original_content)
            return edit
        except Exception:
            self._history.append(edit)
            return None

    def get_recent(self, limit: int = 10) -> List[FileEdit]:
        """Get recent edits."""
        return list(reversed(self._history[-limit:]))

    def get_diff_summary(self) -> List[Dict]:
        """Get summary of all edits in this session."""
        seen = set()
        result = []
        for edit in reversed(self._history):
            if edit.path not in seen:
                seen.add(edit.path)
                result.append({
                    "path": edit.path,
                    "tool": edit.tool_name,
                    "time": edit.timestamp,
                })
        return result
