"""File operation tools: read, write, edit."""

import os
from pathlib import Path
from .base import Tool, ToolResult


class ReadTool(Tool):
    """Read contents of a file."""

    name = "read_file"
    description = "Read the contents of a file. Can optionally read specific line ranges."
    requires_permission = False

    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to read"},
            "start_line": {
                "type": "integer",
                "description": "Starting line number (1-indexed, optional)",
            },
            "end_line": {
                "type": "integer",
                "description": "Ending line number (inclusive, optional)",
            },
        },
        "required": ["path"],
    }

    def execute(
        self, path: str, start_line: int = None, end_line: int = None, **kwargs
    ) -> ToolResult:
        try:
            p = Path(path).expanduser().resolve()

            if not p.exists():
                return ToolResult(False, "", f"File not found: {path}")

            if not p.is_file():
                return ToolResult(False, "", f"Not a file: {path}")

            content = p.read_text()
            lines = content.split("\n")

            # Handle line range
            if start_line is not None or end_line is not None:
                start = (start_line or 1) - 1  # Convert to 0-indexed
                end = end_line or len(lines)
                lines = lines[start:end]
                content = "\n".join(lines)

            # Add line numbers
            numbered_lines = []
            base_line = start_line or 1
            for i, line in enumerate(lines):
                numbered_lines.append(f"{base_line + i:4d} | {line}")

            return ToolResult(
                True, "\n".join(numbered_lines), data={"path": str(p), "lines": len(lines)}
            )

        except PermissionError:
            return ToolResult(False, "", f"Permission denied: {path}")
        except Exception as e:
            return ToolResult(False, "", str(e))


class WriteTool(Tool):
    """Write content to a file."""

    name = "write_file"
    description = "Create or overwrite a file with new content."
    requires_permission = True

    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to write"},
            "content": {"type": "string", "description": "Content to write to the file"},
        },
        "required": ["path", "content"],
    }

    def execute(self, path: str, content: str, **kwargs) -> ToolResult:
        try:
            p = Path(path).expanduser().resolve()

            # Create parent directories if needed
            p.parent.mkdir(parents=True, exist_ok=True)

            # Backup existing file
            backup_path = None
            if p.exists():
                backup_path = p.with_suffix(p.suffix + ".bak")
                backup_path.write_text(p.read_text())

            p.write_text(content)

            msg = f"Wrote {len(content)} bytes to {path}"
            if backup_path:
                msg += f" (backup: {backup_path.name})"

            return ToolResult(
                True,
                msg,
                data={
                    "path": str(p),
                    "bytes": len(content),
                    "backup": str(backup_path) if backup_path else None,
                },
            )

        except PermissionError:
            return ToolResult(False, "", f"Permission denied: {path}")
        except Exception as e:
            return ToolResult(False, "", str(e))


class EditTool(Tool):
    """Edit a file by replacing text."""

    name = "edit_file"
    description = "Edit a file by replacing specific text. Use for precise modifications."
    requires_permission = True

    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to edit"},
            "old_text": {"type": "string", "description": "The exact text to find and replace"},
            "new_text": {"type": "string", "description": "The text to replace it with"},
        },
        "required": ["path", "old_text", "new_text"],
    }

    def execute(self, path: str, old_text: str, new_text: str, **kwargs) -> ToolResult:
        try:
            p = Path(path).expanduser().resolve()

            if not p.exists():
                return ToolResult(False, "", f"File not found: {path}")

            content = p.read_text()

            if old_text not in content:
                return ToolResult(False, "", f"Text not found in file: '{old_text[:50]}...'")

            # Count occurrences
            count = content.count(old_text)

            # Backup
            backup_path = p.with_suffix(p.suffix + ".bak")
            backup_path.write_text(content)

            # Replace
            new_content = content.replace(old_text, new_text)
            p.write_text(new_content)

            return ToolResult(
                True,
                f"Replaced {count} occurrence(s) in {path}",
                data={"path": str(p), "replacements": count, "backup": str(backup_path)},
            )

        except PermissionError:
            return ToolResult(False, "", f"Permission denied: {path}")
        except Exception as e:
            return ToolResult(False, "", str(e))
