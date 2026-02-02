"""Search tools: glob, grep, ls."""

import os
import re
import fnmatch
from pathlib import Path
from .base import Tool, ToolResult


class GlobTool(Tool):
    """Find files matching a pattern."""

    name = "glob"
    description = "Find files matching a glob pattern (e.g., '*.py', 'src/**/*.ts')."
    requires_permission = False

    parameters = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Glob pattern to match (e.g., '*.py', '**/*.js')"
            },
            "path": {
                "type": "string",
                "description": "Directory to search in (default: current directory)"
            }
        },
        "required": ["pattern"]
    }

    def execute(self, pattern: str, path: str = ".", **kwargs) -> ToolResult:
        try:
            base = Path(path).expanduser().resolve()

            if not base.exists():
                return ToolResult(False, "", f"Directory not found: {path}")

            matches = list(base.glob(pattern))
            matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)

            if not matches:
                return ToolResult(True, f"No files matching '{pattern}'", data={"count": 0})

            # Format output
            output_lines = [f"Found {len(matches)} file(s) matching '{pattern}':", ""]
            for p in matches[:50]:  # Limit to 50
                rel_path = p.relative_to(base) if p.is_relative_to(base) else p
                size = p.stat().st_size if p.is_file() else 0
                output_lines.append(f"  {rel_path} ({size:,} bytes)")

            if len(matches) > 50:
                output_lines.append(f"  ... and {len(matches) - 50} more")

            return ToolResult(
                True,
                '\n'.join(output_lines),
                data={"count": len(matches), "files": [str(m) for m in matches[:50]]}
            )

        except Exception as e:
            return ToolResult(False, "", str(e))


class GrepTool(Tool):
    """Search file contents with regex."""

    name = "grep"
    description = "Search for a pattern in file contents. Supports regex."
    requires_permission = False

    parameters = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Regex pattern to search for"
            },
            "path": {
                "type": "string",
                "description": "File or directory to search in"
            },
            "file_pattern": {
                "type": "string",
                "description": "Only search files matching this glob (e.g., '*.py')"
            }
        },
        "required": ["pattern", "path"]
    }

    def execute(self, pattern: str, path: str, file_pattern: str = "*", **kwargs) -> ToolResult:
        try:
            base = Path(path).expanduser().resolve()
            regex = re.compile(pattern, re.IGNORECASE)

            matches = []

            if base.is_file():
                files = [base]
            else:
                files = list(base.rglob(file_pattern))

            for file_path in files:
                if not file_path.is_file():
                    continue

                try:
                    content = file_path.read_text()
                    for i, line in enumerate(content.split('\n'), 1):
                        if regex.search(line):
                            rel_path = file_path.relative_to(base) if file_path.is_relative_to(base) else file_path
                            matches.append((str(rel_path), i, line.strip()[:100]))
                except (UnicodeDecodeError, PermissionError):
                    continue

            if not matches:
                return ToolResult(True, f"No matches for '{pattern}'", data={"count": 0})

            # Format output
            output_lines = [f"Found {len(matches)} match(es) for '{pattern}':", ""]
            for file_path, line_num, line_content in matches[:30]:
                output_lines.append(f"  {file_path}:{line_num}: {line_content}")

            if len(matches) > 30:
                output_lines.append(f"  ... and {len(matches) - 30} more matches")

            return ToolResult(
                True,
                '\n'.join(output_lines),
                data={"count": len(matches), "matches": matches[:30]}
            )

        except re.error as e:
            return ToolResult(False, "", f"Invalid regex: {e}")
        except Exception as e:
            return ToolResult(False, "", str(e))


class LsTool(Tool):
    """List directory contents."""

    name = "ls"
    description = "List files and directories in a path."
    requires_permission = False

    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Directory to list (default: current directory)"
            },
            "all": {
                "type": "boolean",
                "description": "Include hidden files (default: false)"
            }
        },
        "required": []
    }

    def execute(self, path: str = ".", all: bool = False, **kwargs) -> ToolResult:
        try:
            p = Path(path).expanduser().resolve()

            if not p.exists():
                return ToolResult(False, "", f"Path not found: {path}")

            if not p.is_dir():
                return ToolResult(False, "", f"Not a directory: {path}")

            entries = []
            for entry in sorted(p.iterdir()):
                if not all and entry.name.startswith('.'):
                    continue

                if entry.is_dir():
                    entries.append(f"  📁 {entry.name}/")
                else:
                    size = entry.stat().st_size
                    entries.append(f"  📄 {entry.name} ({size:,} bytes)")

            output = f"Contents of {p}:\n\n" + '\n'.join(entries)

            return ToolResult(
                True,
                output,
                data={"path": str(p), "count": len(entries)}
            )

        except PermissionError:
            return ToolResult(False, "", f"Permission denied: {path}")
        except Exception as e:
            return ToolResult(False, "", str(e))
