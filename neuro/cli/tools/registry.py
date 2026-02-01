"""
Tool Registry - Register and manage tools.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Callable, Optional, List
import os
import subprocess


@dataclass
class Tool:
    """A registered tool."""
    name: str
    description: str
    func: Callable
    schema: Dict[str, Any] = field(default_factory=dict)


class ToolRegistry:
    """
    Registry for tools.

    Includes built-in tools and supports custom tool registration.
    """

    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self._register_builtins()

    def _register_builtins(self):
        """Register built-in tools."""
        # File operations
        self.register(
            name="read_file",
            description="Read contents of a file",
            func=self._read_file,
            schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"}
                },
                "required": ["path"]
            }
        )

        self.register(
            name="write_file",
            description="Write content to a file",
            func=self._write_file,
            schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["path", "content"]
            }
        )

        self.register(
            name="edit_file",
            description="Edit specific lines in a file",
            func=self._edit_file,
            schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"}
                },
                "required": ["path", "old_text", "new_text"]
            }
        )

        self.register(
            name="list_files",
            description="List files in a directory",
            func=self._list_files,
            schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "default": "."}
                }
            }
        )

        # Shell
        self.register(
            name="run_command",
            description="Run a shell command",
            func=self._run_command,
            schema={
                "type": "object",
                "properties": {
                    "command": {"type": "string"}
                },
                "required": ["command"]
            }
        )

        # Git
        self.register(
            name="git_status",
            description="Get git repository status",
            func=self._git_status,
        )

        self.register(
            name="git_diff",
            description="Get git diff",
            func=self._git_diff,
        )

        # Web (placeholder - needs actual implementation)
        self.register(
            name="web_search",
            description="Search the web",
            func=self._web_search,
            schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        )

    def register(
        self,
        name: str,
        description: str,
        func: Callable,
        schema: Dict[str, Any] = None,
    ):
        """Register a tool."""
        self.tools[name] = Tool(
            name=name,
            description=description,
            func=func,
            schema=schema or {},
        )

    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self.tools.get(name)

    def list_tools(self) -> List[str]:
        """List all tool names."""
        return list(self.tools.keys())

    def get_tools_schema(self) -> List[Dict]:
        """Get schema for all tools (for LLM)."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.schema,
            }
            for t in self.tools.values()
        ]

    # Built-in tool implementations

    def _read_file(self, path: str) -> str:
        """Read a file."""
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            return f"Error: File not found: {path}"
        try:
            with open(path, 'r') as f:
                content = f.read()
            return content[:50000]  # Limit size
        except Exception as e:
            return f"Error reading file: {e}"

    def _write_file(self, path: str, content: str) -> str:
        """Write to a file."""
        path = os.path.expanduser(path)
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w') as f:
                f.write(content)
            return f"Successfully wrote {len(content)} characters to {path}"
        except Exception as e:
            return f"Error writing file: {e}"

    def _edit_file(self, path: str, old_text: str, new_text: str) -> str:
        """Edit a file by replacing text."""
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            return f"Error: File not found: {path}"
        try:
            with open(path, 'r') as f:
                content = f.read()

            if old_text not in content:
                return f"Error: Text not found in file"

            new_content = content.replace(old_text, new_text, 1)

            with open(path, 'w') as f:
                f.write(new_content)

            return f"Successfully edited {path}"
        except Exception as e:
            return f"Error editing file: {e}"

    def _list_files(self, path: str = ".") -> str:
        """List files in directory."""
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            return f"Error: Path not found: {path}"
        try:
            items = os.listdir(path)
            result = []
            for item in sorted(items)[:100]:
                full_path = os.path.join(path, item)
                if os.path.isdir(full_path):
                    result.append(f"[DIR] {item}/")
                else:
                    size = os.path.getsize(full_path)
                    result.append(f"      {item} ({size} bytes)")
            return "\n".join(result)
        except Exception as e:
            return f"Error listing files: {e}"

    def _run_command(self, command: str) -> str:
        """Run a shell command."""
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
            output = result.stdout
            if result.stderr:
                output += f"\nSTDERR:\n{result.stderr}"
            return output[:10000]
        except subprocess.TimeoutExpired:
            return "Error: Command timed out"
        except Exception as e:
            return f"Error running command: {e}"

    def _git_status(self) -> str:
        """Get git status."""
        return self._run_command("git status --porcelain")

    def _git_diff(self) -> str:
        """Get git diff."""
        return self._run_command("git diff")

    def _web_search(self, query: str) -> str:
        """Search the web (placeholder)."""
        # TODO: Implement actual web search
        return f"Web search for '{query}' not yet implemented. Use web_fetch with a specific URL instead."
