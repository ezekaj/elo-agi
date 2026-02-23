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
    read_only: bool = False


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
            description="Read contents of a file. Supports optional offset and limit for large files.",
            func=self._read_file,
            schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                    "offset": {"type": "integer", "description": "Line number to start from (1-indexed)", "default": 0},
                    "limit": {"type": "integer", "description": "Number of lines to read", "default": 0},
                },
                "required": ["path"],
            },
            read_only=True,
        )

        self.register(
            name="write_file",
            description="Write content to a file",
            func=self._write_file,
            schema={
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
            },
        )

        self.register(
            name="edit_file",
            description="Edit specific text in a file by replacing old_text with new_text. Use replace_all=true to replace all occurrences.",
            func=self._edit_file,
            schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_text": {"type": "string", "description": "Text to find and replace"},
                    "new_text": {"type": "string", "description": "Replacement text"},
                    "replace_all": {"type": "boolean", "default": False, "description": "Replace all occurrences"},
                },
                "required": ["path", "old_text", "new_text"],
            },
        )

        self.register(
            name="list_files",
            description="List files in a directory. Use recursive=true to list all files recursively.",
            func=self._list_files,
            schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "default": "."},
                    "recursive": {"type": "boolean", "default": False, "description": "List recursively"},
                },
            },
            read_only=True,
        )

        # Shell
        self.register(
            name="run_command",
            description="Run a shell command. Supports timeout and background execution.",
            func=self._run_command,
            schema={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to run"},
                    "timeout": {"type": "integer", "default": 60, "description": "Timeout in seconds"},
                    "background": {"type": "boolean", "default": False, "description": "Run in background"},
                },
                "required": ["command"],
            },
        )

        # Git
        self.register(
            name="git_status",
            description="Get git repository status",
            func=self._git_status,
            read_only=True,
        )

        self.register(
            name="git_diff",
            description="Get git diff",
            func=self._git_diff,
            read_only=True,
        )

        # Web search
        self.register(
            name="web_search",
            description="Search the web for information",
            func=self._web_search,
            schema={
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search query"}},
                "required": ["query"],
            },
            read_only=True,
        )

        # Web fetch
        self.register(
            name="web_fetch",
            description="Fetch content from a URL",
            func=self._web_fetch,
            schema={
                "type": "object",
                "properties": {"url": {"type": "string", "description": "URL to fetch"}},
                "required": ["url"],
            },
            read_only=True,
        )

        # Self-improvement
        self.register(
            name="improve_self",
            description="Analyze and improve ELO's own code",
            func=self._improve_self,
            schema={
                "type": "object",
                "properties": {
                    "area": {
                        "type": "string",
                        "description": "Area to improve (e.g., 'learning', 'tools', 'ui')",
                    }
                },
                "required": ["area"],
            },
        )

        # Glob (file search)
        from .builtins.search import glob_files
        self.register(
            name="glob_files",
            description="Fast file pattern matching (e.g., '**/*.py', 'src/**/*.ts'). Returns matching paths sorted by modification time.",
            func=glob_files,
            schema={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern (e.g., '**/*.py')"},
                    "path": {"type": "string", "description": "Base directory to search in", "default": "."},
                },
                "required": ["pattern"],
            },
            read_only=True,
        )

        # Grep (content search)
        from .builtins.search import grep_content
        self.register(
            name="grep_content",
            description="Search file contents with regex. Uses ripgrep if available, falls back to Python. Supports output modes: content (matching lines), files_with_matches (file paths only), count.",
            func=grep_content,
            schema={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern to search for"},
                    "path": {"type": "string", "description": "Directory to search in", "default": "."},
                    "output_mode": {
                        "type": "string",
                        "enum": ["content", "files_with_matches", "count"],
                        "default": "content",
                        "description": "Output format",
                    },
                    "glob_filter": {"type": "string", "description": "File pattern filter (e.g., '*.py')"},
                    "case_insensitive": {"type": "boolean", "default": False},
                    "context_lines": {"type": "integer", "default": 0, "description": "Lines of context around matches"},
                },
                "required": ["pattern"],
            },
            read_only=True,
        )

        # Notebook edit
        from .builtins.notebook import notebook_edit
        self.register(
            name="notebook_edit",
            description="Edit a Jupyter notebook (.ipynb) cell. Supports replace, insert, and delete operations.",
            func=notebook_edit,
            schema={
                "type": "object",
                "properties": {
                    "notebook_path": {"type": "string", "description": "Path to .ipynb file"},
                    "cell_number": {"type": "integer", "description": "0-indexed cell number"},
                    "new_source": {"type": "string", "description": "New cell content"},
                    "edit_mode": {
                        "type": "string",
                        "enum": ["replace", "insert", "delete"],
                        "default": "replace",
                    },
                    "cell_type": {
                        "type": "string",
                        "enum": ["code", "markdown"],
                        "default": "code",
                    },
                },
                "required": ["notebook_path", "cell_number"],
            },
        )

    def register(
        self,
        name: str,
        description: str,
        func: Callable,
        schema: Dict[str, Any] = None,
        read_only: bool = False,
    ):
        """Register a tool."""
        self.tools[name] = Tool(
            name=name,
            description=description,
            func=func,
            schema=schema or {},
            read_only=read_only,
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

    def get_ollama_tools(self) -> List[Dict]:
        """Generate Ollama-format tool schemas for all registered tools."""
        result = []
        for tool in self.tools.values():
            result.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.schema or {"type": "object", "properties": {}},
                },
            })
        return result

    async def execute_tool(self, name: str, args: Dict[str, Any]) -> Any:
        """Execute a tool by name with args dict, mapping to function params."""
        import asyncio
        import inspect

        tool = self.tools.get(name)
        if not tool:
            raise ValueError(f"Tool not found: {name}")

        sig = inspect.signature(tool.func)
        params = sig.parameters

        # Map args dict to function parameters
        call_args = {}
        for param_name, param in params.items():
            if param_name in args:
                call_args[param_name] = args[param_name]
            elif param.default is not inspect.Parameter.empty:
                pass  # Use default

        if asyncio.iscoroutinefunction(tool.func):
            return await tool.func(**call_args)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: tool.func(**call_args))

    def get_read_only_tools(self) -> List[str]:
        """Get names of all read-only (safe) tools."""
        return [name for name, tool in self.tools.items() if tool.read_only]

    # Built-in tool implementations

    def _read_file(self, path: str, offset: int = 0, limit: int = 0) -> str:
        """Read a file with optional offset and limit (line-based)."""
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            return f"Error: File not found: {path}"
        if os.path.isdir(path):
            return self._list_files(path)
        try:
            with open(path, "r") as f:
                if offset > 0 or limit > 0:
                    lines = f.readlines()
                    start = max(0, offset - 1) if offset > 0 else 0
                    end = start + limit if limit > 0 else len(lines)
                    selected = lines[start:end]
                    numbered = [f"{i+start+1:6d}\t{line}" for i, line in enumerate(selected)]
                    return "".join(numbered)[:50000]
                else:
                    content = f.read()
                    return content[:50000]
        except Exception as e:
            return f"Error reading file: {e}"

    def _write_file(self, path: str, content: str) -> str:
        """Write to a file."""
        path = os.path.expanduser(path)
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(content)
            return f"Successfully wrote {len(content)} characters to {path}"
        except Exception as e:
            return f"Error writing file: {e}"

    def _edit_file(self, path: str, old_text: str, new_text: str, replace_all: bool = False) -> str:
        """Edit a file by replacing text."""
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            return f"Error: File not found: {path}"
        try:
            with open(path, "r") as f:
                content = f.read()

            if old_text not in content:
                return "Error: Text not found in file"

            if replace_all:
                count = content.count(old_text)
                new_content = content.replace(old_text, new_text)
            else:
                count = 1
                new_content = content.replace(old_text, new_text, 1)

            with open(path, "w") as f:
                f.write(new_content)

            return f"Successfully edited {path} ({count} replacement{'s' if count > 1 else ''})"
        except Exception as e:
            return f"Error editing file: {e}"

    def _list_files(self, path: str = ".", recursive: bool = False) -> str:
        """List files in directory."""
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            return f"Error: Path not found: {path}"
        try:
            if recursive:
                results = []
                for root, dirs, files in os.walk(path):
                    dirs[:] = [d for d in dirs if not d.startswith(".")]
                    for f in sorted(files)[:500]:
                        rel = os.path.relpath(os.path.join(root, f), path)
                        results.append(rel)
                    if len(results) >= 500:
                        break
                return "\n".join(results[:500])
            else:
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

    def _run_command(self, command: str, timeout: int = 60, background: bool = False) -> str:
        """Run a shell command with configurable timeout."""
        if background:
            import threading
            def run_bg():
                subprocess.run(command, shell=True, capture_output=True, text=True)
            threading.Thread(target=run_bg, daemon=True).start()
            return f"Command started in background: {command}"

        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True, timeout=timeout,
            )
            output = result.stdout
            if result.stderr:
                output += f"\nSTDERR:\n{result.stderr}"
            return output[:10000]
        except subprocess.TimeoutExpired:
            return f"Error: Command timed out after {timeout}s"
        except Exception as e:
            return f"Error running command: {e}"

    def _git_status(self) -> str:
        """Get git status."""
        return self._run_command("git status --porcelain")

    def _git_diff(self) -> str:
        """Get git diff."""
        return self._run_command("git diff")

    def _web_search(self, query: str) -> str:
        """Search the web using DuckDuckGo."""
        import urllib.request
        import urllib.parse
        import json
        import re

        # Try DuckDuckGo Instant Answer API first (good for factual queries)
        try:
            encoded_query = urllib.parse.quote(query)
            url = f"https://api.duckduckgo.com/?q={encoded_query}&format=json&no_html=1"

            req = urllib.request.Request(url, headers={"User-Agent": "ELO-AGI/1.0"})
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())

            results = []

            if data.get("Abstract"):
                results.append(f"Summary: {data['Abstract']}")
                if data.get("AbstractURL"):
                    results.append(f"Source: {data['AbstractURL']}")

            for topic in data.get("RelatedTopics", [])[:5]:
                if isinstance(topic, dict) and topic.get("Text"):
                    results.append(f"- {topic['Text'][:200]}")

            if results:
                return "\n".join(results)
        except Exception:
            pass

        # Fallback: DuckDuckGo HTML search (better for news/general queries)
        try:
            encoded_query = urllib.parse.quote(query)
            url = f"https://html.duckduckgo.com/html/?q={encoded_query}"

            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    )
                },
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                html_content = response.read().decode("utf-8", errors="ignore")

            results = []

            # Extract titles and URLs
            title_matches = re.findall(
                r'class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
                html_content,
                re.DOTALL,
            )

            # Extract snippets (full inner HTML, then strip tags)
            snippet_matches = re.findall(
                r'class="result__snippet"[^>]*>(.*?)</(?:a|td|span)',
                html_content,
                re.DOTALL,
            )

            for i, (url_match, title_html) in enumerate(title_matches[:5]):
                title = re.sub(r"<[^>]+>", "", title_html).strip()
                # Decode DuckDuckGo redirect URL
                if "uddg=" in url_match:
                    actual_url = urllib.parse.unquote(
                        re.search(r"uddg=([^&]+)", url_match).group(1)
                    )
                else:
                    actual_url = url_match

                snippet = ""
                if i < len(snippet_matches):
                    snippet = re.sub(r"<[^>]+>", "", snippet_matches[i]).strip()
                    snippet = snippet.replace("&quot;", '"').replace("&amp;", "&")
                    snippet = snippet.replace("&lt;", "<").replace("&gt;", ">")

                entry = f"- {title}"
                if snippet:
                    entry += f"\n  {snippet[:200]}"
                entry += f"\n  {actual_url}"
                results.append(entry)

            if results:
                return f"Search results for '{query}':\n\n" + "\n\n".join(results)

            return f"No results found for '{query}'"

        except Exception as e:
            return f"Web search error: {e}"

    def _improve_self(self, area: str) -> str:
        """Analyze and return info about ELO's code for self-improvement."""
        import os

        # Get path to neuro/ directory (this file is in neuro/cli/tools/)
        this_file = os.path.abspath(__file__)  # .../neuro/cli/tools/registry.py
        neuro_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(this_file)))
        )  # .../neuro project root

        areas = {
            "learning": [
                "neuro/active_learning.py",
                "neuro/self_training.py",
            ],
            "tools": [
                "neuro/cli/tools/registry.py",
                "neuro/cli/tools/executor.py",
            ],
            "ui": [
                "neuro/cli/ui/renderer.py",
                "neuro/cli/app.py",
            ],
            "core": [
                "neuro/cli/app.py",
                "neuro/cli/main.py",
                "neuro/cli/core/stream.py",
            ],
            "all": [],  # Will scan everything
        }

        target_files = areas.get(area.lower(), areas.get("core", []))

        result = [f"ELO Self-Improvement Analysis: {area}"]
        result.append(f"Root: {neuro_root}")
        result.append("")

        for rel_path in target_files:
            full_path = os.path.join(neuro_root, rel_path)
            if os.path.exists(full_path):
                try:
                    with open(full_path) as f:
                        content = f.read()
                    lines = len(content.split("\n"))
                    result.append(f"File: {rel_path} ({lines} lines)")

                    # Find TODOs and FIXMEs
                    for i, line in enumerate(content.split("\n"), 1):
                        if "TODO" in line or "FIXME" in line or "HACK" in line:
                            result.append(f"  Line {i}: {line.strip()[:80]}")

                except Exception as e:
                    result.append(f"Error reading {rel_path}: {e}")

        result.append("")
        result.append("Use read_file to examine specific files, then edit_file to improve them.")
        result.append("After improvements, run: run_command('python test_cli.py') to verify.")

        return "\n".join(result)

    def _web_fetch(self, url: str) -> str:
        """Fetch content from a URL."""
        try:
            import urllib.request
            import re

            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
                },
            )
            with urllib.request.urlopen(req, timeout=15) as response:
                html = response.read().decode("utf-8", errors="ignore")

            # Strip HTML tags for basic text extraction
            text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
            text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text).strip()

            return text[:15000]  # Limit size

        except Exception as e:
            return f"Error fetching URL: {e}"
