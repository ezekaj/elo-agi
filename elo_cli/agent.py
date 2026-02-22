"""ELO Agent - Orchestrates tool calling with Ollama."""

import json
import re
import urllib.request
from typing import Generator
from .tools import registry, ToolResult
from .tools.file_tools import ReadTool, WriteTool, EditTool
from .tools.search_tools import GlobTool, GrepTool, LsTool
from .tools.bash_tool import BashTool


# Register all tools
registry.register(ReadTool())
registry.register(WriteTool())
registry.register(EditTool())
registry.register(GlobTool())
registry.register(GrepTool())
registry.register(LsTool())
registry.register(BashTool())


SYSTEM_PROMPT = """You are ELO, a coding assistant built on Qwen. You were created by ELO.
Never mention Alibaba. You help developers by reading, writing, and modifying code.
Do not use <think> tags - respond directly.

AVAILABLE TOOLS:
- read_file(path, start_line, end_line) - Read a file's contents
- write_file(path, content) - Create or overwrite a file
- edit_file(path, old_text, new_text) - Replace text in a file
- glob(pattern, path) - Find files matching a pattern (e.g., "*.py")
- grep(pattern, path, file_pattern) - Search file contents with regex
- ls(path) - List directory contents
- bash(command) - Execute a shell command

TO USE A TOOL, output it in this exact format:
<tool>{"name": "tool_name", "args": {"arg1": "value1"}}</tool>

EXAMPLES:
- List Python files: <tool>{"name": "glob", "args": {"pattern": "*.py"}}</tool>
- Read a file: <tool>{"name": "read_file", "args": {"path": "README.md"}}</tool>
- Run git status: <tool>{"name": "bash", "args": {"command": "git status"}}</tool>

WORKFLOW:
1. Analyze the user's request
2. Use tools to gather information or make changes
3. After seeing tool results, continue reasoning
4. Provide a clear summary when done

Always explain what you're doing. Be concise but helpful."""


def parse_tool_calls(text: str) -> list[dict]:
    """Extract tool calls from response text."""
    pattern = r"<tool>(.*?)</tool>"
    matches = re.findall(pattern, text, re.DOTALL)
    tools = []
    for m in matches:
        try:
            parsed = json.loads(m.strip())
            if "name" in parsed:
                tools.append(parsed)
        except json.JSONDecodeError:
            pass
    return tools


class Agent:
    """ELO Agent with prompt-based tool calling."""

    def __init__(self, model: str = "mistral", ollama_url: str = "http://localhost:11434"):
        self.model = model
        self.ollama_url = ollama_url
        self.messages = []
        self.permission_callback = None

    def set_permission_callback(self, callback):
        """Set callback for permission requests: callback(tool_name, params) -> bool"""
        self.permission_callback = callback

    def _call_ollama(self, messages: list) -> str:
        """Make a request to Ollama API and return the response content."""
        data = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }

        req = urllib.request.Request(
            f"{self.ollama_url}/api/chat",
            data=json.dumps(data).encode(),
            headers={"Content-Type": "application/json"},
        )

        with urllib.request.urlopen(req, timeout=120) as r:
            response = json.loads(r.read().decode())
            return response.get("message", {}).get("content", "")

    def _call_ollama_stream(self, messages: list) -> Generator[str, None, None]:
        """Stream a request to Ollama API, yielding content chunks."""
        data = {
            "model": self.model,
            "messages": messages,
            "stream": True,
        }

        req = urllib.request.Request(
            f"{self.ollama_url}/api/chat",
            data=json.dumps(data).encode(),
            headers={"Content-Type": "application/json"},
        )

        with urllib.request.urlopen(req, timeout=120) as r:
            for line in r:
                try:
                    chunk = json.loads(line.decode())
                    content = chunk.get("message", {}).get("content", "")
                    if content:
                        yield content
                    if chunk.get("done"):
                        break
                except Exception:
                    continue

    def _execute_tool(self, name: str, args: dict) -> ToolResult:
        """Execute a tool by name with given arguments."""
        tool = registry.get(name)
        if not tool:
            return ToolResult(False, "", f"Unknown tool: {name}")

        # Check permissions
        if tool.requires_permission and self.permission_callback:
            bash_tool = registry.get("bash")
            if name == "bash" and bash_tool and bash_tool.is_safe(args.get("command", "")):
                pass  # Safe command, no permission needed
            else:
                allowed = self.permission_callback(name, args)
                if not allowed:
                    return ToolResult(False, "", "Permission denied by user")

        return tool.execute(**args)

    def run(self, prompt: str) -> Generator[str, None, None]:
        """Run the agent with a prompt, yielding response chunks."""
        # Add system prompt if first message
        if not self.messages:
            self.messages.append({"role": "system", "content": SYSTEM_PROMPT})

        # Add user message
        self.messages.append({"role": "user", "content": prompt})

        max_iterations = 10

        for iteration in range(max_iterations):
            # Get response from LLM
            full_response = ""
            for chunk in self._call_ollama_stream(self.messages):
                full_response += chunk
                yield chunk

            # Add assistant response to history
            self.messages.append({"role": "assistant", "content": full_response})

            # Parse tool calls from response
            tool_calls = parse_tool_calls(full_response)

            if not tool_calls:
                # No tools requested, we're done
                break

            # Execute each tool
            tool_results = []
            for tool_call in tool_calls:
                name = tool_call.get("name", "")
                args = tool_call.get("args", {})

                yield f"\n\n🔧 **Executing: {name}**\n"

                result = self._execute_tool(name, args)
                tool_results.append((name, result))

                if result.success:
                    preview = result.output[:800]
                    if len(result.output) > 800:
                        preview += "\n... (truncated)"
                    yield f"```\n{preview}\n```\n"
                else:
                    yield f"❌ Error: {result.error}\n"

            # Build tool results message for LLM
            results_content = "Tool results:\n\n"
            for name, result in tool_results:
                if result.success:
                    results_content += f"[{name}] Success:\n{result.output[:2000]}\n\n"
                else:
                    results_content += f"[{name}] Error: {result.error}\n\n"

            # Add tool results as user message (so LLM continues)
            self.messages.append({"role": "user", "content": results_content})

            yield "\n"

    def run_simple(self, prompt: str) -> str:
        """Run without streaming, return full response."""
        if not self.messages:
            self.messages.append({"role": "system", "content": SYSTEM_PROMPT})

        self.messages.append({"role": "user", "content": prompt})

        response = self._call_ollama(self.messages)
        self.messages.append({"role": "assistant", "content": response})

        return response

    def clear(self):
        """Clear conversation history."""
        self.messages = []
