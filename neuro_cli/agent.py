"""NEURO Agent - Orchestrates tool calling with Ollama."""

import json
import urllib.request
from typing import Generator, Optional
from .tools import registry, Tool, ToolResult
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


SYSTEM_PROMPT = """You are NEURO, an AI coding assistant running locally. You help developers by reading, writing, and modifying code.

You have access to these tools:
- read_file: Read file contents
- write_file: Create or overwrite files
- edit_file: Replace specific text in files
- glob: Find files by pattern
- grep: Search file contents
- ls: List directory contents
- bash: Execute shell commands

When asked to do something with files or code:
1. First use glob/ls to understand the project structure
2. Use read_file to examine relevant files
3. Use edit_file for modifications or write_file for new files
4. Use bash for running tests, builds, or git commands

Always explain what you're doing. Be concise but helpful."""


class Agent:
    """NEURO Agent with tool calling capabilities."""

    def __init__(self, model: str = "mistral", ollama_url: str = "http://localhost:11434"):
        self.model = model
        self.ollama_url = ollama_url
        self.messages = []
        self.permission_callback = None  # Set externally for permission prompts

    def set_permission_callback(self, callback):
        """Set callback for permission requests: callback(tool_name, params) -> bool"""
        self.permission_callback = callback

    def _call_ollama(self, messages: list, tools: list = None) -> dict:
        """Make a request to Ollama API."""
        data = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }

        if tools:
            data["tools"] = tools

        req = urllib.request.Request(
            f"{self.ollama_url}/api/chat",
            data=json.dumps(data).encode(),
            headers={"Content-Type": "application/json"}
        )

        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read().decode())

    def _call_ollama_stream(self, messages: list, tools: list = None) -> Generator[dict, None, None]:
        """Stream a request to Ollama API."""
        data = {
            "model": self.model,
            "messages": messages,
            "stream": True,
        }

        if tools:
            data["tools"] = tools

        req = urllib.request.Request(
            f"{self.ollama_url}/api/chat",
            data=json.dumps(data).encode(),
            headers={"Content-Type": "application/json"}
        )

        with urllib.request.urlopen(req, timeout=120) as r:
            for line in r:
                try:
                    chunk = json.loads(line.decode())
                    yield chunk
                    if chunk.get("done"):
                        break
                except:
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
        # Add system prompt if this is the first message
        if not self.messages:
            self.messages.append({
                "role": "system",
                "content": SYSTEM_PROMPT
            })

        # Add user message
        self.messages.append({
            "role": "user",
            "content": prompt
        })

        tools = registry.get_ollama_tools()
        max_iterations = 10

        for _ in range(max_iterations):
            # Call Ollama with tools
            response = self._call_ollama(self.messages, tools)

            message = response.get("message", {})
            content = message.get("content", "")
            tool_calls = message.get("tool_calls", [])

            # Add assistant message
            self.messages.append(message)

            # If there's content, yield it
            if content:
                yield content

            # If no tool calls, we're done
            if not tool_calls:
                break

            # Execute tool calls
            for tool_call in tool_calls:
                func = tool_call.get("function", {})
                name = func.get("name", "")
                args = func.get("arguments", {})

                # Parse arguments if they're a string
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except:
                        args = {}

                yield f"\n\n🔧 Using tool: **{name}**"
                if args:
                    yield f"\n```json\n{json.dumps(args, indent=2)}\n```\n"

                # Execute the tool
                result = self._execute_tool(name, args)

                # Add tool result to messages
                self.messages.append({
                    "role": "tool",
                    "content": str(result)
                })

                # Yield result preview
                if result.success:
                    preview = result.output[:500]
                    if len(result.output) > 500:
                        preview += "..."
                    yield f"\n✅ Result:\n```\n{preview}\n```\n"
                else:
                    yield f"\n❌ Error: {result.error}\n"

    def run_stream(self, prompt: str) -> Generator[str, None, None]:
        """Run the agent with streaming response."""
        # Add system prompt if this is the first message
        if not self.messages:
            self.messages.append({
                "role": "system",
                "content": SYSTEM_PROMPT
            })

        # Add user message
        self.messages.append({
            "role": "user",
            "content": prompt
        })

        tools = registry.get_ollama_tools()
        max_iterations = 10

        for _ in range(max_iterations):
            full_content = ""
            tool_calls = []

            # Stream the response
            for chunk in self._call_ollama_stream(self.messages, tools):
                message = chunk.get("message", {})
                content = message.get("content", "")
                if content:
                    full_content += content
                    yield content

                # Collect tool calls
                if "tool_calls" in message:
                    tool_calls.extend(message["tool_calls"])

            # Build assistant message
            assistant_msg = {"role": "assistant", "content": full_content}
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            self.messages.append(assistant_msg)

            # If no tool calls, we're done
            if not tool_calls:
                break

            # Execute tool calls
            for tool_call in tool_calls:
                func = tool_call.get("function", {})
                name = func.get("name", "")
                args = func.get("arguments", {})

                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except:
                        args = {}

                yield f"\n\n🔧 Using tool: **{name}**\n"

                result = self._execute_tool(name, args)

                self.messages.append({
                    "role": "tool",
                    "content": str(result)
                })

                if result.success:
                    preview = result.output[:500]
                    if len(result.output) > 500:
                        preview += "..."
                    yield f"✅ Result:\n```\n{preview}\n```\n"
                else:
                    yield f"❌ Error: {result.error}\n"

    def clear(self):
        """Clear conversation history."""
        self.messages = []
