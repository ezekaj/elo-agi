"""
Stream Handler - Token-by-token streaming from LLM.

Supports both Ollama (local) and OpenAI-compatible APIs (cloud).
"""

from dataclasses import dataclass, field
from typing import AsyncGenerator, Callable, Optional, Dict, Any, List
from enum import Enum
import asyncio
import json

try:
    import aiohttp
except ImportError:
    aiohttp = None


class StreamEventType(Enum):
    """Types of stream events."""

    TOKEN = "token"
    TOOL_USE_START = "tool_use_start"
    TOOL_USE_END = "tool_use_end"
    THINKING = "thinking"
    ERROR = "error"
    DONE = "done"


@dataclass
class StreamEvent:
    """A single event in the response stream."""

    type: StreamEventType
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class StreamHandler:
    """
    Handles streaming responses from LLM with real-time token output.

    Supports:
    - Ollama API (local, /api/chat)
    - OpenAI-compatible APIs (cloud, /chat/completions)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "ministral-3:8b",
        api_key: Optional[str] = None,
        api_type: str = "ollama",
        timeout: int = 120,
        on_token: Optional[Callable[[str], None]] = None,
        on_tool_start: Optional[Callable[[str], None]] = None,
        on_tool_end: Optional[Callable[[str, Any], None]] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.api_type = api_type  # "ollama" or "openai"
        self.timeout = timeout

        # Callbacks
        self.on_token = on_token
        self.on_tool_start = on_tool_start
        self.on_tool_end = on_tool_end

        self._session: Optional[Any] = None

        # Native function calling tools
        self.tools = None

    def set_tools(self, tools: List[Dict[str, Any]]):
        """Set tools for native function calling."""
        self.tools = tools

    async def stream(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        ultrathink: bool = False,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream response token-by-token."""
        if aiohttp is None:
            yield StreamEvent(
                type=StreamEventType.ERROR,
                content="aiohttp not installed. Run: pip install aiohttp",
            )
            return

        if self.api_type == "openai":
            async for event in self._stream_openai(messages, system_prompt, ultrathink):
                yield event
        else:
            async for event in self._stream_ollama(messages, system_prompt, ultrathink):
                yield event

    async def _stream_openai(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        ultrathink: bool = False,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream from OpenAI-compatible API."""
        session = await self._get_session()

        all_messages = []
        if system_prompt:
            all_messages.append({"role": "system", "content": system_prompt})
        all_messages.extend(messages)

        payload = {
            "model": self.model,
            "messages": all_messages,
            "stream": True,
        }

        if ultrathink:
            payload["temperature"] = 0.4
            payload["max_tokens"] = 8192
        else:
            payload["temperature"] = 0.7
            payload["max_tokens"] = 2048

        if self.tools:
            payload["tools"] = self.tools

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        url = f"{self.base_url}/chat/completions"

        try:
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    yield StreamEvent(
                        type=StreamEventType.ERROR,
                        content=f"API error {response.status}: {error_text}",
                    )
                    return

                buffer = ""
                tool_parsing = False

                async for line in response.content:
                    line_str = line.decode("utf-8").strip()

                    if not line_str or line_str == "data: [DONE]":
                        if line_str == "data: [DONE]":
                            yield StreamEvent(type=StreamEventType.DONE, metadata={})
                            break
                        continue

                    # Strip "data: " prefix (SSE format)
                    if line_str.startswith("data: "):
                        line_str = line_str[6:]

                    try:
                        data = json.loads(line_str)
                    except json.JSONDecodeError:
                        continue

                    choices = data.get("choices", [])
                    if not choices:
                        continue

                    delta = choices[0].get("delta", {})
                    finish_reason = choices[0].get("finish_reason")

                    # Handle tool calls
                    tool_calls = delta.get("tool_calls", [])
                    if tool_calls:
                        for tc in tool_calls:
                            func = tc.get("function", {})
                            name = func.get("name", "")
                            args_str = func.get("arguments", "")
                            if name:
                                try:
                                    args = json.loads(args_str) if args_str else {}
                                except json.JSONDecodeError:
                                    args = {}
                                yield StreamEvent(
                                    type=StreamEventType.TOOL_USE_START,
                                    content=name,
                                    metadata={
                                        "name": name,
                                        "arguments": args,
                                        "native": True,
                                    },
                                )

                    # Handle content
                    content = delta.get("content", "")
                    if content:
                        buffer += content

                        # Detect XML-style tool use
                        if "<tool>" in buffer and not tool_parsing:
                            tool_parsing = True
                            yield StreamEvent(type=StreamEventType.TOOL_USE_START, content=buffer)
                        elif "</args>" in buffer and tool_parsing:
                            tool_parsing = False
                            yield StreamEvent(type=StreamEventType.TOOL_USE_END, content=buffer)
                            buffer = ""
                        elif not tool_parsing:
                            if self.on_token:
                                self.on_token(content)
                            yield StreamEvent(type=StreamEventType.TOKEN, content=content)

                    if finish_reason == "stop":
                        yield StreamEvent(type=StreamEventType.DONE, metadata={})
                        break

        except asyncio.TimeoutError:
            yield StreamEvent(type=StreamEventType.ERROR, content="Request timed out")
        except Exception as e:
            yield StreamEvent(type=StreamEventType.ERROR, content=str(e))

    async def _stream_ollama(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        ultrathink: bool = False,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream from Ollama API."""
        session = await self._get_session()

        # Build payload
        all_messages = []

        # Enhanced system prompt for ultrathink mode
        if ultrathink:
            ultrathink_prefix = """You are in ULTRATHINK mode. This means:
1. Think step-by-step through EVERY aspect of the problem
2. Consider multiple approaches before settling on one
3. Show your reasoning process explicitly
4. Break complex problems into smaller parts
5. Question your assumptions
6. Consider edge cases and potential issues
7. Provide comprehensive, detailed responses
8. Take your time - depth over brevity

Begin your response with <thinking> to show your reasoning process, then provide your final answer.

"""
            if system_prompt:
                all_messages.append(
                    {"role": "system", "content": ultrathink_prefix + system_prompt}
                )
            else:
                all_messages.append({"role": "system", "content": ultrathink_prefix})
        elif system_prompt:
            all_messages.append({"role": "system", "content": system_prompt})

        all_messages.extend(messages)

        # Configure options based on mode
        if ultrathink:
            options = {
                "temperature": 0.4,
                "num_ctx": 32768,
                "num_predict": 8192,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
            }
        else:
            options = {"temperature": 0.7}

        payload = {
            "model": self.model,
            "messages": all_messages,
            "stream": True,
            "options": options,
        }

        # Add native function calling tools if available
        if hasattr(self, "tools") and self.tools:
            payload["tools"] = self.tools

        try:
            async with session.post(f"{self.base_url}/api/chat", json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    yield StreamEvent(
                        type=StreamEventType.ERROR,
                        content=f"API error {response.status}: {error_text}",
                    )
                    return

                buffer = ""
                tool_parsing = False

                async for line in response.content:
                    if not line:
                        continue

                    try:
                        data = json.loads(line.decode("utf-8"))
                    except json.JSONDecodeError:
                        continue

                    message = data.get("message", {})
                    content = message.get("content", "")
                    tool_calls = message.get("tool_calls", [])
                    done = data.get("done", False)

                    # Handle native function calls (Ollama API)
                    if tool_calls:
                        for tool_call in tool_calls:
                            func = tool_call.get("function", {})
                            tool_name = func.get("name", "")
                            tool_args = func.get("arguments", {})

                            yield StreamEvent(
                                type=StreamEventType.TOOL_USE_START,
                                content=tool_name,
                                metadata={
                                    "name": tool_name,
                                    "arguments": tool_args,
                                    "native": True,
                                },
                            )

                    if content:
                        buffer += content

                        # Detect tool use start
                        if "<tool>" in buffer and not tool_parsing:
                            tool_parsing = True
                            if self.on_tool_start:
                                self.on_tool_start(buffer)
                            yield StreamEvent(type=StreamEventType.TOOL_USE_START, content=buffer)

                        # Detect tool use end
                        elif "</args>" in buffer and tool_parsing:
                            tool_parsing = False
                            yield StreamEvent(type=StreamEventType.TOOL_USE_END, content=buffer)
                            buffer = ""

                        # Normal token
                        elif not tool_parsing:
                            if self.on_token:
                                self.on_token(content)
                            yield StreamEvent(type=StreamEventType.TOKEN, content=content)

                    if done:
                        yield StreamEvent(
                            type=StreamEventType.DONE,
                            metadata={
                                "total_duration": data.get("total_duration"),
                                "eval_count": data.get("eval_count"),
                                "model": data.get("model"),
                            },
                        )
                        break

        except asyncio.TimeoutError:
            yield StreamEvent(type=StreamEventType.ERROR, content="Request timed out")
        except Exception as e:
            yield StreamEvent(type=StreamEventType.ERROR, content=str(e))

    async def _get_session(self):
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self):
        """Close the session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None


# Sync wrapper for non-async contexts
def stream_sync(
    messages: List[Dict[str, str]],
    model: str = "ministral-3:8b",
    system_prompt: Optional[str] = None,
) -> str:
    """Synchronous streaming (collects all tokens)."""
    handler = StreamHandler(model=model)

    async def _collect():
        tokens = []
        async for event in handler.stream(messages, system_prompt):
            if event.type == StreamEventType.TOKEN:
                tokens.append(event.content)
        await handler.close()
        return "".join(tokens)

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_collect())
    finally:
        loop.close()
