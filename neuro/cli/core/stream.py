"""
Stream Handler - Token-by-token streaming from LLM.
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

    Features:
    - Token-by-token streaming via SSE
    - Tool use detection mid-stream
    - Graceful error handling
    - Callback support
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "ministral-3:8b",
        timeout: int = 120,
        on_token: Optional[Callable[[str], None]] = None,
        on_tool_start: Optional[Callable[[str], None]] = None,
        on_tool_end: Optional[Callable[[str, Any], None]] = None,
    ):
        self.base_url = base_url
        self.model = model
        self.timeout = timeout

        # Callbacks
        self.on_token = on_token
        self.on_tool_start = on_tool_start
        self.on_tool_end = on_tool_end

        self._session: Optional[Any] = None

    async def stream(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream response token-by-token."""
        if aiohttp is None:
            yield StreamEvent(
                type=StreamEventType.ERROR,
                content="aiohttp not installed. Run: pip install aiohttp"
            )
            return

        session = await self._get_session()

        # Build payload
        all_messages = []
        if system_prompt:
            all_messages.append({"role": "system", "content": system_prompt})
        all_messages.extend(messages)

        payload = {
            "model": self.model,
            "messages": all_messages,
            "stream": True,
            "options": {"temperature": 0.7}
        }

        try:
            async with session.post(
                f"{self.base_url}/api/chat",
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    yield StreamEvent(
                        type=StreamEventType.ERROR,
                        content=f"API error {response.status}: {error_text}"
                    )
                    return

                buffer = ""
                tool_parsing = False

                async for line in response.content:
                    if not line:
                        continue

                    try:
                        data = json.loads(line.decode('utf-8'))
                    except json.JSONDecodeError:
                        continue

                    content = data.get("message", {}).get("content", "")
                    done = data.get("done", False)

                    if content:
                        buffer += content

                        # Detect tool use start
                        if "<tool>" in buffer and not tool_parsing:
                            tool_parsing = True
                            if self.on_tool_start:
                                self.on_tool_start(buffer)
                            yield StreamEvent(
                                type=StreamEventType.TOOL_USE_START,
                                content=buffer
                            )

                        # Detect tool use end
                        elif "</args>" in buffer and tool_parsing:
                            tool_parsing = False
                            yield StreamEvent(
                                type=StreamEventType.TOOL_USE_END,
                                content=buffer
                            )
                            buffer = ""

                        # Normal token
                        elif not tool_parsing:
                            if self.on_token:
                                self.on_token(content)
                            yield StreamEvent(
                                type=StreamEventType.TOKEN,
                                content=content
                            )

                    if done:
                        yield StreamEvent(
                            type=StreamEventType.DONE,
                            metadata={
                                "total_duration": data.get("total_duration"),
                                "eval_count": data.get("eval_count"),
                                "model": data.get("model"),
                            }
                        )
                        break

        except asyncio.TimeoutError:
            yield StreamEvent(
                type=StreamEventType.ERROR,
                content="Request timed out"
            )
        except Exception as e:
            yield StreamEvent(
                type=StreamEventType.ERROR,
                content=str(e)
            )

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
