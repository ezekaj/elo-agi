"""
NEURO Streaming Engine - Real-time LLM Response Streaming

Provides async streaming of tokens from Ollama for instant feedback.
Uses aiohttp for efficient async HTTP.
"""

import asyncio
import json
from typing import AsyncGenerator, Optional, List, Dict, Any, Callable
from dataclasses import dataclass

try:
    import aiohttp

    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False


@dataclass
class StreamConfig:
    """Configuration for streaming."""

    base_url: str = "http://localhost:11434"
    model: str = "ministral-3:8b"
    timeout: int = 120
    temperature: float = 0.7
    max_tokens: Optional[int] = None


@dataclass
class StreamChunk:
    """A single chunk from the stream."""

    content: str
    done: bool = False
    model: str = ""
    total_duration: Optional[int] = None
    eval_count: Optional[int] = None


class StreamHandler:
    """
    Handles streaming responses from Ollama.

    Features:
    - Real-time token streaming
    - Progress callbacks
    - Graceful error handling
    - Sync fallback if aiohttp unavailable
    """

    def __init__(self, config: Optional[StreamConfig] = None):
        self.config = config or StreamConfig()
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self):
        """Close the session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def stream(
        self,
        messages: List[Dict[str, str]],
        on_token: Optional[Callable[[str], None]] = None,
        on_done: Optional[Callable[[Dict], None]] = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        """
        Stream tokens from Ollama.

        Args:
            messages: Chat messages in OpenAI format
            on_token: Callback for each token (for side effects like printing)
            on_done: Callback when stream completes

        Yields:
            StreamChunk objects with content
        """
        if not AIOHTTP_AVAILABLE:
            # Fallback to sync if aiohttp not installed
            async for chunk in self._sync_fallback(messages):
                if on_token and chunk.content:
                    on_token(chunk.content)
                yield chunk
            return

        session = await self._get_session()
        url = f"{self.config.base_url}/api/chat"

        payload = {
            "model": self.config.model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": self.config.temperature,
            },
        }

        if self.config.max_tokens:
            payload["options"]["num_predict"] = self.config.max_tokens

        try:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    yield StreamChunk(content=f"Error: {response.status} - {error_text}", done=True)
                    return

                full_response = ""
                async for line in response.content:
                    if not line:
                        continue

                    try:
                        data = json.loads(line.decode("utf-8"))
                    except json.JSONDecodeError:
                        continue

                    # Extract content
                    content = data.get("message", {}).get("content", "")
                    done = data.get("done", False)

                    if content:
                        full_response += content
                        if on_token:
                            on_token(content)

                    chunk = StreamChunk(
                        content=content,
                        done=done,
                        model=data.get("model", ""),
                        total_duration=data.get("total_duration"),
                        eval_count=data.get("eval_count"),
                    )

                    yield chunk

                    if done:
                        if on_done:
                            on_done(
                                {
                                    "full_response": full_response,
                                    "model": chunk.model,
                                    "total_duration": chunk.total_duration,
                                    "eval_count": chunk.eval_count,
                                }
                            )
                        break

        except aiohttp.ClientError as e:
            yield StreamChunk(content=f"Connection error: {e}", done=True)
        except asyncio.TimeoutError:
            yield StreamChunk(content="Request timed out", done=True)

    async def _sync_fallback(
        self, messages: List[Dict[str, str]]
    ) -> AsyncGenerator[StreamChunk, None]:
        """Sync fallback using requests (yields chunks from full response)."""
        import requests

        url = f"{self.config.base_url}/api/chat"
        payload = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
        }

        try:
            response = requests.post(url, json=payload, timeout=self.config.timeout)
            if response.status_code == 200:
                data = response.json()
                content = data.get("message", {}).get("content", "")
                # Yield content in chunks to simulate streaming
                words = content.split()
                for i, word in enumerate(words):
                    yield StreamChunk(
                        content=word + (" " if i < len(words) - 1 else ""),
                        done=(i == len(words) - 1),
                        model=data.get("model", ""),
                    )
            else:
                yield StreamChunk(content=f"Error: {response.status_code}", done=True)
        except Exception as e:
            yield StreamChunk(content=f"Error: {e}", done=True)

    async def stream_with_context(
        self,
        query: str,
        system_prompt: str,
        context: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        """
        Stream with full context building.

        Args:
            query: User's query
            system_prompt: System instructions
            context: Additional context (from cognitive pipeline)
            history: Conversation history
        """
        messages = []

        # Build system message
        full_system = system_prompt
        if context:
            full_system += f"\n\n[Context]\n{context}"

        messages.append({"role": "system", "content": full_system})

        # Add history
        if history:
            messages.extend(history)

        # Add current query
        messages.append({"role": "user", "content": query})

        async for chunk in self.stream(messages):
            yield chunk


class TerminalStreamer:
    """
    Pretty terminal output for streaming.

    Handles:
    - Real-time character printing
    - Progress indicators
    - Color support
    """

    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    DIM = "\033[2m"
    RESET = "\033[0m"
    BOLD = "\033[1m"

    def __init__(self, show_thinking: bool = True):
        self.show_thinking = show_thinking
        self.token_count = 0
        self.start_time = None

    def on_start(self, phase: str = "Thinking"):
        """Called when streaming starts."""
        import time

        self.start_time = time.time()
        self.token_count = 0
        if self.show_thinking:
            print(f"\n  {self.CYAN}{self.BOLD}Neuro:{self.RESET} ", end="", flush=True)

    def on_token(self, token: str):
        """Called for each token."""
        self.token_count += 1
        print(token, end="", flush=True)

    def on_done(self, stats: Dict[str, Any]):
        """Called when streaming completes."""
        import time

        duration = time.time() - self.start_time if self.start_time else 0
        print()  # New line after response
        if self.show_thinking:
            tokens_per_sec = self.token_count / max(0.1, duration)
            print(
                f"  {self.DIM}[{self.token_count} tokens, {tokens_per_sec:.1f} tok/s]{self.RESET}"
            )

    async def stream_to_terminal(
        self,
        handler: StreamHandler,
        messages: List[Dict[str, str]],
    ) -> str:
        """Stream response directly to terminal."""
        self.on_start()
        full_response = ""

        async for chunk in handler.stream(messages, on_token=self.on_token):
            full_response += chunk.content
            if chunk.done:
                self.on_done({"eval_count": chunk.eval_count})

        return full_response


# Convenience functions for simple usage


async def stream_chat(
    query: str,
    system_prompt: str = "You are a helpful AI assistant.",
    model: str = "ministral-3:8b",
    print_tokens: bool = True,
) -> str:
    """
    Simple streaming chat function.

    Args:
        query: User's query
        system_prompt: System instructions
        model: Ollama model to use
        print_tokens: Whether to print tokens as they arrive

    Returns:
        Full response text
    """
    config = StreamConfig(model=model)
    handler = StreamHandler(config)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]

    full_response = ""

    def on_token(token: str):
        nonlocal full_response
        full_response += token
        if print_tokens:
            print(token, end="", flush=True)

    async for chunk in handler.stream(messages, on_token=on_token):
        if chunk.done:
            break

    await handler.close()

    if print_tokens:
        print()  # New line

    return full_response


def stream_chat_sync(
    query: str,
    system_prompt: str = "You are a helpful AI assistant.",
    model: str = "ministral-3:8b",
    print_tokens: bool = True,
) -> str:
    """Synchronous wrapper for stream_chat."""
    return asyncio.run(stream_chat(query, system_prompt, model, print_tokens))


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("NEURO STREAMING ENGINE TEST")
    print("=" * 60)

    async def test():
        # Test basic streaming
        print("\nTest 1: Basic streaming")
        print("-" * 40)

        response = await stream_chat("What is 2 + 2? Answer in one sentence.", print_tokens=True)

        print(f"\nFull response: {response}")

        # Test with context
        print("\n" + "=" * 60)
        print("Test 2: Streaming with cognitive context")
        print("-" * 40)

        config = StreamConfig()
        handler = StreamHandler(config)
        streamer = TerminalStreamer(show_thinking=True)

        messages = [
            {
                "role": "system",
                "content": "You are NEURO, a neuroscience-inspired AI. Be helpful and direct.",
            },
            {"role": "user", "content": "Explain recursion briefly."},
        ]

        full = await streamer.stream_to_terminal(handler, messages)
        await handler.close()

        print(f"\n\nTotal length: {len(full)} characters")

    asyncio.run(test())
