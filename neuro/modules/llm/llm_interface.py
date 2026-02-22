"""
LLM Interface: API wrapper for Large Language Models.

Provides a unified interface to query LLMs as frozen semantic
oracles. The cognitive system uses LLMs for:
- Natural language understanding
- Semantic embedding
- Knowledge retrieval
- Response generation

No fine-tuning is required - the LLM is used as-is.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from abc import ABC, abstractmethod
import numpy as np
import time
import hashlib
import json

# Try to import API clients
try:
    import anthropic

    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False
    anthropic = None

try:
    import openai

    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    openai = None


@dataclass
class LLMConfig:
    """Configuration for LLM interface."""

    provider: str = "mock"  # "anthropic", "openai", "mock"
    model: str = "claude-3-haiku-20240307"
    api_key: Optional[str] = None
    max_tokens: int = 1024
    temperature: float = 0.7
    embedding_dim: int = 64
    cache_responses: bool = True
    cache_ttl: float = 3600.0  # 1 hour


@dataclass
class LLMResponse:
    """Response from an LLM query."""

    text: str
    tokens_used: int = 0
    latency: float = 0.0
    model: str = ""
    cached: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class LLMOracle(ABC):
    """
    Abstract base class for LLM oracles.

    The oracle provides:
    1. query() - Get text responses to prompts
    2. embed() - Get semantic embeddings of text
    3. parse_action() - Extract structured actions from text
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self._cache: Dict[str, tuple] = {}  # prompt_hash -> (response, timestamp)
        self._query_count = 0
        self._cache_hits = 0

    @abstractmethod
    def query(self, prompt: str, system: Optional[str] = None) -> LLMResponse:
        """
        Query the LLM with a prompt.

        Args:
            prompt: The user prompt
            system: Optional system message

        Returns:
            LLMResponse with text and metadata
        """
        pass

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """
        Get semantic embedding of text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector of shape (embedding_dim,)
        """
        pass

    def parse_action(self, response: str) -> Dict[str, Any]:
        """
        Parse LLM response into structured action.

        Args:
            response: Text response from LLM

        Returns:
            Dictionary with action type and parameters
        """
        # Default implementation - look for common patterns
        response_lower = response.lower()

        action = {
            "type": "unknown",
            "confidence": 0.5,
            "raw_text": response,
        }

        # Detect action types
        if any(w in response_lower for w in ["move", "go", "walk", "travel"]):
            action["type"] = "move"
            # Extract direction
            for direction in ["north", "south", "east", "west", "up", "down"]:
                if direction in response_lower:
                    action["direction"] = direction
                    break

        elif any(w in response_lower for w in ["take", "grab", "pick", "get"]):
            action["type"] = "take"

        elif any(w in response_lower for w in ["say", "tell", "speak", "respond"]):
            action["type"] = "speak"
            # Extract quoted text if present
            if '"' in response:
                parts = response.split('"')
                if len(parts) >= 2:
                    action["message"] = parts[1]

        elif any(w in response_lower for w in ["wait", "stay", "pause"]):
            action["type"] = "wait"

        elif any(w in response_lower for w in ["look", "observe", "examine"]):
            action["type"] = "observe"

        return action

    def _get_cache_key(self, prompt: str, system: Optional[str] = None) -> str:
        """Generate cache key from prompt."""
        content = f"{system or ''}:{prompt}"
        return hashlib.md5(content.encode()).hexdigest()

    def _check_cache(self, key: str) -> Optional[LLMResponse]:
        """Check if response is cached and valid."""
        if not self.config.cache_responses:
            return None

        if key in self._cache:
            response, timestamp = self._cache[key]
            if time.time() - timestamp < self.config.cache_ttl:
                self._cache_hits += 1
                cached_response = LLMResponse(
                    text=response.text,
                    tokens_used=response.tokens_used,
                    latency=0.0,
                    model=response.model,
                    cached=True,
                    metadata=response.metadata,
                )
                return cached_response

        return None

    def _store_cache(self, key: str, response: LLMResponse) -> None:
        """Store response in cache."""
        if self.config.cache_responses:
            self._cache[key] = (response, time.time())

    def get_statistics(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "query_count": self._query_count,
            "cache_hits": self._cache_hits,
            "cache_size": len(self._cache),
            "cache_hit_rate": self._cache_hits / max(1, self._query_count),
            "provider": self.config.provider,
            "model": self.config.model,
        }

    def clear_cache(self) -> None:
        """Clear the response cache."""
        self._cache = {}


class MockLLM(LLMOracle):
    """
    Mock LLM for testing without API calls.

    Generates deterministic responses based on input.
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        config = config or LLMConfig(provider="mock")
        super().__init__(config)
        self._responses = self._build_responses()

    def _build_responses(self) -> Dict[str, str]:
        """Build canned responses for common prompts."""
        return {
            "greeting": "Hello! I'm here to help. What would you like to discuss?",
            "question": "That's an interesting question. Let me think about it...",
            "action": "I would suggest taking a careful approach to this situation.",
            "description": "I can see a complex environment with multiple elements.",
            "default": "I understand. Please tell me more about what you need.",
        }

    def query(self, prompt: str, system: Optional[str] = None) -> LLMResponse:
        self._query_count += 1
        start_time = time.time()

        # Check cache
        cache_key = self._get_cache_key(prompt, system)
        cached = self._check_cache(cache_key)
        if cached:
            return cached

        # Generate mock response
        prompt_lower = prompt.lower()

        if any(w in prompt_lower for w in ["hello", "hi", "greet"]):
            text = self._responses["greeting"]
        elif "?" in prompt:
            text = self._responses["question"]
        elif any(w in prompt_lower for w in ["do", "action", "move", "take"]):
            text = self._responses["action"]
        elif any(w in prompt_lower for w in ["describe", "what", "see"]):
            text = self._responses["description"]
        else:
            text = self._responses["default"]

        # Add some variation based on prompt hash
        prompt_hash = hash(prompt) % 100
        if prompt_hash > 50:
            text += " Would you like me to elaborate?"

        response = LLMResponse(
            text=text,
            tokens_used=len(text.split()),
            latency=time.time() - start_time,
            model="mock-llm",
            cached=False,
        )

        self._store_cache(cache_key, response)
        return response

    def embed(self, text: str) -> np.ndarray:
        """Generate deterministic embedding from text."""
        # Use character-level hashing for deterministic embeddings
        embedding = np.zeros(self.config.embedding_dim, dtype=np.float32)

        for i, char in enumerate(text):
            idx = (ord(char) + i) % self.config.embedding_dim
            embedding[idx] += ord(char) / 256.0

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding


class AnthropicLLM(LLMOracle):
    """
    LLM Oracle using Anthropic's Claude API.
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        if not HAS_ANTHROPIC:
            raise ImportError("anthropic package required. Install with: pip install anthropic")

        config = config or LLMConfig(provider="anthropic")
        super().__init__(config)

        self._client = anthropic.Anthropic(api_key=config.api_key)

    def query(self, prompt: str, system: Optional[str] = None) -> LLMResponse:
        self._query_count += 1
        start_time = time.time()

        # Check cache
        cache_key = self._get_cache_key(prompt, system)
        cached = self._check_cache(cache_key)
        if cached:
            return cached

        # Make API call
        messages = [{"role": "user", "content": prompt}]

        kwargs = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system

        try:
            result = self._client.messages.create(**kwargs)
            text = result.content[0].text
            tokens = result.usage.input_tokens + result.usage.output_tokens
        except Exception as e:
            text = f"Error: {str(e)}"
            tokens = 0

        response = LLMResponse(
            text=text,
            tokens_used=tokens,
            latency=time.time() - start_time,
            model=self.config.model,
            cached=False,
        )

        self._store_cache(cache_key, response)
        return response

    def embed(self, text: str) -> np.ndarray:
        """
        Generate embedding using Claude.

        Note: Claude doesn't have a native embedding API,
        so we use a prompt-based approach.
        """
        # For now, fall back to hash-based embedding
        # In production, could use Claude to describe semantic features
        embedding = np.zeros(self.config.embedding_dim, dtype=np.float32)

        for i, char in enumerate(text):
            idx = (ord(char) + i) % self.config.embedding_dim
            embedding[idx] += ord(char) / 256.0

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding


class OpenAILLM(LLMOracle):
    """
    LLM Oracle using OpenAI's API.
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        if not HAS_OPENAI:
            raise ImportError("openai package required. Install with: pip install openai")

        config = config or LLMConfig(provider="openai", model="gpt-4o-mini")
        super().__init__(config)

        self._client = openai.OpenAI(api_key=config.api_key)

    def query(self, prompt: str, system: Optional[str] = None) -> LLMResponse:
        self._query_count += 1
        start_time = time.time()

        # Check cache
        cache_key = self._get_cache_key(prompt, system)
        cached = self._check_cache(cache_key)
        if cached:
            return cached

        # Build messages
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            result = self._client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )
            text = result.choices[0].message.content
            tokens = result.usage.total_tokens
        except Exception as e:
            text = f"Error: {str(e)}"
            tokens = 0

        response = LLMResponse(
            text=text,
            tokens_used=tokens,
            latency=time.time() - start_time,
            model=self.config.model,
            cached=False,
        )

        self._store_cache(cache_key, response)
        return response

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding using OpenAI's embedding API."""
        try:
            result = self._client.embeddings.create(
                model="text-embedding-3-small",
                input=text,
            )
            full_embedding = np.array(result.data[0].embedding, dtype=np.float32)

            # Resize to configured dimension
            if len(full_embedding) > self.config.embedding_dim:
                embedding = full_embedding[: self.config.embedding_dim]
            else:
                embedding = np.pad(
                    full_embedding, (0, self.config.embedding_dim - len(full_embedding))
                )

            return embedding

        except Exception:
            # Fallback to hash-based
            embedding = np.zeros(self.config.embedding_dim, dtype=np.float32)
            for i, char in enumerate(text):
                idx = (ord(char) + i) % self.config.embedding_dim
                embedding[idx] += ord(char) / 256.0
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            return embedding


def create_llm(config: Optional[LLMConfig] = None) -> LLMOracle:
    """
    Factory function to create appropriate LLM oracle.

    Args:
        config: LLM configuration

    Returns:
        LLMOracle instance
    """
    config = config or LLMConfig()

    if config.provider == "anthropic":
        return AnthropicLLM(config)
    elif config.provider == "openai":
        return OpenAILLM(config)
    else:
        return MockLLM(config)
