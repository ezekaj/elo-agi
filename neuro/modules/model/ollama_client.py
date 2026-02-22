"""
Ollama Client - Local LLM API Wrapper

Provides direct integration with Ollama for local model inference.
"""

import requests
import json
from dataclasses import dataclass
from typing import Optional, List, Dict, Iterator


@dataclass
class OllamaConfig:
    """Configuration for Ollama client."""

    base_url: str = "http://localhost:11434"
    default_model: str = "ministral-3:8b"  # Ministral 3 8B - ONLY model
    timeout: int = 120
    temperature: float = 0.7
    max_tokens: int = 2048


class OllamaClient:
    """
    Local LLM client using Ollama.

    Provides both streaming and non-streaming interfaces
    for local model inference.
    """

    def __init__(self, config: Optional[OllamaConfig] = None):
        self.config = config or OllamaConfig()

    def is_available(self) -> bool:
        """Check if Ollama is running and accessible."""
        try:
            response = requests.get(f"{self.config.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def list_models(self) -> List[str]:
        """List available models."""
        try:
            response = requests.get(f"{self.config.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                data = response.json()
                return [m["name"] for m in data.get("models", [])]
            return []
        except Exception:
            return []

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate response from local model.

        Args:
            prompt: The user prompt
            model: Model to use (defaults to config default)
            system: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text response
        """
        model = model or self.config.default_model
        temperature = temperature if temperature is not None else self.config.temperature

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }

        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
        if system:
            payload["system"] = system

        response = requests.post(
            f"{self.config.base_url}/api/generate", json=payload, timeout=self.config.timeout
        )

        if response.status_code != 200:
            raise RuntimeError(f"Ollama error: {response.text}")

        return response.json()["response"]

    def generate_stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> Iterator[str]:
        """
        Generate response with streaming.

        Yields:
            Tokens as they are generated
        """
        model = model or self.config.default_model
        temperature = temperature if temperature is not None else self.config.temperature

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
            },
        }

        if system:
            payload["system"] = system

        response = requests.post(
            f"{self.config.base_url}/api/generate",
            json=payload,
            timeout=self.config.timeout,
            stream=True,
        )

        if response.status_code != 200:
            raise RuntimeError(f"Ollama error: {response.text}")

        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                if "response" in data:
                    yield data["response"]
                if data.get("done", False):
                    break

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Chat completion with local model.

        Args:
            messages: List of {"role": "user/assistant/system", "content": "..."}
            model: Model to use
            temperature: Sampling temperature

        Returns:
            Assistant response
        """
        model = model or self.config.default_model
        temperature = temperature if temperature is not None else self.config.temperature

        response = requests.post(
            f"{self.config.base_url}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                },
            },
            timeout=self.config.timeout,
        )

        if response.status_code != 200:
            raise RuntimeError(f"Ollama error: {response.text}")

        return response.json()["message"]["content"]

    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> Iterator[str]:
        """
        Chat completion with streaming.

        Yields:
            Tokens as they are generated
        """
        model = model or self.config.default_model
        temperature = temperature if temperature is not None else self.config.temperature

        response = requests.post(
            f"{self.config.base_url}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": True,
                "options": {
                    "temperature": temperature,
                },
            },
            timeout=self.config.timeout,
            stream=True,
        )

        if response.status_code != 200:
            raise RuntimeError(f"Ollama error: {response.text}")

        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                if "message" in data and "content" in data["message"]:
                    yield data["message"]["content"]
                if data.get("done", False):
                    break
