"""
NeuroLLM - Unified LLM Interface for Neuro AGI

Combines local model inference with cognitive architecture orchestration.
"""

import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Iterator, Tuple

from .ollama_client import OllamaClient, OllamaConfig
from .model_router import ModelRouter, ModelSelection


@dataclass
class NeuroLLMConfig:
    """Configuration for NeuroLLM."""
    default_model: str = "mistral"
    enable_routing: bool = True
    system_prompt: str = """You are Neuro, an advanced AGI assistant built on a neuroscience-inspired cognitive architecture. You combine multiple reasoning styles (deductive, inductive, abductive, analogical, causal) to provide thoughtful, well-reasoned responses.

You are helpful, direct, and concise. You think through problems carefully before responding."""
    temperature: float = 0.7
    max_history: int = 20


@dataclass
class Message:
    """A chat message."""
    role: str
    content: str
    timestamp: float = field(default_factory=time.time)
    model_used: Optional[str] = None
    reasoning_style: Optional[str] = None


class NeuroLLM:
    """
    Unified LLM interface for Neuro AGI.

    Integrates:
    - Local model inference via Ollama
    - Intelligent model routing
    - Conversation history management
    - Cognitive architecture hooks
    """

    def __init__(self, config: Optional[NeuroLLMConfig] = None):
        self.config = config or NeuroLLMConfig()
        self._ollama = OllamaClient(OllamaConfig(
            default_model=self.config.default_model,
            temperature=self.config.temperature,
        ))
        self._router = ModelRouter(fallback_model=self.config.default_model)
        self._history: List[Message] = []
        self._total_tokens = 0

    def is_ready(self) -> Tuple[bool, str]:
        """Check if the system is ready."""
        if not self._ollama.is_available():
            return False, "Ollama is not running. Start with: ollama serve"

        models = self._ollama.list_models()
        if not models:
            return False, "No models installed. Run: ollama pull mistral"

        return True, f"Ready with models: {', '.join(models)}"

    def get_models(self) -> List[str]:
        """Get available models."""
        return self._ollama.list_models()

    def chat(
        self,
        message: str,
        model: Optional[str] = None,
        stream: bool = False,
    ) -> str | Iterator[str]:
        """
        Send a message and get a response.

        Args:
            message: User message
            model: Override model selection
            stream: Whether to stream response

        Returns:
            Response string or iterator of tokens
        """
        selection = None
        if not model and self.config.enable_routing:
            available = self._ollama.list_models()
            selection = self._router.select_model(message, available_models=available)
            model = selection.model

        model = model or self.config.default_model

        self._history.append(Message(
            role="user",
            content=message,
        ))

        messages = self._build_messages()

        if stream:
            return self._stream_response(messages, model, selection)
        else:
            response = self._ollama.chat(messages, model=model)
            self._history.append(Message(
                role="assistant",
                content=response,
                model_used=model,
                reasoning_style=selection.reasoning_style if selection else None,
            ))
            return response

    def _stream_response(
        self,
        messages: List[Dict[str, str]],
        model: str,
        selection: Optional[ModelSelection],
    ) -> Iterator[str]:
        """Stream response tokens."""
        full_response = []

        for token in self._ollama.chat_stream(messages, model=model):
            full_response.append(token)
            yield token

        response = "".join(full_response)
        self._history.append(Message(
            role="assistant",
            content=response,
            model_used=model,
            reasoning_style=selection.reasoning_style if selection else None,
        ))

    def _build_messages(self) -> List[Dict[str, str]]:
        """Build messages for API call."""
        messages = []

        if self.config.system_prompt:
            messages.append({
                "role": "system",
                "content": self.config.system_prompt,
            })

        recent = self._history[-self.config.max_history:]
        for msg in recent:
            messages.append({
                "role": msg.role,
                "content": msg.content,
            })

        return messages

    def clear_history(self) -> None:
        """Clear conversation history."""
        self._history = []

    def get_history(self) -> List[Message]:
        """Get conversation history."""
        return list(self._history)

    def get_last_model(self) -> Optional[str]:
        """Get the model used for the last response."""
        for msg in reversed(self._history):
            if msg.role == "assistant" and msg.model_used:
                return msg.model_used
        return None

    def statistics(self) -> Dict[str, Any]:
        """Get usage statistics."""
        models_used = {}
        styles_used = {}

        for msg in self._history:
            if msg.role == "assistant":
                if msg.model_used:
                    models_used[msg.model_used] = models_used.get(msg.model_used, 0) + 1
                if msg.reasoning_style:
                    styles_used[msg.reasoning_style] = styles_used.get(msg.reasoning_style, 0) + 1

        return {
            "total_messages": len(self._history),
            "user_messages": sum(1 for m in self._history if m.role == "user"),
            "assistant_messages": sum(1 for m in self._history if m.role == "assistant"),
            "models_used": models_used,
            "reasoning_styles": styles_used,
        }
