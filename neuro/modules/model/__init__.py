"""
Neuro Model - Local LLM Integration

Provides local model integration via Ollama:
- OllamaClient: Direct Ollama API wrapper
- ModelRouter: Problem-based model selection using meta-reasoning
- NeuroLLM: Unified interface for the cognitive architecture
"""

from .ollama_client import OllamaClient, OllamaConfig
from .model_router import ModelRouter
from .neuro_llm import NeuroLLM, NeuroLLMConfig

__all__ = [
    "OllamaClient",
    "OllamaConfig",
    "ModelRouter",
    "NeuroLLM",
    "NeuroLLMConfig",
]
