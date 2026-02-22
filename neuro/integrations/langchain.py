"""
LangChain integration for NEURO cognitive architecture.

Provides a LangChain-compatible LLM that routes every call through
NEURO's 38 cognitive modules (memory, reasoning, causal analysis, etc.)
before returning a response.

Usage:
    from neuro.integrations.langchain import NeuroCognitiveLLM

    llm = NeuroCognitiveLLM()
    print(llm.invoke("What causes inflation?"))

    # With explicit provider
    llm = NeuroCognitiveLLM(provider="ollama", model="llama3.2")
"""

from typing import Any, List, Mapping, Optional

try:
    from langchain_core.language_models.llms import BaseLLM
    from langchain_core.callbacks.manager import CallbackManagerForLLMRun

    _HAS_LANGCHAIN = True
except ImportError:

    class BaseLLM:  # type: ignore[no-redef]
        """Stub so the module can be imported without langchain-core installed."""

        def __init__(self, **kwargs: Any) -> None:
            pass

    CallbackManagerForLLMRun = Any  # type: ignore[misc,assignment]
    _HAS_LANGCHAIN = False

from neuro.wrapper import SmartWrapper


class NeuroCognitiveLLM(BaseLLM):
    """LangChain LLM that routes queries through NEURO's cognitive pipeline.

    Every call passes through SmartWrapper which adds memory retrieval,
    reasoning analysis, and cognitive processing before hitting the
    underlying LLM provider.
    """

    provider: str = "auto"
    model: Optional[str] = None
    _wrapper: Optional[SmartWrapper] = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, provider: str = "auto", model: Optional[str] = None, **kwargs: Any) -> None:
        if _HAS_LANGCHAIN:
            super().__init__(provider=provider, model=model, **kwargs)
        else:
            super().__init__(**kwargs)
            self.provider = provider
            self.model = model
        self._wrapper = SmartWrapper(provider=provider, model=model)

    @property
    def _llm_type(self) -> str:
        return "neuro-cognitive"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        result = self._wrapper.query(prompt)
        text = result.text

        if stop:
            for token in stop:
                idx = text.find(token)
                if idx != -1:
                    text = text[:idx]

        return text

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "llm_type": self._llm_type,
        }


__all__ = ["NeuroCognitiveLLM"]
