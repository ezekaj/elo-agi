"""Brain API: Dead-simple interface to NEURO's cognitive architecture."""

from typing import List, Optional

from neuro.wrapper import SmartWrapper, SmartResponse


class Brain:
    """Simple wrapper around SmartWrapper for intuitive cognitive queries."""

    def __init__(self, provider: str = "auto", model: Optional[str] = None):
        self._wrapper = SmartWrapper(provider=provider, model=model)

    def think(self, question: str) -> SmartResponse:
        """Ask the brain a question and get a cognitively-enriched response."""
        return self._wrapper.query(question)

    def learn(self, topic: str) -> None:
        """Store context by processing a learning prompt."""
        self._wrapper.query(
            f"Learn and remember the following information for future reference: {topic}"
        )

    def reason(self, question: str) -> SmartResponse:
        """Alias for think()."""
        return self.think(question)

    def reset(self) -> None:
        """Reset conversation history and module state."""
        self._wrapper.reset()

    @property
    def modules(self) -> List[str]:
        """List available cognitive module names."""
        try:
            core = self._wrapper._get_core()
            state = core.get_state()
            return state.active_modules
        except Exception:
            return []


def think(question: str, provider: str = "auto", model: Optional[str] = None) -> SmartResponse:
    """One-shot brain query."""
    brain = Brain(provider=provider, model=model)
    return brain.think(question)
