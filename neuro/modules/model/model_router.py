"""
Model Router - Intelligent Model Selection

Uses meta-reasoning to select the optimal model for each query.
"""

import sys
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any
from enum import Enum

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from neuro_meta_reasoning.src.problem_classifier import (
        ProblemClassifier,
        ProblemType,
        ProblemDifficulty,
    )
    from neuro_meta_reasoning.src.style_selector import (
        StyleSelector,
        ReasoningStyle,
    )
    META_REASONING_AVAILABLE = True
except ImportError:
    META_REASONING_AVAILABLE = False


class ModelTier(Enum):
    """Model capability tiers."""
    FAST = "fast"
    BALANCED = "balanced"
    POWERFUL = "powerful"


@dataclass
class ModelSelection:
    """Result of model selection."""
    model: str
    tier: ModelTier
    reasoning_style: Optional[str]
    confidence: float
    rationale: str


class ModelRouter:
    """
    Routes queries to the best local model based on problem analysis.

    Uses the meta-reasoning module to classify problems and select
    appropriate models for different task types.
    """

    MODEL_MAP = {
        ModelTier.FAST: "phi3",
        ModelTier.BALANCED: "mistral",
        ModelTier.POWERFUL: "llama3",
    }

    STYLE_TO_MODEL = {
        "deductive": "mistral",
        "inductive": "mistral",
        "abductive": "llama3",
        "analogical": "mistral",
        "causal": "mistral",
        "spatial": "mistral",
        "temporal": "mistral",
        "heuristic": "phi3",
        "systematic": "mistral",
        "creative": "llama3",
    }

    PROBLEM_TO_TIER = {
        "logical": ModelTier.BALANCED,
        "mathematical": ModelTier.BALANCED,
        "causal": ModelTier.BALANCED,
        "spatial": ModelTier.BALANCED,
        "creative": ModelTier.POWERFUL,
        "planning": ModelTier.BALANCED,
        "linguistic": ModelTier.BALANCED,
        "analogical": ModelTier.BALANCED,
    }

    def __init__(self, fallback_model: str = "mistral"):
        self.fallback_model = fallback_model
        self._classifier = None
        self._style_selector = None

        if META_REASONING_AVAILABLE:
            try:
                self._classifier = ProblemClassifier()
                self._style_selector = StyleSelector()
            except Exception:
                pass

    def select_model(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        available_models: Optional[list] = None,
    ) -> ModelSelection:
        """
        Select the best model for a query.

        Args:
            query: User query
            context: Optional context
            available_models: List of available models

        Returns:
            ModelSelection with chosen model and rationale
        """
        available = available_models or list(self.MODEL_MAP.values())

        query_lower = query.lower().strip()
        query_len = len(query)

        if query_len < 20:
            return ModelSelection(
                model=self._get_available(ModelTier.FAST, available),
                tier=ModelTier.FAST,
                reasoning_style=None,
                confidence=0.9,
                rationale="Short query, using fast model"
            )

        if any(word in query_lower for word in [
            "code", "program", "function", "algorithm", "debug",
            "math", "calculate", "solve", "equation", "proof"
        ]):
            return ModelSelection(
                model=self._get_available(ModelTier.BALANCED, available),
                tier=ModelTier.BALANCED,
                reasoning_style="systematic",
                confidence=0.85,
                rationale="Technical/mathematical query"
            )

        if any(word in query_lower for word in [
            "creative", "story", "imagine", "design", "invent",
            "brainstorm", "novel", "unique", "innovative"
        ]):
            return ModelSelection(
                model=self._get_available(ModelTier.POWERFUL, available),
                tier=ModelTier.POWERFUL,
                reasoning_style="creative",
                confidence=0.85,
                rationale="Creative task requiring powerful model"
            )

        if any(word in query_lower for word in [
            "analyze", "explain", "compare", "evaluate", "reason",
            "why", "how", "complex", "detailed"
        ]):
            return ModelSelection(
                model=self._get_available(ModelTier.BALANCED, available),
                tier=ModelTier.BALANCED,
                reasoning_style="deductive",
                confidence=0.8,
                rationale="Analytical task"
            )

        return ModelSelection(
            model=self._get_available(ModelTier.BALANCED, available),
            tier=ModelTier.BALANCED,
            reasoning_style="inductive",
            confidence=0.7,
            rationale="General query"
        )

    def _get_available(
        self,
        tier: ModelTier,
        available: list,
    ) -> str:
        """Get best available model for tier."""
        preferred = self.MODEL_MAP.get(tier, self.fallback_model)

        if preferred in available:
            return preferred

        for t in [ModelTier.BALANCED, ModelTier.FAST, ModelTier.POWERFUL]:
            model = self.MODEL_MAP.get(t)
            if model in available:
                return model

        return available[0] if available else self.fallback_model
