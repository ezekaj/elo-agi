"""
Conflict Resolution: Handles disagreements between cognitive modules.

When different modules produce conflicting outputs or beliefs,
this system resolves the conflict using various strategies.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from enum import Enum
import numpy as np

from .shared_space import SemanticEmbedding, ModalityType


class ConflictType(Enum):
    """Types of conflicts between modules."""

    BELIEF = "belief"  # Conflicting beliefs about world state
    ACTION = "action"  # Different action recommendations
    ATTENTION = "attention"  # Competition for attention
    GOAL = "goal"  # Conflicting goals
    INTERPRETATION = "interpretation"  # Different interpretations of input


class ResolutionStrategy(Enum):
    """Strategies for resolving conflicts."""

    WEIGHTED_AVERAGE = "weighted_average"  # Blend based on confidence
    WINNER_TAKE_ALL = "winner_take_all"  # Highest confidence wins
    VOTING = "voting"  # Majority/plurality
    ARBITRATION = "arbitration"  # Meta-module decides
    NEGOTIATION = "negotiation"  # Iterative refinement
    CONTEXT_DEPENDENT = "context_dependent"  # Based on current context


@dataclass
class Conflict:
    """Represents a conflict between modules."""

    conflict_id: int
    conflict_type: ConflictType
    modules: List[str]
    embeddings: Dict[str, SemanticEmbedding]
    confidences: Dict[str, float]
    context: Optional[SemanticEmbedding] = None
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def severity(self) -> float:
        """Compute conflict severity based on embedding disagreement."""
        if len(self.embeddings) < 2:
            return 0.0

        embeddings = list(self.embeddings.values())
        total_disagreement = 0.0
        n_pairs = 0

        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                # Higher similarity = lower disagreement
                similarity = embeddings[i].similarity(embeddings[j])
                total_disagreement += 1 - similarity
                n_pairs += 1

        if n_pairs == 0:
            return 0.0

        return total_disagreement / n_pairs


@dataclass
class Resolution:
    """Result of conflict resolution."""

    conflict: Conflict
    strategy_used: ResolutionStrategy
    resolved_embedding: SemanticEmbedding
    winning_modules: List[str]
    resolution_confidence: float
    reasoning: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConflictResolver:
    """
    Resolves conflicts between cognitive modules.

    Uses multiple strategies depending on conflict type and context.
    Learns from past resolutions to improve future decisions.
    """

    def __init__(
        self,
        default_strategy: ResolutionStrategy = ResolutionStrategy.WEIGHTED_AVERAGE,
        random_seed: Optional[int] = None,
    ):
        self.default_strategy = default_strategy
        self._rng = np.random.default_rng(random_seed)

        # Module reliability scores (learned)
        self._reliability: Dict[str, float] = {}

        # Strategy performance history
        self._strategy_history: Dict[ResolutionStrategy, List[bool]] = {
            s: [] for s in ResolutionStrategy
        }

        # Conflict history
        self._conflicts: List[Conflict] = []
        self._resolutions: List[Resolution] = []

        # Conflict counter
        self._conflict_counter = 0

        # Strategy preferences per conflict type
        self._type_strategies: Dict[ConflictType, ResolutionStrategy] = {
            ConflictType.BELIEF: ResolutionStrategy.WEIGHTED_AVERAGE,
            ConflictType.ACTION: ResolutionStrategy.WINNER_TAKE_ALL,
            ConflictType.ATTENTION: ResolutionStrategy.VOTING,
            ConflictType.GOAL: ResolutionStrategy.NEGOTIATION,
            ConflictType.INTERPRETATION: ResolutionStrategy.CONTEXT_DEPENDENT,
        }

    def detect_conflict(
        self,
        module_outputs: Dict[str, SemanticEmbedding],
        threshold: float = 0.3,
    ) -> Optional[Conflict]:
        """Detect if there's a conflict between module outputs."""
        if len(module_outputs) < 2:
            return None

        # Compute pairwise similarities
        modules = list(module_outputs.keys())
        embeddings = list(module_outputs.values())

        max_disagreement = 0.0
        conflicting_pair = None

        for i in range(len(modules)):
            for j in range(i + 1, len(modules)):
                similarity = embeddings[i].similarity(embeddings[j])
                disagreement = 1 - similarity

                if disagreement > max_disagreement:
                    max_disagreement = disagreement
                    conflicting_pair = (modules[i], modules[j])

        if max_disagreement > threshold:
            self._conflict_counter += 1

            conflict = Conflict(
                conflict_id=self._conflict_counter,
                conflict_type=ConflictType.BELIEF,  # Default type
                modules=modules,
                embeddings=module_outputs.copy(),
                confidences={m: e.confidence for m, e in module_outputs.items()},
                timestamp=float(self._conflict_counter),
            )
            self._conflicts.append(conflict)
            return conflict

        return None

    def resolve(
        self,
        conflict: Conflict,
        strategy: Optional[ResolutionStrategy] = None,
    ) -> Resolution:
        """Resolve a conflict using the specified strategy."""
        if strategy is None:
            strategy = self._type_strategies.get(conflict.conflict_type, self.default_strategy)

        if strategy == ResolutionStrategy.WEIGHTED_AVERAGE:
            resolution = self._resolve_weighted_average(conflict)
        elif strategy == ResolutionStrategy.WINNER_TAKE_ALL:
            resolution = self._resolve_winner_take_all(conflict)
        elif strategy == ResolutionStrategy.VOTING:
            resolution = self._resolve_voting(conflict)
        elif strategy == ResolutionStrategy.ARBITRATION:
            resolution = self._resolve_arbitration(conflict)
        elif strategy == ResolutionStrategy.NEGOTIATION:
            resolution = self._resolve_negotiation(conflict)
        elif strategy == ResolutionStrategy.CONTEXT_DEPENDENT:
            resolution = self._resolve_context_dependent(conflict)
        else:
            resolution = self._resolve_weighted_average(conflict)

        resolution.strategy_used = strategy
        self._resolutions.append(resolution)

        return resolution

    def _resolve_weighted_average(self, conflict: Conflict) -> Resolution:
        """Resolve by weighted average of embeddings."""
        embeddings = conflict.embeddings
        confidences = conflict.confidences

        # Adjust confidences by reliability
        adjusted_conf = {}
        for module, conf in confidences.items():
            reliability = self._reliability.get(module, 1.0)
            adjusted_conf[module] = conf * reliability

        total_weight = sum(adjusted_conf.values())
        if total_weight < 1e-8:
            total_weight = 1.0

        # Weighted average
        dim = list(embeddings.values())[0].vector.shape[0]
        result_vector = np.zeros(dim)

        for module, embedding in embeddings.items():
            weight = adjusted_conf[module] / total_weight
            result_vector += weight * embedding.vector

        # Normalize
        norm = np.linalg.norm(result_vector)
        if norm > 1e-8:
            result_vector = result_vector / norm

        resolved = SemanticEmbedding(
            vector=result_vector,
            modality=ModalityType.ABSTRACT,
            source_module="conflict_resolution",
            confidence=float(np.mean(list(adjusted_conf.values()))),
        )

        return Resolution(
            conflict=conflict,
            strategy_used=ResolutionStrategy.WEIGHTED_AVERAGE,
            resolved_embedding=resolved,
            winning_modules=list(embeddings.keys()),
            resolution_confidence=resolved.confidence,
            reasoning="Blended all module outputs weighted by confidence and reliability",
        )

    def _resolve_winner_take_all(self, conflict: Conflict) -> Resolution:
        """Resolve by selecting highest confidence module."""
        embeddings = conflict.embeddings
        confidences = conflict.confidences

        # Find winner
        winner = max(
            confidences.keys(), key=lambda m: confidences[m] * self._reliability.get(m, 1.0)
        )

        resolved = SemanticEmbedding(
            vector=embeddings[winner].vector.copy(),
            modality=embeddings[winner].modality,
            source_module=winner,
            confidence=confidences[winner],
        )

        return Resolution(
            conflict=conflict,
            strategy_used=ResolutionStrategy.WINNER_TAKE_ALL,
            resolved_embedding=resolved,
            winning_modules=[winner],
            resolution_confidence=confidences[winner],
            reasoning=f"Selected {winner} as winner with highest adjusted confidence",
        )

    def _resolve_voting(self, conflict: Conflict) -> Resolution:
        """Resolve by clustering similar outputs and voting."""
        embeddings = list(conflict.embeddings.values())
        modules = list(conflict.embeddings.keys())

        # Simple clustering: group by similarity
        clusters: List[List[int]] = []
        assigned = set()

        for i in range(len(embeddings)):
            if i in assigned:
                continue

            cluster = [i]
            assigned.add(i)

            for j in range(i + 1, len(embeddings)):
                if j in assigned:
                    continue
                if embeddings[i].similarity(embeddings[j]) > 0.7:
                    cluster.append(j)
                    assigned.add(j)

            clusters.append(cluster)

        # Find largest cluster
        largest = max(clusters, key=len)
        winning_modules = [modules[i] for i in largest]

        # Average within winning cluster
        dim = embeddings[0].vector.shape[0]
        result_vector = np.zeros(dim)
        for i in largest:
            result_vector += embeddings[i].vector
        result_vector /= len(largest)
        norm = np.linalg.norm(result_vector)
        if norm > 1e-8:
            result_vector /= norm

        resolved = SemanticEmbedding(
            vector=result_vector,
            modality=ModalityType.ABSTRACT,
            source_module="voting",
            confidence=len(largest) / len(modules),
        )

        return Resolution(
            conflict=conflict,
            strategy_used=ResolutionStrategy.VOTING,
            resolved_embedding=resolved,
            winning_modules=winning_modules,
            resolution_confidence=resolved.confidence,
            reasoning=f"Majority cluster of {len(largest)} modules won",
        )

    def _resolve_arbitration(self, conflict: Conflict) -> Resolution:
        """Resolve using meta-level arbitration rules."""
        embeddings = conflict.embeddings
        confidences = conflict.confidences

        # Arbitration rules based on conflict type
        if conflict.conflict_type == ConflictType.ACTION:
            # For actions, prefer safer/more cautious option
            # (represented by lower norm activation)
            safest = min(embeddings.keys(), key=lambda m: np.linalg.norm(embeddings[m].vector))
            winner = safest
        elif conflict.conflict_type == ConflictType.GOAL:
            # For goals, prefer higher-level goal (if detectable)
            winner = max(confidences.keys(), key=lambda m: confidences[m])
        else:
            # Default to highest confidence
            winner = max(confidences.keys(), key=lambda m: confidences[m])

        resolved = SemanticEmbedding(
            vector=embeddings[winner].vector.copy(),
            modality=embeddings[winner].modality,
            source_module=winner,
            confidence=confidences[winner],
        )

        return Resolution(
            conflict=conflict,
            strategy_used=ResolutionStrategy.ARBITRATION,
            resolved_embedding=resolved,
            winning_modules=[winner],
            resolution_confidence=confidences[winner],
            reasoning=f"Arbitration selected {winner} based on conflict type rules",
        )

    def _resolve_negotiation(self, conflict: Conflict) -> Resolution:
        """Resolve through iterative negotiation between modules."""
        embeddings = dict(conflict.embeddings)
        confidences = dict(conflict.confidences)

        # Iterative refinement (simplified)
        max_iterations = 10
        convergence_threshold = 0.01

        dim = list(embeddings.values())[0].vector.shape[0]
        current = np.mean([e.vector for e in embeddings.values()], axis=0)

        for _ in range(max_iterations):
            # Each module adjusts toward consensus
            new_vectors = {}
            for module, emb in embeddings.items():
                # Move toward mean weighted by confidence
                weight = 0.5 * confidences[module]
                new_vectors[module] = (1 - weight) * emb.vector + weight * current

            # Update embeddings
            for module in embeddings:
                embeddings[module] = SemanticEmbedding(
                    vector=new_vectors[module],
                    modality=embeddings[module].modality,
                    source_module=module,
                    confidence=confidences[module],
                )

            # Update consensus
            new_current = np.mean([e.vector for e in embeddings.values()], axis=0)

            # Check convergence
            if np.linalg.norm(new_current - current) < convergence_threshold:
                break

            current = new_current

        # Final consensus
        norm = np.linalg.norm(current)
        if norm > 1e-8:
            current = current / norm

        resolved = SemanticEmbedding(
            vector=current,
            modality=ModalityType.ABSTRACT,
            source_module="negotiation",
            confidence=float(np.mean(list(confidences.values()))),
        )

        return Resolution(
            conflict=conflict,
            strategy_used=ResolutionStrategy.NEGOTIATION,
            resolved_embedding=resolved,
            winning_modules=list(embeddings.keys()),
            resolution_confidence=resolved.confidence,
            reasoning="Iterative negotiation converged to consensus",
        )

    def _resolve_context_dependent(self, conflict: Conflict) -> Resolution:
        """Resolve based on current context."""
        embeddings = conflict.embeddings
        confidences = conflict.confidences
        context = conflict.context

        if context is None:
            # Fall back to weighted average
            return self._resolve_weighted_average(conflict)

        # Find module most aligned with current context
        best_module = None
        best_alignment = -1.0

        for module, emb in embeddings.items():
            alignment = emb.similarity(context)
            weighted_alignment = alignment * confidences[module]

            if weighted_alignment > best_alignment:
                best_alignment = weighted_alignment
                best_module = module

        if best_module is None:
            return self._resolve_weighted_average(conflict)

        resolved = SemanticEmbedding(
            vector=embeddings[best_module].vector.copy(),
            modality=embeddings[best_module].modality,
            source_module=best_module,
            confidence=confidences[best_module],
        )

        return Resolution(
            conflict=conflict,
            strategy_used=ResolutionStrategy.CONTEXT_DEPENDENT,
            resolved_embedding=resolved,
            winning_modules=[best_module],
            resolution_confidence=confidences[best_module],
            reasoning=f"Context alignment selected {best_module}",
        )

    def update_reliability(
        self,
        module: str,
        success: bool,
        learning_rate: float = 0.05,
    ) -> None:
        """Update reliability score for a module."""
        current = self._reliability.get(module, 1.0)
        if success:
            self._reliability[module] = min(2.0, current + learning_rate)
        else:
            self._reliability[module] = max(0.1, current - learning_rate)

    def record_outcome(
        self,
        resolution: Resolution,
        success: bool,
    ) -> None:
        """Record resolution outcome for learning."""
        self._strategy_history[resolution.strategy_used].append(success)

        # Update reliability for winning modules
        for module in resolution.winning_modules:
            self.update_reliability(module, success)

    def get_best_strategy(
        self,
        conflict_type: ConflictType,
    ) -> ResolutionStrategy:
        """Get best strategy based on historical performance."""
        best_strategy = self._type_strategies.get(conflict_type, self.default_strategy)
        best_success_rate = 0.0

        for strategy, history in self._strategy_history.items():
            if len(history) > 5:  # Need enough data
                success_rate = sum(history[-20:]) / len(history[-20:])
                if success_rate > best_success_rate:
                    best_success_rate = success_rate
                    best_strategy = strategy

        return best_strategy

    def statistics(self) -> Dict[str, Any]:
        """Get conflict resolution statistics."""
        strategy_stats = {}
        for strategy, history in self._strategy_history.items():
            if history:
                strategy_stats[strategy.value] = {
                    "count": len(history),
                    "success_rate": sum(history) / len(history) if history else 0,
                }

        return {
            "total_conflicts": len(self._conflicts),
            "total_resolutions": len(self._resolutions),
            "module_reliability": dict(self._reliability),
            "strategy_performance": strategy_stats,
            "default_strategy": self.default_strategy.value,
        }
