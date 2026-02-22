"""
IntegratedAGIAgent: Combines all AGI modules into a unified agent.

This agent demonstrates how neuro-causal, neuro-abstract, neuro-robust,
and sleep-consolidation work together for intelligent behavior.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import sys
from pathlib import Path


@dataclass
class AgentConfig:
    """Configuration for the integrated agent."""

    embedding_dim: int = 256
    uncertainty_threshold: float = 0.3
    ood_threshold: float = 0.7
    consolidation_strength_target: float = 0.8
    random_seed: Optional[int] = 42


@dataclass
class Concept:
    """A learned concept in the agent's knowledge base."""

    name: str
    embedding: np.ndarray
    strength: float  # Memory strength [0, 1]
    category: str  # "episodic", "semantic", "procedural"
    causal_links: List[Tuple[str, str, float]] = field(
        default_factory=list
    )  # (cause, effect, strength)
    uncertainty: float = 0.0
    last_accessed: float = 0.0


@dataclass
class ReasoningResult:
    """Result of a reasoning operation."""

    answer: Any
    confidence: float
    uncertainty: float
    reasoning_trace: List[str]
    abstained: bool = False


@dataclass
class ConsolidationStats:
    """Statistics from a consolidation cycle."""

    memories_replayed: int
    memories_strengthened: int
    interference_resolved: int
    schemas_updated: int
    total_replay_time: float


class IntegratedAGIAgent:
    """
    An integrated AGI agent combining:
    - neuro-abstract: Symbol binding and compositional reasoning
    - neuro-causal: Causal models and counterfactual reasoning
    - sleep-consolidation: Memory consolidation and spaced repetition
    - neuro-robust: Uncertainty quantification and safe inference
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self._rng = np.random.default_rng(self.config.random_seed)

        # Knowledge base
        self._concepts: Dict[str, Concept] = {}
        self._schemas: Dict[str, np.ndarray] = {}
        self._causal_graph: Dict[str, List[str]] = {}

        # Consolidation state
        self._consolidation_history: List[ConsolidationStats] = []
        self._current_time: float = 0.0

        # Statistics
        self._total_queries: int = 0
        self._abstentions: int = 0
        self._successful_predictions: int = 0

    def learn_concept(
        self, name: str, examples: List[np.ndarray], category: str = "semantic"
    ) -> Concept:
        """
        Learn a new concept from examples using abstraction.

        Args:
            name: Concept name
            examples: List of example embeddings
            category: Type of memory ("episodic", "semantic", "procedural")

        Returns:
            Learned concept
        """
        if not examples:
            raise ValueError("Need at least one example")

        # Compute prototype (centroid of examples)
        examples_arr = np.array(examples)
        prototype = np.mean(examples_arr, axis=0)

        # Normalize
        norm = np.linalg.norm(prototype)
        if norm > 1e-8:
            prototype = prototype / norm

        # Compute uncertainty based on example variance
        if len(examples) > 1:
            variance = np.mean(np.var(examples_arr, axis=0))
            uncertainty = min(1.0, variance * 10)
        else:
            uncertainty = 0.5  # High uncertainty for single example

        concept = Concept(
            name=name,
            embedding=prototype,
            strength=0.3,  # Initial strength
            category=category,
            uncertainty=uncertainty,
            last_accessed=self._current_time,
        )

        self._concepts[name] = concept
        return concept

    def add_causal_relation(self, cause: str, effect: str, strength: float = 1.0) -> None:
        """
        Add a causal relation to the agent's world model.

        Args:
            cause: Name of cause concept
            effect: Name of effect concept
            strength: Causal strength [0, 1]
        """
        if cause not in self._concepts:
            raise ValueError(f"Unknown cause concept: {cause}")
        if effect not in self._concepts:
            raise ValueError(f"Unknown effect concept: {effect}")

        # Add to causal graph
        if cause not in self._causal_graph:
            self._causal_graph[cause] = []
        if effect not in self._causal_graph[cause]:
            self._causal_graph[cause].append(effect)

        # Add to concept's causal links
        self._concepts[cause].causal_links.append((cause, effect, strength))

    def reason_causally(
        self, query_type: str, cause: str, effect: str, intervention_value: Optional[float] = None
    ) -> ReasoningResult:
        """
        Perform causal reasoning.

        Args:
            query_type: "association", "intervention", or "counterfactual"
            cause: Cause concept name
            effect: Effect concept name
            intervention_value: Value for intervention queries

        Returns:
            ReasoningResult with answer and confidence
        """
        self._total_queries += 1
        trace = []

        # Check if concepts exist
        if cause not in self._concepts:
            return ReasoningResult(
                answer=None,
                confidence=0.0,
                uncertainty=1.0,
                reasoning_trace=[f"Unknown concept: {cause}"],
                abstained=True,
            )

        if effect not in self._concepts:
            return ReasoningResult(
                answer=None,
                confidence=0.0,
                uncertainty=1.0,
                reasoning_trace=[f"Unknown concept: {effect}"],
                abstained=True,
            )

        cause_concept = self._concepts[cause]
        effect_concept = self._concepts[effect]

        # Check for causal path
        has_path = self._has_causal_path(cause, effect)
        trace.append(f"Causal path exists: {has_path}")

        if not has_path:
            trace.append(f"No causal link from {cause} to {effect}")
            return ReasoningResult(
                answer=0.0,
                confidence=0.8,
                uncertainty=0.2,
                reasoning_trace=trace,
            )

        # Compute uncertainty from both concepts
        combined_uncertainty = (cause_concept.uncertainty + effect_concept.uncertainty) / 2

        # Check if should abstain due to uncertainty
        if combined_uncertainty > self.config.uncertainty_threshold:
            self._abstentions += 1
            trace.append(f"High uncertainty: {combined_uncertainty:.2f}")
            return ReasoningResult(
                answer=None,
                confidence=0.0,
                uncertainty=combined_uncertainty,
                reasoning_trace=trace,
                abstained=True,
            )

        # Find causal strength
        causal_strength = self._get_causal_strength(cause, effect)
        trace.append(f"Causal strength: {causal_strength:.2f}")

        # Compute result based on query type
        if query_type == "association":
            # Simple correlation-like query
            similarity = self._concept_similarity(cause_concept, effect_concept)
            answer = similarity * causal_strength
            trace.append(f"Association: {answer:.2f}")

        elif query_type == "intervention":
            # do(cause = value)
            if intervention_value is None:
                intervention_value = 1.0
            answer = intervention_value * causal_strength
            trace.append(f"Intervention effect: {answer:.2f}")

        elif query_type == "counterfactual":
            # What if cause had been different?
            answer = causal_strength * 0.9  # Slight discount for counterfactual
            trace.append(f"Counterfactual estimate: {answer:.2f}")

        else:
            trace.append(f"Unknown query type: {query_type}")
            return ReasoningResult(
                answer=None, confidence=0.0, uncertainty=1.0, reasoning_trace=trace, abstained=True
            )

        self._successful_predictions += 1
        confidence = 1.0 - combined_uncertainty

        return ReasoningResult(
            answer=answer,
            confidence=confidence,
            uncertainty=combined_uncertainty,
            reasoning_trace=trace,
        )

    def sleep_consolidate(self, n_cycles: int = 1) -> ConsolidationStats:
        """
        Run sleep consolidation cycles.

        Args:
            n_cycles: Number of sleep cycles to run

        Returns:
            Consolidation statistics
        """
        total_stats = ConsolidationStats(
            memories_replayed=0,
            memories_strengthened=0,
            interference_resolved=0,
            schemas_updated=0,
            total_replay_time=0.0,
        )

        for _ in range(n_cycles):
            stats = self._run_consolidation_cycle()
            total_stats.memories_replayed += stats.memories_replayed
            total_stats.memories_strengthened += stats.memories_strengthened
            total_stats.interference_resolved += stats.interference_resolved
            total_stats.schemas_updated += stats.schemas_updated
            total_stats.total_replay_time += stats.total_replay_time

        self._consolidation_history.append(total_stats)
        return total_stats

    def predict_with_uncertainty(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, float, bool]:
        """
        Make a prediction with uncertainty quantification.

        Args:
            query: Query string (concept name or relation)
            context: Optional context information

        Returns:
            Tuple of (prediction, confidence, abstained)
        """
        self._total_queries += 1

        # Check if query is about a known concept
        if query in self._concepts:
            concept = self._concepts[query]
            concept.last_accessed = self._current_time

            # Check OOD and uncertainty
            if concept.uncertainty > self.config.ood_threshold:
                self._abstentions += 1
                return None, 0.0, True

            self._successful_predictions += 1
            return concept.embedding, 1.0 - concept.uncertainty, False

        # Unknown query
        self._abstentions += 1
        return None, 0.0, True

    def get_concept(self, name: str) -> Optional[Concept]:
        """Get a concept by name."""
        return self._concepts.get(name)

    def list_concepts(self) -> List[str]:
        """List all known concepts."""
        return list(self._concepts.keys())

    def statistics(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            "total_concepts": len(self._concepts),
            "total_schemas": len(self._schemas),
            "causal_relations": sum(len(v) for v in self._causal_graph.values()),
            "total_queries": self._total_queries,
            "abstentions": self._abstentions,
            "abstention_rate": self._abstentions / max(1, self._total_queries),
            "successful_predictions": self._successful_predictions,
            "consolidation_cycles": len(self._consolidation_history),
            "current_time": self._current_time,
        }

    def advance_time(self, delta: float = 1.0) -> None:
        """Advance the agent's internal clock."""
        self._current_time += delta

        # Decay memory strengths
        for concept in self._concepts.values():
            time_since_access = self._current_time - concept.last_accessed
            decay = np.exp(-time_since_access / 10.0)  # Exponential decay
            concept.strength *= decay

    # Private methods

    def _has_causal_path(self, start: str, end: str, visited: Optional[set] = None) -> bool:
        """Check if there's a causal path from start to end."""
        if visited is None:
            visited = set()

        if start == end:
            return True

        if start in visited:
            return False

        visited.add(start)

        for child in self._causal_graph.get(start, []):
            if self._has_causal_path(child, end, visited):
                return True

        return False

    def _get_causal_strength(self, cause: str, effect: str) -> float:
        """Get the strength of the causal link."""
        concept = self._concepts.get(cause)
        if not concept:
            return 0.0

        for c, e, strength in concept.causal_links:
            if c == cause and e == effect:
                return strength

        # Check indirect path
        if cause in self._causal_graph:
            for intermediate in self._causal_graph[cause]:
                if intermediate == effect:
                    return 0.8  # Direct link
                if self._has_causal_path(intermediate, effect):
                    return 0.5  # Indirect link

        return 0.0

    def _concept_similarity(self, c1: Concept, c2: Concept) -> float:
        """Compute similarity between two concepts."""
        norm1 = np.linalg.norm(c1.embedding)
        norm2 = np.linalg.norm(c2.embedding)

        if norm1 < 1e-8 or norm2 < 1e-8:
            return 0.0

        return float(np.dot(c1.embedding, c2.embedding) / (norm1 * norm2))

    def _run_consolidation_cycle(self) -> ConsolidationStats:
        """Run a single consolidation cycle."""
        memories_replayed = 0
        memories_strengthened = 0
        interference_resolved = 0
        schemas_updated = 0

        # Priority replay: focus on weak memories
        weak_memories = [
            c
            for c in self._concepts.values()
            if c.strength < self.config.consolidation_strength_target
        ]

        for concept in weak_memories:
            # Replay
            memories_replayed += 1

            # Strengthen
            old_strength = concept.strength
            concept.strength = min(1.0, concept.strength + 0.1)
            if concept.strength > old_strength:
                memories_strengthened += 1

            # Reduce uncertainty through consolidation
            concept.uncertainty = max(0.1, concept.uncertainty * 0.9)

        # Check for interference between similar concepts
        concept_list = list(self._concepts.values())
        for i, c1 in enumerate(concept_list):
            for c2 in concept_list[i + 1 :]:
                sim = self._concept_similarity(c1, c2)
                if sim > 0.9 and c1.category == c2.category:
                    # High similarity - potential interference
                    # Differentiate by adding small noise
                    noise = self._rng.normal(0, 0.01, c1.embedding.shape)
                    c1.embedding = c1.embedding + noise
                    c1.embedding = c1.embedding / np.linalg.norm(c1.embedding)
                    interference_resolved += 1

        return ConsolidationStats(
            memories_replayed=memories_replayed,
            memories_strengthened=memories_strengthened,
            interference_resolved=interference_resolved,
            schemas_updated=schemas_updated,
            total_replay_time=memories_replayed * 0.1,  # Simulated time
        )
