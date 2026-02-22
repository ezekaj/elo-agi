"""
Interference Resolution for Catastrophic Forgetting Prevention.

Implements mechanisms to detect and resolve interference
between similar memories during consolidation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum
import numpy as np


class ResolutionStrategy(Enum):
    """Strategies for resolving memory interference."""

    INTERLEAVE = "interleave"  # Alternate replay of conflicting memories
    DIFFERENTIATE = "differentiate"  # Emphasize distinctive features
    INHIBIT = "inhibit"  # Temporarily suppress one memory
    MERGE = "merge"  # Combine into single memory
    SEPARATE = "separate"  # Create distinct contexts


class InterferenceType(Enum):
    """Types of memory interference."""

    PROACTIVE = "proactive"  # Old memory interferes with new
    RETROACTIVE = "retroactive"  # New memory interferes with old
    BIDIRECTIONAL = "bidirectional"


@dataclass
class MemoryVector:
    """Representation of a memory for interference analysis."""

    memory_id: str
    content: np.ndarray
    encoding_time: float
    strength: float = 1.0
    context: Optional[np.ndarray] = None
    distinctive_features: Optional[np.ndarray] = None

    def similarity(self, other: "MemoryVector") -> float:
        """Compute cosine similarity with another memory."""
        norm_self = np.linalg.norm(self.content)
        norm_other = np.linalg.norm(other.content)
        if norm_self < 1e-8 or norm_other < 1e-8:
            return 0.0
        return float(np.dot(self.content, other.content) / (norm_self * norm_other))


@dataclass
class InterferenceEvent:
    """Detected interference between memories."""

    memory_a: str
    memory_b: str
    similarity: float
    interference_type: InterferenceType
    severity: float  # 0-1, how much interference
    resolution_strategy: ResolutionStrategy
    detected_at: float
    resolved: bool = False
    resolution_success: Optional[float] = None


@dataclass
class InterleaveSchedule:
    """Schedule for interleaved replay of conflicting memories."""

    memories: List[str]
    pattern: List[int]  # Indices into memories list
    current_position: int = 0
    repetitions: int = 3

    def next_memory(self) -> str:
        """Get next memory in interleave pattern."""
        if not self.pattern:
            return self.memories[0] if self.memories else ""

        idx = self.pattern[self.current_position % len(self.pattern)]
        self.current_position += 1
        return self.memories[idx]

    def is_complete(self) -> bool:
        """Check if interleave schedule is complete."""
        return self.current_position >= len(self.pattern) * self.repetitions


class InterferenceResolver:
    """
    Resolves interference between similar memories.

    Implements:
    - Similarity-based interference detection
    - Multiple resolution strategies
    - Proactive and retroactive interference handling
    - Interleaved replay scheduling
    """

    def __init__(
        self,
        similarity_threshold: float = 0.7,
        severity_threshold: float = 0.3,
        random_seed: Optional[int] = None,
    ):
        self.similarity_threshold = similarity_threshold
        self.severity_threshold = severity_threshold
        self._rng = np.random.default_rng(random_seed)

        # Memory storage
        self._memories: Dict[str, MemoryVector] = {}

        # Detected interference events
        self._events: List[InterferenceEvent] = []

        # Active interleave schedules
        self._interleave_schedules: Dict[str, InterleaveSchedule] = {}

        # Similarity cache
        self._similarity_cache: Dict[Tuple[str, str], float] = {}

        # Statistics
        self._n_detections = 0
        self._n_resolutions = 0
        self._n_successful = 0

    def register_memory(
        self,
        memory_id: str,
        content: np.ndarray,
        encoding_time: float,
        strength: float = 1.0,
        context: Optional[np.ndarray] = None,
    ) -> MemoryVector:
        """Register a memory for interference analysis."""
        memory = MemoryVector(
            memory_id=memory_id,
            content=content.copy(),
            encoding_time=encoding_time,
            strength=strength,
            context=context.copy() if context is not None else None,
        )
        self._memories[memory_id] = memory
        return memory

    def compute_similarity(
        self,
        memory_a: str,
        memory_b: str,
    ) -> float:
        """Compute similarity between two memories."""
        cache_key = (min(memory_a, memory_b), max(memory_a, memory_b))
        if cache_key in self._similarity_cache:
            return self._similarity_cache[cache_key]

        mem_a = self._memories.get(memory_a)
        mem_b = self._memories.get(memory_b)

        if mem_a is None or mem_b is None:
            return 0.0

        similarity = mem_a.similarity(mem_b)
        self._similarity_cache[cache_key] = similarity
        return similarity

    def detect_interference(
        self,
        memory_ids: Optional[List[str]] = None,
        current_time: float = 0.0,
    ) -> List[InterferenceEvent]:
        """
        Detect interference between memories.

        Returns list of interference events above threshold.
        """
        if memory_ids is None:
            memory_ids = list(self._memories.keys())

        events = []

        for i, mem_a in enumerate(memory_ids):
            for mem_b in memory_ids[i + 1 :]:
                similarity = self.compute_similarity(mem_a, mem_b)

                if similarity >= self.similarity_threshold:
                    # Determine interference type
                    time_a = self._memories[mem_a].encoding_time
                    time_b = self._memories[mem_b].encoding_time
                    strength_a = self._memories[mem_a].strength
                    strength_b = self._memories[mem_b].strength

                    if abs(time_a - time_b) < 0.1:
                        interference_type = InterferenceType.BIDIRECTIONAL
                    elif time_a < time_b:
                        interference_type = InterferenceType.PROACTIVE
                    else:
                        interference_type = InterferenceType.RETROACTIVE

                    # Compute severity based on similarity and strength difference
                    severity = similarity * (1.0 - abs(strength_a - strength_b))

                    # Choose resolution strategy
                    strategy = self._choose_strategy(similarity, strength_a, strength_b)

                    event = InterferenceEvent(
                        memory_a=mem_a,
                        memory_b=mem_b,
                        similarity=similarity,
                        interference_type=interference_type,
                        severity=severity,
                        resolution_strategy=strategy,
                        detected_at=current_time,
                    )
                    events.append(event)
                    self._events.append(event)
                    self._n_detections += 1

        return events

    def _choose_strategy(
        self,
        similarity: float,
        strength_a: float,
        strength_b: float,
    ) -> ResolutionStrategy:
        """Choose resolution strategy based on memory characteristics."""
        # Very high similarity -> differentiate or merge
        if similarity > 0.9:
            if abs(strength_a - strength_b) < 0.2:
                return ResolutionStrategy.MERGE
            else:
                return ResolutionStrategy.DIFFERENTIATE

        # Moderate similarity -> interleave
        if similarity > 0.7:
            return ResolutionStrategy.INTERLEAVE

        # Lower similarity but still interfering
        if strength_a > strength_b + 0.3:
            return ResolutionStrategy.INHIBIT  # Inhibit weaker
        elif strength_b > strength_a + 0.3:
            return ResolutionStrategy.INHIBIT
        else:
            return ResolutionStrategy.SEPARATE

    def resolve_proactive(
        self,
        new_memory: str,
        existing_memories: List[str],
    ) -> Tuple[MemoryVector, List[str]]:
        """
        Resolve proactive interference (old affecting new).

        Returns modified new memory and list of affected existing memories.
        """
        self._n_resolutions += 1
        new_mem = self._memories.get(new_memory)
        if new_mem is None:
            return None, []

        affected = []
        modified_content = new_mem.content.copy()

        for existing_id in existing_memories:
            existing = self._memories.get(existing_id)
            if existing is None:
                continue

            similarity = self.compute_similarity(new_memory, existing_id)

            if similarity >= self.similarity_threshold:
                affected.append(existing_id)

                # Differentiate by emphasizing distinctive features
                diff = new_mem.content - existing.content
                diff_norm = np.linalg.norm(diff)

                if diff_norm > 1e-8:
                    # Enhance distinctive features
                    distinctive = diff / diff_norm
                    enhancement = 0.2 * similarity  # More enhancement for more similar
                    modified_content = modified_content + enhancement * distinctive

        # Normalize
        norm = np.linalg.norm(modified_content)
        if norm > 1e-8:
            modified_content = modified_content / norm

        # Update memory
        new_mem.content = modified_content
        new_mem.distinctive_features = modified_content - new_mem.content

        self._n_successful += 1
        return new_mem, affected

    def resolve_retroactive(
        self,
        existing_memory: str,
        new_memories: List[str],
    ) -> Tuple[MemoryVector, float]:
        """
        Resolve retroactive interference (new affecting old).

        Returns modified existing memory and protection factor.
        """
        self._n_resolutions += 1
        existing = self._memories.get(existing_memory)
        if existing is None:
            return None, 0.0

        protection = 1.0
        modified_content = existing.content.copy()

        for new_id in new_memories:
            new_mem = self._memories.get(new_id)
            if new_mem is None:
                continue

            similarity = self.compute_similarity(existing_memory, new_id)

            if similarity >= self.similarity_threshold:
                # Reduce protection based on similarity
                protection *= 1.0 - similarity * 0.3

                # Strengthen distinctive features of existing memory
                diff = existing.content - new_mem.content
                diff_norm = np.linalg.norm(diff)

                if diff_norm > 1e-8:
                    distinctive = diff / diff_norm
                    # Strengthen existing memory's unique aspects
                    modified_content = modified_content + 0.1 * existing.strength * distinctive

        # Normalize
        norm = np.linalg.norm(modified_content)
        if norm > 1e-8:
            modified_content = modified_content / norm

        existing.content = modified_content
        self._n_successful += 1

        return existing, protection

    def interleave_replays(
        self,
        conflicting_memories: List[str],
        repetitions: int = 3,
    ) -> InterleaveSchedule:
        """
        Create interleaved replay schedule for conflicting memories.

        Interleaving helps reduce interference during consolidation.
        """
        if len(conflicting_memories) < 2:
            return InterleaveSchedule(
                memories=conflicting_memories,
                pattern=list(range(len(conflicting_memories))),
                repetitions=repetitions,
            )

        # Create ABAB... pattern for 2 memories
        # Or ABCABC... for 3+
        n_memories = len(conflicting_memories)
        pattern = []

        for _ in range(repetitions):
            indices = list(range(n_memories))
            self._rng.shuffle(indices)
            pattern.extend(indices)

        schedule = InterleaveSchedule(
            memories=conflicting_memories,
            pattern=pattern,
            repetitions=repetitions,
        )

        # Store for tracking
        schedule_id = "_".join(conflicting_memories)
        self._interleave_schedules[schedule_id] = schedule

        return schedule

    def get_interference_risk(
        self,
        memory_id: str,
    ) -> float:
        """
        Get interference risk score for a memory.

        Higher score = more vulnerable to interference.
        """
        memory = self._memories.get(memory_id)
        if memory is None:
            return 0.0

        risk = 0.0
        n_similar = 0

        for other_id, other in self._memories.items():
            if other_id == memory_id:
                continue

            similarity = self.compute_similarity(memory_id, other_id)
            if similarity >= self.similarity_threshold:
                n_similar += 1
                # Risk increases with similarity and time proximity
                time_factor = 1.0 / (1.0 + abs(memory.encoding_time - other.encoding_time))
                strength_factor = 1.0 - memory.strength
                risk += similarity * time_factor * (1 + strength_factor)

        # Normalize by number of similar memories
        if n_similar > 0:
            risk = risk / n_similar

        return min(1.0, risk)

    def get_vulnerable_memories(
        self,
        threshold: float = 0.5,
    ) -> List[Tuple[str, float]]:
        """Get memories with high interference risk."""
        vulnerable = []

        for memory_id in self._memories:
            risk = self.get_interference_risk(memory_id)
            if risk >= threshold:
                vulnerable.append((memory_id, risk))

        vulnerable.sort(key=lambda x: x[1], reverse=True)
        return vulnerable

    def clear_similarity_cache(self) -> None:
        """Clear the similarity cache."""
        self._similarity_cache = {}

    def remove_memory(self, memory_id: str) -> None:
        """Remove a memory from tracking."""
        if memory_id in self._memories:
            del self._memories[memory_id]
            # Clear related cache entries
            self._similarity_cache = {
                k: v for k, v in self._similarity_cache.items() if memory_id not in k
            }

    def statistics(self) -> Dict[str, Any]:
        """Get interference resolution statistics."""
        if not self._memories:
            return {
                "total_memories": 0,
                "n_detections": self._n_detections,
            }

        # Count active interference events
        active_events = [e for e in self._events if not e.resolved]
        resolved_events = [e for e in self._events if e.resolved]

        # Strategy distribution
        strategy_counts = {}
        for event in self._events:
            strategy = event.resolution_strategy.value
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

        # Type distribution
        type_counts = {}
        for event in self._events:
            t = event.interference_type.value
            type_counts[t] = type_counts.get(t, 0) + 1

        # Average similarity of interference events
        avg_similarity = np.mean([e.similarity for e in self._events]) if self._events else 0.0

        # Vulnerable memories
        vulnerable = self.get_vulnerable_memories(threshold=0.5)

        return {
            "total_memories": len(self._memories),
            "n_detections": self._n_detections,
            "n_resolutions": self._n_resolutions,
            "n_successful": self._n_successful,
            "success_rate": (
                self._n_successful / self._n_resolutions if self._n_resolutions > 0 else 0.0
            ),
            "active_interference_events": len(active_events),
            "resolved_events": len(resolved_events),
            "strategy_distribution": strategy_counts,
            "type_distribution": type_counts,
            "average_similarity": float(avg_similarity),
            "n_vulnerable_memories": len(vulnerable),
            "active_interleave_schedules": len(self._interleave_schedules),
        }


__all__ = [
    "ResolutionStrategy",
    "InterferenceType",
    "MemoryVector",
    "InterferenceEvent",
    "InterleaveSchedule",
    "InterferenceResolver",
]
