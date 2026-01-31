"""
Hippocampal Memory Replay System

Implements memory reactivation during sleep:
- Memories replay in compressed time (~20x faster than real-time)
- Sharp-wave ripples coordinate replay timing
- Recent and emotional memories prioritized
"""

import numpy as np
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass, field
from enum import Enum


class MemoryType(Enum):
    """Types of memories"""
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


@dataclass
class MemoryTrace:
    """Single memory representation in the hippocampus.

    Memories are stored as activation patterns that can be
    replayed during sleep for consolidation.
    """
    content: np.ndarray                    # Memory encoding (pattern)
    strength: float = 1.0                  # Connection strength
    hippocampal_index: bool = True         # Indexed in hippocampus
    cortical_index: bool = False           # Transferred to cortex
    encoding_time: float = 0.0             # When originally encoded
    last_replay: float = 0.0               # Time of last reactivation
    replay_count: int = 0                  # Number of replays
    emotional_salience: float = 0.0        # -1 to 1 (negative to positive)
    memory_type: MemoryType = MemoryType.EPISODIC
    learning_complete: float = 0.0         # 0-1, how well learned

    # Context information
    context: Optional[Dict] = None

    def similarity_to(self, other: 'MemoryTrace') -> float:
        """Compute similarity to another memory trace."""
        if self.content.shape != other.content.shape:
            return 0.0
        norm_self = np.linalg.norm(self.content)
        norm_other = np.linalg.norm(other.content)
        if norm_self < 1e-8 or norm_other < 1e-8:
            return 0.0
        return float(np.dot(self.content, other.content) / (norm_self * norm_other))


class ReplayPrioritizer:
    """Determines which memories to replay based on research criteria.

    Selection criteria:
    - Recent experiences prioritized
    - Emotional salience increases replay probability
    - Incomplete learning triggers more replay
    """

    def __init__(
        self,
        recency_weight: float = 0.4,
        emotion_weight: float = 0.3,
        incompleteness_weight: float = 0.3
    ):
        self.recency_weight = recency_weight
        self.emotion_weight = emotion_weight
        self.incompleteness_weight = incompleteness_weight

    def compute_priority(
        self,
        memory: MemoryTrace,
        current_time: float
    ) -> float:
        """Compute replay priority for a memory.

        Args:
            memory: Memory trace to evaluate
            current_time: Current simulation time

        Returns:
            Priority score (higher = more likely to replay)
        """
        # Recency: more recent = higher priority
        age = current_time - memory.encoding_time
        recency_score = np.exp(-age / 100.0)  # Decay with time

        # Emotional salience: absolute value matters
        emotion_score = np.abs(memory.emotional_salience)

        # Incompleteness: less learned = needs more replay
        incompleteness_score = 1.0 - memory.learning_complete

        priority = (
            self.recency_weight * recency_score +
            self.emotion_weight * emotion_score +
            self.incompleteness_weight * incompleteness_score
        )

        return priority

    def select_for_replay(
        self,
        memories: List[MemoryTrace],
        current_time: float,
        n_select: int = 5
    ) -> List[Tuple[MemoryTrace, float]]:
        """Select memories for replay based on priority.

        Args:
            memories: All available memories
            current_time: Current simulation time
            n_select: Number of memories to select

        Returns:
            List of (memory, priority) tuples, sorted by priority
        """
        if not memories:
            return []

        # Compute priorities
        priorities = [
            (mem, self.compute_priority(mem, current_time))
            for mem in memories
            if mem.hippocampal_index  # Only replay hippocampal memories
        ]

        # Sort by priority
        priorities.sort(key=lambda x: x[1], reverse=True)

        return priorities[:n_select]


class HippocampalReplay:
    """Reactivates memories during sleep at accelerated rate.

    Key features:
    - Compression factor ~20x (replay faster than real-time)
    - Coordination with sharp-wave ripples
    - Strengthens memories through replay
    """

    def __init__(
        self,
        compression_factor: float = 20.0,
        replay_strength_boost: float = 0.1,
        learning_rate: float = 0.05
    ):
        """Initialize hippocampal replay system.

        Args:
            compression_factor: How much faster than real-time (default 20x)
            replay_strength_boost: How much replay strengthens memory
            learning_rate: Rate of learning completion improvement
        """
        self.compression_factor = compression_factor
        self.replay_strength_boost = replay_strength_boost
        self.learning_rate = learning_rate

        # Memory storage
        self.memory_traces: List[MemoryTrace] = []

        # Prioritizer for selection
        self.prioritizer = ReplayPrioritizer()

        # Replay statistics
        self.total_replays = 0
        self.current_time = 0.0

        # Sharp-wave ripple coordination
        self.ripple_active = False
        self.ripple_start_time = 0.0

    def add_memory(self, memory: MemoryTrace) -> None:
        """Add a memory trace to the replay system."""
        memory.encoding_time = self.current_time
        self.memory_traces.append(memory)

    def encode_experience(
        self,
        pattern: np.ndarray,
        emotional_salience: float = 0.0,
        context: Optional[Dict] = None,
        memory_type: MemoryType = MemoryType.EPISODIC
    ) -> MemoryTrace:
        """Encode a new experience as a memory trace.

        Args:
            pattern: Neural activation pattern
            emotional_salience: Emotional significance (-1 to 1)
            context: Contextual information
            memory_type: Type of memory

        Returns:
            The created memory trace
        """
        memory = MemoryTrace(
            content=pattern.copy(),
            strength=1.0,
            hippocampal_index=True,
            cortical_index=False,
            encoding_time=self.current_time,
            emotional_salience=emotional_salience,
            memory_type=memory_type,
            context=context
        )
        self.add_memory(memory)
        return memory

    def select_for_replay(self, n_select: int = 5) -> List[MemoryTrace]:
        """Select memories for replay based on priority.

        Args:
            n_select: Number of memories to select

        Returns:
            List of selected memories
        """
        selected = self.prioritizer.select_for_replay(
            self.memory_traces,
            self.current_time,
            n_select
        )
        return [mem for mem, _ in selected]

    def replay_memory(
        self,
        memory: MemoryTrace,
        ripple_present: bool = False
    ) -> np.ndarray:
        """Replay a single memory in compressed time.

        Args:
            memory: Memory to replay
            ripple_present: Whether sharp-wave ripple is active

        Returns:
            Replayed activation pattern (may be slightly modified)
        """
        # Strength boost is larger if ripple is present
        boost_multiplier = 1.5 if ripple_present else 1.0

        # Update memory statistics
        memory.last_replay = self.current_time
        memory.replay_count += 1
        memory.strength += self.replay_strength_boost * boost_multiplier

        # Improve learning completion
        memory.learning_complete = min(
            1.0,
            memory.learning_complete + self.learning_rate * boost_multiplier
        )

        self.total_replays += 1

        # Return the replayed pattern (could add noise for variation)
        return memory.content.copy()

    def replay_sequence(
        self,
        memories: List[MemoryTrace],
        ripple_times: Optional[List[float]] = None
    ) -> List[np.ndarray]:
        """Replay a sequence of memories in compressed time.

        Args:
            memories: Memories to replay
            ripple_times: Optional list of times when ripples occur

        Returns:
            List of replayed patterns
        """
        if ripple_times is None:
            ripple_times = []

        replayed_patterns = []
        replay_duration = 0.0

        for i, memory in enumerate(memories):
            # Check if ripple is present
            ripple_present = any(
                abs(replay_duration - rt) < 0.05  # Within 50ms of ripple
                for rt in ripple_times
            )

            pattern = self.replay_memory(memory, ripple_present)
            replayed_patterns.append(pattern)

            # Advance replay time (compressed)
            replay_duration += 0.05 / self.compression_factor  # ~2.5ms per memory

        return replayed_patterns

    def coordinate_with_ripple(self, ripple_start: float) -> List[MemoryTrace]:
        """Coordinate replay with a sharp-wave ripple event.

        Sharp-wave ripples (~50ms) are ideal windows for replay.
        Multiple memories can be replayed during one ripple.

        Args:
            ripple_start: Start time of the ripple

        Returns:
            List of memories replayed during this ripple
        """
        self.ripple_active = True
        self.ripple_start_time = ripple_start

        # Select memories for this ripple
        memories = self.select_for_replay(n_select=3)

        # Replay during the ripple window
        for memory in memories:
            self.replay_memory(memory, ripple_present=True)

        self.ripple_active = False
        return memories

    def get_compressed_duration(self, real_duration: float) -> float:
        """Convert real duration to compressed replay duration.

        Args:
            real_duration: Duration in real time (seconds)

        Returns:
            Compressed duration
        """
        return real_duration / self.compression_factor

    def advance_time(self, dt: float) -> None:
        """Advance the simulation time.

        Args:
            dt: Time step
        """
        self.current_time += dt

    def get_memory_statistics(self) -> Dict:
        """Get statistics about stored memories."""
        if not self.memory_traces:
            return {
                "total_memories": 0,
                "mean_strength": 0.0,
                "mean_replay_count": 0.0,
                "hippocampal_count": 0,
                "cortical_count": 0,
            }

        return {
            "total_memories": len(self.memory_traces),
            "mean_strength": np.mean([m.strength for m in self.memory_traces]),
            "mean_replay_count": np.mean([m.replay_count for m in self.memory_traces]),
            "hippocampal_count": sum(1 for m in self.memory_traces if m.hippocampal_index),
            "cortical_count": sum(1 for m in self.memory_traces if m.cortical_index),
            "total_replays": self.total_replays,
        }

    def find_similar_memories(
        self,
        pattern: np.ndarray,
        threshold: float = 0.5
    ) -> List[Tuple[MemoryTrace, float]]:
        """Find memories similar to a given pattern.

        Args:
            pattern: Pattern to match
            threshold: Minimum similarity threshold

        Returns:
            List of (memory, similarity) tuples
        """
        query_memory = MemoryTrace(content=pattern)
        similar = []

        for memory in self.memory_traces:
            sim = query_memory.similarity_to(memory)
            if sim >= threshold:
                similar.append((memory, sim))

        similar.sort(key=lambda x: x[1], reverse=True)
        return similar

    def reset(self) -> None:
        """Reset the replay system."""
        self.memory_traces = []
        self.total_replays = 0
        self.current_time = 0.0
        self.ripple_active = False


class ReplayScheduler:
    """Schedules replay sessions across sleep stages.

    Different stages have different replay characteristics:
    - SWS: Most intense replay, with ripples
    - NREM2: Moderate replay, with spindles
    - REM: Emotional memory processing
    """

    def __init__(self, replay_system: HippocampalReplay):
        self.replay_system = replay_system

        # Stage-specific parameters
        self.sws_replays_per_minute = 10
        self.nrem2_replays_per_minute = 5
        self.rem_replays_per_minute = 2

    def run_sws_replay(
        self,
        duration_minutes: float,
        slow_oscillation_phase: str = "up"
    ) -> List[MemoryTrace]:
        """Run replay during slow-wave sleep.

        Most consolidation happens during SWS with ripple-coordinated replay.

        Args:
            duration_minutes: Duration of SWS period
            slow_oscillation_phase: "up" or "down" state

        Returns:
            All memories replayed during this period
        """
        all_replayed = []

        if slow_oscillation_phase != "up":
            # Replay mainly during up-states
            return all_replayed

        n_replays = int(duration_minutes * self.sws_replays_per_minute)

        for _ in range(n_replays):
            # Simulate ripple-coordinated replay
            memories = self.replay_system.coordinate_with_ripple(
                self.replay_system.current_time
            )
            all_replayed.extend(memories)
            self.replay_system.advance_time(60.0 / self.sws_replays_per_minute)

        return all_replayed

    def run_nrem2_replay(self, duration_minutes: float) -> List[MemoryTrace]:
        """Run replay during NREM2 sleep.

        Moderate replay associated with sleep spindles.
        """
        all_replayed = []
        n_replays = int(duration_minutes * self.nrem2_replays_per_minute)

        for _ in range(n_replays):
            memories = self.replay_system.select_for_replay(n_select=2)
            for mem in memories:
                self.replay_system.replay_memory(mem, ripple_present=False)
            all_replayed.extend(memories)
            self.replay_system.advance_time(60.0 / self.nrem2_replays_per_minute)

        return all_replayed

    def run_rem_replay(self, duration_minutes: float) -> List[MemoryTrace]:
        """Run replay during REM sleep.

        Focused on emotional memories and schema formation.
        """
        all_replayed = []
        n_replays = int(duration_minutes * self.rem_replays_per_minute)

        # Prioritize emotional memories during REM
        emotional_memories = [
            m for m in self.replay_system.memory_traces
            if np.abs(m.emotional_salience) > 0.3
        ]

        for _ in range(n_replays):
            if emotional_memories:
                # Prefer emotional memories
                mem = np.random.choice(emotional_memories)
            else:
                # Fall back to any memory
                selected = self.replay_system.select_for_replay(n_select=1)
                if selected:
                    mem = selected[0]
                else:
                    continue

            self.replay_system.replay_memory(mem, ripple_present=False)
            all_replayed.append(mem)
            self.replay_system.advance_time(60.0 / self.rem_replays_per_minute)

        return all_replayed
