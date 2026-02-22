"""
Memory Processes: The six core memory operations

Based on research defining:
- Encoding: Converting input to storable form
- Consolidation: Short-term → Long-term transfer
- Storage: Maintaining information
- Retrieval: Accessing stored information
- Reconsolidation: Updating reactivated memories
- Forgetting: Filtering irrelevant information
"""

import numpy as np
import time
from typing import Optional, Any, List, Dict, Tuple, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from .engram import Engram
    from .long_term_memory.episodic_memory import Episode


@dataclass
class EncodedMemory:
    """Result of encoding process"""

    pattern: np.ndarray
    modality: str
    timestamp: float
    content: Any
    metadata: Dict[str, Any] = None


class MemoryEncoder:
    """
    Converting input to storable neural patterns.

    Transforms various input modalities into patterns
    suitable for engram storage.
    """

    def __init__(self, pattern_size: int = 100):
        """
        Initialize encoder.

        Args:
            pattern_size: Size of output patterns
        """
        self.pattern_size = pattern_size
        self._time_fn = time.time

    def encode_visual(self, image: np.ndarray) -> EncodedMemory:
        """
        Encode visual input to pattern.

        Args:
            image: Visual data (any shape)

        Returns:
            EncodedMemory with visual pattern
        """
        # Flatten and normalize
        flat = image.flatten().astype(float)
        flat = (flat - flat.min()) / (flat.max() - flat.min() + 1e-8)

        # Resize to pattern size
        pattern = np.interp(
            np.linspace(0, 1, self.pattern_size), np.linspace(0, 1, len(flat)), flat
        )

        return EncodedMemory(
            pattern=pattern,
            modality="visual",
            timestamp=self._time_fn(),
            content=image,
            metadata={"shape": image.shape},
        )

    def encode_auditory(self, audio: np.ndarray) -> EncodedMemory:
        """
        Encode auditory input to pattern.

        Args:
            audio: Audio data

        Returns:
            EncodedMemory with auditory pattern
        """
        # Normalize
        flat = audio.flatten().astype(float)
        if flat.max() != flat.min():
            flat = (flat - flat.min()) / (flat.max() - flat.min())

        # Resize to pattern size
        pattern = np.interp(
            np.linspace(0, 1, self.pattern_size), np.linspace(0, 1, len(flat)), flat
        )

        return EncodedMemory(
            pattern=pattern,
            modality="auditory",
            timestamp=self._time_fn(),
            content=audio,
            metadata={"length": len(audio)},
        )

    def encode_semantic(self, text: str) -> EncodedMemory:
        """
        Encode text to pattern.

        Simple encoding based on character frequencies.
        In practice, would use embeddings.

        Args:
            text: Text content

        Returns:
            EncodedMemory with semantic pattern
        """
        # Simple bag-of-characters encoding
        pattern = np.zeros(self.pattern_size)

        for i, char in enumerate(text.lower()):
            idx = ord(char) % self.pattern_size
            pattern[idx] += 1

        # Normalize
        if pattern.max() > 0:
            pattern = pattern / pattern.max()

        return EncodedMemory(
            pattern=pattern,
            modality="semantic",
            timestamp=self._time_fn(),
            content=text,
            metadata={"length": len(text)},
        )

    def encode_episodic(self, experience: Any, context: Dict[str, Any]) -> EncodedMemory:
        """
        Encode full episode with context.

        Args:
            experience: What happened
            context: Contextual information

        Returns:
            EncodedMemory with episodic pattern
        """
        # Combine experience and context into pattern
        exp_str = str(experience)
        ctx_str = str(context)
        combined = exp_str + ctx_str

        # Hash-based encoding for consistent patterns
        pattern = np.zeros(self.pattern_size)
        for i, char in enumerate(combined):
            idx = (ord(char) * (i + 1)) % self.pattern_size
            pattern[idx] += 1

        # Normalize
        if pattern.max() > 0:
            pattern = pattern / pattern.max()

        return EncodedMemory(
            pattern=pattern,
            modality="episodic",
            timestamp=self._time_fn(),
            content=experience,
            metadata={"context": context},
        )

    def encode_procedural(self, trigger: Dict, actions: List) -> EncodedMemory:
        """
        Encode skill/procedure.

        Args:
            trigger: Trigger features
            actions: Action sequence

        Returns:
            EncodedMemory with procedural pattern
        """
        # Encode trigger and actions
        combined = str(trigger) + str(actions)

        pattern = np.zeros(self.pattern_size)
        for i, char in enumerate(combined):
            idx = (ord(char) * (i + 1)) % self.pattern_size
            pattern[idx] += 1

        if pattern.max() > 0:
            pattern = pattern / pattern.max()

        return EncodedMemory(
            pattern=pattern,
            modality="procedural",
            timestamp=self._time_fn(),
            content={"trigger": trigger, "actions": actions},
            metadata={"n_actions": len(actions)},
        )

    def set_time_function(self, time_fn) -> None:
        self._time_fn = time_fn


class MemoryConsolidator:
    """
    Short-term → Long-term transfer via hippocampal replay.

    Simulates sleep consolidation through replay cycles.
    """

    def __init__(self, consolidation_threshold: float = 0.5):
        """
        Initialize consolidator.

        Args:
            consolidation_threshold: Minimum strength for transfer
        """
        self.consolidation_threshold = consolidation_threshold
        self.replay_queue: List[Any] = []
        self._time_fn = time.time

    def queue_for_consolidation(self, memory: Any) -> None:
        """
        Add memory to replay queue.

        Args:
            memory: Memory to consolidate
        """
        self.replay_queue.append(memory)

    def replay_cycle(self, iterations: int = 3) -> int:
        """
        Simulate sleep consolidation.

        Replays queued memories to strengthen them.

        Args:
            iterations: Number of replay iterations per memory

        Returns:
            Number of memories consolidated
        """
        consolidated = 0

        for memory in self.replay_queue:
            for _ in range(iterations):
                # If memory has engram, consolidate it
                if hasattr(memory, "engram") and memory.engram is not None:
                    memory.engram.consolidate()

                # If memory has strength attribute, increase it
                if hasattr(memory, "strength"):
                    memory.strength = min(1.0, memory.strength + 0.1)

            consolidated += 1

        # Clear queue after consolidation
        self.replay_queue.clear()

        return consolidated

    def transfer_to_ltm(self, memory: Any, memory_store: Any) -> bool:
        """
        Move memory from WM to appropriate LTM store.

        Args:
            memory: Memory to transfer
            memory_store: Target LTM store (episodic, semantic, or procedural)

        Returns:
            True if transfer successful
        """
        if hasattr(memory_store, "encode"):
            # Episodic memory
            memory_store.encode(memory)
            return True
        elif hasattr(memory_store, "store"):
            # Semantic or procedural
            memory_store.store(memory)
            return True
        return False

    def interleave_replay(self, memories: List[Any]) -> None:
        """
        Replay memories in interleaved order (better consolidation).

        Args:
            memories: Memories to replay
        """
        # Shuffle for interleaved replay
        indices = np.random.permutation(len(memories))

        for idx in indices:
            memory = memories[idx]
            if hasattr(memory, "engram") and memory.engram is not None:
                memory.engram.consolidate()

    def set_time_function(self, time_fn) -> None:
        self._time_fn = time_fn


class MemoryRetriever:
    """
    Accessing stored information via pattern completion.

    Uses partial cues to reconstruct full memories.
    """

    def __init__(self, similarity_threshold: float = 0.3):
        """
        Initialize retriever.

        Args:
            similarity_threshold: Minimum similarity for match
        """
        self.similarity_threshold = similarity_threshold

    def retrieve(self, cue: Any, memory_store: Any) -> Optional[Any]:
        """
        Find matching memory in store.

        Args:
            cue: Query cue
            memory_store: Memory store to search

        Returns:
            Matching memory or None
        """
        # Try different retrieval methods based on store type
        if hasattr(memory_store, "retrieve_by_cue"):
            results = memory_store.retrieve_by_cue(cue)
            return results[0] if results else None

        if hasattr(memory_store, "retrieve"):
            return memory_store.retrieve(cue)

        return None

    def pattern_complete(self, partial_cue: np.ndarray, engram: "Engram") -> np.ndarray:
        """
        Fill in missing information from partial cue.

        Args:
            partial_cue: Incomplete pattern
            engram: Engram to use for completion

        Returns:
            Completed pattern
        """
        return engram.reactivate(partial_cue)

    def reconstruct(self, fragments: List[np.ndarray], engrams: List["Engram"]) -> np.ndarray:
        """
        Piece together memory from multiple sources.

        Args:
            fragments: Partial patterns
            engrams: Associated engrams

        Returns:
            Reconstructed pattern
        """
        if not fragments or not engrams:
            return np.array([])

        # Average completed patterns
        completed = []
        for frag, eng in zip(fragments, engrams):
            completed.append(eng.reactivate(frag))

        return np.mean(completed, axis=0)

    def context_reinstate(self, context: Dict[str, Any], memory_store: Any) -> List[Any]:
        """
        Use context to aid retrieval.

        Args:
            context: Context features
            memory_store: Memory store to search

        Returns:
            List of matching memories
        """
        if hasattr(memory_store, "retrieve_by_context"):
            return memory_store.retrieve_by_context(context)
        return []


class MemoryReconsolidator:
    """
    Updating reactivated memories.

    When a memory is retrieved, it enters a labile state
    where it can be modified before re-stabilizing.
    """

    def __init__(self, labile_window: float = 6.0):
        """
        Initialize reconsolidator.

        Args:
            labile_window: Hours that memory remains labile after retrieval
        """
        self.labile_window = labile_window * 3600  # Convert to seconds
        self._time_fn = time.time

    def reactivate(self, memory: Any) -> Any:
        """
        Bring memory into labile state.

        Args:
            memory: Memory to reactivate

        Returns:
            The reactivated memory
        """
        if hasattr(memory, "engram") and memory.engram is not None:
            memory.engram.destabilize()

        if hasattr(memory, "last_retrieval"):
            memory.last_retrieval = self._time_fn()

        return memory

    def modify(self, memory: Any, update: Dict[str, Any]) -> Any:
        """
        Change reactivated memory.

        Memory must be in labile state.

        Args:
            memory: Memory to modify
            update: Changes to apply

        Returns:
            Modified memory
        """
        # Check if memory is labile
        if hasattr(memory, "engram") and memory.engram is not None:
            if not memory.engram.is_labile():
                return memory  # Cannot modify consolidated memory

        # Apply updates
        if hasattr(memory, "content") and "content" in update:
            memory.content = update["content"]

        if hasattr(memory, "context") and "context" in update:
            memory.context.update(update["context"])

        if hasattr(memory, "emotional_valence") and "emotional_valence" in update:
            memory.emotional_valence = update["emotional_valence"]

        return memory

    def restabilize(self, memory: Any) -> None:
        """
        Re-consolidate modified memory.

        Args:
            memory: Memory to restabilize
        """
        if hasattr(memory, "engram") and memory.engram is not None:
            # Get current content as new pattern
            if hasattr(memory, "content"):
                # Re-encode content (simplified)
                content_str = str(memory.content)
                pattern = np.array([ord(c) for c in content_str[:100]])
                pattern = pattern / pattern.max() if pattern.max() > 0 else pattern
                memory.engram.restabilize(pattern)
            else:
                memory.engram.consolidate()

    def is_labile(self, memory: Any) -> bool:
        """
        Check if memory is in labile state.

        Args:
            memory: Memory to check

        Returns:
            True if memory can be modified
        """
        if hasattr(memory, "engram") and memory.engram is not None:
            return memory.engram.is_labile()

        # Check time-based lability
        if hasattr(memory, "last_retrieval") and memory.last_retrieval is not None:
            elapsed = self._time_fn() - memory.last_retrieval
            return elapsed < self.labile_window

        return False

    def set_time_function(self, time_fn) -> None:
        self._time_fn = time_fn


class Forgetter:
    """
    Filtering irrelevant information via synaptic pruning.

    Implements multiple forgetting mechanisms:
    - Decay: Gradual weakening over time
    - Interference: Competition between similar memories
    - Pruning: Removal of weak connections
    - Active forgetting: Intentional suppression
    """

    def __init__(self, decay_rate: float = 0.01, prune_threshold: float = 0.05):
        """
        Initialize forgetter.

        Args:
            decay_rate: Rate of passive decay
            prune_threshold: Threshold for pruning
        """
        self.decay_rate = decay_rate
        self.prune_threshold = prune_threshold
        self._time_fn = time.time

    def decay(self, memory: Any, rate: Optional[float] = None) -> float:
        """
        Gradual weakening over time.

        Args:
            memory: Memory to decay
            rate: Decay rate (uses default if None)

        Returns:
            New strength value
        """
        rate = rate or self.decay_rate

        if hasattr(memory, "strength"):
            memory.strength = max(0.0, memory.strength - rate)
            return memory.strength

        if hasattr(memory, "vividness"):
            memory.vividness = max(0.0, memory.vividness - rate)
            return memory.vividness

        return 0.0

    def interfere(self, old_memory: Any, new_memory: Any, similarity: float = 0.5) -> float:
        """
        Retroactive interference from similar new memory.

        Args:
            old_memory: Existing memory
            new_memory: New similar memory
            similarity: Similarity between memories (0-1)

        Returns:
            Interference strength
        """
        interference = similarity * 0.3  # Max 30% interference

        if hasattr(old_memory, "strength"):
            old_memory.strength = max(0.0, old_memory.strength - interference)

        if hasattr(old_memory, "vividness"):
            old_memory.vividness = max(0.0, old_memory.vividness - interference * 0.5)

        return interference

    def prune(self, engram: "Engram", threshold: Optional[float] = None) -> int:
        """
        Remove weak connections from engram.

        Args:
            engram: Engram to prune
            threshold: Pruning threshold (uses default if None)

        Returns:
            Number of connections pruned
        """
        threshold = threshold or self.prune_threshold
        return engram.prune(threshold)

    def active_forget(self, memory: Any, suppression_strength: float = 0.5) -> bool:
        """
        Intentional suppression of memory.

        Args:
            memory: Memory to suppress
            suppression_strength: How strongly to suppress (0-1)

        Returns:
            True if suppression applied
        """
        if hasattr(memory, "strength"):
            memory.strength *= 1 - suppression_strength
            return True

        if hasattr(memory, "engram") and memory.engram is not None:
            # Prune more aggressively
            memory.engram.prune(self.prune_threshold * 2)
            return True

        return False

    def retrieve_induced_forgetting(
        self, retrieved: Any, competitors: List[Any], rif_strength: float = 0.2
    ) -> int:
        """
        Recalling one memory weakens similar competing memories.

        Args:
            retrieved: Memory that was recalled
            competitors: Similar memories
            rif_strength: Strength of forgetting effect

        Returns:
            Number of competitors affected
        """
        affected = 0

        for comp in competitors:
            if comp is not retrieved:
                if hasattr(comp, "strength"):
                    comp.strength = max(0.0, comp.strength - rif_strength)
                    affected += 1

        return affected

    def set_time_function(self, time_fn) -> None:
        self._time_fn = time_fn
