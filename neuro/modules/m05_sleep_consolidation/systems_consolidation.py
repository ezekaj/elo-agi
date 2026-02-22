"""
Systems Consolidation: Hippocampal-Cortical Memory Transfer

Implements the Active Systems Consolidation model:
- Fast hippocampal encoding → slow cortical consolidation
- Hippocampal-cortical dialogue during SWS
- Memory transformation (gist extraction, schema integration)
"""

import numpy as np
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field
from .memory_replay import MemoryTrace, MemoryType


@dataclass
class StoredMemory:
    """Memory representation in a store (hippocampus or cortex)"""

    trace: MemoryTrace
    consolidation_level: float = 0.0  # 0-1, how consolidated
    schema_integration: float = 0.0  # 0-1, how integrated with schemas
    abstraction_level: float = 0.0  # 0-1, how abstract (gist vs detail)


class HippocampalStore:
    """Fast-learning hippocampal memory system.

    Rapidly encodes new memories with rich episodic detail.
    Acts as temporary storage before cortical consolidation.
    """

    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.memories: Dict[int, StoredMemory] = {}
        self._next_id = 0

    def encode(self, trace: MemoryTrace) -> int:
        """Rapidly encode a new memory.

        Args:
            trace: Memory trace to encode

        Returns:
            Memory ID
        """
        trace.hippocampal_index = True
        memory = StoredMemory(
            trace=trace,
            consolidation_level=0.0,  # Not yet consolidated
            abstraction_level=0.0,  # Full detail initially
        )

        memory_id = self._next_id
        self.memories[memory_id] = memory
        self._next_id += 1

        # Handle capacity limits (oldest memories fade)
        if len(self.memories) > self.capacity:
            self._remove_weakest()

        return memory_id

    def retrieve(self, memory_id: int) -> Optional[StoredMemory]:
        """Retrieve a memory by ID."""
        return self.memories.get(memory_id)

    def get_unconsolidated(self, threshold: float = 0.5) -> List[StoredMemory]:
        """Get memories that need consolidation.

        Args:
            threshold: Consolidation level below which memories are returned

        Returns:
            List of memories needing consolidation
        """
        return [mem for mem in self.memories.values() if mem.consolidation_level < threshold]

    def update_consolidation(self, memory_id: int, amount: float) -> None:
        """Update consolidation level for a memory.

        Args:
            memory_id: ID of memory to update
            amount: Amount to increase consolidation
        """
        if memory_id in self.memories:
            mem = self.memories[memory_id]
            mem.consolidation_level = min(1.0, mem.consolidation_level + amount)

    def mark_transferred(self, memory_id: int) -> None:
        """Mark a memory as transferred to cortex."""
        if memory_id in self.memories:
            self.memories[memory_id].trace.cortical_index = True

    def _remove_weakest(self) -> None:
        """Remove weakest memory when capacity exceeded."""
        if not self.memories:
            return

        # Find memory with lowest strength * (1 - consolidation)
        weakest_id = min(
            self.memories.keys(),
            key=lambda k: (
                self.memories[k].trace.strength * (1 - self.memories[k].consolidation_level)
            ),
        )
        del self.memories[weakest_id]

    def get_memory_count(self) -> int:
        """Get number of stored memories."""
        return len(self.memories)


class CorticalStore:
    """Slow-learning cortical memory system.

    Gradually acquires memories from hippocampus.
    Stores more abstract, schema-integrated representations.
    """

    def __init__(self):
        self.memories: Dict[int, StoredMemory] = {}
        self.schemas: Dict[str, np.ndarray] = {}  # Named schemas
        self._next_id = 0

    def receive_transfer(self, memory: StoredMemory, abstraction: float = 0.0) -> int:
        """Receive a memory transferred from hippocampus.

        Args:
            memory: Memory being transferred
            abstraction: How much to abstract (0 = full detail, 1 = pure gist)

        Returns:
            Memory ID in cortical store
        """
        # Create cortical version (potentially more abstract)
        cortical_memory = StoredMemory(
            trace=MemoryTrace(
                content=self._abstract_content(memory.trace.content, abstraction),
                strength=memory.trace.strength,
                hippocampal_index=memory.trace.hippocampal_index,
                cortical_index=True,
                encoding_time=memory.trace.encoding_time,
                emotional_salience=memory.trace.emotional_salience,
                memory_type=memory.trace.memory_type,
                context=memory.trace.context,
            ),
            consolidation_level=memory.consolidation_level,
            abstraction_level=abstraction,
        )

        memory_id = self._next_id
        self.memories[memory_id] = cortical_memory
        self._next_id += 1

        return memory_id

    def _abstract_content(self, content: np.ndarray, abstraction: float) -> np.ndarray:
        """Apply abstraction to memory content.

        Higher abstraction = more smoothing/averaging.
        """
        if abstraction <= 0:
            return content.copy()

        # Simulate gist extraction via low-pass filtering
        from scipy.ndimage import gaussian_filter1d

        sigma = abstraction * 2  # More abstraction = more smoothing
        return gaussian_filter1d(content.astype(float), sigma=max(0.1, sigma))

    def integrate_with_schema(self, memory_id: int, schema_name: str) -> float:
        """Integrate a memory with an existing schema.

        Args:
            memory_id: Memory to integrate
            schema_name: Name of schema to integrate with

        Returns:
            Integration score (similarity to schema)
        """
        if memory_id not in self.memories:
            return 0.0

        if schema_name not in self.schemas:
            return 0.0

        memory = self.memories[memory_id]
        schema = self.schemas[schema_name]

        # Compute similarity
        mem_norm = np.linalg.norm(memory.trace.content)
        schema_norm = np.linalg.norm(schema)
        if mem_norm < 1e-8 or schema_norm < 1e-8:
            return 0.0

        similarity = np.dot(memory.trace.content, schema) / (mem_norm * schema_norm)

        # Update schema integration level
        memory.schema_integration = max(memory.schema_integration, similarity)

        return similarity

    def add_schema(self, name: str, pattern: np.ndarray) -> None:
        """Add or update a schema.

        Args:
            name: Schema name
            pattern: Schema pattern
        """
        self.schemas[name] = pattern.copy()

    def extract_schema(self, memories: List[StoredMemory]) -> np.ndarray:
        """Extract a schema from multiple memories.

        Schema = commonalities across memories (statistical regularities).

        Args:
            memories: Memories to extract schema from

        Returns:
            Extracted schema pattern
        """
        if not memories:
            return np.array([])

        # Schema = mean of memory patterns
        patterns = [m.trace.content for m in memories]

        # Ensure same shape
        if len(set(p.shape for p in patterns)) > 1:
            return np.array([])

        return np.mean(patterns, axis=0)

    def retrieve(self, memory_id: int) -> Optional[StoredMemory]:
        """Retrieve a memory by ID."""
        return self.memories.get(memory_id)

    def get_memory_count(self) -> int:
        """Get number of stored memories."""
        return len(self.memories)


@dataclass
class ConsolidationWindow:
    """Tracks the optimal window for consolidation.

    Consolidation is maximized when slow oscillations, spindles,
    and ripples are temporally aligned.
    """

    slow_oscillation_phase: str = "down"  # "up" or "down"
    spindle_present: bool = False
    ripple_present: bool = False
    timestamp: float = 0.0

    def is_optimal(self) -> bool:
        """Check if this is an optimal consolidation window.

        Optimal = up-state of slow oscillation + spindle + ripple
        """
        return self.slow_oscillation_phase == "up" and self.spindle_present and self.ripple_present

    def get_consolidation_boost(self) -> float:
        """Get consolidation boost factor based on current state."""
        boost = 1.0

        if self.slow_oscillation_phase == "up":
            boost *= 1.5

        if self.spindle_present:
            boost *= 1.3

        if self.ripple_present:
            boost *= 1.4

        return boost


class MemoryTransformation:
    """Transforms memories during consolidation.

    Memories change during consolidation:
    - Gist extraction: semantic core preserved, details fade
    - Schema integration: fit to existing knowledge
    - Abstraction: increase generality over time
    """

    def __init__(self, gist_extraction_rate: float = 0.1, schema_weight: float = 0.3):
        self.gist_extraction_rate = gist_extraction_rate
        self.schema_weight = schema_weight

    def extract_gist(self, memory: StoredMemory, amount: float = 0.1) -> np.ndarray:
        """Extract semantic gist from memory.

        Gist = core meaning without episodic details.

        Args:
            memory: Memory to extract gist from
            amount: How much to extract (0-1)

        Returns:
            Gist representation
        """
        content = memory.trace.content

        # Gist extraction = keep strongest features
        threshold = np.percentile(np.abs(content), (1 - amount) * 100)
        gist = content.copy()
        gist[np.abs(gist) < threshold] = 0

        return gist

    def integrate_with_schema(self, memory: StoredMemory, schema: np.ndarray) -> np.ndarray:
        """Integrate memory with existing schema.

        Memory is pulled toward schema structure.

        Args:
            memory: Memory to integrate
            schema: Schema to integrate with

        Returns:
            Integrated memory content
        """
        content = memory.trace.content

        if content.shape != schema.shape:
            return content

        # Blend memory with schema
        integrated = (1 - self.schema_weight) * content + self.schema_weight * schema

        return integrated

    def increase_abstraction(self, memory: StoredMemory, amount: float = 0.1) -> None:
        """Increase abstraction level of memory.

        Higher abstraction = more generalized representation.

        Args:
            memory: Memory to abstract
            amount: Amount to increase abstraction
        """
        memory.abstraction_level = min(1.0, memory.abstraction_level + amount)


class HippocampalCorticalDialogue:
    """Coordinates memory transfer between hippocampus and cortex.

    Implements the Active Systems Consolidation model:
    1. Hippocampus rapidly encodes new experiences
    2. During SWS, hippocampus replays memories
    3. Cortex gradually learns from replayed memories
    4. Eventually cortex can support retrieval independently
    """

    def __init__(
        self,
        hippocampus: Optional[HippocampalStore] = None,
        cortex: Optional[CorticalStore] = None,
        transfer_threshold: float = 0.7,
    ):
        self.hippocampus = hippocampus or HippocampalStore()
        self.cortex = cortex or CorticalStore()
        self.transfer_threshold = transfer_threshold

        # Consolidation tracking
        self.consolidation_window = ConsolidationWindow()
        self.transformer = MemoryTransformation()

        # Statistics
        self.memories_transferred = 0
        self.total_consolidation_events = 0

    def initiate_dialogue(
        self, slow_osc_phase: str = "up", spindle: bool = False, ripple: bool = False
    ) -> float:
        """Update consolidation window state.

        Args:
            slow_osc_phase: Phase of slow oscillation
            spindle: Whether spindle is present
            ripple: Whether ripple is present

        Returns:
            Consolidation boost factor
        """
        self.consolidation_window.slow_oscillation_phase = slow_osc_phase
        self.consolidation_window.spindle_present = spindle
        self.consolidation_window.ripple_present = ripple

        return self.consolidation_window.get_consolidation_boost()

    def consolidate_memory(self, memory_id: int, replay_strength: float = 1.0) -> bool:
        """Consolidate a single memory.

        Args:
            memory_id: Hippocampal memory ID
            replay_strength: Strength of the replay event

        Returns:
            True if memory was consolidated
        """
        memory = self.hippocampus.retrieve(memory_id)
        if memory is None:
            return False

        # Get consolidation boost from current window
        boost = self.consolidation_window.get_consolidation_boost()

        # Update consolidation level
        consolidation_amount = 0.1 * replay_strength * boost
        self.hippocampus.update_consolidation(memory_id, consolidation_amount)

        # Extract gist during consolidation
        if memory.consolidation_level > 0.3:
            self.transformer.extract_gist(memory, amount=0.05 * boost)

        # Check if ready for cortical transfer
        if memory.consolidation_level >= self.transfer_threshold:
            self.transfer_memory(memory_id)

        self.total_consolidation_events += 1
        return True

    def transfer_memory(self, memory_id: int) -> Optional[int]:
        """Transfer memory from hippocampus to cortex.

        Args:
            memory_id: Hippocampal memory ID

        Returns:
            Cortical memory ID if transferred, None otherwise
        """
        memory = self.hippocampus.retrieve(memory_id)
        if memory is None:
            return None

        if memory.trace.cortical_index:
            return None  # Already transferred

        # Transfer with abstraction based on consolidation level
        abstraction = memory.consolidation_level * 0.5
        cortical_id = self.cortex.receive_transfer(memory, abstraction)

        # Mark hippocampal copy as transferred
        self.hippocampus.mark_transferred(memory_id)

        self.memories_transferred += 1
        return cortical_id

    def run_consolidation_cycle(
        self, duration_seconds: float, oscillation_events: Optional[List[Dict]] = None
    ) -> Dict:
        """Run a consolidation cycle (e.g., during one slow oscillation).

        Args:
            duration_seconds: Duration of the cycle
            oscillation_events: List of oscillation events with timing

        Returns:
            Statistics about the consolidation cycle
        """
        if oscillation_events is None:
            oscillation_events = []

        consolidated_count = 0
        transferred_count = 0

        # Get memories needing consolidation
        to_consolidate = self.hippocampus.get_unconsolidated()

        for memory in to_consolidate:
            # Find memory ID
            for mid, stored in self.hippocampus.memories.items():
                if stored is memory:
                    if self.consolidate_memory(mid):
                        consolidated_count += 1
                        if memory.consolidation_level >= self.transfer_threshold:
                            transferred_count += 1
                    break

        return {
            "consolidated": consolidated_count,
            "transferred": transferred_count,
            "duration": duration_seconds,
        }

    def update_indices(self) -> None:
        """Update hippocampal/cortical indices for all memories."""
        for memory in self.hippocampus.memories.values():
            if memory.consolidation_level >= self.transfer_threshold:
                memory.trace.cortical_index = True

    def get_statistics(self) -> Dict:
        """Get consolidation statistics."""
        return {
            "hippocampal_memories": self.hippocampus.get_memory_count(),
            "cortical_memories": self.cortex.get_memory_count(),
            "total_transferred": self.memories_transferred,
            "total_consolidation_events": self.total_consolidation_events,
        }

    def reset(self) -> None:
        """Reset the dialogue system."""
        self.hippocampus = HippocampalStore()
        self.cortex = CorticalStore()
        self.consolidation_window = ConsolidationWindow()
        self.memories_transferred = 0
        self.total_consolidation_events = 0
