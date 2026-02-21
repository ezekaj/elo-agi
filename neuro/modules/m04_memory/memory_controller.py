"""
Memory Controller: Orchestrates the multi-store pipeline

Coordinates:
- Sensory → Working Memory → Long-Term Memory flow
- Attention gating
- Consolidation cycles
- Unified recall interface
"""

import time
from typing import Optional, Any, List, Dict, Union
import numpy as np

from .sensory_memory import IconicBuffer, EchoicBuffer
from .working_memory import WorkingMemory
from .long_term_memory import EpisodicMemory, SemanticMemory, ProceduralMemory
from .memory_processes import (
    MemoryEncoder,
    MemoryConsolidator,
    MemoryRetriever,
    MemoryReconsolidator,
    Forgetter
)


class MemoryController:
    """
    Orchestrates the multi-store memory pipeline.

    Implements the flow:
    Input → Sensory Memory → (attention) → Working Memory → (consolidation) → Long-Term Memory
    """

    def __init__(
        self,
        wm_capacity: int = 7,
        iconic_decay: float = 0.250,
        echoic_decay: float = 3.5
    ):
        """
        Initialize memory controller with all subsystems.

        Args:
            wm_capacity: Working memory capacity (5-9)
            iconic_decay: Iconic buffer decay time in seconds
            echoic_decay: Echoic buffer decay time in seconds
        """
        # Sensory memory
        self.iconic = IconicBuffer(decay_time=iconic_decay)
        self.echoic = EchoicBuffer(decay_time=echoic_decay)

        # Working memory
        self.working = WorkingMemory(capacity=wm_capacity)

        # Long-term memory stores
        self.episodic = EpisodicMemory()
        self.semantic = SemanticMemory()
        self.procedural = ProceduralMemory()

        # Memory processes
        self.encoder = MemoryEncoder()
        self.consolidator = MemoryConsolidator()
        self.retriever = MemoryRetriever()
        self.reconsolidator = MemoryReconsolidator()
        self.forgetter = Forgetter()

        # Time tracking
        self._time_fn = time.time
        self._last_tick = self._time_fn()

    def process_visual(self, visual_input: np.ndarray) -> None:
        """
        Process visual input into sensory memory.

        Args:
            visual_input: Visual data
        """
        self.iconic.capture(visual_input)

    def process_auditory(self, audio_input: np.ndarray) -> None:
        """
        Process auditory input into sensory memory.

        Args:
            audio_input: Audio data
        """
        self.echoic.capture(audio_input)

    def attend(self, modality: str = "visual") -> Optional[Any]:
        """
        Transfer attended sensory item to working memory.

        Args:
            modality: "visual" or "auditory"

        Returns:
            The attended item, or None if nothing available
        """
        if modality == "visual":
            data = self.iconic.read()
            if data is not None:
                encoded = self.encoder.encode_visual(data)
                self.working.store(encoded)
                return encoded

        elif modality == "auditory":
            data = self.echoic.read_window()
            if data is not None:
                encoded = self.encoder.encode_auditory(data)
                self.working.store(encoded)
                return encoded

        return None

    def store_in_working_memory(self, item: Any) -> bool:
        """
        Directly store item in working memory.

        Args:
            item: Item to store

        Returns:
            True if stored
        """
        return self.working.store(item)

    def consolidate(self) -> int:
        """
        Transfer working memory contents to long-term storage.

        Simulates sleep consolidation cycle.

        Returns:
            Number of items consolidated
        """
        # Get well-rehearsed items from working memory
        items = self.working.get_rehearsed_items()

        # Queue for consolidation
        for item in items:
            self.consolidator.queue_for_consolidation(item)

        # Run replay cycle
        consolidated = self.consolidator.replay_cycle()

        return consolidated

    def encode_experience(
        self,
        experience: Any,
        context: Optional[Dict[str, Any]] = None,
        emotional_valence: float = 0.0
    ) -> Any:
        """
        Encode a new episodic memory.

        Args:
            experience: What happened
            context: Contextual information
            emotional_valence: Emotional coloring (-1 to +1)

        Returns:
            The created episode
        """
        episode = self.episodic.encode(
            experience=experience,
            context=context,
            emotional_valence=emotional_valence
        )

        # Also store in working memory for immediate access
        self.working.store(episode)

        return episode

    def learn_concept(
        self,
        name: str,
        features: Optional[Dict[str, float]] = None,
        relations: Optional[List[tuple]] = None
    ) -> Any:
        """
        Add a concept to semantic memory.

        Args:
            name: Concept name
            features: Attribute-value pairs
            relations: List of (relation_type, target, strength)

        Returns:
            The created concept
        """
        return self.semantic.create_concept(name, features, relations)

    def learn_skill(
        self,
        name: str,
        trigger_features: Dict[str, Any],
        action_names: List[str]
    ) -> Any:
        """
        Encode a new procedural memory.

        Args:
            name: Skill name
            trigger_features: Features that trigger the skill
            action_names: Sequence of action names

        Returns:
            The created procedure
        """
        return self.procedural.encode_simple(name, trigger_features, action_names)

    def recall(
        self,
        cue: Any,
        memory_types: Optional[List[str]] = None
    ) -> Optional[Any]:
        """
        Unified recall interface.

        Searches memory stores in order: working → episodic → semantic

        Args:
            cue: Query cue
            memory_types: List of store types to search (default: all)

        Returns:
            First matching memory, or None
        """
        if memory_types is None:
            memory_types = ["working", "episodic", "semantic"]

        for mem_type in memory_types:
            if mem_type == "working":
                result = self.working.retrieve(cue)
                if result is not None:
                    return result

            elif mem_type == "episodic":
                results = self.episodic.retrieve_by_cue(cue)
                if results:
                    return results[0]

            elif mem_type == "semantic":
                if isinstance(cue, str):
                    result = self.semantic.retrieve(cue)
                    if result is not None:
                        return result

        return None

    def recall_context(self, context: Dict[str, Any]) -> List[Any]:
        """
        Context-dependent recall from episodic memory.

        Args:
            context: Context features

        Returns:
            List of matching episodes
        """
        return self.episodic.retrieve_by_context(context)

    def recall_emotion(self, valence_min: float, valence_max: float) -> List[Any]:
        """
        Mood-congruent recall from episodic memory.

        Args:
            valence_min: Minimum emotional valence
            valence_max: Maximum emotional valence

        Returns:
            List of matching episodes
        """
        return self.episodic.retrieve_by_emotion(valence_min, valence_max)

    def execute_skill(self, stimulus: Dict[str, Any]) -> Optional[List[Any]]:
        """
        Execute matching procedural memory.

        Args:
            stimulus: Current stimulus features

        Returns:
            List of action results, or None
        """
        return self.procedural.execute(stimulus)

    def spread_concepts(self, start: str, depth: int = 3) -> Dict[str, float]:
        """
        Activate related concepts via spreading activation.

        Args:
            start: Starting concept
            depth: Propagation depth

        Returns:
            Dict of activated concepts with activation levels
        """
        return self.semantic.spread_activation(start, depth=depth)

    def update_memory(self, memory: Any, modification: Dict[str, Any]) -> Any:
        """
        Reconsolidate a memory with modifications.

        Args:
            memory: Memory to update
            modification: Changes to apply

        Returns:
            Updated memory
        """
        # Reactivate (makes labile)
        self.reconsolidator.reactivate(memory)

        # Modify
        updated = self.reconsolidator.modify(memory, modification)

        # Restabilize
        self.reconsolidator.restabilize(updated)

        return updated

    def forget(self, memory: Any, method: str = "decay") -> bool:
        """
        Apply forgetting to a memory.

        Args:
            memory: Memory to forget
            method: "decay", "suppress", or "prune"

        Returns:
            True if forgetting applied
        """
        if method == "decay":
            self.forgetter.decay(memory)
            return True
        elif method == "suppress":
            return self.forgetter.active_forget(memory)
        elif method == "prune" and hasattr(memory, 'engram'):
            self.forgetter.prune(memory.engram)
            return True
        return False

    def tick(self, dt: Optional[float] = None) -> None:
        """
        Advance time and apply decay to all memory systems.

        Args:
            dt: Time step in seconds (auto-computed if None)
        """
        current_time = self._time_fn()

        if dt is None:
            dt = current_time - self._last_tick

        self._last_tick = current_time

        # Decay sensory buffers
        self.iconic.decay(current_time)
        self.echoic.decay(current_time)

        # Decay working memory
        self.working.decay_step(dt)

        # Decay procedural memory (skills weaken without practice)
        self.procedural.decay_all(dt * 0.001)  # Slow decay

    def get_state(self) -> Dict[str, Any]:
        """
        Get current state of all memory systems.

        Returns:
            Dict with state information
        """
        return {
            "iconic_available": self.iconic.is_available(),
            "echoic_duration": self.echoic.get_duration(),
            "working_memory_load": self.working.get_load(),
            "working_memory_items": len(self.working),
            "episodic_count": len(self.episodic),
            "semantic_count": len(self.semantic),
            "procedural_count": len(self.procedural),
        }

    def clear_working_memory(self) -> None:
        """Clear working memory"""
        self.working.clear()

    def clear_sensory_buffers(self) -> None:
        """Clear sensory memory buffers"""
        self.iconic.clear()
        self.echoic.clear()

    def set_time_function(self, time_fn) -> None:
        """Set custom time function for all components"""
        self._time_fn = time_fn
        self._last_tick = time_fn()

        self.iconic.set_time_function(time_fn)
        self.echoic.set_time_function(time_fn)
        self.working.set_time_function(time_fn)
        self.episodic.set_time_function(time_fn)
        self.semantic.set_time_function(time_fn)
        self.procedural.set_time_function(time_fn)
        self.encoder.set_time_function(time_fn)
        self.consolidator.set_time_function(time_fn)
        self.reconsolidator.set_time_function(time_fn)
        self.forgetter.set_time_function(time_fn)
