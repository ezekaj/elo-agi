"""
Dream Generator

Dreams as a by-product of memory consolidation:
- Multiple memories reactivate simultaneously for consolidation
- A small subset assembles into conscious narrative
- Creative assembly produces the dream "story"

Key insight: Dreams REFLECT what's being consolidated, not the other way around.
"""

import numpy as np
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .memory_replay import MemoryTrace, MemoryType


class DreamEmotionTone(Enum):
    """Emotional tone of dreams"""

    NEUTRAL = "neutral"
    POSITIVE = "positive"
    NEGATIVE = "negative"
    ANXIOUS = "anxious"
    BIZARRE = "bizarre"


@dataclass
class DreamElement:
    """Single element/fragment in a dream"""

    source_memory: MemoryTrace
    content_fragment: np.ndarray
    vividness: float  # 0-1, how vivid this element is
    distortion: float  # 0-1, how distorted from original
    emotional_contribution: float


@dataclass
class DreamReport:
    """Output of dream generation - the "dream" itself"""

    source_memories: List[MemoryTrace]
    elements: List[DreamElement]
    narrative_coherence: float  # 0-1, how coherent the narrative
    emotional_tone: DreamEmotionTone
    bizarreness_index: float  # 0-1, how bizarre
    duration: float  # Subjective dream duration
    timestamp: float  # When dream occurred

    def get_summary(self) -> Dict:
        """Get summary statistics."""
        return {
            "n_source_memories": len(self.source_memories),
            "n_elements": len(self.elements),
            "coherence": self.narrative_coherence,
            "tone": self.emotional_tone.value,
            "bizarreness": self.bizarreness_index,
            "duration": self.duration,
        }


class NarrativeAssembler:
    """Creates dream narrative from memory fragments.

    The brain attempts to create a coherent story from
    simultaneously activated memory fragments.
    """

    def __init__(self, coherence_threshold: float = 0.3, association_strength: float = 0.5):
        self.coherence_threshold = coherence_threshold
        self.association_strength = association_strength

    def find_associations(self, memories: List[MemoryTrace]) -> List[Tuple[int, int, float]]:
        """Find associations between memories.

        Args:
            memories: List of memory traces

        Returns:
            List of (idx1, idx2, strength) tuples
        """
        associations = []

        for i, mem1 in enumerate(memories):
            for j, mem2 in enumerate(memories[i + 1 :], i + 1):
                # Compute similarity
                sim = mem1.similarity_to(mem2)

                # Check for shared emotional tone
                emotion_match = np.sign(mem1.emotional_salience) == np.sign(mem2.emotional_salience)

                # Stronger association if similar or same emotion
                strength = sim
                if emotion_match:
                    strength *= 1.3

                if strength > 0.1:
                    associations.append((i, j, strength))

        return associations

    def generate_visual_scene(
        self, memories: List[MemoryTrace], blend_weights: Optional[List[float]] = None
    ) -> np.ndarray:
        """Generate visual scene from memory blend.

        Dreams have vivid imagery generated from memory combinations.

        Args:
            memories: Memories to blend
            blend_weights: Optional weights for blending

        Returns:
            Blended "visual" pattern
        """
        if not memories:
            return np.array([])

        if blend_weights is None:
            blend_weights = [1.0 / len(memories)] * len(memories)

        # Normalize weights
        total = sum(blend_weights)
        blend_weights = [w / total for w in blend_weights]

        # Find common shape
        shapes = [m.content.shape for m in memories]
        if len(set(shapes)) > 1:
            # Different shapes - use first memory's shape
            target_shape = shapes[0]
            contents = []
            for m in memories:
                if m.content.shape == target_shape:
                    contents.append(m.content)
                else:
                    # Resize or pad
                    resized = np.resize(m.content, target_shape)
                    contents.append(resized)
        else:
            contents = [m.content for m in memories]

        # Blend memories
        blended = np.zeros_like(contents[0], dtype=float)
        for content, weight in zip(contents, blend_weights):
            blended += weight * content

        return blended

    def fill_gaps(
        self, elements: List[DreamElement], gap_fill_creativity: float = 0.5
    ) -> List[DreamElement]:
        """Fill narrative gaps with creative content.

        Dreams often have gaps filled by the brain to create coherence.

        Args:
            elements: Existing dream elements
            gap_fill_creativity: How creative the gap filling is

        Returns:
            Elements with gaps filled (may add new elements)
        """
        if len(elements) < 2:
            return elements

        filled = []
        for i, elem in enumerate(elements):
            filled.append(elem)

            # Check for gap to next element
            if i < len(elements) - 1:
                next_elem = elements[i + 1]

                # Compute similarity
                current_content = elem.content_fragment
                next_content = next_elem.content_fragment

                if current_content.shape == next_content.shape:
                    similarity = np.corrcoef(current_content.flatten(), next_content.flatten())[
                        0, 1
                    ]

                    if np.isnan(similarity):
                        similarity = 0

                    # Add bridge element if gap is large
                    if similarity < 0.3:
                        bridge = self._create_bridge_element(elem, next_elem, gap_fill_creativity)
                        if bridge is not None:
                            filled.append(bridge)

        return filled

    def _create_bridge_element(
        self, elem1: DreamElement, elem2: DreamElement, creativity: float
    ) -> Optional[DreamElement]:
        """Create a bridging element between two elements."""
        if elem1.content_fragment.shape != elem2.content_fragment.shape:
            return None

        # Interpolate between elements
        bridge_content = (elem1.content_fragment + elem2.content_fragment) / 2

        # Add creative noise
        noise = np.random.randn(*bridge_content.shape) * creativity * 0.1
        bridge_content = bridge_content + noise

        # Create a "synthetic" memory for the bridge
        bridge_memory = MemoryTrace(
            content=bridge_content,
            strength=0.5,
            emotional_salience=(elem1.emotional_contribution + elem2.emotional_contribution) / 2,
        )

        return DreamElement(
            source_memory=bridge_memory,
            content_fragment=bridge_content,
            vividness=(elem1.vividness + elem2.vividness) / 2,
            distortion=0.7,  # Bridge elements are more distorted
            emotional_contribution=(elem1.emotional_contribution + elem2.emotional_contribution)
            / 2,
        )

    def compute_coherence(self, elements: List[DreamElement]) -> float:
        """Compute narrative coherence of dream elements.

        Args:
            elements: Dream elements

        Returns:
            Coherence score (0-1)
        """
        if len(elements) < 2:
            return 1.0

        # Coherence = average similarity between consecutive elements
        similarities = []
        for i in range(len(elements) - 1):
            c1 = elements[i].content_fragment
            c2 = elements[i + 1].content_fragment

            if c1.shape == c2.shape:
                sim = np.corrcoef(c1.flatten(), c2.flatten())[0, 1]
                if not np.isnan(sim):
                    similarities.append((sim + 1) / 2)  # Normalize to 0-1

        return np.mean(similarities) if similarities else 0.5


class DreamGenerator:
    """Generates dreams from simultaneous memory reactivation.

    Key mechanisms:
    1. Sample concurrently replaying memories
    2. Assemble into narrative (with reduced prefrontal oversight)
    3. Accept bizarre logic (PFC is suppressed during REM)
    """

    def __init__(self, bizarreness_tolerance: float = 0.7, pfc_suppression: float = 0.6):
        """Initialize dream generator.

        Args:
            bizarreness_tolerance: How much bizarreness is accepted
            pfc_suppression: How much prefrontal cortex is suppressed (0-1)
        """
        self.bizarreness_tolerance = bizarreness_tolerance
        self.pfc_suppression = pfc_suppression

        # Components
        self.narrative_assembler = NarrativeAssembler()

        # Active memories during current replay
        self.active_memories: List[MemoryTrace] = []

        # Generated dreams
        self.dream_history: List[DreamReport] = []

        # Current time
        self.current_time = 0.0

    def sample_concurrent_replays(
        self, replaying_memories: List[MemoryTrace], n_sample: int = 3
    ) -> List[MemoryTrace]:
        """Sample from currently replaying memories.

        Not all replaying memories enter dream consciousness.

        Args:
            replaying_memories: All memories being replayed
            n_sample: How many to sample for dream

        Returns:
            Sampled memories that enter dream
        """
        if not replaying_memories:
            return []

        n_sample = min(n_sample, len(replaying_memories))

        # Weight by emotional salience (emotional memories more likely in dreams)
        weights = [1.0 + np.abs(m.emotional_salience) for m in replaying_memories]
        total = sum(weights)
        probs = [w / total for w in weights]

        # Sample without replacement
        indices = np.random.choice(len(replaying_memories), size=n_sample, replace=False, p=probs)

        return [replaying_memories[i] for i in indices]

    def create_dream_elements(self, memories: List[MemoryTrace]) -> List[DreamElement]:
        """Create dream elements from memories.

        Each memory contributes an element, potentially distorted.

        Args:
            memories: Source memories

        Returns:
            Dream elements
        """
        elements = []

        for memory in memories:
            # Distortion increases with PFC suppression
            distortion = self.pfc_suppression * np.random.random()

            # Apply distortion to content
            if distortion > 0:
                noise = np.random.randn(*memory.content.shape) * distortion * 0.2
                distorted_content = memory.content + noise
            else:
                distorted_content = memory.content.copy()

            # Vividness related to emotional salience
            vividness = 0.5 + 0.5 * np.abs(memory.emotional_salience)

            element = DreamElement(
                source_memory=memory,
                content_fragment=distorted_content,
                vividness=vividness,
                distortion=distortion,
                emotional_contribution=memory.emotional_salience,
            )
            elements.append(element)

        return elements

    def bizarre_logic_filter(
        self, elements: List[DreamElement]
    ) -> Tuple[List[DreamElement], float]:
        """Apply reduced logical filtering (accept bizarre content).

        During REM, prefrontal cortex is suppressed, so bizarre
        content is accepted without critical evaluation.

        Args:
            elements: Dream elements

        Returns:
            Tuple of (filtered elements, bizarreness score)
        """
        # With high PFC suppression, even bizarre combinations pass
        acceptance_threshold = 1.0 - self.pfc_suppression

        # Compute bizarreness (inverse of coherence)
        coherence = self.narrative_assembler.compute_coherence(elements)
        bizarreness = 1.0 - coherence

        # Accept elements if bizarreness is tolerable or PFC suppressed
        if bizarreness <= self.bizarreness_tolerance or self.pfc_suppression > 0.5:
            return elements, bizarreness

        # Otherwise, try to filter most bizarre elements
        # (This rarely happens given typical PFC suppression during REM)
        filtered = []
        for elem in elements:
            if elem.distortion < acceptance_threshold:
                filtered.append(elem)

        return filtered if filtered else elements, bizarreness

    def determine_emotional_tone(self, elements: List[DreamElement]) -> DreamEmotionTone:
        """Determine overall emotional tone of dream.

        Args:
            elements: Dream elements

        Returns:
            Emotional tone classification
        """
        if not elements:
            return DreamEmotionTone.NEUTRAL

        # Average emotional contribution
        avg_emotion = np.mean([e.emotional_contribution for e in elements])

        # Average distortion (contributes to bizarreness)
        avg_distortion = np.mean([e.distortion for e in elements])

        if avg_distortion > 0.6:
            return DreamEmotionTone.BIZARRE

        if avg_emotion > 0.3:
            return DreamEmotionTone.POSITIVE
        elif avg_emotion < -0.3:
            if avg_emotion < -0.5:
                return DreamEmotionTone.ANXIOUS
            return DreamEmotionTone.NEGATIVE

        return DreamEmotionTone.NEUTRAL

    def generate_dream(
        self, replaying_memories: List[MemoryTrace], duration: float = 10.0
    ) -> DreamReport:
        """Generate a dream from replaying memories.

        Args:
            replaying_memories: Memories currently being replayed
            duration: Subjective dream duration (minutes)

        Returns:
            Generated dream report
        """
        # Sample memories for dream content
        sampled = self.sample_concurrent_replays(replaying_memories)
        self.active_memories = sampled

        # Create dream elements
        elements = self.create_dream_elements(sampled)

        # Fill narrative gaps
        elements = self.narrative_assembler.fill_gaps(elements)

        # Apply bizarre logic acceptance
        elements, bizarreness = self.bizarre_logic_filter(elements)

        # Compute coherence
        coherence = self.narrative_assembler.compute_coherence(elements)

        # Determine emotional tone
        tone = self.determine_emotional_tone(elements)

        # Create dream report
        dream = DreamReport(
            source_memories=sampled,
            elements=elements,
            narrative_coherence=coherence,
            emotional_tone=tone,
            bizarreness_index=bizarreness,
            duration=duration,
            timestamp=self.current_time,
        )

        self.dream_history.append(dream)
        return dream

    def advance_time(self, dt: float) -> None:
        """Advance simulation time."""
        self.current_time += dt

    def get_dream_statistics(self) -> Dict:
        """Get statistics about generated dreams."""
        if not self.dream_history:
            return {
                "n_dreams": 0,
                "mean_bizarreness": 0,
                "mean_coherence": 0,
                "tone_distribution": {},
            }

        tone_counts = {}
        for dream in self.dream_history:
            tone = dream.emotional_tone.value
            tone_counts[tone] = tone_counts.get(tone, 0) + 1

        return {
            "n_dreams": len(self.dream_history),
            "mean_bizarreness": np.mean([d.bizarreness_index for d in self.dream_history]),
            "mean_coherence": np.mean([d.narrative_coherence for d in self.dream_history]),
            "mean_n_elements": np.mean([len(d.elements) for d in self.dream_history]),
            "tone_distribution": tone_counts,
        }

    def clear_history(self) -> None:
        """Clear dream history."""
        self.dream_history = []

    def reset(self) -> None:
        """Reset generator state."""
        self.active_memories = []
        self.dream_history = []
        self.current_time = 0.0
