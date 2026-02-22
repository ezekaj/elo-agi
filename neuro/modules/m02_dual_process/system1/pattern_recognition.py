"""
Pattern Recognition - Neocortex Simulation

Implements parallel pattern matching against learned statistical regularities.
This is the core of System 1's ability to recognize familiar situations instantly.

Based on research showing neocortex performs massively parallel pattern matching
without explicit rule representation - patterns are implicit in connection weights.
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class Pattern:
    """A learned pattern template"""

    id: str
    template: np.ndarray
    exemplar_count: int = 1
    variance: Optional[np.ndarray] = None
    contexts: List[Any] = field(default_factory=list)


@dataclass
class PatternMatch:
    """Result of pattern matching"""

    pattern_id: str
    confidence: float
    template: np.ndarray


class PatternRecognition:
    """
    Parallel pattern matching against learned regularities.

    Simulates neocortex statistical pattern detection:
    - All patterns checked SIMULTANEOUSLY (parallel)
    - No explicit rules - patterns are implicit
    - Context-dependent activation
    - Learns statistical regularities from examples
    """

    def __init__(self, similarity_threshold: float = 0.7, max_patterns: int = 10000):
        self.patterns: Dict[str, Pattern] = {}
        self.similarity_threshold = similarity_threshold
        self.max_patterns = max_patterns
        self._pattern_matrix: Optional[np.ndarray] = None
        self._pattern_ids: List[str] = []
        self._matrix_dirty = True

    def learn_pattern(
        self, pattern_id: str, examples: List[np.ndarray], contexts: Optional[List[Any]] = None
    ) -> Pattern:
        """
        Extract statistical regularity from examples.

        The brain learns patterns by extracting what's common across examples,
        not by storing explicit rules.
        """
        if len(examples) == 0:
            raise ValueError("Need at least one example to learn pattern")

        examples_array = np.array(examples)
        template = np.mean(examples_array, axis=0)
        variance = np.var(examples_array, axis=0) if len(examples) > 1 else None

        pattern = Pattern(
            id=pattern_id,
            template=template,
            exemplar_count=len(examples),
            variance=variance,
            contexts=contexts or [],
        )

        self.patterns[pattern_id] = pattern
        self._matrix_dirty = True

        return pattern

    def update_pattern(self, pattern_id: str, new_example: np.ndarray) -> Pattern:
        """
        Incrementally update pattern with new example.
        Online learning - brain continuously updates patterns.
        """
        if pattern_id not in self.patterns:
            return self.learn_pattern(pattern_id, [new_example])

        pattern = self.patterns[pattern_id]
        n = pattern.exemplar_count

        # Running mean update
        pattern.template = (pattern.template * n + new_example) / (n + 1)
        pattern.exemplar_count = n + 1

        self._matrix_dirty = True
        return pattern

    def _build_pattern_matrix(self):
        """Build matrix for vectorized parallel matching"""
        if not self._matrix_dirty:
            return

        self._pattern_ids = list(self.patterns.keys())
        if self._pattern_ids:
            self._pattern_matrix = np.array(
                [self.patterns[pid].template for pid in self._pattern_ids]
            )
        else:
            self._pattern_matrix = None
        self._matrix_dirty = False

    def match(
        self, input_vector: np.ndarray, top_k: Optional[int] = None, context: Optional[Any] = None
    ) -> List[PatternMatch]:
        """
        Find all matching patterns - PARALLEL operation.

        This is the key System 1 capability: all patterns are checked
        simultaneously, not sequentially. Returns in order of confidence.
        """
        self._build_pattern_matrix()

        if self._pattern_matrix is None or len(self._pattern_ids) == 0:
            return []

        # Vectorized similarity computation - ALL patterns at once
        # This simulates parallel neural processing
        input_normalized = input_vector / (np.linalg.norm(input_vector) + 1e-8)
        patterns_normalized = self._pattern_matrix / (
            np.linalg.norm(self._pattern_matrix, axis=1, keepdims=True) + 1e-8
        )

        similarities = np.dot(patterns_normalized, input_normalized)

        # Context modulation - patterns associated with current context get boost
        if context is not None:
            for i, pid in enumerate(self._pattern_ids):
                if context in self.patterns[pid].contexts:
                    similarities[i] *= 1.2  # Context boost

        # Filter by threshold
        matches = []
        for i, (pid, sim) in enumerate(zip(self._pattern_ids, similarities)):
            if sim >= self.similarity_threshold:
                matches.append(
                    PatternMatch(
                        pattern_id=pid, confidence=float(sim), template=self.patterns[pid].template
                    )
                )

        # Sort by confidence (highest first)
        matches.sort(key=lambda m: m.confidence, reverse=True)

        if top_k is not None:
            matches = matches[:top_k]

        return matches

    def confidence(self, input_vector: np.ndarray, pattern_id: str) -> float:
        """Get confidence score for specific pattern match"""
        if pattern_id not in self.patterns:
            return 0.0

        pattern = self.patterns[pattern_id]
        input_norm = input_vector / (np.linalg.norm(input_vector) + 1e-8)
        pattern_norm = pattern.template / (np.linalg.norm(pattern.template) + 1e-8)

        return float(np.dot(input_norm, pattern_norm))

    def best_match(self, input_vector: np.ndarray) -> Optional[PatternMatch]:
        """Get single best matching pattern"""
        matches = self.match(input_vector, top_k=1)
        return matches[0] if matches else None

    def has_match(self, input_vector: np.ndarray) -> bool:
        """Quick check if any pattern matches"""
        return len(self.match(input_vector, top_k=1)) > 0

    def generalize(self, novel_input: np.ndarray) -> List[PatternMatch]:
        """
        Generalization - find partial matches for novel input.

        System 1 can recognize things it hasn't seen exactly before
        by matching to similar known patterns.
        """
        # Lower threshold for generalization
        original_threshold = self.similarity_threshold
        self.similarity_threshold = original_threshold * 0.7

        matches = self.match(novel_input)

        self.similarity_threshold = original_threshold
        return matches
