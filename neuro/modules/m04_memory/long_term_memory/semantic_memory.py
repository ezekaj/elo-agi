"""
Semantic Memory: Facts and concepts as a network

Based on research showing decontextualized knowledge stored
in cerebral cortex as interconnected concept network.

Brain region: Cerebral cortex (distributed)
"""

import time
from typing import Optional, Any, List, Dict, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import heapq


@dataclass
class Concept:
    """Semantic knowledge unit"""

    name: str
    features: Dict[str, float] = field(default_factory=dict)
    relations: List[Tuple[str, str, float]] = field(default_factory=list)
    activation: float = 0.0
    frequency: int = 0
    last_access: Optional[float] = None

    def add_feature(self, feature: str, value: float) -> None:
        """Add or update a feature"""
        self.features[feature] = value

    def add_relation(self, relation_type: str, target: str, strength: float = 1.0) -> None:
        """Add relation to another concept"""
        # Remove existing relation of same type to same target
        self.relations = [
            r for r in self.relations if not (r[0] == relation_type and r[1] == target)
        ]
        self.relations.append((relation_type, target, strength))


class SemanticMemory:
    """
    Cerebral cortex - facts & concepts as network

    Implements semantic memory with spreading activation
    and multiple relation types.
    """

    RELATION_TYPES = {
        "is-a",  # Category membership (dog is-a animal)
        "has-a",  # Part-whole (car has-a wheel)
        "part-of",  # Inverse of has-a
        "causes",  # Causal relation
        "caused-by",  # Inverse causal
        "similar-to",  # Similarity
        "opposite-of",  # Antonymy
        "related-to",  # General association
    }

    def __init__(self):
        self._concepts: Dict[str, Concept] = {}
        self._relation_index: Dict[str, Dict[str, List[Tuple[str, float]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self._time_fn = time.time

    def store(self, concept: Concept) -> None:
        """
        Add concept to semantic network.

        Args:
            concept: Concept to store
        """
        self._concepts[concept.name] = concept

        # Index relations
        for rel_type, target, strength in concept.relations:
            self._relation_index[concept.name][rel_type].append((target, strength))

    def create_concept(
        self,
        name: str,
        features: Optional[Dict[str, float]] = None,
        relations: Optional[List[Tuple[str, str, float]]] = None,
    ) -> Concept:
        """
        Create and store a new concept.

        Args:
            name: Concept name
            features: Attribute-value pairs
            relations: List of (relation_type, target_name, strength)

        Returns:
            The created Concept
        """
        concept = Concept(name=name, features=features or {}, relations=relations or [])
        self.store(concept)
        return concept

    def retrieve(self, name: str) -> Optional[Concept]:
        """
        Direct lookup by name.

        Args:
            name: Concept name

        Returns:
            Concept if found, None otherwise
        """
        concept = self._concepts.get(name)
        if concept:
            concept.frequency += 1
            concept.last_access = self._time_fn()
        return concept

    def spread_activation(
        self, source: str, initial_strength: float = 1.0, decay: float = 0.5, depth: int = 3
    ) -> Dict[str, float]:
        """
        Activate related concepts via spreading activation.

        Args:
            source: Starting concept name
            initial_strength: Initial activation level
            decay: Decay factor per step
            depth: Maximum propagation depth

        Returns:
            Dict of concept_name -> activation_level
        """
        if source not in self._concepts:
            return {}

        activations = {source: initial_strength}
        frontier = [(source, initial_strength, 0)]

        while frontier:
            current, strength, current_depth = frontier.pop(0)

            if current_depth >= depth:
                continue

            concept = self._concepts.get(current)
            if not concept:
                continue

            # Spread to related concepts
            for rel_type, target, rel_strength in concept.relations:
                if target not in self._concepts:
                    continue

                # Compute propagated activation
                new_activation = strength * decay * rel_strength

                # Update if stronger than existing
                if target not in activations or new_activation > activations[target]:
                    activations[target] = new_activation
                    frontier.append((target, new_activation, current_depth + 1))

        # Update concept activations
        for name, activation in activations.items():
            if name in self._concepts:
                self._concepts[name].activation = activation

        return activations

    def get_related(
        self, concept_name: str, relation_type: Optional[str] = None
    ) -> List[Tuple[str, str, float]]:
        """
        Find linked concepts.

        Args:
            concept_name: Source concept
            relation_type: Filter by relation type (None for all)

        Returns:
            List of (relation_type, target_name, strength)
        """
        concept = self._concepts.get(concept_name)
        if not concept:
            return []

        if relation_type:
            return [r for r in concept.relations if r[0] == relation_type]
        return concept.relations.copy()

    def find_path(
        self, start: str, end: str, max_depth: int = 5
    ) -> Optional[List[Tuple[str, str, str]]]:
        """
        Find shortest relational path between concepts.

        Args:
            start: Starting concept
            end: Target concept
            max_depth: Maximum path length

        Returns:
            List of (source, relation, target) tuples, or None if no path
        """
        if start not in self._concepts or end not in self._concepts:
            return None

        if start == end:
            return []

        # BFS
        visited = {start}
        queue = [(start, [])]

        while queue:
            current, path = queue.pop(0)

            if len(path) >= max_depth:
                continue

            concept = self._concepts.get(current)
            if not concept:
                continue

            for rel_type, target, _ in concept.relations:
                if target == end:
                    return path + [(current, rel_type, target)]

                if target not in visited and target in self._concepts:
                    visited.add(target)
                    queue.append((target, path + [(current, rel_type, target)]))

        return None

    def abstract_from_episodes(self, episodes: List[Any]) -> Optional[Concept]:
        """
        Extract semantic knowledge from episodic memories.

        Args:
            episodes: List of episodes with common features

        Returns:
            Abstracted concept, or None if insufficient commonality
        """
        if len(episodes) < 2:
            return None

        # Extract common features (simplified)
        all_contents = [str(ep.content) if hasattr(ep, "content") else str(ep) for ep in episodes]

        # Find common words (very simplified)
        word_sets = [set(content.lower().split()) for content in all_contents]
        common_words = set.intersection(*word_sets) if word_sets else set()

        if not common_words:
            return None

        # Create abstract concept
        concept_name = "_".join(sorted(common_words)[:3])
        features = {word: 1.0 for word in common_words}

        return self.create_concept(concept_name, features)

    def add_relation(self, source: str, relation: str, target: str, strength: float = 1.0) -> bool:
        """
        Create link between concepts.

        Args:
            source: Source concept name
            relation: Relation type
            target: Target concept name
            strength: Relation strength

        Returns:
            True if relation created
        """
        if source not in self._concepts:
            return False

        concept = self._concepts[source]
        concept.add_relation(relation, target, strength)

        # Update index
        self._relation_index[source][relation].append((target, strength))

        return True

    def prune(self, frequency_threshold: int = 0, strength_threshold: float = 0.1) -> int:
        """
        Remove weak/unused concepts.

        Args:
            frequency_threshold: Minimum access count
            strength_threshold: Minimum relation strength

        Returns:
            Number of concepts removed
        """
        to_remove = []

        for name, concept in self._concepts.items():
            if concept.frequency <= frequency_threshold:
                to_remove.append(name)
                continue

            # Prune weak relations
            concept.relations = [r for r in concept.relations if r[2] >= strength_threshold]

        for name in to_remove:
            del self._concepts[name]
            if name in self._relation_index:
                del self._relation_index[name]

        return len(to_remove)

    def find_by_feature(self, feature: str, min_value: float = 0.0) -> List[Concept]:
        """Find concepts with a specific feature"""
        return [
            c
            for c in self._concepts.values()
            if feature in c.features and c.features[feature] >= min_value
        ]

    def get_categories(self, concept_name: str) -> List[str]:
        """Get all categories a concept belongs to (via is-a relations)"""
        categories = []
        concept = self._concepts.get(concept_name)
        if concept:
            for rel_type, target, _ in concept.relations:
                if rel_type == "is-a":
                    categories.append(target)
        return categories

    def get_members(self, category_name: str) -> List[str]:
        """Get all members of a category"""
        members = []
        for name, concept in self._concepts.items():
            for rel_type, target, _ in concept.relations:
                if rel_type == "is-a" and target == category_name:
                    members.append(name)
        return members

    def __len__(self) -> int:
        return len(self._concepts)

    def __contains__(self, name: str) -> bool:
        return name in self._concepts

    def set_time_function(self, time_fn) -> None:
        """Set custom time function for simulation"""
        self._time_fn = time_fn
