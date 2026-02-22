"""
Semantic Network: Concept graphs with spreading activation.

Implements Collins & Quillian style semantic networks with
modern spreading activation for associative retrieval.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum
import numpy as np


class RelationType(Enum):
    """Types of semantic relations."""

    IS_A = "is_a"  # Taxonomic (dog IS_A animal)
    HAS_A = "has_a"  # Part-whole (car HAS_A wheel)
    PART_OF = "part_of"  # Meronymic (wheel PART_OF car)
    PROPERTY = "property"  # Attribute (apple PROPERTY red)
    CAN = "can"  # Capability (bird CAN fly)
    CAUSES = "causes"  # Causal (fire CAUSES smoke)
    LOCATED_IN = "located_in"  # Spatial (Paris LOCATED_IN France)
    SIMILAR_TO = "similar_to"  # Similarity (dog SIMILAR_TO wolf)
    OPPOSITE_OF = "opposite_of"  # Antonymy (hot OPPOSITE_OF cold)
    RELATED_TO = "related_to"  # General association
    INSTANCE_OF = "instance_of"  # Instance (Fido INSTANCE_OF dog)
    USED_FOR = "used_for"  # Function (hammer USED_FOR nail)


@dataclass
class Concept:
    """A concept node in the semantic network."""

    name: str
    embedding: Optional[np.ndarray] = None
    activation: float = 0.0
    base_activation: float = 0.0
    frequency: int = 0  # How often accessed
    properties: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, Concept):
            return self.name == other.name
        return False


@dataclass
class SemanticRelation:
    """A relation between concepts."""

    source: str
    target: str
    relation_type: RelationType
    weight: float = 1.0
    bidirectional: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActivationPattern:
    """Pattern of activation across network."""

    activations: Dict[str, float]
    source_concept: str
    spread_depth: int
    timestamp: float = 0.0

    def top_k(self, k: int = 10) -> List[Tuple[str, float]]:
        """Get top k activated concepts."""
        sorted_items = sorted(self.activations.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:k]


class SemanticNetwork:
    """
    Semantic network with spreading activation.

    Implements:
    - Concept storage with embeddings
    - Multiple relation types
    - Spreading activation for retrieval
    - Semantic similarity queries
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        decay_rate: float = 0.8,
        activation_threshold: float = 0.01,
        max_spread_depth: int = 5,
        random_seed: Optional[int] = None,
    ):
        self.embedding_dim = embedding_dim
        self.decay_rate = decay_rate
        self.activation_threshold = activation_threshold
        self.max_spread_depth = max_spread_depth
        self._rng = np.random.default_rng(random_seed)

        # Concept storage
        self._concepts: Dict[str, Concept] = {}

        # Relations (adjacency lists)
        self._outgoing: Dict[str, List[SemanticRelation]] = {}
        self._incoming: Dict[str, List[SemanticRelation]] = {}

        # Relation type weights for spreading
        self._relation_weights: Dict[RelationType, float] = {
            RelationType.IS_A: 0.9,
            RelationType.HAS_A: 0.7,
            RelationType.PART_OF: 0.7,
            RelationType.PROPERTY: 0.6,
            RelationType.CAN: 0.5,
            RelationType.CAUSES: 0.6,
            RelationType.SIMILAR_TO: 0.8,
            RelationType.RELATED_TO: 0.4,
            RelationType.INSTANCE_OF: 0.9,
            RelationType.USED_FOR: 0.5,
            RelationType.LOCATED_IN: 0.5,
            RelationType.OPPOSITE_OF: 0.3,
        }

        # Statistics
        self._activation_count = 0

    def add_concept(
        self,
        name: str,
        embedding: Optional[np.ndarray] = None,
        properties: Optional[Dict[str, Any]] = None,
    ) -> Concept:
        """Add a concept to the network."""
        if embedding is None:
            embedding = self._rng.normal(0, 0.1, self.embedding_dim)
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

        concept = Concept(
            name=name,
            embedding=embedding,
            properties=properties or {},
        )

        self._concepts[name] = concept
        self._outgoing[name] = []
        self._incoming[name] = []

        return concept

    def get_concept(self, name: str) -> Optional[Concept]:
        """Get a concept by name."""
        return self._concepts.get(name)

    def has_concept(self, name: str) -> bool:
        """Check if concept exists."""
        return name in self._concepts

    def add_relation(
        self,
        source: str,
        target: str,
        relation_type: RelationType,
        weight: float = 1.0,
        bidirectional: bool = False,
    ) -> SemanticRelation:
        """Add a relation between concepts."""
        # Ensure concepts exist
        if source not in self._concepts:
            self.add_concept(source)
        if target not in self._concepts:
            self.add_concept(target)

        relation = SemanticRelation(
            source=source,
            target=target,
            relation_type=relation_type,
            weight=weight,
            bidirectional=bidirectional,
        )

        self._outgoing[source].append(relation)
        self._incoming[target].append(relation)

        if bidirectional:
            reverse = SemanticRelation(
                source=target,
                target=source,
                relation_type=relation_type,
                weight=weight,
                bidirectional=True,
            )
            self._outgoing[target].append(reverse)
            self._incoming[source].append(reverse)

        return relation

    def get_relations(
        self,
        concept: str,
        relation_type: Optional[RelationType] = None,
        direction: str = "outgoing",
    ) -> List[SemanticRelation]:
        """Get relations for a concept."""
        if direction == "outgoing":
            relations = self._outgoing.get(concept, [])
        elif direction == "incoming":
            relations = self._incoming.get(concept, [])
        else:
            relations = self._outgoing.get(concept, []) + self._incoming.get(concept, [])

        if relation_type:
            relations = [r for r in relations if r.relation_type == relation_type]

        return relations

    def activate(
        self,
        concept_name: str,
        initial_activation: float = 1.0,
    ) -> ActivationPattern:
        """
        Activate a concept and spread activation through network.

        Uses spreading activation to find related concepts.
        """
        if concept_name not in self._concepts:
            return ActivationPattern({}, concept_name, 0)

        self._activation_count += 1

        # Reset activations
        for concept in self._concepts.values():
            concept.activation = 0.0

        # Initialize source activation
        self._concepts[concept_name].activation = initial_activation
        self._concepts[concept_name].frequency += 1

        # Spread activation
        activations = {concept_name: initial_activation}
        frontier = [(concept_name, initial_activation, 0)]
        visited = {concept_name}

        while frontier:
            current_name, current_activation, depth = frontier.pop(0)

            if depth >= self.max_spread_depth:
                continue

            # Spread to neighbors
            for relation in self._outgoing.get(current_name, []):
                target = relation.target

                # Compute spread activation
                relation_weight = self._relation_weights.get(relation.relation_type, 0.5)
                spread = current_activation * self.decay_rate * relation.weight * relation_weight

                if spread < self.activation_threshold:
                    continue

                # Update target activation
                if target in activations:
                    activations[target] = max(activations[target], spread)
                else:
                    activations[target] = spread

                self._concepts[target].activation = activations[target]

                # Add to frontier if not visited at this level
                if target not in visited:
                    visited.add(target)
                    frontier.append((target, spread, depth + 1))

        return ActivationPattern(
            activations=activations,
            source_concept=concept_name,
            spread_depth=self.max_spread_depth,
            timestamp=float(self._activation_count),
        )

    def find_similar(
        self,
        concept_name: str,
        top_k: int = 10,
        use_embedding: bool = True,
    ) -> List[Tuple[str, float]]:
        """Find concepts similar to the given concept."""
        if concept_name not in self._concepts:
            return []

        concept = self._concepts[concept_name]
        similarities = []

        for other_name, other in self._concepts.items():
            if other_name == concept_name:
                continue

            if use_embedding and concept.embedding is not None and other.embedding is not None:
                # Cosine similarity
                sim = np.dot(concept.embedding, other.embedding)
            else:
                # Structure-based similarity (shared neighbors)
                neighbors1 = {r.target for r in self._outgoing.get(concept_name, [])}
                neighbors2 = {r.target for r in self._outgoing.get(other_name, [])}

                if neighbors1 or neighbors2:
                    sim = len(neighbors1 & neighbors2) / len(neighbors1 | neighbors2)
                else:
                    sim = 0.0

            similarities.append((other_name, float(sim)))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def find_path(
        self,
        source: str,
        target: str,
        max_depth: int = 5,
    ) -> Optional[List[SemanticRelation]]:
        """Find shortest path between two concepts."""
        if source not in self._concepts or target not in self._concepts:
            return None

        # BFS
        queue = [(source, [])]
        visited = {source}

        while queue:
            current, path = queue.pop(0)

            if current == target:
                return path

            if len(path) >= max_depth:
                continue

            for relation in self._outgoing.get(current, []):
                if relation.target not in visited:
                    visited.add(relation.target)
                    queue.append((relation.target, path + [relation]))

        return None

    def get_ancestors(
        self,
        concept_name: str,
        relation_type: RelationType = RelationType.IS_A,
    ) -> List[str]:
        """Get all ancestors via IS_A (or specified) relation."""
        ancestors = []
        visited = {concept_name}
        queue = [concept_name]

        while queue:
            current = queue.pop(0)

            for relation in self._outgoing.get(current, []):
                if relation.relation_type == relation_type:
                    if relation.target not in visited:
                        visited.add(relation.target)
                        ancestors.append(relation.target)
                        queue.append(relation.target)

        return ancestors

    def get_descendants(
        self,
        concept_name: str,
        relation_type: RelationType = RelationType.IS_A,
    ) -> List[str]:
        """Get all descendants via IS_A (or specified) relation."""
        descendants = []
        visited = {concept_name}
        queue = [concept_name]

        while queue:
            current = queue.pop(0)

            for relation in self._incoming.get(current, []):
                if relation.relation_type == relation_type:
                    if relation.source not in visited:
                        visited.add(relation.source)
                        descendants.append(relation.source)
                        queue.append(relation.source)

        return descendants

    def common_ancestor(
        self,
        concept1: str,
        concept2: str,
        relation_type: RelationType = RelationType.IS_A,
    ) -> Optional[str]:
        """Find lowest common ancestor of two concepts."""
        ancestors1 = set(self.get_ancestors(concept1, relation_type))
        ancestors1.add(concept1)

        # BFS from concept2 to find first shared ancestor
        visited = {concept2}
        queue = [concept2]

        while queue:
            current = queue.pop(0)

            if current in ancestors1:
                return current

            for relation in self._outgoing.get(current, []):
                if relation.relation_type == relation_type:
                    if relation.target not in visited:
                        visited.add(relation.target)
                        queue.append(relation.target)

        return None

    def semantic_distance(
        self,
        concept1: str,
        concept2: str,
    ) -> float:
        """Compute semantic distance between concepts."""
        # Find path
        path = self.find_path(concept1, concept2)

        if path is None:
            # No path, use embedding distance
            c1 = self._concepts.get(concept1)
            c2 = self._concepts.get(concept2)

            if c1 and c2 and c1.embedding is not None and c2.embedding is not None:
                return 1.0 - np.dot(c1.embedding, c2.embedding)
            return float("inf")

        # Path-based distance with relation weights
        distance = 0.0
        for relation in path:
            weight = self._relation_weights.get(relation.relation_type, 0.5)
            distance += 1.0 / weight

        return distance

    def query(
        self,
        subject: Optional[str] = None,
        relation: Optional[RelationType] = None,
        obj: Optional[str] = None,
    ) -> List[SemanticRelation]:
        """Query relations matching pattern."""
        results = []

        if subject and subject in self._outgoing:
            for rel in self._outgoing[subject]:
                if relation and rel.relation_type != relation:
                    continue
                if obj and rel.target != obj:
                    continue
                results.append(rel)

        elif obj and obj in self._incoming:
            for rel in self._incoming[obj]:
                if relation and rel.relation_type != relation:
                    continue
                if subject and rel.source != subject:
                    continue
                results.append(rel)

        elif relation:
            for concept_name in self._outgoing:
                for rel in self._outgoing[concept_name]:
                    if rel.relation_type == relation:
                        results.append(rel)

        return results

    def statistics(self) -> Dict[str, Any]:
        """Get network statistics."""
        n_relations = sum(len(rels) for rels in self._outgoing.values())

        relation_counts = {}
        for rels in self._outgoing.values():
            for rel in rels:
                rt = rel.relation_type.value
                relation_counts[rt] = relation_counts.get(rt, 0) + 1

        return {
            "n_concepts": len(self._concepts),
            "n_relations": n_relations,
            "relation_type_counts": relation_counts,
            "activation_count": self._activation_count,
            "embedding_dim": self.embedding_dim,
        }
