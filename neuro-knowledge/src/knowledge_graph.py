"""
Knowledge Graph: Entity-relation graph with embeddings.

Implements knowledge graph embeddings and link prediction
for neural-symbolic integration.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum
import numpy as np


@dataclass
class Entity:
    """An entity in the knowledge graph."""
    name: str
    entity_type: str = "entity"
    embedding: Optional[np.ndarray] = None
    properties: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, Entity):
            return self.name == other.name
        return False


@dataclass
class Relation:
    """A relation type in the knowledge graph."""
    name: str
    embedding: Optional[np.ndarray] = None
    symmetric: bool = False
    transitive: bool = False
    inverse: Optional[str] = None
    domain: Optional[str] = None  # Entity type for subject
    range: Optional[str] = None   # Entity type for object

    def __hash__(self):
        return hash(self.name)


@dataclass
class GraphEmbedding:
    """Embedding of an entity or relation."""
    name: str
    vector: np.ndarray
    is_entity: bool = True


@dataclass
class GraphQuery:
    """A query against the knowledge graph."""
    head: Optional[str] = None
    relation: Optional[str] = None
    tail: Optional[str] = None
    entity_type: Optional[str] = None
    top_k: int = 10


class KnowledgeGraph:
    """
    Knowledge graph with neural embeddings.

    Supports:
    - Entity and relation storage
    - TransE-style embeddings
    - Link prediction
    - Path queries
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        margin: float = 1.0,
        learning_rate: float = 0.01,
        random_seed: Optional[int] = None,
    ):
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.learning_rate = learning_rate
        self._rng = np.random.default_rng(random_seed)

        # Entity storage
        self._entities: Dict[str, Entity] = {}

        # Relation storage
        self._relations: Dict[str, Relation] = {}

        # Edges (head, relation, tail)
        self._edges: Set[Tuple[str, str, str]] = set()

        # Adjacency lists
        self._outgoing: Dict[str, Dict[str, List[str]]] = {}  # head -> relation -> [tails]
        self._incoming: Dict[str, Dict[str, List[str]]] = {}  # tail -> relation -> [heads]

        # Entity type index
        self._by_type: Dict[str, Set[str]] = {}

    def add_entity(
        self,
        name: str,
        entity_type: str = "entity",
        embedding: Optional[np.ndarray] = None,
        properties: Optional[Dict[str, Any]] = None,
    ) -> Entity:
        """Add an entity to the graph."""
        if embedding is None:
            embedding = self._rng.normal(0, 0.1, self.embedding_dim)
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

        entity = Entity(
            name=name,
            entity_type=entity_type,
            embedding=embedding,
            properties=properties or {},
        )

        self._entities[name] = entity
        self._outgoing[name] = {}
        self._incoming[name] = {}

        # Index by type
        if entity_type not in self._by_type:
            self._by_type[entity_type] = set()
        self._by_type[entity_type].add(name)

        return entity

    def get_entity(self, name: str) -> Optional[Entity]:
        """Get an entity by name."""
        return self._entities.get(name)

    def has_entity(self, name: str) -> bool:
        """Check if entity exists."""
        return name in self._entities

    def add_relation(
        self,
        name: str,
        embedding: Optional[np.ndarray] = None,
        symmetric: bool = False,
        transitive: bool = False,
        inverse: Optional[str] = None,
        domain: Optional[str] = None,
        range_type: Optional[str] = None,
    ) -> Relation:
        """Add a relation type."""
        if embedding is None:
            embedding = self._rng.normal(0, 0.1, self.embedding_dim)

        relation = Relation(
            name=name,
            embedding=embedding,
            symmetric=symmetric,
            transitive=transitive,
            inverse=inverse,
            domain=domain,
            range=range_type,
        )

        self._relations[name] = relation
        return relation

    def get_relation(self, name: str) -> Optional[Relation]:
        """Get a relation by name."""
        return self._relations.get(name)

    def add_edge(
        self,
        head: str,
        relation: str,
        tail: str,
    ) -> bool:
        """Add an edge to the graph."""
        # Ensure entities exist
        if head not in self._entities:
            self.add_entity(head)
        if tail not in self._entities:
            self.add_entity(tail)
        if relation not in self._relations:
            self.add_relation(relation)

        edge = (head, relation, tail)
        if edge in self._edges:
            return False

        self._edges.add(edge)

        # Update adjacency
        if relation not in self._outgoing[head]:
            self._outgoing[head][relation] = []
        self._outgoing[head][relation].append(tail)

        if relation not in self._incoming[tail]:
            self._incoming[tail][relation] = []
        self._incoming[tail][relation].append(head)

        # Handle symmetric relations
        rel_obj = self._relations[relation]
        if rel_obj.symmetric:
            reverse = (tail, relation, head)
            if reverse not in self._edges:
                self._edges.add(reverse)
                if relation not in self._outgoing[tail]:
                    self._outgoing[tail][relation] = []
                self._outgoing[tail][relation].append(head)
                if relation not in self._incoming[head]:
                    self._incoming[head][relation] = []
                self._incoming[head][relation].append(tail)

        return True

    def has_edge(self, head: str, relation: str, tail: str) -> bool:
        """Check if edge exists."""
        return (head, relation, tail) in self._edges

    def get_neighbors(
        self,
        entity: str,
        relation: Optional[str] = None,
        direction: str = "outgoing",
    ) -> List[Tuple[str, str]]:
        """Get neighbors of an entity."""
        neighbors = []

        if direction in ["outgoing", "both"]:
            adj = self._outgoing.get(entity, {})
            for rel, tails in adj.items():
                if relation is None or rel == relation:
                    for tail in tails:
                        neighbors.append((rel, tail))

        if direction in ["incoming", "both"]:
            adj = self._incoming.get(entity, {})
            for rel, heads in adj.items():
                if relation is None or rel == relation:
                    for head in heads:
                        neighbors.append((rel, head))

        return neighbors

    def score_triple(
        self,
        head: str,
        relation: str,
        tail: str,
    ) -> float:
        """Score a triple using TransE scoring."""
        h = self._entities.get(head)
        r = self._relations.get(relation)
        t = self._entities.get(tail)

        if not h or not r or not t:
            return float('inf')

        if h.embedding is None or r.embedding is None or t.embedding is None:
            return float('inf')

        # TransE: h + r ≈ t
        # Score = ||h + r - t||
        score = np.linalg.norm(h.embedding + r.embedding - t.embedding)

        return float(score)

    def predict_tail(
        self,
        head: str,
        relation: str,
        top_k: int = 10,
        exclude_known: bool = True,
    ) -> List[Tuple[str, float]]:
        """Predict tail entities for (head, relation, ?)."""
        h = self._entities.get(head)
        r = self._relations.get(relation)

        if not h or not r:
            return []

        if h.embedding is None or r.embedding is None:
            return []

        # Compute target vector
        target = h.embedding + r.embedding

        # Score all entities
        scores = []
        known_tails = set(self._outgoing.get(head, {}).get(relation, []))

        for name, entity in self._entities.items():
            if exclude_known and name in known_tails:
                continue
            if entity.embedding is None:
                continue

            score = np.linalg.norm(target - entity.embedding)
            scores.append((name, float(score)))

        # Sort by score (lower is better)
        scores.sort(key=lambda x: x[1])

        return scores[:top_k]

    def predict_head(
        self,
        relation: str,
        tail: str,
        top_k: int = 10,
        exclude_known: bool = True,
    ) -> List[Tuple[str, float]]:
        """Predict head entities for (?, relation, tail)."""
        r = self._relations.get(relation)
        t = self._entities.get(tail)

        if not r or not t:
            return []

        if r.embedding is None or t.embedding is None:
            return []

        # Compute target vector: h + r = t => h = t - r
        target = t.embedding - r.embedding

        # Score all entities
        scores = []
        known_heads = set(self._incoming.get(tail, {}).get(relation, []))

        for name, entity in self._entities.items():
            if exclude_known and name in known_heads:
                continue
            if entity.embedding is None:
                continue

            score = np.linalg.norm(entity.embedding - target)
            scores.append((name, float(score)))

        scores.sort(key=lambda x: x[1])

        return scores[:top_k]

    def find_path(
        self,
        source: str,
        target: str,
        max_length: int = 3,
    ) -> Optional[List[Tuple[str, str, str]]]:
        """Find shortest path between entities."""
        if source not in self._entities or target not in self._entities:
            return None

        # BFS
        queue = [(source, [])]
        visited = {source}

        while queue:
            current, path = queue.pop(0)

            if len(path) >= max_length:
                continue

            for rel, neighbor in self.get_neighbors(current, direction="outgoing"):
                edge = (current, rel, neighbor)

                if neighbor == target:
                    return path + [edge]

                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [edge]))

        return None

    def train_step(
        self,
        positive_triples: List[Tuple[str, str, str]],
        negative_ratio: int = 1,
    ) -> float:
        """One training step with contrastive learning."""
        total_loss = 0.0
        n_updates = 0

        for head, relation, tail in positive_triples:
            h = self._entities.get(head)
            r = self._relations.get(relation)
            t = self._entities.get(tail)

            if not h or not r or not t:
                continue
            if h.embedding is None or r.embedding is None or t.embedding is None:
                continue

            # Positive score
            pos_score = self.score_triple(head, relation, tail)

            # Generate negative samples
            for _ in range(negative_ratio):
                # Corrupt tail
                neg_tail = self._rng.choice(list(self._entities.keys()))
                neg_score = self.score_triple(head, relation, neg_tail)

                # Margin-based loss
                loss = max(0, self.margin + pos_score - neg_score)

                if loss > 0:
                    # Gradient update
                    neg_t = self._entities.get(neg_tail)
                    if neg_t and neg_t.embedding is not None:
                        gradient = self.learning_rate * (
                            (h.embedding + r.embedding - t.embedding) / (pos_score + 1e-8) -
                            (h.embedding + r.embedding - neg_t.embedding) / (neg_score + 1e-8)
                        )

                        h.embedding -= gradient
                        r.embedding -= gradient
                        t.embedding += gradient
                        neg_t.embedding -= gradient

                        # Normalize
                        h.embedding /= np.linalg.norm(h.embedding) + 1e-8
                        t.embedding /= np.linalg.norm(t.embedding) + 1e-8
                        neg_t.embedding /= np.linalg.norm(neg_t.embedding) + 1e-8

                total_loss += loss
                n_updates += 1

        return total_loss / max(1, n_updates)

    def get_similar_entities(
        self,
        entity: str,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """Find similar entities by embedding distance."""
        e = self._entities.get(entity)
        if not e or e.embedding is None:
            return []

        similarities = []
        for name, other in self._entities.items():
            if name == entity or other.embedding is None:
                continue

            sim = float(np.dot(e.embedding, other.embedding))
            similarities.append((name, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def get_entities_by_type(self, entity_type: str) -> List[Entity]:
        """Get all entities of a type."""
        names = self._by_type.get(entity_type, set())
        return [self._entities[n] for n in names if n in self._entities]

    def query(self, query: GraphQuery) -> List[Tuple[str, str, str]]:
        """Query the graph."""
        results = []

        for head, relation, tail in self._edges:
            if query.head and head != query.head:
                continue
            if query.relation and relation != query.relation:
                continue
            if query.tail and tail != query.tail:
                continue
            if query.entity_type:
                h_entity = self._entities.get(head)
                if not h_entity or h_entity.entity_type != query.entity_type:
                    continue

            results.append((head, relation, tail))

            if len(results) >= query.top_k:
                break

        return results

    def statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        type_counts = {t: len(e) for t, e in self._by_type.items()}
        relation_counts = {}

        for head, relation, tail in self._edges:
            relation_counts[relation] = relation_counts.get(relation, 0) + 1

        return {
            "n_entities": len(self._entities),
            "n_relations": len(self._relations),
            "n_edges": len(self._edges),
            "entity_types": type_counts,
            "relation_counts": relation_counts,
            "embedding_dim": self.embedding_dim,
        }
