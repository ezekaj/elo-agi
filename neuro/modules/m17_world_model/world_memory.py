"""
World Memory: Persistent storage of world state knowledge.

The world memory maintains the agent's understanding of the current
world state, including object representations, spatial relationships,
and temporal changes.

Based on:
- Object-based attention and memory
- Situation models in cognitive science
- Knowledge representation systems
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum
import numpy as np
import time


class EntityType(Enum):
    """Types of entities in the world model."""

    OBJECT = "object"
    AGENT = "agent"
    LOCATION = "location"
    EVENT = "event"
    CONCEPT = "concept"
    SELF = "self"


class RelationType(Enum):
    """Types of relationships between entities."""

    SPATIAL = "spatial"  # X is near/far from Y
    TEMPORAL = "temporal"  # X happened before/after Y
    CAUSAL = "causal"  # X caused Y
    PART_OF = "part_of"  # X is part of Y
    OWNS = "owns"  # X owns Y
    SIMILAR = "similar"  # X is similar to Y
    INTERACTS = "interacts"  # X interacts with Y


@dataclass
class MemoryParams:
    """Parameters for world memory."""

    n_features: int = 128  # Feature dimensionality
    max_entities: int = 1000  # Maximum tracked entities
    max_relations: int = 10000  # Maximum relations
    decay_rate: float = 0.999  # Activation decay
    consolidation_threshold: float = 0.3  # Below this, memories fade
    attention_boost: float = 0.2  # Boost from attention


@dataclass
class Entity:
    """An entity in the world model."""

    entity_id: str
    entity_type: EntityType
    features: np.ndarray
    activation: float
    confidence: float
    last_updated: float = field(default_factory=time.time)
    created: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def decay(self, rate: float) -> None:
        """Apply activation decay."""
        self.activation *= rate

    def refresh(self, new_features: np.ndarray, confidence: float) -> None:
        """Update entity with new observation."""
        blend = confidence * 0.3
        self.features = (1 - blend) * self.features + blend * new_features
        self.activation = min(1.0, self.activation + 0.2)
        self.confidence = 0.7 * self.confidence + 0.3 * confidence
        self.last_updated = time.time()


@dataclass
class Relation:
    """A relation between entities."""

    source_id: str
    target_id: str
    relation_type: RelationType
    strength: float
    evidence: float
    timestamp: float = field(default_factory=time.time)

    def decay(self, rate: float) -> None:
        """Apply strength decay."""
        self.strength *= rate


@dataclass
class WorldState:
    """Snapshot of the current world state."""

    entities: Dict[str, Entity]
    relations: List[Relation]
    self_state: np.ndarray
    current_location: Optional[str]
    attention_focus: Optional[str]
    timestamp: float = field(default_factory=time.time)

    def get_active_entities(self, threshold: float = 0.1) -> List[Entity]:
        """Get entities with activation above threshold."""
        return [e for e in self.entities.values() if e.activation > threshold]


class WorldMemory:
    """
    Persistent memory of world state and knowledge.

    The world memory maintains a structured representation of the
    agent's understanding of the world, including:

    1. **Entities**: Objects, agents, locations in the world
    2. **Relations**: How entities relate to each other
    3. **Self-state**: The agent's own state
    4. **Dynamics**: How the world changes over time

    Key features:

    1. **Object permanence**: Entities persist even when not observed
    2. **Decay and consolidation**: Important memories strengthen, others fade
    3. **Relational reasoning**: Can query relationships between entities
    4. **Predictive**: Can predict likely states of unobserved entities

    This implements a form of situation model from cognitive science.
    """

    def __init__(self, params: Optional[MemoryParams] = None):
        self.params = params or MemoryParams()

        # Entity storage
        self._entities: Dict[str, Entity] = {}

        # Relation storage
        self._relations: List[Relation] = []
        self._relation_index: Dict[str, Set[int]] = {}  # entity_id -> relation indices

        # Self state
        self._self_state = np.zeros(self.params.n_features)
        self._self_entity = Entity(
            entity_id="self",
            entity_type=EntityType.SELF,
            features=self._self_state.copy(),
            activation=1.0,
            confidence=1.0,
        )
        self._entities["self"] = self._self_entity

        # Attention and focus
        self._attention_focus: Optional[str] = None
        self._current_location: Optional[str] = None

        # History
        self._state_history: List[WorldState] = []
        self._update_count = 0

    def add_entity(
        self,
        entity_id: str,
        entity_type: EntityType,
        features: np.ndarray,
        confidence: float = 0.8,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Entity:
        """Add or update an entity in the world model."""
        # Resize features if needed
        if len(features) != self.params.n_features:
            features = np.resize(features, self.params.n_features)

        if entity_id in self._entities:
            # Update existing entity
            self._entities[entity_id].refresh(features, confidence)
            if metadata:
                self._entities[entity_id].metadata.update(metadata)
        else:
            # Create new entity
            if len(self._entities) >= self.params.max_entities:
                self._evict_weakest_entity()

            entity = Entity(
                entity_id=entity_id,
                entity_type=entity_type,
                features=features.copy(),
                activation=0.8,
                confidence=confidence,
                metadata=metadata or {},
            )
            self._entities[entity_id] = entity
            self._relation_index[entity_id] = set()

        self._update_count += 1
        return self._entities[entity_id]

    def remove_entity(self, entity_id: str) -> bool:
        """Remove an entity and its relations."""
        if entity_id not in self._entities or entity_id == "self":
            return False

        # Remove relations involving this entity
        self._relations = [
            r for r in self._relations if r.source_id != entity_id and r.target_id != entity_id
        ]

        # Update relation index
        if entity_id in self._relation_index:
            del self._relation_index[entity_id]
        for other_id in self._relation_index:
            self._relation_index[other_id] = {
                i
                for i, r in enumerate(self._relations)
                if r.source_id == other_id or r.target_id == other_id
            }

        del self._entities[entity_id]
        return True

    def add_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: RelationType,
        strength: float = 0.5,
        evidence: float = 0.8,
    ) -> Optional[Relation]:
        """Add or update a relation between entities."""
        if source_id not in self._entities or target_id not in self._entities:
            return None

        # Check for existing relation
        for r in self._relations:
            if (
                r.source_id == source_id
                and r.target_id == target_id
                and r.relation_type == relation_type
            ):
                # Update existing
                r.strength = 0.7 * r.strength + 0.3 * strength
                r.evidence = 0.7 * r.evidence + 0.3 * evidence
                r.timestamp = time.time()
                return r

        # Create new relation
        if len(self._relations) >= self.params.max_relations:
            self._evict_weakest_relation()

        relation = Relation(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            strength=strength,
            evidence=evidence,
        )
        idx = len(self._relations)
        self._relations.append(relation)

        # Update indices
        if source_id in self._relation_index:
            self._relation_index[source_id].add(idx)
        if target_id in self._relation_index:
            self._relation_index[target_id].add(idx)

        return relation

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by ID."""
        return self._entities.get(entity_id)

    def get_related_entities(
        self,
        entity_id: str,
        relation_type: Optional[RelationType] = None,
    ) -> List[Tuple[Entity, Relation]]:
        """Get entities related to a given entity."""
        if entity_id not in self._relation_index:
            return []

        results = []
        for idx in self._relation_index[entity_id]:
            if idx >= len(self._relations):
                continue
            r = self._relations[idx]

            if relation_type is not None and r.relation_type != relation_type:
                continue

            # Get the other entity
            other_id = r.target_id if r.source_id == entity_id else r.source_id
            if other_id in self._entities:
                results.append((self._entities[other_id], r))

        return results

    def query_entities(
        self,
        query_features: np.ndarray,
        entity_type: Optional[EntityType] = None,
        top_k: int = 5,
    ) -> List[Tuple[Entity, float]]:
        """Query entities by feature similarity."""
        if len(query_features) != self.params.n_features:
            query_features = np.resize(query_features, self.params.n_features)

        results = []
        for entity in self._entities.values():
            if entity_type is not None and entity.entity_type != entity_type:
                continue

            # Cosine similarity
            similarity = self._cosine_similarity(query_features, entity.features)
            # Weight by activation and confidence
            score = similarity * entity.activation * entity.confidence
            results.append((entity, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def update_self_state(self, state: np.ndarray) -> None:
        """Update the agent's own state."""
        if len(state) != self.params.n_features:
            state = np.resize(state, self.params.n_features)
        self._self_state = state.copy()
        self._self_entity.features = state.copy()
        self._self_entity.last_updated = time.time()

    def set_attention(self, entity_id: str) -> None:
        """Set attention focus to an entity."""
        if entity_id in self._entities:
            self._attention_focus = entity_id
            self._entities[entity_id].activation = min(
                1.0, self._entities[entity_id].activation + self.params.attention_boost
            )

    def set_location(self, location_id: str) -> None:
        """Set current location."""
        if location_id in self._entities:
            self._current_location = location_id

    def step(self) -> None:
        """Advance world memory by one time step."""
        # Decay all activations
        for entity in self._entities.values():
            if entity.entity_id != "self":
                entity.decay(self.params.decay_rate)

        # Decay relations
        for relation in self._relations:
            relation.decay(self.params.decay_rate)

        # Consolidate or evict weak entities
        to_remove = []
        for entity_id, entity in self._entities.items():
            if (
                entity.entity_id != "self"
                and entity.activation < self.params.consolidation_threshold
            ):
                to_remove.append(entity_id)

        for entity_id in to_remove[:10]:  # Limit removals per step
            self.remove_entity(entity_id)

        self._update_count += 1

    def _evict_weakest_entity(self) -> None:
        """Evict the entity with lowest activation."""
        weakest_id = None
        weakest_activation = float("inf")

        for entity_id, entity in self._entities.items():
            if entity_id == "self":
                continue
            if entity.activation < weakest_activation:
                weakest_activation = entity.activation
                weakest_id = entity_id

        if weakest_id:
            self.remove_entity(weakest_id)

    def _evict_weakest_relation(self) -> None:
        """Evict the relation with lowest strength."""
        if not self._relations:
            return

        weakest_idx = min(range(len(self._relations)), key=lambda i: self._relations[i].strength)

        r = self._relations[weakest_idx]
        if r.source_id in self._relation_index:
            self._relation_index[r.source_id].discard(weakest_idx)
        if r.target_id in self._relation_index:
            self._relation_index[r.target_id].discard(weakest_idx)

        self._relations.pop(weakest_idx)

        # Update indices
        for entity_id in self._relation_index:
            self._relation_index[entity_id] = {
                (i - 1 if i > weakest_idx else i)
                for i in self._relation_index[entity_id]
                if i != weakest_idx
            }

    def get_current_state(self) -> WorldState:
        """Get current world state snapshot."""
        return WorldState(
            entities=self._entities.copy(),
            relations=self._relations.copy(),
            self_state=self._self_state.copy(),
            current_location=self._current_location,
            attention_focus=self._attention_focus,
        )

    def predict_entity_state(
        self,
        entity_id: str,
        time_delta: float = 1.0,
    ) -> Optional[np.ndarray]:
        """
        Predict likely state of an entity in the future.

        Simple linear prediction based on recent updates.
        """
        if entity_id not in self._entities:
            return None

        entity = self._entities[entity_id]

        # Get related entities for context
        related = self.get_related_entities(entity_id, RelationType.CAUSAL)

        # Simple prediction: current state with decay
        predicted = entity.features * (self.params.decay_rate**time_delta)

        # Add influence from causal relations
        for related_entity, relation in related:
            if relation.target_id == entity_id:
                influence = related_entity.features * relation.strength * 0.1
                predicted = predicted + influence

        return predicted

    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics."""
        active_entities = [
            e for e in self._entities.values() if e.activation > self.params.consolidation_threshold
        ]

        return {
            "n_entities": len(self._entities),
            "n_active_entities": len(active_entities),
            "n_relations": len(self._relations),
            "update_count": self._update_count,
            "avg_activation": float(np.mean([e.activation for e in self._entities.values()])),
            "attention_focus": self._attention_focus,
            "current_location": self._current_location,
            "entity_types": {
                et.value: sum(1 for e in self._entities.values() if e.entity_type == et)
                for et in EntityType
            },
        }

    def reset(self) -> None:
        """Reset world memory."""
        self._entities = {"self": self._self_entity}
        self._relations = []
        self._relation_index = {"self": set()}
        self._self_state = np.zeros(self.params.n_features)
        self._self_entity.features = self._self_state.copy()
        self._attention_focus = None
        self._current_location = None
        self._state_history = []
        self._update_count = 0
