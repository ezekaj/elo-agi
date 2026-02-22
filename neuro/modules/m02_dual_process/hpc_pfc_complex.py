"""
HPC-PFC Complex - Hippocampus-Prefrontal Cortex Integration

Implements the circuit crucial for compositional thinking:
- Hippocampus: Rapid episodic encoding, cognitive maps
- Prefrontal Cortex: Schema extraction, abstraction
- Integration: Combine episodic + schematic for novel compositions

This enables combining known concepts into novel configurations
(e.g., "jump twice" requires binding "jump" with "twice").
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import time
from collections import defaultdict
import hashlib


@dataclass
class Episode:
    """An episodic memory - specific experience with context"""

    id: str
    content: Any
    context: Dict[str, Any]
    timestamp: float
    encoding_strength: float = 1.0
    retrieval_count: int = 0
    associated_episodes: List[str] = field(default_factory=list)


@dataclass
class CognitiveMapNode:
    """A node in the cognitive map (place cell analog)"""

    id: str
    position: np.ndarray  # Abstract position in cognitive space
    associations: Dict[str, float]  # Connected nodes with weights
    content: Any = None


@dataclass
class Schema:
    """An abstracted pattern extracted from episodes"""

    id: str
    structure: Dict[str, Any]  # Abstract structure with slots
    slot_fillers: Dict[str, List[Any]]  # Examples of what fills each slot
    source_episodes: List[str]
    confidence: float = 0.5
    use_count: int = 0


class Hippocampus:
    """
    Rapid episodic encoding and cognitive maps.

    The hippocampus rapidly binds disparate elements into coherent
    episodes and creates cognitive maps of relationships.
    """

    def __init__(self, max_episodes: int = 10000, decay_rate: float = 0.001):
        self.episodes: Dict[str, Episode] = {}
        self.cognitive_map: Dict[str, CognitiveMapNode] = {}
        self.max_episodes = max_episodes
        self.decay_rate = decay_rate

        # Index for fast retrieval by content features
        self._content_index: Dict[str, List[str]] = defaultdict(list)
        self._context_index: Dict[str, List[str]] = defaultdict(list)

    def encode_episode(
        self, content: Any, context: Dict[str, Any], strength: float = 1.0
    ) -> Episode:
        """
        Rapidly encode an episodic memory.

        Hippocampus excels at one-shot learning - single exposure
        can create lasting memory.
        """
        episode_id = self._generate_id(content, context)
        timestamp = time.time()

        episode = Episode(
            id=episode_id,
            content=content,
            context=context,
            timestamp=timestamp,
            encoding_strength=strength,
        )

        self.episodes[episode_id] = episode

        # Index for retrieval
        self._index_episode(episode)

        # Consolidation limit
        if len(self.episodes) > self.max_episodes:
            self._consolidate_oldest()

        # Update cognitive map
        self._update_cognitive_map(episode)

        return episode

    def _generate_id(self, content: Any, context: Dict) -> str:
        """Generate unique ID for episode"""
        combined = str(content) + str(sorted(context.items()))
        return hashlib.md5(combined.encode()).hexdigest()[:12]

    def _index_episode(self, episode: Episode):
        """Index episode for fast retrieval"""
        # Content-based indexing
        content_key = str(type(episode.content).__name__)
        self._content_index[content_key].append(episode.id)

        # Context-based indexing
        for key, value in episode.context.items():
            context_key = f"{key}:{value}"
            self._context_index[context_key].append(episode.id)

    def retrieve_episode(
        self, cue: Any, context: Optional[Dict[str, Any]] = None, top_k: int = 5
    ) -> List[Episode]:
        """
        Retrieve episodes by cue (pattern completion).

        Hippocampus can reconstruct full episode from partial cue.
        """
        candidates = []

        # Content-based retrieval
        content_key = str(type(cue).__name__)
        if content_key in self._content_index:
            for ep_id in self._content_index[content_key]:
                if ep_id in self.episodes:
                    candidates.append(self.episodes[ep_id])

        # Context-based retrieval
        if context:
            for key, value in context.items():
                context_key = f"{key}:{value}"
                if context_key in self._context_index:
                    for ep_id in self._context_index[context_key]:
                        if ep_id in self.episodes and self.episodes[ep_id] not in candidates:
                            candidates.append(self.episodes[ep_id])

        # Score and rank candidates
        scored = []
        for ep in candidates:
            score = self._compute_retrieval_score(ep, cue, context)
            scored.append((ep, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        # Update retrieval counts
        results = []
        for ep, score in scored[:top_k]:
            ep.retrieval_count += 1
            results.append(ep)

        return results

    def _compute_retrieval_score(
        self, episode: Episode, cue: Any, context: Optional[Dict]
    ) -> float:
        """Score episode match to cue"""
        score = episode.encoding_strength

        # Recency bonus
        age = time.time() - episode.timestamp
        recency_factor = np.exp(-self.decay_rate * age / 3600)  # Decay over hours
        score *= recency_factor

        # Retrieval strength bonus (frequently retrieved = stronger)
        score *= 1.0 + 0.1 * min(episode.retrieval_count, 10)

        # Context match bonus
        if context:
            matches = sum(1 for k, v in context.items() if episode.context.get(k) == v)
            score *= 1.0 + 0.2 * matches

        return score

    def _update_cognitive_map(self, episode: Episode):
        """Update cognitive map with new episode"""
        # Create node for episode if needed
        if episode.id not in self.cognitive_map:
            # Position based on context features
            position = self._context_to_position(episode.context)
            self.cognitive_map[episode.id] = CognitiveMapNode(
                id=episode.id, position=position, associations={}, content=episode.content
            )

        # Link to nearby episodes in cognitive space
        node = self.cognitive_map[episode.id]
        for other_id, other_node in self.cognitive_map.items():
            if other_id != episode.id:
                distance = np.linalg.norm(node.position - other_node.position)
                if distance < 2.0:  # Close in cognitive space
                    weight = 1.0 / (1.0 + distance)
                    node.associations[other_id] = weight
                    other_node.associations[episode.id] = weight

    def _context_to_position(self, context: Dict[str, Any]) -> np.ndarray:
        """Map context to position in cognitive space"""
        # Simple hashing approach - real implementation would be learned
        features = []
        for key, value in sorted(context.items()):
            hash_val = hash(f"{key}:{value}") % 1000
            features.append(hash_val / 1000.0)

        # Pad or truncate to fixed size
        while len(features) < 8:
            features.append(0.0)
        features = features[:8]

        return np.array(features)

    def replay(self, episodes: Optional[List[Episode]] = None) -> List[Episode]:
        """
        Replay episodes for consolidation.

        Hippocampal replay strengthens memories and enables
        transfer to neocortex (schemas).
        """
        if episodes is None:
            # Replay recent episodes
            recent = sorted(self.episodes.values(), key=lambda e: e.timestamp, reverse=True)[:10]
            episodes = recent

        for ep in episodes:
            ep.encoding_strength = min(1.0, ep.encoding_strength * 1.1)

        return episodes

    def _consolidate_oldest(self):
        """Remove oldest, least accessed episodes"""
        if len(self.episodes) <= self.max_episodes:
            return

        # Score by recency and access
        scored = []
        for ep_id, ep in self.episodes.items():
            age = time.time() - ep.timestamp
            score = ep.encoding_strength * (1 + ep.retrieval_count) / (1 + age / 3600)
            scored.append((ep_id, score))

        scored.sort(key=lambda x: x[1])

        # Remove bottom 10%
        to_remove = scored[: len(scored) // 10]
        for ep_id, _ in to_remove:
            del self.episodes[ep_id]
            if ep_id in self.cognitive_map:
                del self.cognitive_map[ep_id]

    def find_path(self, start_id: str, end_id: str) -> List[str]:
        """
        Find path through cognitive map.

        Navigation through cognitive space - used for inference.
        """
        if start_id not in self.cognitive_map or end_id not in self.cognitive_map:
            return []

        # BFS for shortest path
        visited = {start_id}
        queue = [(start_id, [start_id])]

        while queue:
            current, path = queue.pop(0)
            if current == end_id:
                return path

            node = self.cognitive_map.get(current)
            if node:
                for neighbor in node.associations:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, path + [neighbor]))

        return []


class PrefrontalCortex:
    """
    Schema extraction and abstraction.

    PFC extracts generalizable patterns from episodes,
    creating reusable schemas that can be applied to novel situations.
    """

    def __init__(self, min_examples_for_schema: int = 3):
        self.schemas: Dict[str, Schema] = {}
        self.min_examples = min_examples_for_schema

    def extract_schema(self, episodes: List[Episode]) -> Optional[Schema]:
        """
        Extract common structure from episodes.

        Finds what's constant across episodes (structure) vs
        what varies (slots that can be filled differently).
        """
        if len(episodes) < self.min_examples:
            return None

        # Find common context keys
        all_context_keys = [set(ep.context.keys()) for ep in episodes]
        common_keys = set.intersection(*all_context_keys) if all_context_keys else set()

        # Build structure with slots
        structure = {}
        slot_fillers = {}

        for key in common_keys:
            values = [ep.context.get(key) for ep in episodes]
            unique_values = set(str(v) for v in values)

            if len(unique_values) == 1:
                # Constant across episodes - part of structure
                structure[key] = values[0]
            else:
                # Variable - this is a slot
                structure[key] = f"<SLOT:{key}>"
                slot_fillers[key] = values

        if not structure:
            return None

        schema_id = hashlib.md5(str(sorted(structure.items())).encode()).hexdigest()[:12]

        schema = Schema(
            id=schema_id,
            structure=structure,
            slot_fillers=slot_fillers,
            source_episodes=[ep.id for ep in episodes],
            confidence=min(1.0, len(episodes) / 10.0),
        )

        self.schemas[schema_id] = schema
        return schema

    def consolidate(self, schema: Schema):
        """
        Strengthen schema for long-term use.

        With repeated use, schemas become more reliable.
        """
        schema.confidence = min(1.0, schema.confidence + 0.1)
        schema.use_count += 1

    def apply_schema(self, schema: Schema, slot_values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply schema to novel input by filling slots.

        This is generalization - using abstract pattern
        with new specific content.
        """
        result = schema.structure.copy()

        for key, value in result.items():
            if isinstance(value, str) and value.startswith("<SLOT:"):
                slot_name = value[6:-1]  # Extract slot name
                if slot_name in slot_values:
                    result[key] = slot_values[slot_name]

        schema.use_count += 1
        return result

    def find_matching_schema(self, context: Dict[str, Any]) -> Optional[Tuple[Schema, float]]:
        """Find schema that best matches given context"""
        best_schema = None
        best_score = 0.0

        for schema in self.schemas.values():
            score = self._schema_match_score(schema, context)
            if score > best_score:
                best_score = score
                best_schema = schema

        if best_schema and best_score > 0.5:
            return (best_schema, best_score)
        return None

    def _schema_match_score(self, schema: Schema, context: Dict[str, Any]) -> float:
        """Score how well context matches schema"""
        matches = 0
        total = 0

        for key, value in schema.structure.items():
            if isinstance(value, str) and value.startswith("<SLOT:"):
                # Slot - just check key exists
                if key in context:
                    matches += 0.5  # Half credit for slot match
                total += 1
            else:
                # Fixed value - must match exactly
                if context.get(key) == value:
                    matches += 1
                total += 1

        return matches / total if total > 0 else 0.0

    def merge_schemas(self, schema1: Schema, schema2: Schema) -> Optional[Schema]:
        """Merge two related schemas into more abstract one"""
        common_structure = {}

        for key in set(schema1.structure.keys()) & set(schema2.structure.keys()):
            v1 = schema1.structure[key]
            v2 = schema2.structure[key]

            if v1 == v2:
                common_structure[key] = v1
            else:
                # Differ - becomes slot
                common_structure[key] = f"<SLOT:{key}>"

        if not common_structure:
            return None

        return Schema(
            id=hashlib.md5(str(sorted(common_structure.items())).encode()).hexdigest()[:12],
            structure=common_structure,
            slot_fillers={},
            source_episodes=schema1.source_episodes + schema2.source_episodes,
            confidence=(schema1.confidence + schema2.confidence) / 2,
        )


class HPCPFCComplex:
    """
    Integrated Hippocampus-Prefrontal Cortex system.

    Combines episodic encoding with schema extraction
    for compositional thinking.
    """

    def __init__(self):
        self.hippocampus = Hippocampus()
        self.pfc = PrefrontalCortex()
        self._consolidation_buffer: List[Episode] = []

    def encode_and_abstract(
        self, content: Any, context: Dict[str, Any]
    ) -> Tuple[Episode, Optional[Schema]]:
        """
        Full pipeline: encode episode and extract/update schemas.

        This is the HPC-PFC interaction - HPC rapidly encodes,
        PFC gradually abstracts.
        """
        # HPC: Rapid encoding
        episode = self.hippocampus.encode_episode(content, context)
        self._consolidation_buffer.append(episode)

        # PFC: Try to find matching schema
        schema_match = self.pfc.find_matching_schema(context)
        if schema_match:
            schema, score = schema_match
            self.pfc.consolidate(schema)
            return episode, schema

        # Try to extract new schema from buffer
        if len(self._consolidation_buffer) >= self.pfc.min_examples:
            new_schema = self.pfc.extract_schema(self._consolidation_buffer)
            if new_schema:
                self._consolidation_buffer = []
                return episode, new_schema

        return episode, None

    def compose_novel(self, concepts: List[Any], relation: str = "combined") -> Dict[str, Any]:
        """
        Combine known concepts into novel configuration.

        This is the key compositional ability - "jump" + "twice" = "jump twice"
        """
        # Retrieve episodes for each concept
        all_contexts = []
        for concept in concepts:
            episodes = self.hippocampus.retrieve_episode(concept, top_k=3)
            for ep in episodes:
                all_contexts.append(ep.context)

        # Find or create schema that can combine them
        combined_context = {"components": concepts, "relation": relation, "novel": True}

        # Merge contexts from components
        for i, ctx in enumerate(all_contexts):
            for key, value in ctx.items():
                combined_context[f"component_{i}_{key}"] = value

        # Encode novel combination
        self.hippocampus.encode_episode(
            content={"composed": concepts, "relation": relation},
            context=combined_context,
            strength=0.8,  # Novel combinations start slightly weaker
        )

        return combined_context

    def retrieve_by_schema(self, schema: Schema, partial_context: Dict[str, Any]) -> List[Episode]:
        """Retrieve episodes that match a schema with partial slot filling"""
        # Apply schema to get full context pattern
        filled = self.pfc.apply_schema(schema, partial_context)

        # Retrieve matching episodes
        episodes = self.hippocampus.retrieve_episode(cue=None, context=filled, top_k=10)

        return episodes

    def sleep_consolidation(self):
        """
        Simulate sleep-based memory consolidation.

        During sleep, HPC replays episodes and PFC extracts schemas.
        """
        # Replay recent episodes
        replayed = self.hippocampus.replay()

        # Try to extract schemas from replayed episodes
        if len(replayed) >= 3:
            new_schema = self.pfc.extract_schema(replayed)
            if new_schema:
                # Link episodes to schema
                for ep in replayed:
                    ep.associated_episodes.append(f"schema:{new_schema.id}")

        # Strengthen frequently accessed schemas
        for schema in self.pfc.schemas.values():
            if schema.use_count > 5:
                schema.confidence = min(1.0, schema.confidence + 0.05)
