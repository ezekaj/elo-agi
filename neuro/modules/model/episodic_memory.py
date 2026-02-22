"""
Episodic Memory System

Ported from AGIELO with enhancements:
- Hot/cold storage tiers for efficient access
- Multi-index retrieval (temporal, entity, topic)
- Importance-based decay for forgetting
- Consolidation to long-term storage
"""

import json
import hashlib
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class Episode:
    """A single episodic memory."""

    id: str
    content: str
    embedding: np.ndarray
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)
    entities: List[str] = field(default_factory=list)
    topic: str = ""
    importance: float = 0.5
    access_count: int = 0
    last_accessed: datetime = None
    consolidated: bool = False

    def __post_init__(self):
        if self.last_accessed is None:
            self.last_accessed = self.timestamp


class EpisodicMemoryStore:
    """
    Episodic memory with hot/cold storage and multi-index retrieval.

    Features:
    - Hot cache for recent/frequent memories
    - Cold store for older memories
    - Multi-index retrieval (temporal, entity, topic)
    - Importance decay for forgetting
    - Consolidation for long-term storage
    """

    def __init__(
        self,
        storage_path: str = None,
        hot_cache_size: int = 1000,
        cold_threshold_days: int = 7,
        decay_rate: float = 0.01,
    ):
        self.storage_path = Path(storage_path or "~/.neuro/episodic").expanduser()
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.hot_cache_size = hot_cache_size
        self.cold_threshold_days = cold_threshold_days
        self.decay_rate = decay_rate

        # Storage tiers
        self.hot_cache: Dict[str, Episode] = {}
        self.cold_store: Dict[str, Episode] = {}

        # Multi-indices
        self.temporal_index: Dict[str, List[str]] = defaultdict(list)  # date -> episode_ids
        self.entity_index: Dict[str, List[str]] = defaultdict(list)  # entity -> episode_ids
        self.topic_index: Dict[str, List[str]] = defaultdict(list)  # topic -> episode_ids

        # Load existing memories
        self._load()

    def _create_embedding(self, text: str, dim: int = 128) -> np.ndarray:
        """Create embedding for text."""
        h = hashlib.sha256(text.encode()).digest()
        full_hash = h * (dim // len(h) + 1)
        return np.array([b / 255.0 for b in full_hash[:dim]]) * 2 - 1

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity."""
        norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def store(
        self,
        content: str,
        context: Dict[str, Any] = None,
        entities: List[str] = None,
        topic: str = "",
        importance: float = 0.5,
    ) -> str:
        """Store an episodic memory."""
        episode_id = f"ep_{int(datetime.now().timestamp())}_{len(self.hot_cache)}"

        episode = Episode(
            id=episode_id,
            content=content,
            embedding=self._create_embedding(content),
            timestamp=datetime.now(),
            context=context or {},
            entities=entities or [],
            topic=topic,
            importance=importance,
        )

        # Add to hot cache
        self.hot_cache[episode_id] = episode

        # Update indices
        self._index_episode(episode)

        # Move old episodes to cold storage if needed
        self._manage_cache()

        return episode_id

    def _index_episode(self, episode: Episode) -> None:
        """Add episode to all indices."""
        # Temporal index (by date)
        date_key = episode.timestamp.strftime("%Y-%m-%d")
        self.temporal_index[date_key].append(episode.id)

        # Entity index
        for entity in episode.entities:
            self.entity_index[entity.lower()].append(episode.id)

        # Topic index
        if episode.topic:
            self.topic_index[episode.topic.lower()].append(episode.id)

    def _manage_cache(self) -> None:
        """Move old/unused episodes from hot to cold storage."""
        if len(self.hot_cache) <= self.hot_cache_size:
            return

        now = datetime.now()
        threshold = now - timedelta(days=self.cold_threshold_days)

        # Find candidates for cold storage
        to_move = []
        for eid, episode in self.hot_cache.items():
            # Move if old and not frequently accessed
            if episode.last_accessed < threshold and episode.access_count < 5:
                to_move.append(eid)

        # Move to cold storage
        for eid in to_move[: len(self.hot_cache) - self.hot_cache_size]:
            self.cold_store[eid] = self.hot_cache.pop(eid)

    def retrieve(
        self,
        query: str = None,
        embedding: np.ndarray = None,
        entities: List[str] = None,
        topic: str = None,
        start_date: datetime = None,
        end_date: datetime = None,
        k: int = 5,
    ) -> List[Tuple[Episode, float]]:
        """
        Retrieve episodes using multi-index lookup.

        Can filter by:
        - Semantic similarity (query/embedding)
        - Entities mentioned
        - Topic
        - Time range
        """
        candidate_ids = set()
        scores: Dict[str, float] = defaultdict(float)

        # Entity-based retrieval
        if entities:
            for entity in entities:
                entity_lower = entity.lower()
                if entity_lower in self.entity_index:
                    for eid in self.entity_index[entity_lower]:
                        candidate_ids.add(eid)
                        scores[eid] += 2.0

        # Topic-based retrieval
        if topic:
            topic_lower = topic.lower()
            if topic_lower in self.topic_index:
                for eid in self.topic_index[topic_lower]:
                    candidate_ids.add(eid)
                    scores[eid] += 3.0

        # Time-based retrieval
        if start_date or end_date:
            for date_str, eids in self.temporal_index.items():
                try:
                    date = datetime.strptime(date_str, "%Y-%m-%d")
                    if start_date and date < start_date:
                        continue
                    if end_date and date > end_date:
                        continue
                    for eid in eids:
                        candidate_ids.add(eid)
                        scores[eid] += 1.0
                except ValueError:
                    pass

        # If no filters, consider all episodes
        if not candidate_ids:
            candidate_ids = set(self.hot_cache.keys()) | set(self.cold_store.keys())

        # Semantic similarity
        if query or embedding is not None:
            query_emb = embedding if embedding is not None else self._create_embedding(query)

            for eid in candidate_ids:
                episode = self._get_episode(eid)
                if episode:
                    sim = self._cosine_similarity(query_emb, episode.embedding)
                    scores[eid] += sim * 5.0  # High weight for semantic similarity

        # Apply importance and recency weights
        now = datetime.now()
        for eid in candidate_ids:
            episode = self._get_episode(eid)
            if episode:
                # Importance weight
                scores[eid] *= 0.5 + episode.importance

                # Recency decay
                days_old = (now - episode.timestamp).days
                recency = max(0.1, 1.0 - days_old * self.decay_rate)
                scores[eid] *= recency

        # Sort and return top k
        sorted_episodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for eid, score in sorted_episodes[:k]:
            episode = self._get_episode(eid)
            if episode:
                episode.access_count += 1
                episode.last_accessed = now
                results.append((episode, score))

        return results

    def _get_episode(self, episode_id: str) -> Optional[Episode]:
        """Get an episode from hot or cold storage."""
        if episode_id in self.hot_cache:
            return self.hot_cache[episode_id]
        return self.cold_store.get(episode_id)

    def consolidate(self) -> int:
        """
        Consolidate memories (like sleep consolidation).

        Moves important, frequently accessed memories to long-term storage.
        Returns number of memories consolidated.
        """
        consolidated = 0
        datetime.now()

        for episode in list(self.hot_cache.values()):
            if episode.consolidated:
                continue

            # Consolidate if important and accessed multiple times
            if episode.importance > 0.7 or episode.access_count > 3:
                episode.consolidated = True
                episode.importance = min(1.0, episode.importance * 1.2)  # Boost importance
                consolidated += 1

        return consolidated

    def forget(self, threshold: float = 0.1) -> int:
        """
        Forget low-importance, old memories.

        Returns number of memories forgotten.
        """
        forgotten = 0
        now = datetime.now()

        # Only forget from cold storage
        to_forget = []
        for eid, episode in self.cold_store.items():
            days_old = (now - episode.timestamp).days
            decayed_importance = episode.importance * (1 - self.decay_rate * days_old)

            if decayed_importance < threshold and episode.access_count < 2:
                to_forget.append(eid)

        for eid in to_forget:
            del self.cold_store[eid]
            forgotten += 1

        return forgotten

    def save(self) -> None:
        """Save all memories to disk."""
        # Save hot cache
        hot_data = {
            eid: {
                "id": ep.id,
                "content": ep.content,
                "embedding": ep.embedding.tolist(),
                "timestamp": ep.timestamp.isoformat(),
                "context": ep.context,
                "entities": ep.entities,
                "topic": ep.topic,
                "importance": ep.importance,
                "access_count": ep.access_count,
                "last_accessed": ep.last_accessed.isoformat(),
                "consolidated": ep.consolidated,
            }
            for eid, ep in self.hot_cache.items()
        }

        with open(self.storage_path / "hot_cache.json", "w") as f:
            json.dump(hot_data, f)

        # Save cold store
        cold_data = {
            eid: {
                "id": ep.id,
                "content": ep.content,
                "embedding": ep.embedding.tolist(),
                "timestamp": ep.timestamp.isoformat(),
                "context": ep.context,
                "entities": ep.entities,
                "topic": ep.topic,
                "importance": ep.importance,
                "access_count": ep.access_count,
                "last_accessed": ep.last_accessed.isoformat(),
                "consolidated": ep.consolidated,
            }
            for eid, ep in self.cold_store.items()
        }

        with open(self.storage_path / "cold_store.json", "w") as f:
            json.dump(cold_data, f)

        # Save indices
        indices = {
            "temporal": dict(self.temporal_index),
            "entity": dict(self.entity_index),
            "topic": dict(self.topic_index),
        }

        with open(self.storage_path / "indices.json", "w") as f:
            json.dump(indices, f)

    def _load(self) -> None:
        """Load memories from disk."""
        # Load hot cache
        hot_file = self.storage_path / "hot_cache.json"
        if hot_file.exists():
            try:
                with open(hot_file) as f:
                    hot_data = json.load(f)
                for eid, data in hot_data.items():
                    self.hot_cache[eid] = Episode(
                        id=data["id"],
                        content=data["content"],
                        embedding=np.array(data["embedding"]),
                        timestamp=datetime.fromisoformat(data["timestamp"]),
                        context=data.get("context", {}),
                        entities=data.get("entities", []),
                        topic=data.get("topic", ""),
                        importance=data.get("importance", 0.5),
                        access_count=data.get("access_count", 0),
                        last_accessed=datetime.fromisoformat(
                            data.get("last_accessed", data["timestamp"])
                        ),
                        consolidated=data.get("consolidated", False),
                    )
            except Exception:
                pass

        # Load cold store
        cold_file = self.storage_path / "cold_store.json"
        if cold_file.exists():
            try:
                with open(cold_file) as f:
                    cold_data = json.load(f)
                for eid, data in cold_data.items():
                    self.cold_store[eid] = Episode(
                        id=data["id"],
                        content=data["content"],
                        embedding=np.array(data["embedding"]),
                        timestamp=datetime.fromisoformat(data["timestamp"]),
                        context=data.get("context", {}),
                        entities=data.get("entities", []),
                        topic=data.get("topic", ""),
                        importance=data.get("importance", 0.5),
                        access_count=data.get("access_count", 0),
                        last_accessed=datetime.fromisoformat(
                            data.get("last_accessed", data["timestamp"])
                        ),
                        consolidated=data.get("consolidated", False),
                    )
            except Exception:
                pass

        # Load indices
        indices_file = self.storage_path / "indices.json"
        if indices_file.exists():
            try:
                with open(indices_file) as f:
                    indices = json.load(f)
                self.temporal_index = defaultdict(list, indices.get("temporal", {}))
                self.entity_index = defaultdict(list, indices.get("entity", {}))
                self.topic_index = defaultdict(list, indices.get("topic", {}))
            except Exception:
                pass

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            "hot_cache_size": len(self.hot_cache),
            "cold_store_size": len(self.cold_store),
            "total_memories": len(self.hot_cache) + len(self.cold_store),
            "topics": len(self.topic_index),
            "entities": len(self.entity_index),
            "storage_path": str(self.storage_path),
        }


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("EPISODIC MEMORY TEST")
    print("=" * 60)

    store = EpisodicMemoryStore()

    # Store some episodes
    store.store(
        "Had a conversation about machine learning with Claude",
        entities=["Claude", "machine learning"],
        topic="AI",
        importance=0.8,
    )
    store.store(
        "Learned about neural network architectures",
        entities=["neural networks"],
        topic="AI",
        importance=0.7,
    )
    store.store(
        "Discussed Python programming best practices",
        entities=["Python"],
        topic="programming",
        importance=0.5,
    )

    # Test retrieval
    print("\nRetrieving by query 'machine learning':")
    results = store.retrieve(query="machine learning", k=3)
    for ep, score in results:
        print(f"  [{score:.2f}] {ep.content[:50]}...")

    print("\nRetrieving by entity 'Python':")
    results = store.retrieve(entities=["Python"], k=3)
    for ep, score in results:
        print(f"  [{score:.2f}] {ep.content[:50]}...")

    print("\nRetrieving by topic 'AI':")
    results = store.retrieve(topic="AI", k=3)
    for ep, score in results:
        print(f"  [{score:.2f}] {ep.content[:50]}...")

    # Consolidate
    consolidated = store.consolidate()
    print(f"\nConsolidated {consolidated} memories")

    # Save
    store.save()

    print(f"\nStats: {store.get_stats()}")
