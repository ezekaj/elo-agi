"""
Collective Memory: Shared memory enabling coordination.

Implements a shared knowledge base that:
- Stores contributions from individual agents
- Tracks who contributed what
- Decays unused memories
- Consolidates redundant information

Based on:
- Emergent Collective Memory (arXiv:2512.10166)
- BMAM: Brain-inspired Multi-Agent Memory (arXiv:2601.20465)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import time


@dataclass
class MemoryParams:
    """Parameters for collective memory."""

    capacity: int = 1000
    decay_rate: float = 0.01
    consolidation_threshold: float = 0.8  # Similarity for merging
    min_confidence: float = 0.1  # Below this, memory is forgotten
    retrieval_k: int = 5  # Default number of retrievals


@dataclass
class MemoryEntry:
    """A single entry in collective memory."""

    key: str
    value: Any
    embedding: np.ndarray
    contributors: List[str]
    confidence: float
    access_count: int
    created_at: float
    last_accessed: float

    def access(self) -> None:
        """Record an access to this memory."""
        self.access_count += 1
        self.last_accessed = time.time()

    def age(self) -> float:
        """Get age of memory in seconds."""
        return time.time() - self.created_at


class CollectiveMemory:
    """
    Shared memory enabling coordination without direct communication.

    Agents contribute knowledge to the collective memory, which:
    - Stores and indexes information for retrieval
    - Tracks contributions from each agent
    - Consolidates similar memories
    - Decays unused memories over time

    This enables stigmergic coordination where agents
    coordinate through the shared memory rather than
    direct communication.
    """

    def __init__(self, params: Optional[MemoryParams] = None):
        self.params = params or MemoryParams()

        # Memory storage
        self._memories: Dict[str, MemoryEntry] = {}

        # Index for retrieval
        self._embeddings: Dict[str, np.ndarray] = {}

        # Contribution tracking
        self._contributions: Dict[str, List[str]] = {}  # agent_id -> [keys]
        self._contribution_scores: Dict[str, float] = {}  # agent_id -> score

        # Statistics
        self._store_count = 0
        self._retrieve_count = 0
        self._consolidation_count = 0
        self._forget_count = 0

    def store(
        self,
        agent_id: str,
        key: str,
        value: Any,
        confidence: float = 0.5,
        embedding: Optional[np.ndarray] = None,
    ) -> bool:
        """
        Store knowledge from an agent.

        Args:
            agent_id: ID of contributing agent
            key: Unique key for the memory
            value: Content to store
            confidence: Confidence in the value
            embedding: Vector representation for retrieval

        Returns:
            True if stored successfully
        """
        # Generate embedding if not provided
        if embedding is None:
            embedding = self._generate_embedding(value)

        # Check capacity
        if len(self._memories) >= self.params.capacity:
            self._evict_oldest()

        # Check if key exists
        if key in self._memories:
            # Update existing memory
            existing = self._memories[key]
            if agent_id not in existing.contributors:
                existing.contributors.append(agent_id)
            # Increase confidence with multiple contributors
            existing.confidence = min(1.0, existing.confidence + confidence * 0.1)
            existing.last_accessed = time.time()
        else:
            # Create new memory
            entry = MemoryEntry(
                key=key,
                value=value,
                embedding=embedding,
                contributors=[agent_id],
                confidence=confidence,
                access_count=0,
                created_at=time.time(),
                last_accessed=time.time(),
            )
            self._memories[key] = entry
            self._embeddings[key] = embedding

        # Track contribution
        if agent_id not in self._contributions:
            self._contributions[agent_id] = []
        if key not in self._contributions[agent_id]:
            self._contributions[agent_id].append(key)

        # Update contribution score
        self._contribution_scores[agent_id] = (
            self._contribution_scores.get(agent_id, 0) + confidence
        )

        self._store_count += 1
        return True

    def retrieve(
        self,
        query: np.ndarray,
        n: Optional[int] = None,
    ) -> List[MemoryEntry]:
        """
        Retrieve relevant memories.

        Args:
            query: Query vector
            n: Number of memories to retrieve

        Returns:
            List of most relevant memory entries
        """
        if n is None:
            n = self.params.retrieval_k

        if not self._memories:
            return []

        self._retrieve_count += 1

        # Compute similarities
        similarities = []
        for key, embedding in self._embeddings.items():
            if key in self._memories:
                # Cosine similarity
                sim = np.dot(query, embedding) / (
                    np.linalg.norm(query) * np.linalg.norm(embedding) + 1e-8
                )
                similarities.append((sim, key))

        # Sort by similarity
        similarities.sort(key=lambda x: x[0], reverse=True)

        # Return top-n
        results = []
        for sim, key in similarities[:n]:
            entry = self._memories[key]
            entry.access()
            results.append(entry)

        return results

    def retrieve_by_key(self, key: str) -> Optional[MemoryEntry]:
        """Retrieve a specific memory by key."""
        if key in self._memories:
            entry = self._memories[key]
            entry.access()
            self._retrieve_count += 1
            return entry
        return None

    def consolidate(self) -> int:
        """
        Merge redundant memories, strengthen consistent ones.

        Returns:
            Number of memories consolidated
        """
        if len(self._memories) < 2:
            return 0

        keys = list(self._memories.keys())
        to_merge: List[Tuple[str, str]] = []

        # Find similar memories
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                key_i, key_j = keys[i], keys[j]
                emb_i = self._embeddings.get(key_i)
                emb_j = self._embeddings.get(key_j)

                if emb_i is not None and emb_j is not None:
                    sim = np.dot(emb_i, emb_j) / (
                        np.linalg.norm(emb_i) * np.linalg.norm(emb_j) + 1e-8
                    )
                    if sim > self.params.consolidation_threshold:
                        to_merge.append((key_i, key_j))

        # Merge similar memories
        merged_count = 0
        merged_keys = set()

        for key_i, key_j in to_merge:
            if key_i in merged_keys or key_j in merged_keys:
                continue

            mem_i = self._memories.get(key_i)
            mem_j = self._memories.get(key_j)

            if mem_i and mem_j:
                # Keep the one with more contributors
                if len(mem_i.contributors) >= len(mem_j.contributors):
                    keeper, remove = mem_i, mem_j
                    keeper_key, remove_key = key_i, key_j
                else:
                    keeper, remove = mem_j, mem_i
                    keeper_key, remove_key = key_j, key_i

                # Merge contributors
                for c in remove.contributors:
                    if c not in keeper.contributors:
                        keeper.contributors.append(c)

                # Boost confidence
                keeper.confidence = min(1.0, keeper.confidence + remove.confidence * 0.5)

                # Remove the merged memory
                del self._memories[remove_key]
                if remove_key in self._embeddings:
                    del self._embeddings[remove_key]

                merged_keys.add(remove_key)
                merged_count += 1

        self._consolidation_count += merged_count
        return merged_count

    def forget(self) -> int:
        """
        Decay unused memories.

        Returns:
            Number of memories forgotten
        """
        current_time = time.time()
        to_forget = []

        for key, entry in self._memories.items():
            # Apply decay based on age and access
            age = current_time - entry.last_accessed
            decay = self.params.decay_rate * age / 60  # Per minute
            entry.confidence *= 1 - decay

            # Mark for removal if below threshold
            if entry.confidence < self.params.min_confidence:
                to_forget.append(key)

        # Remove forgotten memories
        for key in to_forget:
            del self._memories[key]
            if key in self._embeddings:
                del self._embeddings[key]

        self._forget_count += len(to_forget)
        return len(to_forget)

    def _generate_embedding(self, value: Any) -> np.ndarray:
        """Generate embedding from value."""
        if isinstance(value, np.ndarray):
            # Normalize and resize
            emb = value.flatten()[:64]
            if len(emb) < 64:
                emb = np.pad(emb, (0, 64 - len(emb)))
            return emb / (np.linalg.norm(emb) + 1e-8)

        elif isinstance(value, (list, tuple)):
            arr = np.array(value, dtype=float).flatten()[:64]
            if len(arr) < 64:
                arr = np.pad(arr, (0, 64 - len(arr)))
            return arr / (np.linalg.norm(arr) + 1e-8)

        elif isinstance(value, str):
            # Simple hash-based embedding
            emb = np.zeros(64)
            for i, char in enumerate(value[:64]):
                emb[i % 64] += ord(char) / 256
            return emb / (np.linalg.norm(emb) + 1e-8)

        elif isinstance(value, (int, float)):
            emb = np.zeros(64)
            emb[0] = float(value)
            return emb / (np.linalg.norm(emb) + 1e-8)

        else:
            # Random embedding as fallback
            return np.random.randn(64)

    def _evict_oldest(self) -> None:
        """Evict oldest memory to make space."""
        if not self._memories:
            return

        # Find oldest
        oldest_key = min(self._memories.keys(), key=lambda k: self._memories[k].last_accessed)

        del self._memories[oldest_key]
        if oldest_key in self._embeddings:
            del self._embeddings[oldest_key]

    def get_agent_contribution(self, agent_id: str) -> float:
        """Get contribution score for an agent."""
        return self._contribution_scores.get(agent_id, 0.0)

    def get_agent_memories(self, agent_id: str) -> List[MemoryEntry]:
        """Get all memories contributed by an agent."""
        keys = self._contributions.get(agent_id, [])
        return [self._memories[k] for k in keys if k in self._memories]

    def get_top_contributors(self, n: int = 5) -> List[Tuple[str, float]]:
        """Get top contributing agents."""
        sorted_contributors = sorted(
            self._contribution_scores.items(), key=lambda x: x[1], reverse=True
        )
        return sorted_contributors[:n]

    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics."""
        if not self._memories:
            return {
                "size": 0,
                "store_count": self._store_count,
                "retrieve_count": self._retrieve_count,
            }

        confidences = [m.confidence for m in self._memories.values()]
        ages = [m.age() for m in self._memories.values()]
        access_counts = [m.access_count for m in self._memories.values()]

        return {
            "size": len(self._memories),
            "capacity": self.params.capacity,
            "utilization": len(self._memories) / self.params.capacity,
            "store_count": self._store_count,
            "retrieve_count": self._retrieve_count,
            "consolidation_count": self._consolidation_count,
            "forget_count": self._forget_count,
            "mean_confidence": float(np.mean(confidences)),
            "mean_age": float(np.mean(ages)),
            "mean_access_count": float(np.mean(access_counts)),
            "n_contributors": len(self._contributions),
        }

    def reset(self) -> None:
        """Reset all memory."""
        self._memories = {}
        self._embeddings = {}
        self._contributions = {}
        self._contribution_scores = {}
        self._store_count = 0
        self._retrieve_count = 0
        self._consolidation_count = 0
        self._forget_count = 0
