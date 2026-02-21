"""
Knowledge Base - Persistent Self-Training System

Ported from AGIELO with enhancements:
- Embedding-based semantic retrieval
- Stop-word filtering for better matching
- Multi-concept matching (understands whole queries)
- Auto-save with persistence
- Knowledge decay for relevance
"""

import os
import json
import pickle
import hashlib
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class Fact:
    """A single fact in the knowledge base."""
    id: str
    topic: str
    content: str
    source: str
    embedding: Optional[np.ndarray] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    access_count: int = 0
    importance: float = 0.5
    confidence: float = 0.8


class KnowledgeBase:
    """
    Persistent knowledge base that grows over time.

    Features:
    - Semantic search using embeddings
    - Stop-word filtering for better matching
    - Multi-index retrieval
    - Importance-based ranking
    - Automatic persistence
    """

    # Stop words for filtering
    STOP_WORDS = {
        'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
        'ought', 'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
        'from', 'as', 'into', 'through', 'during', 'before', 'after', 'above',
        'below', 'between', 'under', 'again', 'further', 'then', 'once',
        'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each', 'few',
        'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
        'own', 'same', 'so', 'than', 'too', 'very', 'just', 'and', 'but',
        'if', 'or', 'because', 'until', 'while', 'about', 'against', 'what',
        'which', 'who', 'this', 'that', 'these', 'those', 'am', 'it', 'its',
        'i', 'me', 'my', 'myself', 'we', 'our', 'you', 'your', 'he', 'him',
        'his', 'she', 'her', 'they', 'them', 'their', 'find', 'search',
        'look', 'get', 'show', 'tell', 'give', 'know', 'think', 'want'
    }

    def __init__(self, storage_path: str = None):
        self.storage_path = Path(storage_path or os.path.expanduser("~/.neuro/knowledge"))
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.facts_file = self.storage_path / "facts.json"
        self.embeddings_file = self.storage_path / "embeddings.pkl"
        self.stats_file = self.storage_path / "stats.json"
        self.index_file = self.storage_path / "index.json"

        # Load existing knowledge
        self.facts: List[Fact] = []
        self.embeddings: Dict[str, np.ndarray] = {}
        self.topic_index: Dict[str, List[str]] = {}  # topic -> fact_ids
        self.word_index: Dict[str, List[str]] = {}   # word -> fact_ids
        self.stats = {
            'total_facts': 0,
            'total_searches': 0,
            'total_recalls': 0,
            'created': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat()
        }

        self._load()

    def _load(self) -> None:
        """Load knowledge from disk."""
        # Load facts
        if self.facts_file.exists():
            try:
                with open(self.facts_file, 'r') as f:
                    data = json.load(f)
                    self.facts = [Fact(**fact) for fact in data]
            except Exception:
                pass

        # Load embeddings
        if self.embeddings_file.exists():
            try:
                with open(self.embeddings_file, 'rb') as f:
                    self.embeddings = pickle.load(f)
            except Exception:
                pass

        # Load stats
        if self.stats_file.exists():
            try:
                with open(self.stats_file, 'r') as f:
                    self.stats = json.load(f)
            except Exception:
                pass

        # Load/rebuild indices
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    indices = json.load(f)
                    self.topic_index = indices.get('topics', {})
                    self.word_index = indices.get('words', {})
            except Exception:
                self._rebuild_indices()
        else:
            self._rebuild_indices()

    def _rebuild_indices(self) -> None:
        """Rebuild all indices from facts."""
        self.topic_index = {}
        self.word_index = {}

        for fact in self.facts:
            self._index_fact(fact)

    def _index_fact(self, fact: Fact) -> None:
        """Add a fact to all indices."""
        # Topic index
        topic_lower = fact.topic.lower()
        if topic_lower not in self.topic_index:
            self.topic_index[topic_lower] = []
        if fact.id not in self.topic_index[topic_lower]:
            self.topic_index[topic_lower].append(fact.id)

        # Word index
        words = self._extract_words(fact.content)
        words.extend(self._extract_words(fact.topic))

        for word in words:
            if word not in self.word_index:
                self.word_index[word] = []
            if fact.id not in self.word_index[word]:
                self.word_index[word].append(fact.id)

    def _extract_words(self, text: str) -> List[str]:
        """Extract meaningful words from text."""
        words = []
        for word in text.lower().split():
            # Clean punctuation
            word = ''.join(c for c in word if c.isalnum())
            if word and len(word) > 2 and word not in self.STOP_WORDS:
                words.append(word)
        return words

    def save(self) -> None:
        """Save all knowledge to disk."""
        # Save facts (without embeddings)
        facts_data = []
        for fact in self.facts:
            fact_dict = {
                'id': fact.id,
                'topic': fact.topic,
                'content': fact.content,
                'source': fact.source,
                'timestamp': fact.timestamp,
                'access_count': fact.access_count,
                'importance': fact.importance,
                'confidence': fact.confidence
            }
            facts_data.append(fact_dict)

        with open(self.facts_file, 'w') as f:
            json.dump(facts_data, f, indent=2)

        # Save embeddings
        with open(self.embeddings_file, 'wb') as f:
            pickle.dump(self.embeddings, f)

        # Save stats
        self.stats['total_facts'] = len(self.facts)
        self.stats['last_updated'] = datetime.now().isoformat()
        with open(self.stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)

        # Save indices
        indices = {
            'topics': self.topic_index,
            'words': self.word_index
        }
        with open(self.index_file, 'w') as f:
            json.dump(indices, f)

    def _create_embedding(self, text: str, dim: int = 128) -> np.ndarray:
        """Create embedding for text using hash-based method."""
        h = hashlib.sha256(text.encode()).digest()
        full_hash = h * (dim // len(h) + 1)
        emb = np.array([b / 255.0 for b in full_hash[:dim]])
        return (emb - 0.5) * 2  # Normalize to [-1, 1]

    def add_fact(
        self,
        topic: str,
        content: str,
        source: str,
        importance: float = 0.5,
        confidence: float = 0.8
    ) -> str:
        """Add a new fact to the knowledge base."""
        # Generate unique ID
        fact_id = f"fact_{len(self.facts)}_{int(datetime.now().timestamp())}"

        # Create embedding
        embedding = self._create_embedding(f"{topic} {content}")

        # Create fact
        fact = Fact(
            id=fact_id,
            topic=topic,
            content=content,
            source=source,
            embedding=embedding,
            importance=importance,
            confidence=confidence
        )

        # Add to storage
        self.facts.append(fact)
        self.embeddings[fact_id] = embedding

        # Update indices
        self._index_fact(fact)

        # Auto-save every 10 facts
        if len(self.facts) % 10 == 0:
            self.save()

        return fact_id

    def search(
        self,
        query: str,
        k: int = 5,
        min_score: float = 0.0
    ) -> List[Tuple[Fact, float]]:
        """
        Search for relevant facts using multi-strategy matching.

        Combines:
        1. Exact topic matching
        2. Word-based index lookup
        3. Embedding similarity
        """
        self.stats['total_searches'] = self.stats.get('total_searches', 0) + 1

        query_lower = query.lower()
        query_words = self._extract_words(query)
        query_embedding = self._create_embedding(query)

        scored_facts: Dict[str, float] = {}

        # 1. Topic matching (highest weight)
        for topic, fact_ids in self.topic_index.items():
            if topic in query_lower or query_lower in topic:
                for fid in fact_ids:
                    scored_facts[fid] = scored_facts.get(fid, 0) + 5.0

        # 2. Word index matching
        for word in query_words:
            if word in self.word_index:
                for fid in self.word_index[word]:
                    scored_facts[fid] = scored_facts.get(fid, 0) + 1.0

        # Bonus for multiple word matches
        for fid in scored_facts:
            fact = self._get_fact_by_id(fid)
            if fact:
                fact_words = set(self._extract_words(fact.content))
                matching = len(set(query_words) & fact_words)
                if matching > 1:
                    scored_facts[fid] += matching * 2

        # 3. Embedding similarity for top candidates
        candidate_ids = list(scored_facts.keys())[:100]  # Limit for efficiency

        for fid in candidate_ids:
            if fid in self.embeddings:
                sim = self._cosine_similarity(query_embedding, self.embeddings[fid])
                scored_facts[fid] += sim * 3  # Embedding weight

        # 4. Apply importance and recency boosts
        now = datetime.now()
        for fid in scored_facts:
            fact = self._get_fact_by_id(fid)
            if fact:
                # Importance boost
                scored_facts[fid] *= (0.5 + fact.importance)

                # Recency boost (newer facts slightly preferred)
                try:
                    fact_time = datetime.fromisoformat(fact.timestamp)
                    days_old = (now - fact_time).days
                    recency = max(0.5, 1.0 - days_old / 365)  # Decay over a year
                    scored_facts[fid] *= recency
                except Exception:
                    pass

        # Sort and return top k
        sorted_facts = sorted(scored_facts.items(), key=lambda x: x[1], reverse=True)

        results = []
        for fid, score in sorted_facts[:k]:
            if score >= min_score:
                fact = self._get_fact_by_id(fid)
                if fact:
                    fact.access_count += 1
                    results.append((fact, score))

        return results

    def _get_fact_by_id(self, fact_id: str) -> Optional[Fact]:
        """Get a fact by its ID."""
        for fact in self.facts:
            if fact.id == fact_id:
                return fact
        return None

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def get_recent(self, n: int = 10) -> List[Fact]:
        """Get most recent facts."""
        return self.facts[-n:]

    def get_by_topic(self, topic: str) -> List[Fact]:
        """Get all facts for a topic."""
        topic_lower = topic.lower()
        fact_ids = self.topic_index.get(topic_lower, [])
        return [f for f in self.facts if f.id in fact_ids]

    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        return {
            **self.stats,
            'total_facts': len(self.facts),
            'total_embeddings': len(self.embeddings),
            'total_topics': len(self.topic_index),
            'total_indexed_words': len(self.word_index),
            'storage_path': str(self.storage_path)
        }


class SelfTrainer:
    """
    Self-training system that improves the AGI over time.

    Wraps KnowledgeBase with training-specific features:
    - Learns facts from conversations
    - Retrieves relevant knowledge for prompts
    - Tracks learning progress
    """

    def __init__(self, storage_path: str = None):
        self.kb = KnowledgeBase(storage_path)
        self.session_learning: List[Dict] = []

    def learn(
        self,
        topic: str,
        content: str,
        source: str = "conversation",
        importance: float = 0.5
    ) -> str:
        """Learn a new fact."""
        fact_id = self.kb.add_fact(topic, content, source, importance)

        self.session_learning.append({
            'topic': topic,
            'content': content[:100],
            'source': source,
            'time': datetime.now().isoformat()
        })

        return fact_id

    def recall(self, query: str, k: int = 5) -> List[str]:
        """Recall relevant knowledge for a query."""
        results = self.kb.search(query, k=k)
        self.kb.stats['total_recalls'] = self.kb.stats.get('total_recalls', 0) + 1
        return [fact.content for fact, _ in results]

    def get_knowledge_for_prompt(self, query: str, k: int = 3) -> str:
        """Get formatted knowledge to inject into system prompt."""
        results = self.kb.search(query, k=k)

        if not results:
            return ""

        knowledge = "\n[Relevant knowledge from memory:]"
        for i, (fact, score) in enumerate(results, 1):
            # Truncate long content
            content = fact.content[:200]
            if len(fact.content) > 200:
                content += "..."
            knowledge += f"\n{i}. [{fact.topic}] {content}"

        return knowledge

    def save(self) -> None:
        """Save all knowledge to disk."""
        self.kb.save()

    def get_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        return {
            **self.kb.get_stats(),
            'session_learning': len(self.session_learning)
        }

    def get_session_summary(self) -> str:
        """Get summary of what was learned this session."""
        if not self.session_learning:
            return "No new knowledge learned this session."

        summary = f"Learned {len(self.session_learning)} new facts this session:\n"
        for item in self.session_learning[-5:]:
            summary += f"  [{item['topic']}]: {item['content']}...\n"

        return summary


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("KNOWLEDGE BASE TEST")
    print("=" * 60)

    trainer = SelfTrainer()

    # Learn some facts
    trainer.learn("AI", "Artificial intelligence is the simulation of human intelligence by machines", "test")
    trainer.learn("Memory", "Human memory uses hippocampus for episodic storage", "test")
    trainer.learn("Python", "Python is a high-level programming language", "test")
    trainer.learn("Neural Networks", "Neural networks are computing systems inspired by biological neurons", "test")
    trainer.learn("Machine Learning", "Machine learning is a subset of AI that learns from data", "test")

    # Test search
    print("\nSearching for 'intelligence':")
    results = trainer.recall("intelligence")
    for r in results:
        print(f"  - {r[:60]}...")

    print("\nSearching for 'python programming':")
    results = trainer.recall("python programming")
    for r in results:
        print(f"  - {r[:60]}...")

    # Get knowledge for prompt
    print("\nKnowledge for prompt (query: 'how does AI work'):")
    print(trainer.get_knowledge_for_prompt("how does AI work"))

    # Save
    trainer.save()

    # Stats
    print(f"\nStats: {trainer.get_stats()}")
    print(f"\nSession summary:\n{trainer.get_session_summary()}")

    print("\n" + "=" * 60)
    print(f"Knowledge saved to: {trainer.kb.storage_path}")
    print("=" * 60)
