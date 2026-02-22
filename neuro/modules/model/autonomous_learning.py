"""
Autonomous Learning System for Neuro AGI

Integrates curiosity module with tools to enable autonomous exploration.
The AGI will:
- Track topics from conversations
- Identify knowledge gaps
- Autonomously search and learn
- Remember discoveries for later use
"""

import json
import threading
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from collections import deque
import hashlib
import numpy as np

# Import curiosity module
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "neuro-module-06-motivation" / "src"))

try:
    from curiosity_drive import CuriosityModule, CuriosityType  # noqa: F401
    from intrinsic_motivation import PathEntropyMaximizer, DriveType  # noqa: F401

    CURIOSITY_AVAILABLE = True
except ImportError:
    CURIOSITY_AVAILABLE = False


@dataclass
class Topic:
    """A topic the AGI is curious about."""

    name: str
    first_mentioned: float
    times_discussed: int = 1
    curiosity_level: float = 0.5
    last_searched: float = 0.0
    knowledge: List[str] = field(default_factory=list)
    related_topics: Set[str] = field(default_factory=set)


@dataclass
class Memory:
    """A piece of learned information."""

    content: str
    source: str
    topic: str
    timestamp: float
    importance: float = 0.5


class AutonomousLearner:
    """
    Autonomous learning system that uses curiosity to drive exploration.

    Runs in background, tracking topics from conversations and
    autonomously searching for related information.
    """

    def __init__(
        self,
        memory_path: str = "~/.neuro/memories",
        search_interval: float = 300.0,  # 5 minutes between auto-searches
        max_memories: int = 1000,
    ):
        self.memory_path = Path(memory_path).expanduser()
        self.memory_path.mkdir(parents=True, exist_ok=True)

        self.search_interval = search_interval
        self.max_memories = max_memories

        # Topic tracking
        self.topics: Dict[str, Topic] = {}
        self.conversation_history: deque = deque(maxlen=100)

        # Memory storage
        self.memories: List[Memory] = []
        self.knowledge_graph: Dict[str, Set[str]] = {}  # topic -> related topics

        # Curiosity module
        if CURIOSITY_AVAILABLE:
            self.curiosity = CuriosityModule(state_dim=128, base_curiosity=1.0)  # MAX CURIOSITY
            self.motivation = PathEntropyMaximizer(state_dim=128, action_dim=32)
        else:
            self.curiosity = None
            self.motivation = None

        # Background learning
        self.running = False
        self.learning_thread = None
        self.pending_searches: deque = deque(maxlen=20)

        # Load existing memories
        self._load_memories()

    def start_background_learning(self) -> None:
        """Start the autonomous learning loop."""
        if self.running:
            return

        self.running = True
        self.learning_thread = threading.Thread(target=self._learning_loop, daemon=True)
        self.learning_thread.start()

    def stop_background_learning(self) -> None:
        """Stop the autonomous learning loop."""
        self.running = False
        if self.learning_thread:
            self.learning_thread.join(timeout=2.0)

    def _learning_loop(self) -> None:
        """Background loop that autonomously explores topics."""
        while self.running:
            # Check if there are high-curiosity topics to explore
            top_topics = self.get_curious_topics(limit=3)

            for topic in top_topics:
                if not self.running:
                    break

                # Only search if enough time has passed
                if time.time() - topic.last_searched < self.search_interval:
                    continue

                # Add to pending searches
                self.pending_searches.append(topic.name)

            time.sleep(30)  # Check every 30 seconds

    def process_message(self, message: str, role: str = "user") -> Dict:
        """Process a conversation message to extract topics and update curiosity.

        Args:
            message: The message content
            role: 'user' or 'assistant'

        Returns:
            Dict with extracted topics and curiosity state
        """
        # Store in history
        self.conversation_history.append(
            {"role": role, "content": message, "timestamp": time.time()}
        )

        # Extract potential topics (simple keyword extraction)
        topics = self._extract_topics(message)

        # Update topic tracking
        for topic_name in topics:
            if topic_name in self.topics:
                self.topics[topic_name].times_discussed += 1
                self.topics[topic_name].curiosity_level = min(
                    1.0, self.topics[topic_name].curiosity_level + 0.1
                )
            else:
                self.topics[topic_name] = Topic(
                    name=topic_name, first_mentioned=time.time(), curiosity_level=0.5
                )

        # Update curiosity module if available
        if self.curiosity:
            embedding = self._text_to_embedding(message)
            self.curiosity.process_stimulus(embedding)

            # Register new topics as curiosity targets
            for topic_name in topics:
                if topic_name not in self.topics or self.topics[topic_name].times_discussed <= 2:
                    self.curiosity.register_curiosity_target(topic_name, 0.6)

        return {
            "extracted_topics": list(topics),
            "total_topics": len(self.topics),
            "curiosity_level": self.curiosity.curiosity_level if self.curiosity else 0.5,
            "knowledge_gaps": self.curiosity.get_knowledge_gaps() if self.curiosity else [],
        }

    def _extract_topics(self, text: str) -> Set[str]:
        """Extract potential topics from text."""
        topics = set()

        # Simple extraction: look for capitalized words, technical terms, etc.
        words = text.split()

        for i, word in enumerate(words):
            # Clean word
            clean = word.strip(".,!?()[]{}\"':;").lower()

            if len(clean) < 3:
                continue

            # Skip common words
            common = {
                "the",
                "and",
                "for",
                "are",
                "but",
                "not",
                "you",
                "all",
                "can",
                "had",
                "her",
                "was",
                "one",
                "our",
                "out",
                "has",
                "have",
                "been",
                "would",
                "could",
                "should",
                "this",
                "that",
                "with",
                "they",
                "from",
                "what",
                "when",
                "where",
                "which",
                "their",
                "will",
                "each",
                "about",
                "into",
                "than",
                "them",
            }

            if clean in common:
                continue

            # Technical terms (contains numbers, underscores, or is camelCase)
            if "_" in clean or any(c.isdigit() for c in clean):
                topics.add(clean)

            # Capitalized words in middle of sentence might be important
            elif word[0].isupper() and i > 0:
                topics.add(clean)

            # Longer words are often more specific topics
            elif len(clean) > 7:
                topics.add(clean)

        return topics

    def _text_to_embedding(self, text: str) -> np.ndarray:
        """Convert text to embedding for curiosity module."""
        hash_bytes = hashlib.sha256(text.encode()).digest()
        embedding = np.array([b / 255.0 for b in hash_bytes[:128]])
        return embedding

    def get_curious_topics(self, limit: int = 5) -> List[Topic]:
        """Get topics with highest curiosity level."""
        sorted_topics = sorted(
            self.topics.values(),
            key=lambda t: t.curiosity_level * (1 + 0.1 * t.times_discussed),
            reverse=True,
        )
        return sorted_topics[:limit]

    def get_knowledge_gaps(self) -> List[str]:
        """Get identified knowledge gaps."""
        if self.curiosity:
            return self.curiosity.get_knowledge_gaps()

        # Fallback: topics with few associated memories
        gaps = []
        for topic_name, topic in self.topics.items():
            if len(topic.knowledge) < 2 and topic.times_discussed > 1:
                gaps.append(topic_name)
        return gaps[:10]

    def add_memory(self, content: str, source: str, topic: str, importance: float = 0.5) -> None:
        """Add a learned piece of information to memory."""
        memory = Memory(
            content=content,
            source=source,
            topic=topic,
            timestamp=time.time(),
            importance=importance,
        )

        self.memories.append(memory)

        # Update topic knowledge
        if topic in self.topics:
            self.topics[topic].knowledge.append(content[:200])
            self.topics[topic].curiosity_level *= 0.9  # Partially satisfied

        # Prune if too many memories
        if len(self.memories) > self.max_memories:
            # Remove least important
            self.memories.sort(key=lambda m: m.importance, reverse=True)
            self.memories = self.memories[: self.max_memories]

        # Save to disk
        self._save_memories()

    def recall(self, query: str, limit: int = 5) -> List[Memory]:
        """Recall memories relevant to a query."""
        query_lower = query.lower()
        query_words = set(query_lower.split())

        scored_memories = []
        for memory in self.memories:
            content_lower = memory.content.lower()

            # Simple relevance scoring
            word_matches = sum(1 for w in query_words if w in content_lower)
            topic_match = 1 if memory.topic.lower() in query_lower else 0

            score = word_matches + topic_match * 2 + memory.importance

            if score > 0:
                scored_memories.append((memory, score))

        # Sort by score
        scored_memories.sort(key=lambda x: x[1], reverse=True)

        return [m for m, _ in scored_memories[:limit]]

    def get_pending_search(self) -> Optional[str]:
        """Get a pending topic to search (for tool integration)."""
        if self.pending_searches:
            topic = self.pending_searches.popleft()
            if topic in self.topics:
                self.topics[topic].last_searched = time.time()
            return topic
        return None

    def mark_topic_searched(self, topic: str, found_info: bool = True) -> None:
        """Mark a topic as having been searched."""
        if topic in self.topics:
            self.topics[topic].last_searched = time.time()
            if found_info:
                self.topics[topic].curiosity_level *= 0.8  # Reduce curiosity

    def get_state(self) -> Dict:
        """Get current state of the learning system."""
        return {
            "total_topics": len(self.topics),
            "total_memories": len(self.memories),
            "top_curious_topics": [t.name for t in self.get_curious_topics(5)],
            "knowledge_gaps": self.get_knowledge_gaps()[:5],
            "pending_searches": len(self.pending_searches),
            "curiosity_level": self.curiosity.curiosity_level if self.curiosity else 0.5,
            "running": self.running,
        }

    def _save_memories(self) -> None:
        """Save memories to disk."""
        memory_file = self.memory_path / "memories.json"

        data = {
            "memories": [
                {
                    "content": m.content,
                    "source": m.source,
                    "topic": m.topic,
                    "timestamp": m.timestamp,
                    "importance": m.importance,
                }
                for m in self.memories[-500:]  # Save last 500
            ],
            "topics": {
                name: {
                    "name": t.name,
                    "first_mentioned": t.first_mentioned,
                    "times_discussed": t.times_discussed,
                    "curiosity_level": t.curiosity_level,
                    "last_searched": t.last_searched,
                    "knowledge": t.knowledge[-10:],  # Last 10 pieces
                    "related_topics": list(t.related_topics),
                }
                for name, t in self.topics.items()
            },
        }

        try:
            with open(memory_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def _load_memories(self) -> None:
        """Load memories from disk."""
        memory_file = self.memory_path / "memories.json"

        if not memory_file.exists():
            return

        try:
            with open(memory_file) as f:
                data = json.load(f)

            # Load memories
            for m in data.get("memories", []):
                self.memories.append(
                    Memory(
                        content=m["content"],
                        source=m["source"],
                        topic=m["topic"],
                        timestamp=m["timestamp"],
                        importance=m.get("importance", 0.5),
                    )
                )

            # Load topics
            for name, t in data.get("topics", {}).items():
                self.topics[name] = Topic(
                    name=t["name"],
                    first_mentioned=t["first_mentioned"],
                    times_discussed=t["times_discussed"],
                    curiosity_level=t["curiosity_level"],
                    last_searched=t.get("last_searched", 0),
                    knowledge=t.get("knowledge", []),
                    related_topics=set(t.get("related_topics", [])),
                )
        except Exception:
            pass

    def generate_curiosity_prompt(self) -> Optional[str]:
        """Generate a prompt for autonomous exploration based on curiosity."""
        gaps = self.get_knowledge_gaps()
        curious_topics = self.get_curious_topics(3)

        if not gaps and not curious_topics:
            return None

        prompt_parts = []

        if gaps:
            prompt_parts.append(f"I have knowledge gaps about: {', '.join(gaps[:3])}")

        if curious_topics:
            topics_str = ", ".join(t.name for t in curious_topics)
            prompt_parts.append(f"I'm curious about: {topics_str}")

        if self.memories:
            recent = self.memories[-5:]
            context = "; ".join(m.content[:50] for m in recent)
            prompt_parts.append(f"Recently learned: {context}")

        return " | ".join(prompt_parts)
