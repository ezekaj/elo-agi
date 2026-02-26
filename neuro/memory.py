"""
PersistentMemory - Unified memory system with SQLite storage.

Combines:
- Semantic search (embeddings)
- Recency weighting
- Importance scoring
- Automatic pruning
"""

import sqlite3
import json
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class Memory:
    """A single memory entry."""
    content: str
    memory_type: str
    importance: float = 0.5
    recency: float = field(default_factory=time.time)
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        """Generate stable ID from content."""
        return hashlib.sha256(self.content.encode()).hexdigest()[:16]

    def score(self, query_embedding: Optional[List[float]] = None) -> float:
        """
        Calculate memory relevance score.
        
        Combines:
        - Semantic similarity (if query embedding provided)
        - Recency (exponential decay)
        - Importance
        """
        # Recency decay (half-life ~7 days)
        age_hours = (time.time() - self.recency) / 3600
        recency_score = self.importance * (0.99 ** age_hours)
        
        # Semantic similarity (placeholder - use simple keyword match for now)
        semantic_score = 0.0
        if query_embedding and self.embedding:
            # Cosine similarity
            dot = sum(a * b for a, b in zip(query_embedding, self.embedding))
            norm_q = sum(x * x for x in query_embedding) ** 0.5
            norm_m = sum(x * x for x in self.embedding) ** 0.5
            if norm_q > 0 and norm_m > 0:
                semantic_score = dot / (norm_q * norm_m)
        
        # Weighted combination
        if query_embedding:
            return 0.6 * semantic_score + 0.4 * recency_score
        return recency_score


class PersistentMemory:
    """
    Unified memory system with SQLite backend.
    
    Stores and retrieves memories with:
    - Content-based deduplication
    - Type filtering (interaction, fact, pattern, etc.)
    - Ranked retrieval by relevance
    - Automatic pruning of low-importance old memories
    """

    def __init__(self, path: str = "~/.neuro/memory.db"):
        self.db_path = Path(path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        """Initialize database schema."""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                type TEXT NOT NULL,
                importance REAL DEFAULT 0.5,
                recency REAL NOT NULL,
                embedding TEXT,
                metadata TEXT,
                created_at REAL DEFAULT (strftime('%s', 'now'))
            );
            
            CREATE INDEX IF NOT EXISTS idx_type ON memories(type);
            CREATE INDEX IF NOT EXISTS idx_recency ON memories(recency DESC);
            CREATE INDEX IF NOT EXISTS idx_importance ON memories(importance DESC);
        """)
        self.conn.commit()

    def store(
        self,
        content: str,
        memory_type: str = "interaction",
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None,
    ) -> Memory:
        """
        Store a memory.
        
        Args:
            content: Memory content
            memory_type: Type (interaction, fact, pattern, skill, etc.)
            importance: Importance score (0-1)
            metadata: Additional metadata
            embedding: Optional embedding vector
            
        Returns:
            Stored Memory object
        """
        memory = Memory(
            content=content,
            memory_type=memory_type,
            importance=importance,
            metadata=metadata or {},
            embedding=embedding,
        )
        
        # Check for duplicates
        existing = self.conn.execute(
            "SELECT id FROM memories WHERE id = ?", (memory.id,)
        ).fetchone()
        
        if existing:
            # Update recency of existing
            self.conn.execute(
                "UPDATE memories SET recency = ? WHERE id = ?",
                (time.time(), memory.id)
            )
        else:
            # Insert new
            self.conn.execute(
                """
                INSERT INTO memories (id, content, type, importance, recency, embedding, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory.id,
                    content,
                    memory_type,
                    importance,
                    memory.recency,
                    json.dumps(embedding) if embedding else None,
                    json.dumps(metadata),
                )
            )
        
        self.conn.commit()
        return memory

    def retrieve(
        self,
        query: str,
        k: int = 5,
        memory_type: Optional[str] = None,
        min_importance: float = 0.0,
    ) -> List[Memory]:
        """
        Retrieve relevant memories.
        
        Args:
            query: Search query (keyword-based for now)
            k: Number of memories to return
            memory_type: Filter by type (optional)
            min_importance: Minimum importance threshold
            
        Returns:
            List of Memory objects ranked by relevance
        """
        # Build query
        where_clauses = ["importance >= ?"]
        params: List[Any] = [min_importance]
        
        if memory_type:
            where_clauses.append("type = ?")
            params.append(memory_type)
        
        # Keyword search in content (simple approach)
        # TODO: Replace with actual semantic search when embeddings are available
        query_words = query.lower().split()
        
        where_clauses.append("(" + " OR ".join(
            f"LOWER(content) LIKE ?" for _ in query_words
        ) + ")")
        params.extend(f"%{word}%" for word in query_words)
        
        sql = f"""
            SELECT * FROM memories
            WHERE {" AND ".join(where_clauses)}
            ORDER BY recency DESC, importance DESC
            LIMIT ?
        """
        params.append(k * 2)  # Get more, then re-rank
        
        rows = self.conn.execute(sql, params).fetchall()
        
        # Convert to Memory objects and score
        memories = []
        for row in rows:
            memory = Memory(
                content=row["content"],
                memory_type=row["type"],
                importance=row["importance"],
                recency=row["recency"],
                embedding=json.loads(row["embedding"]) if row["embedding"] else None,
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            )
            memories.append(memory)
        
        # Sort by score (recency + importance for now)
        memories.sort(key=lambda m: m.score(), reverse=True)
        
        return memories[:k]

    def get_by_type(self, memory_type: str, k: int = 10) -> List[Memory]:
        """Get memories by type."""
        rows = self.conn.execute(
            """
            SELECT * FROM memories
            WHERE type = ?
            ORDER BY recency DESC
            LIMIT ?
            """,
            (memory_type, k),
        ).fetchall()
        
        return [
            Memory(
                content=row["content"],
                memory_type=row["type"],
                importance=row["importance"],
                recency=row["recency"],
                embedding=json.loads(row["embedding"]) if row["embedding"] else None,
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            )
            for row in rows
        ]

    def prune(self, max_age_days: int = 30, min_importance: float = 0.2) -> int:
        """
        Remove old, low-importance memories.
        
        Args:
            max_age_days: Maximum age in days
            min_importance: Minimum importance to keep
            
        Returns:
            Number of memories removed
        """
        cutoff = time.time() - (max_age_days * 24 * 3600)
        
        cursor = self.conn.execute(
            """
            DELETE FROM memories
            WHERE recency < ? AND importance < ?
            """,
            (cutoff, min_importance),
        )
        
        removed = cursor.rowcount
        self.conn.commit()
        return removed

    def count(self) -> int:
        """Get total memory count."""
        return self.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

    def stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        by_type = self.conn.execute(
            "SELECT type, COUNT(*) as count FROM memories GROUP BY type"
        ).fetchall()
        
        avg_importance = self.conn.execute(
            "SELECT AVG(importance) FROM memories"
        ).fetchone()[0] or 0.0
        
        oldest = self.conn.execute(
            "SELECT MIN(recency) FROM memories"
        ).fetchone()[0]
        
        return {
            "total": self.count(),
            "by_type": {row["type"]: row["count"] for row in by_type},
            "avg_importance": avg_importance,
            "oldest_age_hours": (time.time() - oldest) / 3600 if oldest else 0,
        }

    def close(self):
        """Close database connection."""
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
