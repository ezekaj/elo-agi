"""
Two-Stage Retrieval System

Efficient retrieval for large knowledge bases:
1. Stage 1: Fast sparse retrieval (BM25-like)
2. Stage 2: Re-rank with dense embeddings
"""

import math
import hashlib
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass


@dataclass
class Document:
    """A retrievable document."""

    id: str
    content: str
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BM25:
    """
    BM25 sparse retrieval algorithm.

    Parameters:
    - k1: Term frequency saturation (default 1.5)
    - b: Document length normalization (default 0.75)
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

        self.doc_freqs: Dict[str, int] = defaultdict(int)  # word -> num docs containing it
        self.doc_lens: Dict[str, int] = {}  # doc_id -> length
        self.doc_term_freqs: Dict[str, Counter] = {}  # doc_id -> term frequencies
        self.avg_doc_len: float = 0
        self.num_docs: int = 0

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        return [w.lower() for w in text.split() if len(w) > 2]

    def add_document(self, doc_id: str, content: str) -> None:
        """Add a document to the index."""
        tokens = self._tokenize(content)
        self.doc_lens[doc_id] = len(tokens)
        self.doc_term_freqs[doc_id] = Counter(tokens)

        # Update document frequencies
        for term in set(tokens):
            self.doc_freqs[term] += 1

        self.num_docs += 1
        self.avg_doc_len = sum(self.doc_lens.values()) / max(1, self.num_docs)

    def score(self, query: str, doc_id: str) -> float:
        """Compute BM25 score for a query-document pair."""
        if doc_id not in self.doc_term_freqs:
            return 0.0

        query_tokens = self._tokenize(query)
        doc_len = self.doc_lens[doc_id]
        doc_tf = self.doc_term_freqs[doc_id]

        score = 0.0
        for term in query_tokens:
            if term not in doc_tf:
                continue

            tf = doc_tf[term]
            df = self.doc_freqs.get(term, 1)

            # IDF component
            idf = math.log((self.num_docs - df + 0.5) / (df + 0.5) + 1)

            # TF component with length normalization
            tf_norm = (tf * (self.k1 + 1)) / (
                tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
            )

            score += idf * tf_norm

        return score

    def search(self, query: str, doc_ids: List[str], k: int = 100) -> List[Tuple[str, float]]:
        """Search for top-k documents."""
        scores = [(doc_id, self.score(query, doc_id)) for doc_id in doc_ids]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


class TwoStageRetriever:
    """
    Two-stage retrieval system combining sparse and dense retrieval.

    Stage 1: BM25 for fast candidate selection (sparse)
    Stage 2: Embedding similarity for re-ranking (dense)
    """

    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        self.bm25 = BM25()
        self.documents: Dict[str, Document] = {}

    def _create_embedding(self, text: str) -> np.ndarray:
        """Create embedding for text."""
        h = hashlib.sha256(text.encode()).digest()
        full_hash = h * (self.embedding_dim // len(h) + 1)
        return np.array([b / 255.0 for b in full_hash[: self.embedding_dim]]) * 2 - 1

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity."""
        norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def add_document(
        self,
        doc_id: str,
        content: str,
        embedding: np.ndarray = None,
        metadata: Dict[str, Any] = None,
    ) -> None:
        """Add a document to the retriever."""
        if embedding is None:
            embedding = self._create_embedding(content)

        doc = Document(id=doc_id, content=content, embedding=embedding, metadata=metadata)

        self.documents[doc_id] = doc
        self.bm25.add_document(doc_id, content)

    def retrieve(
        self, query: str, k: int = 10, stage1_k: int = 100, stage2_weight: float = 0.6
    ) -> List[Tuple[Document, float]]:
        """
        Two-stage retrieval.

        Args:
            query: Search query
            k: Number of final results
            stage1_k: Number of candidates from stage 1
            stage2_weight: Weight for dense similarity (0-1)

        Returns:
            List of (document, score) tuples
        """
        if not self.documents:
            return []

        doc_ids = list(self.documents.keys())

        # Stage 1: BM25 sparse retrieval
        stage1_results = self.bm25.search(query, doc_ids, k=stage1_k)

        if not stage1_results:
            return []

        # Normalize stage 1 scores
        max_bm25 = max(score for _, score in stage1_results) if stage1_results else 1
        stage1_scores = {doc_id: score / max(max_bm25, 1e-10) for doc_id, score in stage1_results}

        # Stage 2: Dense re-ranking
        query_embedding = self._create_embedding(query)

        final_scores = []
        for doc_id in stage1_scores:
            doc = self.documents[doc_id]

            # Dense similarity
            dense_sim = self._cosine_similarity(query_embedding, doc.embedding)
            dense_sim = (dense_sim + 1) / 2  # Normalize to [0, 1]

            # Combine sparse and dense scores
            sparse_score = stage1_scores[doc_id]
            combined = (1 - stage2_weight) * sparse_score + stage2_weight * dense_sim

            final_scores.append((doc, combined))

        # Sort by combined score
        final_scores.sort(key=lambda x: x[1], reverse=True)

        return final_scores[:k]

    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics."""
        return {
            "num_documents": len(self.documents),
            "vocabulary_size": len(self.bm25.doc_freqs),
            "avg_doc_length": self.bm25.avg_doc_len,
            "embedding_dim": self.embedding_dim,
        }


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("TWO-STAGE RETRIEVAL TEST")
    print("=" * 60)

    retriever = TwoStageRetriever()

    # Add documents
    docs = [
        ("d1", "Machine learning is a subset of artificial intelligence"),
        ("d2", "Deep learning uses neural networks with many layers"),
        ("d3", "Python is a popular programming language for AI"),
        ("d4", "Natural language processing handles text and speech"),
        ("d5", "Computer vision enables machines to see and interpret images"),
        ("d6", "Reinforcement learning learns through trial and error"),
    ]

    for doc_id, content in docs:
        retriever.add_document(doc_id, content)

    # Test retrieval
    queries = ["artificial intelligence", "neural networks", "Python AI"]

    for query in queries:
        print(f"\nQuery: '{query}'")
        results = retriever.retrieve(query, k=3)
        for doc, score in results:
            print(f"  [{score:.3f}] {doc.id}: {doc.content[:50]}...")

    print(f"\nStats: {retriever.get_stats()}")
