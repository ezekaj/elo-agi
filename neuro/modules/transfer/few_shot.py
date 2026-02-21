"""
Few-Shot: Learn from few examples.

Implements prototype networks, matching networks, and other
few-shot learning methods.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import numpy as np


@dataclass
class SupportSet:
    """Support set for few-shot learning."""
    examples: List[np.ndarray]  # Feature vectors
    labels: List[str]           # Class labels
    embeddings: Optional[np.ndarray] = None  # Computed embeddings

    @property
    def n_examples(self) -> int:
        return len(self.examples)

    @property
    def n_classes(self) -> int:
        return len(set(self.labels))

    def get_class_examples(self, label: str) -> List[np.ndarray]:
        """Get examples for a specific class."""
        return [ex for ex, lab in zip(self.examples, self.labels) if lab == label]


@dataclass
class QueryResult:
    """Result of a few-shot query."""
    query: np.ndarray
    predicted_label: str
    confidence: float
    class_probabilities: Dict[str, float]
    nearest_examples: List[Tuple[str, float]]  # (label, distance)


class EmbeddingNetwork:
    """
    Neural embedding network for few-shot learning.
    """

    def __init__(
        self,
        input_dim: int = 128,
        embedding_dim: int = 64,
        n_layers: int = 2,
    ):
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers

        # Initialize weights (simple MLP)
        self._weights = []
        self._biases = []

        dims = [input_dim] + [embedding_dim] * n_layers

        np.random.seed(42)
        for i in range(len(dims) - 1):
            w = np.random.randn(dims[i], dims[i + 1]) * np.sqrt(2 / dims[i])
            b = np.zeros(dims[i + 1])
            self._weights.append(w)
            self._biases.append(b)

    def embed(self, x: np.ndarray) -> np.ndarray:
        """Embed input into embedding space."""
        # Ensure correct input dimension
        if len(x) != self.input_dim:
            if len(x) < self.input_dim:
                x = np.pad(x, (0, self.input_dim - len(x)))
            else:
                x = x[:self.input_dim]

        h = x
        for i, (w, b) in enumerate(zip(self._weights, self._biases)):
            h = h @ w + b
            if i < len(self._weights) - 1:
                h = np.maximum(0, h)  # ReLU

        # L2 normalize
        h = h / (np.linalg.norm(h) + 1e-8)
        return h

    def embed_batch(self, xs: List[np.ndarray]) -> np.ndarray:
        """Embed a batch of inputs."""
        return np.array([self.embed(x) for x in xs])


class PrototypeNetwork:
    """
    Prototype Network for few-shot classification.

    Computes class prototypes as mean embeddings and classifies
    by distance to prototypes.
    """

    def __init__(
        self,
        input_dim: int = 128,
        embedding_dim: int = 64,
    ):
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        self.encoder = EmbeddingNetwork(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
        )

        # Class prototypes
        self._prototypes: Dict[str, np.ndarray] = {}

    def compute_prototypes(self, support: SupportSet) -> Dict[str, np.ndarray]:
        """Compute class prototypes from support set."""
        prototypes = {}
        classes = set(support.labels)

        for cls in classes:
            class_examples = support.get_class_examples(cls)
            embeddings = self.encoder.embed_batch(class_examples)
            prototype = embeddings.mean(axis=0)
            prototypes[cls] = prototype

        self._prototypes = prototypes
        return prototypes

    def classify(
        self,
        query: np.ndarray,
        support: Optional[SupportSet] = None,
    ) -> QueryResult:
        """
        Classify a query example.

        Args:
            query: Query feature vector
            support: Support set (optional if prototypes already computed)

        Returns:
            QueryResult with prediction and probabilities
        """
        if support is not None:
            self.compute_prototypes(support)

        if not self._prototypes:
            raise ValueError("No prototypes computed. Provide support set.")

        # Embed query
        query_embedding = self.encoder.embed(query)

        # Compute distances to prototypes
        distances = {}
        for cls, proto in self._prototypes.items():
            dist = np.linalg.norm(query_embedding - proto)
            distances[cls] = dist

        # Convert to probabilities (softmax of negative distances)
        neg_distances = {cls: -d for cls, d in distances.items()}
        max_neg_dist = max(neg_distances.values())
        exp_vals = {cls: np.exp(nd - max_neg_dist) for cls, nd in neg_distances.items()}
        total = sum(exp_vals.values())
        probabilities = {cls: ev / total for cls, ev in exp_vals.items()}

        # Get prediction
        predicted = max(probabilities, key=probabilities.get)
        confidence = probabilities[predicted]

        # Nearest examples
        nearest = sorted(distances.items(), key=lambda x: x[1])

        return QueryResult(
            query=query,
            predicted_label=predicted,
            confidence=confidence,
            class_probabilities=probabilities,
            nearest_examples=nearest,
        )

    def classify_batch(
        self,
        queries: List[np.ndarray],
        support: Optional[SupportSet] = None,
    ) -> List[QueryResult]:
        """Classify multiple queries."""
        return [self.classify(q, support if i == 0 else None) for i, q in enumerate(queries)]


class MatchingNetwork:
    """
    Matching Network for few-shot classification.

    Uses attention-based comparison with support examples.
    """

    def __init__(
        self,
        input_dim: int = 128,
        embedding_dim: int = 64,
    ):
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        self.encoder = EmbeddingNetwork(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
        )

    def classify(
        self,
        query: np.ndarray,
        support: SupportSet,
    ) -> QueryResult:
        """
        Classify using matching network.
        """
        # Embed query and support
        query_emb = self.encoder.embed(query)
        support_embs = self.encoder.embed_batch(support.examples)

        # Compute attention (cosine similarity)
        similarities = []
        for emb in support_embs:
            sim = np.dot(query_emb, emb)
            similarities.append(sim)

        # Softmax
        similarities = np.array(similarities)
        max_sim = similarities.max()
        exp_sims = np.exp(similarities - max_sim)
        attention = exp_sims / exp_sims.sum()

        # Aggregate predictions
        class_scores = {}
        for i, label in enumerate(support.labels):
            if label not in class_scores:
                class_scores[label] = 0.0
            class_scores[label] += attention[i]

        # Normalize
        total = sum(class_scores.values())
        probabilities = {cls: score / total for cls, score in class_scores.items()}

        # Get prediction
        predicted = max(probabilities, key=probabilities.get)
        confidence = probabilities[predicted]

        # Nearest examples
        sorted_indices = np.argsort(similarities)[::-1]
        nearest = [(support.labels[i], float(1 - similarities[i])) for i in sorted_indices[:5]]

        return QueryResult(
            query=query,
            predicted_label=predicted,
            confidence=confidence,
            class_probabilities=probabilities,
            nearest_examples=nearest,
        )


class FewShotLearner:
    """
    Complete few-shot learning system.

    Combines multiple few-shot methods and provides a unified interface.
    """

    def __init__(
        self,
        input_dim: int = 128,
        embedding_dim: int = 64,
        method: str = "prototype",
    ):
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        # Initialize methods
        self.prototype_net = PrototypeNetwork(input_dim, embedding_dim)
        self.matching_net = MatchingNetwork(input_dim, embedding_dim)

        self._method = method
        self._support_sets: Dict[str, SupportSet] = {}

    def set_method(self, method: str) -> None:
        """Set the few-shot learning method."""
        if method not in ["prototype", "matching"]:
            raise ValueError(f"Unknown method: {method}")
        self._method = method

    def register_support_set(
        self,
        task_id: str,
        examples: List[np.ndarray],
        labels: List[str],
    ) -> SupportSet:
        """Register a support set for a task."""
        support = SupportSet(examples=examples, labels=labels)
        self._support_sets[task_id] = support

        # Precompute prototypes if using prototype network
        if self._method == "prototype":
            self.prototype_net.compute_prototypes(support)

        return support

    def classify(
        self,
        query: np.ndarray,
        task_id: Optional[str] = None,
        support: Optional[SupportSet] = None,
    ) -> QueryResult:
        """
        Classify a query using few-shot learning.

        Args:
            query: Query feature vector
            task_id: ID of registered support set
            support: Support set (overrides task_id)

        Returns:
            QueryResult with prediction
        """
        if support is None:
            if task_id is None:
                raise ValueError("Provide task_id or support")
            support = self._support_sets.get(task_id)
            if support is None:
                raise ValueError(f"No support set for task {task_id}")

        if self._method == "prototype":
            return self.prototype_net.classify(query, support)
        else:
            return self.matching_net.classify(query, support)

    def update_support_set(
        self,
        task_id: str,
        new_examples: List[np.ndarray],
        new_labels: List[str],
    ) -> SupportSet:
        """Add new examples to a support set."""
        if task_id not in self._support_sets:
            return self.register_support_set(task_id, new_examples, new_labels)

        support = self._support_sets[task_id]
        support.examples.extend(new_examples)
        support.labels.extend(new_labels)

        # Update prototypes
        if self._method == "prototype":
            self.prototype_net.compute_prototypes(support)

        return support

    def evaluate(
        self,
        test_examples: List[np.ndarray],
        test_labels: List[str],
        task_id: str,
    ) -> Dict[str, float]:
        """
        Evaluate few-shot performance.
        """
        correct = 0
        total = len(test_examples)
        confidences = []

        for example, true_label in zip(test_examples, test_labels):
            result = self.classify(example, task_id=task_id)
            if result.predicted_label == true_label:
                correct += 1
            confidences.append(result.confidence)

        return {
            "accuracy": correct / max(total, 1),
            "avg_confidence": float(np.mean(confidences)),
            "total_examples": total,
        }

    def get_task_classes(self, task_id: str) -> List[str]:
        """Get classes for a task."""
        support = self._support_sets.get(task_id)
        if support is None:
            return []
        return list(set(support.labels))

    def statistics(self) -> Dict[str, Any]:
        """Get few-shot learner statistics."""
        return {
            "method": self._method,
            "n_tasks": len(self._support_sets),
            "tasks": {
                tid: {"n_examples": s.n_examples, "n_classes": s.n_classes}
                for tid, s in self._support_sets.items()
            },
        }
