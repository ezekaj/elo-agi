"""
Shared Semantic Space: Unified embedding space for cross-module integration.

All cognitive modules project their representations into this shared space,
enabling cross-modal reasoning, transfer learning, and unified cognition.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import numpy as np


class ModalityType(Enum):
    """Types of modalities that can project to shared space."""

    VISUAL = "visual"
    AUDITORY = "auditory"
    LINGUISTIC = "linguistic"
    MOTOR = "motor"
    SPATIAL = "spatial"
    TEMPORAL = "temporal"
    EMOTIONAL = "emotional"
    SOCIAL = "social"
    ABSTRACT = "abstract"
    MEMORY = "memory"


@dataclass
class SharedSpaceConfig:
    """Configuration for shared semantic space."""

    embedding_dim: int = 512
    n_attention_heads: int = 8
    temperature: float = 1.0
    similarity_threshold: float = 0.7
    max_active_concepts: int = 100
    decay_rate: float = 0.01
    random_seed: Optional[int] = None


@dataclass
class SemanticEmbedding:
    """A semantic embedding in shared space."""

    vector: np.ndarray
    modality: ModalityType
    source_module: str
    confidence: float = 1.0
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def similarity(self, other: "SemanticEmbedding") -> float:
        """Compute cosine similarity with another embedding."""
        norm_self = np.linalg.norm(self.vector)
        norm_other = np.linalg.norm(other.vector)
        if norm_self < 1e-8 or norm_other < 1e-8:
            return 0.0
        return float(np.dot(self.vector, other.vector) / (norm_self * norm_other))

    def distance(self, other: "SemanticEmbedding") -> float:
        """Compute Euclidean distance to another embedding."""
        return float(np.linalg.norm(self.vector - other.vector))

    def blend(self, other: "SemanticEmbedding", weight: float = 0.5) -> "SemanticEmbedding":
        """Blend this embedding with another."""
        blended_vector = weight * self.vector + (1 - weight) * other.vector
        blended_vector = blended_vector / (np.linalg.norm(blended_vector) + 1e-8)
        return SemanticEmbedding(
            vector=blended_vector,
            modality=self.modality,
            source_module="blended",
            confidence=(self.confidence + other.confidence) / 2,
            timestamp=max(self.timestamp, other.timestamp),
        )


class ProjectionLayer:
    """
    Projects module-specific representations to shared space.

    Each module has its own projection layer that learns to map
    its internal representations to the unified semantic space.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        modality: ModalityType,
        module_name: str,
        random_seed: Optional[int] = None,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.modality = modality
        self.module_name = module_name

        rng = np.random.default_rng(random_seed)

        # Initialize projection weights (Xavier initialization)
        scale = np.sqrt(2.0 / (input_dim + output_dim))
        self.weights = rng.normal(0, scale, (input_dim, output_dim))
        self.bias = np.zeros(output_dim)

        # Learning rate and momentum
        self.learning_rate = 0.01
        self.momentum = 0.9
        self.weight_momentum = np.zeros_like(self.weights)
        self.bias_momentum = np.zeros_like(self.bias)

        # Statistics
        self.n_projections = 0
        self.total_error = 0.0

    def project(self, input_vector: np.ndarray) -> SemanticEmbedding:
        """Project input to shared space."""
        if input_vector.shape[0] != self.input_dim:
            # Pad or truncate
            if input_vector.shape[0] < self.input_dim:
                padded = np.zeros(self.input_dim)
                padded[: input_vector.shape[0]] = input_vector
                input_vector = padded
            else:
                input_vector = input_vector[: self.input_dim]

        # Linear projection + tanh activation
        projected = np.tanh(input_vector @ self.weights + self.bias)

        # L2 normalize to unit sphere
        norm = np.linalg.norm(projected)
        if norm > 1e-8:
            projected = projected / norm

        self.n_projections += 1

        return SemanticEmbedding(
            vector=projected,
            modality=self.modality,
            source_module=self.module_name,
            confidence=1.0,
            timestamp=float(self.n_projections),
        )

    def inverse_project(self, embedding: SemanticEmbedding) -> np.ndarray:
        """Approximate inverse projection from shared space."""
        # Pseudo-inverse for approximate reconstruction
        pinv = np.linalg.pinv(self.weights)
        reconstructed = np.arctanh(np.clip(embedding.vector, -0.999, 0.999)) @ pinv
        return reconstructed

    def update(self, error: np.ndarray, input_vector: np.ndarray) -> None:
        """Update projection weights based on error signal."""
        if input_vector.shape[0] != self.input_dim:
            if input_vector.shape[0] < self.input_dim:
                padded = np.zeros(self.input_dim)
                padded[: input_vector.shape[0]] = input_vector
                input_vector = padded
            else:
                input_vector = input_vector[: self.input_dim]

        # Gradient descent with momentum
        grad_weights = np.outer(input_vector, error)
        grad_bias = error

        self.weight_momentum = (
            self.momentum * self.weight_momentum - self.learning_rate * grad_weights
        )
        self.bias_momentum = self.momentum * self.bias_momentum - self.learning_rate * grad_bias

        self.weights += self.weight_momentum
        self.bias += self.bias_momentum

        self.total_error += np.sum(error**2)


class SharedSpace:
    """
    Central shared semantic space for all cognitive modules.

    Implements:
    - Unified embedding storage
    - Cross-modal similarity search
    - Attention-based activation spreading
    - Concept clustering and binding
    """

    def __init__(self, config: Optional[SharedSpaceConfig] = None):
        self.config = config or SharedSpaceConfig()
        self._rng = np.random.default_rng(self.config.random_seed)

        # Registered projection layers
        self._projections: Dict[str, ProjectionLayer] = {}

        # Active embeddings in workspace
        self._active_embeddings: List[SemanticEmbedding] = []

        # Concept memory (persistent)
        self._concept_memory: Dict[str, SemanticEmbedding] = {}

        # Attention weights for cross-modal binding
        self._attention_weights: Optional[np.ndarray] = None

        # Statistics
        self._n_queries = 0
        self._n_bindings = 0

    def register_module(
        self,
        module_name: str,
        input_dim: int,
        modality: ModalityType,
    ) -> ProjectionLayer:
        """Register a module with its projection layer."""
        projection = ProjectionLayer(
            input_dim=input_dim,
            output_dim=self.config.embedding_dim,
            modality=modality,
            module_name=module_name,
            random_seed=self.config.random_seed,
        )
        self._projections[module_name] = projection
        return projection

    def get_projection(self, module_name: str) -> Optional[ProjectionLayer]:
        """Get projection layer for a module."""
        return self._projections.get(module_name)

    def project(
        self,
        module_name: str,
        vector: np.ndarray,
        activate: bool = True,
    ) -> SemanticEmbedding:
        """Project a module's representation to shared space."""
        if module_name not in self._projections:
            raise ValueError(f"Module '{module_name}' not registered")

        embedding = self._projections[module_name].project(vector)

        if activate:
            self._activate(embedding)

        return embedding

    def _activate(self, embedding: SemanticEmbedding) -> None:
        """Add embedding to active workspace."""
        # Check for similar existing embedding
        for i, existing in enumerate(self._active_embeddings):
            if embedding.similarity(existing) > self.config.similarity_threshold:
                # Merge with existing
                self._active_embeddings[i] = existing.blend(embedding, 0.7)
                return

        # Add new embedding
        self._active_embeddings.append(embedding)

        # Prune if over capacity
        if len(self._active_embeddings) > self.config.max_active_concepts:
            # Remove oldest/lowest confidence
            self._active_embeddings.sort(
                key=lambda e: e.confidence * (1.0 / (1.0 + e.timestamp)), reverse=True
            )
            self._active_embeddings = self._active_embeddings[: self.config.max_active_concepts]

    def query(
        self,
        query_embedding: SemanticEmbedding,
        top_k: int = 5,
        modality_filter: Optional[ModalityType] = None,
    ) -> List[Tuple[SemanticEmbedding, float]]:
        """Find most similar embeddings in active space."""
        self._n_queries += 1

        results = []
        for embedding in self._active_embeddings:
            if modality_filter and embedding.modality != modality_filter:
                continue

            similarity = query_embedding.similarity(embedding)
            results.append((embedding, similarity))

        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]

    def query_memory(
        self,
        query_embedding: SemanticEmbedding,
        top_k: int = 5,
    ) -> List[Tuple[str, SemanticEmbedding, float]]:
        """Query persistent concept memory."""
        results = []
        for name, embedding in self._concept_memory.items():
            similarity = query_embedding.similarity(embedding)
            results.append((name, embedding, similarity))

        results.sort(key=lambda x: x[2], reverse=True)
        return results[:top_k]

    def store_concept(self, name: str, embedding: SemanticEmbedding) -> None:
        """Store a concept in persistent memory."""
        self._concept_memory[name] = embedding

    def get_concept(self, name: str) -> Optional[SemanticEmbedding]:
        """Retrieve a concept from memory."""
        return self._concept_memory.get(name)

    def bind(
        self,
        embeddings: List[SemanticEmbedding],
        binding_strength: float = 1.0,
    ) -> SemanticEmbedding:
        """Bind multiple embeddings into a unified representation."""
        if not embeddings:
            raise ValueError("Cannot bind empty list")

        self._n_bindings += 1

        # Weighted average based on confidence
        total_confidence = sum(e.confidence for e in embeddings)
        if total_confidence < 1e-8:
            total_confidence = 1.0

        bound_vector = np.zeros(self.config.embedding_dim)
        for e in embeddings:
            weight = e.confidence / total_confidence
            bound_vector += weight * e.vector

        # Normalize
        norm = np.linalg.norm(bound_vector)
        if norm > 1e-8:
            bound_vector = bound_vector / norm

        return SemanticEmbedding(
            vector=bound_vector,
            modality=ModalityType.ABSTRACT,
            source_module="binding",
            confidence=binding_strength * np.mean([e.confidence for e in embeddings]),
            timestamp=max(e.timestamp for e in embeddings),
            metadata={"bound_count": len(embeddings)},
        )

    def attention_spread(self, focus: SemanticEmbedding) -> Dict[str, float]:
        """Spread attention from focus to all active embeddings."""
        attention = {}

        for embedding in self._active_embeddings:
            similarity = focus.similarity(embedding)
            # Softmax-like attention
            attention[embedding.source_module] = np.exp(similarity / self.config.temperature)

        # Normalize
        total = sum(attention.values())
        if total > 1e-8:
            attention = {k: v / total for k, v in attention.items()}

        return attention

    def decay(self) -> None:
        """Apply decay to active embeddings."""
        for embedding in self._active_embeddings:
            embedding.confidence *= 1 - self.config.decay_rate

        # Remove very low confidence embeddings
        self._active_embeddings = [e for e in self._active_embeddings if e.confidence > 0.01]

    def clear_active(self) -> None:
        """Clear all active embeddings."""
        self._active_embeddings = []

    def get_active_embeddings(self) -> List[SemanticEmbedding]:
        """Get all active embeddings."""
        return self._active_embeddings.copy()

    def get_modality_embeddings(
        self,
        modality: ModalityType,
    ) -> List[SemanticEmbedding]:
        """Get active embeddings of a specific modality."""
        return [e for e in self._active_embeddings if e.modality == modality]

    def cross_modal_similarity(
        self,
        modality1: ModalityType,
        modality2: ModalityType,
    ) -> np.ndarray:
        """Compute similarity matrix between two modalities."""
        emb1 = self.get_modality_embeddings(modality1)
        emb2 = self.get_modality_embeddings(modality2)

        if not emb1 or not emb2:
            return np.array([[]])

        similarity = np.zeros((len(emb1), len(emb2)))
        for i, e1 in enumerate(emb1):
            for j, e2 in enumerate(emb2):
                similarity[i, j] = e1.similarity(e2)

        return similarity

    def statistics(self) -> Dict[str, Any]:
        """Get statistics about shared space usage."""
        modality_counts = {}
        for emb in self._active_embeddings:
            modality_counts[emb.modality.value] = modality_counts.get(emb.modality.value, 0) + 1

        return {
            "n_registered_modules": len(self._projections),
            "n_active_embeddings": len(self._active_embeddings),
            "n_stored_concepts": len(self._concept_memory),
            "n_queries": self._n_queries,
            "n_bindings": self._n_bindings,
            "modality_distribution": modality_counts,
            "embedding_dim": self.config.embedding_dim,
        }
