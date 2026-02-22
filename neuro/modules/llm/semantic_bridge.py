"""
Semantic Bridge: Translates between language and internal representations.

Provides bidirectional mapping between:
- Natural language ↔ Cognitive state vectors
- LLM embeddings ↔ Module-compatible formats
- Text descriptions ↔ Sensory observations
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np
import time

from .llm_interface import LLMOracle, MockLLM


@dataclass
class SemanticConfig:
    """Configuration for semantic bridge."""

    embedding_dim: int = 64
    internal_dim: int = 64  # Cognitive system dimension
    projection_method: str = "linear"  # "linear", "mlp", "attention"
    normalize_embeddings: bool = True
    use_context: bool = True
    context_window: int = 5


@dataclass
class Embedding:
    """A semantic embedding with metadata."""

    vector: np.ndarray
    text: str
    source: str = "llm"  # "llm", "internal", "projected"
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def similarity(self, other: "Embedding") -> float:
        """Compute cosine similarity with another embedding."""
        dot = np.dot(self.vector, other.vector)
        norm_self = np.linalg.norm(self.vector)
        norm_other = np.linalg.norm(other.vector)
        if norm_self > 0 and norm_other > 0:
            return float(dot / (norm_self * norm_other))
        return 0.0


class SemanticBridge:
    """
    Bridges language and internal cognitive representations.

    The bridge:
    1. Converts text to cognitive-compatible vectors
    2. Converts cognitive states to natural language
    3. Maintains semantic memory for context
    4. Learns alignment between spaces (optional)
    """

    def __init__(
        self,
        llm: Optional[LLMOracle] = None,
        config: Optional[SemanticConfig] = None,
    ):
        self.config = config or SemanticConfig()
        self.llm = llm or MockLLM()

        # Projection matrices (initialized as identity-like)
        self._embedding_to_internal = self._init_projection(
            self.config.embedding_dim, self.config.internal_dim
        )
        self._internal_to_embedding = self._init_projection(
            self.config.internal_dim, self.config.embedding_dim
        )

        # Semantic memory
        self._memory: List[Embedding] = []
        self._concept_cache: Dict[str, Embedding] = {}

        # Statistics
        self._encode_count = 0
        self._decode_count = 0

    def _init_projection(self, in_dim: int, out_dim: int) -> np.ndarray:
        """Initialize projection matrix."""
        # Xavier initialization
        std = np.sqrt(2.0 / (in_dim + out_dim))
        return np.random.randn(out_dim, in_dim).astype(np.float32) * std

    def encode(self, text: str, use_context: bool = True) -> np.ndarray:
        """
        Encode text to internal cognitive representation.

        Args:
            text: Natural language text
            use_context: Whether to incorporate recent context

        Returns:
            Internal representation vector
        """
        self._encode_count += 1

        # Get LLM embedding
        llm_embedding = self.llm.embed(text)

        # Resize if needed
        if len(llm_embedding) != self.config.embedding_dim:
            if len(llm_embedding) < self.config.embedding_dim:
                llm_embedding = np.pad(
                    llm_embedding, (0, self.config.embedding_dim - len(llm_embedding))
                )
            else:
                llm_embedding = llm_embedding[: self.config.embedding_dim]

        # Project to internal space
        internal = self._project_to_internal(llm_embedding)

        # Incorporate context if enabled
        if use_context and self.config.use_context and self._memory:
            context_vec = self._get_context_vector()
            internal = 0.7 * internal + 0.3 * context_vec

        # Normalize
        if self.config.normalize_embeddings:
            norm = np.linalg.norm(internal)
            if norm > 0:
                internal = internal / norm

        # Store in memory
        embedding = Embedding(vector=internal, text=text, source="internal")
        self._memory.append(embedding)
        if len(self._memory) > 100:
            self._memory = self._memory[-50:]

        return internal

    def decode(self, internal: np.ndarray, prompt_hint: Optional[str] = None) -> str:
        """
        Decode internal representation to natural language.

        Args:
            internal: Internal cognitive vector
            prompt_hint: Optional hint for generation

        Returns:
            Natural language description
        """
        self._decode_count += 1

        # Project to embedding space
        embedding = self._project_to_embedding(internal)

        # Find similar concepts in cache
        similar_concepts = self._find_similar_concepts(embedding, k=3)

        # Build prompt for LLM
        if prompt_hint:
            prompt = prompt_hint
        else:
            prompt = "Describe the current state or thought in natural language."

        if similar_concepts:
            context = " ".join([c.text for c in similar_concepts[:2]])
            prompt = f"Context: {context}\n\n{prompt}"

        # Query LLM
        response = self.llm.query(prompt)
        return response.text

    def _project_to_internal(self, embedding: np.ndarray) -> np.ndarray:
        """Project LLM embedding to internal representation."""
        if self.config.projection_method == "linear":
            return np.dot(self._embedding_to_internal, embedding)
        else:
            # Simple nonlinear: tanh activation
            linear = np.dot(self._embedding_to_internal, embedding)
            return np.tanh(linear)

    def _project_to_embedding(self, internal: np.ndarray) -> np.ndarray:
        """Project internal representation to embedding space."""
        if self.config.projection_method == "linear":
            return np.dot(self._internal_to_embedding, internal)
        else:
            linear = np.dot(self._internal_to_embedding, internal)
            return np.tanh(linear)

    def _get_context_vector(self) -> np.ndarray:
        """Get context vector from recent memory."""
        if not self._memory:
            return np.zeros(self.config.internal_dim, dtype=np.float32)

        recent = self._memory[-self.config.context_window :]
        vectors = [e.vector for e in recent]

        # Weighted average (more recent = higher weight)
        weights = np.linspace(0.5, 1.0, len(vectors))
        weighted_sum = sum(w * v for w, v in zip(weights, vectors))
        return weighted_sum / sum(weights)

    def _find_similar_concepts(self, embedding: np.ndarray, k: int = 5) -> List[Embedding]:
        """Find k most similar concepts in cache."""
        if not self._concept_cache:
            return []

        similarities = []
        query_embedding = Embedding(vector=embedding, text="", source="query")

        for name, concept in self._concept_cache.items():
            sim = query_embedding.similarity(concept)
            similarities.append((sim, concept))

        similarities.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in similarities[:k]]

    def register_concept(self, name: str, text: str) -> Embedding:
        """
        Register a named concept for faster retrieval.

        Args:
            name: Concept identifier
            text: Text description of concept

        Returns:
            The created embedding
        """
        vector = self.encode(text, use_context=False)
        embedding = Embedding(vector=vector, text=text, source="concept")
        self._concept_cache[name] = embedding
        return embedding

    def get_concept(self, name: str) -> Optional[Embedding]:
        """Get a registered concept by name."""
        return self._concept_cache.get(name)

    def observation_to_text(self, observation: np.ndarray) -> str:
        """
        Convert observation vector to text description.

        Args:
            observation: Sensory observation vector

        Returns:
            Natural language description
        """
        # Project observation to embedding space
        if len(observation) != self.config.internal_dim:
            if len(observation) < self.config.internal_dim:
                observation = np.pad(observation, (0, self.config.internal_dim - len(observation)))
            else:
                observation = observation[: self.config.internal_dim]

        # Describe based on activation patterns
        activations = []

        # Segment observation into regions
        region_size = len(observation) // 4
        regions = ["perception", "action", "memory", "goal"]

        for i, region in enumerate(regions):
            start = i * region_size
            end = start + region_size
            activation = float(np.mean(np.abs(observation[start:end])))
            if activation > 0.3:
                activations.append(f"high {region} activity")
            elif activation > 0.1:
                activations.append(f"moderate {region} activity")

        if activations:
            description = "Current state shows " + ", ".join(activations) + "."
        else:
            description = "Current state is relatively quiet."

        return description

    def text_to_action(self, text: str) -> np.ndarray:
        """
        Convert text instruction to action vector.

        Args:
            text: Natural language action description

        Returns:
            Action vector
        """
        # Encode text
        encoded = self.encode(text, use_context=False)

        # Parse action structure
        parsed = self.llm.parse_action(text)

        # Modify vector based on action type
        action = encoded.copy()

        action_type = parsed.get("type", "unknown")
        type_weights = {
            "move": np.array([1, 0, 0, 0]),
            "take": np.array([0, 1, 0, 0]),
            "speak": np.array([0, 0, 1, 0]),
            "observe": np.array([0, 0, 0, 1]),
            "wait": np.array([0, 0, 0, 0]),
            "unknown": np.array([0.25, 0.25, 0.25, 0.25]),
        }

        weight = type_weights.get(action_type, type_weights["unknown"])

        # Inject action type into first dimensions
        action[: len(weight)] = weight

        return action

    def align_spaces(
        self,
        text_samples: List[str],
        internal_samples: List[np.ndarray],
        learning_rate: float = 0.01,
    ) -> float:
        """
        Learn alignment between text and internal spaces.

        Uses pairs of (text, internal_vector) to improve projection.

        Args:
            text_samples: List of text samples
            internal_samples: Corresponding internal vectors

        Returns:
            Alignment loss
        """
        if len(text_samples) != len(internal_samples):
            raise ValueError("Samples must have same length")

        total_loss = 0.0

        for text, target in zip(text_samples, internal_samples):
            # Get current projection
            llm_embedding = self.llm.embed(text)
            if len(llm_embedding) < self.config.embedding_dim:
                llm_embedding = np.pad(
                    llm_embedding, (0, self.config.embedding_dim - len(llm_embedding))
                )
            else:
                llm_embedding = llm_embedding[: self.config.embedding_dim]

            predicted = self._project_to_internal(llm_embedding)

            # Resize target if needed
            if len(target) != self.config.internal_dim:
                if len(target) < self.config.internal_dim:
                    target = np.pad(target, (0, self.config.internal_dim - len(target)))
                else:
                    target = target[: self.config.internal_dim]

            # Compute error
            error = target - predicted
            loss = float(np.mean(error**2))
            total_loss += loss

            # Update projection (simple gradient descent)
            gradient = np.outer(error, llm_embedding)
            self._embedding_to_internal += learning_rate * gradient

        return total_loss / len(text_samples)

    def get_statistics(self) -> Dict[str, Any]:
        """Get bridge statistics."""
        return {
            "encode_count": self._encode_count,
            "decode_count": self._decode_count,
            "memory_size": len(self._memory),
            "concept_count": len(self._concept_cache),
            "embedding_dim": self.config.embedding_dim,
            "internal_dim": self.config.internal_dim,
            "llm_stats": self.llm.get_statistics(),
        }

    def reset(self) -> None:
        """Reset bridge state."""
        self._memory = []
        # Keep concept cache
