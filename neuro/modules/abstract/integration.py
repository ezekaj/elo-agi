"""
SharedSpace Integration for neuro-abstract.

Connects abstraction and symbolic binding to the shared semantic space
for cross-module integration with the rest of the Neuro AGI system.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum
import numpy as np

from .symbolic_binder import SymbolicBinder, CompositeBinding, RoleType
from .abstraction_engine import AbstractionEngine, Abstraction, AbstractionLevel
from .composition_types import CompositionType, AtomicType, StructuredType


class SemanticModalityType(Enum):
    """Types of semantic content in shared space."""

    SYMBOLIC = "symbolic"  # Pure symbolic representations
    GROUNDED = "grounded"  # Symbol-percept bindings
    ABSTRACT = "abstract"  # Abstractions
    RELATIONAL = "relational"  # Relations between concepts
    PROCEDURAL = "procedural"  # Programs/procedures


@dataclass
class AbstractSemanticEmbedding:
    """
    A semantic embedding in shared space from neuro-abstract.

    Combines symbolic, structural, and neural information.
    """

    vector: np.ndarray
    modality: SemanticModalityType
    source: str
    symbol: Optional[str] = None
    abstraction_id: Optional[str] = None
    binding_id: Optional[str] = None
    roles: Dict[RoleType, str] = field(default_factory=dict)
    type_info: Optional[CompositionType] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def similarity(self, other: "AbstractSemanticEmbedding") -> float:
        """Compute cosine similarity with another embedding."""
        norm_self = np.linalg.norm(self.vector)
        norm_other = np.linalg.norm(other.vector)
        if norm_self < 1e-8 or norm_other < 1e-8:
            return 0.0
        return float(np.dot(self.vector, other.vector) / (norm_self * norm_other))


@dataclass
class SharedSpaceConfig:
    """Configuration for shared space integration."""

    embedding_dim: int = 512
    projection_dim: int = 256
    n_attention_heads: int = 8
    similarity_threshold: float = 0.6
    max_active_concepts: int = 100
    random_seed: Optional[int] = None


class SharedSpaceProjection:
    """
    Projection layer from abstract representations to shared space.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        random_seed: Optional[int] = None,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim

        rng = np.random.default_rng(random_seed)

        # Projection weights
        scale = np.sqrt(2.0 / (input_dim + output_dim))
        self.weights = rng.normal(0, scale, (input_dim, output_dim))
        self.bias = np.zeros(output_dim)

    def project(self, vector: np.ndarray) -> np.ndarray:
        """Project to shared space."""
        if len(vector) < self.input_dim:
            vector = np.pad(vector, (0, self.input_dim - len(vector)))
        elif len(vector) > self.input_dim:
            vector = vector[: self.input_dim]

        projected = np.tanh(vector @ self.weights + self.bias)
        norm = np.linalg.norm(projected)
        if norm > 1e-8:
            projected = projected / norm
        return projected


class SharedSpaceIntegration:
    """
    Integration layer connecting neuro-abstract to the shared semantic space.

    Provides:
    - Projection of abstract concepts to shared space
    - Query interface for concept retrieval
    - Cross-modal binding with other modules
    - Abstraction-grounded reasoning
    """

    def __init__(
        self,
        config: Optional[SharedSpaceConfig] = None,
        binder: Optional[SymbolicBinder] = None,
        abstraction_engine: Optional[AbstractionEngine] = None,
    ):
        self.config = config or SharedSpaceConfig()

        # Components
        self.binder = binder or SymbolicBinder(
            embedding_dim=self.config.projection_dim,
            random_seed=self.config.random_seed,
        )
        self.abstraction_engine = abstraction_engine or AbstractionEngine(
            embedding_dim=self.config.projection_dim,
            random_seed=self.config.random_seed,
        )

        # Projection layer
        self.projection = SharedSpaceProjection(
            input_dim=self.config.projection_dim,
            output_dim=self.config.embedding_dim,
            random_seed=self.config.random_seed,
        )

        # Active embeddings
        self._active: List[AbstractSemanticEmbedding] = []

        # Concept memory
        self._memory: Dict[str, AbstractSemanticEmbedding] = {}

        # Statistics
        self._n_projections = 0
        self._n_queries = 0
        self._n_bindings = 0

    def project_binding(
        self,
        binding: CompositeBinding,
        activate: bool = True,
    ) -> AbstractSemanticEmbedding:
        """
        Project a symbolic binding to shared space.
        """
        self._n_projections += 1

        # Compose binding to get neural representation
        composed = self.binder.compose([binding])

        # Project to shared space
        projected = self.projection.project(composed)

        # Extract roles
        roles = {role: rb.filler for role, rb in binding.role_bindings.items()}

        embedding = AbstractSemanticEmbedding(
            vector=projected,
            modality=SemanticModalityType.SYMBOLIC,
            source="symbolic_binder",
            symbol=binding.symbol,
            roles=roles,
            confidence=binding.confidence,
        )

        if activate:
            self._activate(embedding)

        return embedding

    def project_abstraction(
        self,
        abstraction: Abstraction,
        activate: bool = True,
    ) -> AbstractSemanticEmbedding:
        """
        Project an abstraction to shared space.
        """
        self._n_projections += 1

        # Use abstraction embedding if available, else create one
        if abstraction.embedding is not None:
            raw = abstraction.embedding
        else:
            raw = np.random.randn(self.config.projection_dim)
            raw = raw / np.linalg.norm(raw)

        # Project to shared space
        projected = self.projection.project(raw)

        embedding = AbstractSemanticEmbedding(
            vector=projected,
            modality=SemanticModalityType.ABSTRACT,
            source="abstraction_engine",
            abstraction_id=abstraction.id,
            confidence=abstraction.confidence,
            metadata={
                "level": abstraction.level.value,
                "variables": abstraction.variables,
                "n_instances": len(abstraction.instances),
            },
        )

        if activate:
            self._activate(embedding)

        return embedding

    def project_type(
        self,
        type_obj: CompositionType,
        symbol: Optional[str] = None,
        activate: bool = True,
    ) -> AbstractSemanticEmbedding:
        """
        Project a compositional type to shared space.
        """
        self._n_projections += 1

        # Get type neural representation
        raw = type_obj.to_neural(self.config.projection_dim)

        # Project to shared space
        projected = self.projection.project(raw)

        embedding = AbstractSemanticEmbedding(
            vector=projected,
            modality=SemanticModalityType.RELATIONAL,
            source="type_system",
            symbol=symbol,
            type_info=type_obj,
            confidence=1.0,
        )

        if activate:
            self._activate(embedding)

        return embedding

    def _activate(self, embedding: AbstractSemanticEmbedding) -> None:
        """Add embedding to active workspace."""
        # Check for similar existing embedding
        for i, existing in enumerate(self._active):
            if embedding.similarity(existing) > self.config.similarity_threshold:
                # Merge
                merged_vector = 0.7 * existing.vector + 0.3 * embedding.vector
                merged_vector = merged_vector / np.linalg.norm(merged_vector)
                self._active[i].vector = merged_vector
                self._active[i].confidence = max(existing.confidence, embedding.confidence)
                return

        self._active.append(embedding)

        # Prune if over capacity
        if len(self._active) > self.config.max_active_concepts:
            self._active.sort(key=lambda e: e.confidence, reverse=True)
            self._active = self._active[: self.config.max_active_concepts]

    def query(
        self,
        query_vector: np.ndarray,
        modality: Optional[SemanticModalityType] = None,
        top_k: int = 5,
    ) -> List[Tuple[AbstractSemanticEmbedding, float]]:
        """
        Query active embeddings.
        """
        self._n_queries += 1

        # Project query if needed
        if len(query_vector) != self.config.embedding_dim:
            query_vector = self.projection.project(query_vector)

        results = []
        for embedding in self._active:
            if modality and embedding.modality != modality:
                continue

            sim = float(np.dot(query_vector, embedding.vector))
            results.append((embedding, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def query_by_symbol(
        self,
        symbol: str,
    ) -> List[AbstractSemanticEmbedding]:
        """Query embeddings by symbol name."""
        return [e for e in self._active if e.symbol == symbol]

    def query_by_role(
        self,
        role: RoleType,
        filler: str,
    ) -> List[AbstractSemanticEmbedding]:
        """Query embeddings by role-filler pair."""
        return [e for e in self._active if e.roles.get(role) == filler]

    def bind_to_shared_space(
        self,
        symbol: str,
        roles: Optional[Dict[RoleType, str]] = None,
    ) -> AbstractSemanticEmbedding:
        """
        Create and project a new binding to shared space.
        """
        self._n_bindings += 1

        # Create binding
        if roles:
            role_tuples = {role: (filler, None) for role, filler in roles.items()}
            binding = self.binder.bind(symbol, roles=role_tuples)
        else:
            binding = self.binder.bind(symbol)

        return self.project_binding(binding)

    def create_grounded_embedding(
        self,
        symbol: str,
        perceptual_embedding: np.ndarray,
        abstraction: Optional[Abstraction] = None,
    ) -> AbstractSemanticEmbedding:
        """
        Create a grounded embedding combining symbol and perception.
        """
        self._n_projections += 1

        # Register symbol
        self.binder.register_symbol(symbol)
        symbol_rep = self.binder.get_symbol(symbol)

        # Combine symbol and perceptual representations
        if len(perceptual_embedding) < self.config.projection_dim:
            perceptual_embedding = np.pad(
                perceptual_embedding, (0, self.config.projection_dim - len(perceptual_embedding))
            )
        elif len(perceptual_embedding) > self.config.projection_dim:
            perceptual_embedding = perceptual_embedding[: self.config.projection_dim]

        # Weighted combination
        combined = 0.5 * symbol_rep + 0.5 * perceptual_embedding
        combined = combined / np.linalg.norm(combined)

        # Project to shared space
        projected = self.projection.project(combined)

        embedding = AbstractSemanticEmbedding(
            vector=projected,
            modality=SemanticModalityType.GROUNDED,
            source="grounding",
            symbol=symbol,
            abstraction_id=abstraction.id if abstraction else None,
            confidence=1.0,
        )

        self._activate(embedding)
        return embedding

    def store_concept(
        self,
        name: str,
        embedding: AbstractSemanticEmbedding,
    ) -> None:
        """Store a concept in persistent memory."""
        self._memory[name] = embedding

    def retrieve_concept(
        self,
        name: str,
    ) -> Optional[AbstractSemanticEmbedding]:
        """Retrieve a concept from memory."""
        return self._memory.get(name)

    def query_memory(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
    ) -> List[Tuple[str, AbstractSemanticEmbedding, float]]:
        """Query persistent memory."""
        results = []

        for name, embedding in self._memory.items():
            sim = float(np.dot(query_vector, embedding.vector))
            results.append((name, embedding, sim))

        results.sort(key=lambda x: x[2], reverse=True)
        return results[:top_k]

    def abstract_and_project(
        self,
        examples: List[Dict[str, Any]],
        level: AbstractionLevel = AbstractionLevel.CONCEPT,
    ) -> List[AbstractSemanticEmbedding]:
        """
        Abstract from examples and project to shared space.
        """
        abstractions = self.abstraction_engine.abstract(examples, level)
        embeddings = []

        for abstraction in abstractions:
            embedding = self.project_abstraction(abstraction)
            embeddings.append(embedding)

        return embeddings

    def find_analogical_concepts(
        self,
        source_embedding: AbstractSemanticEmbedding,
        target_domain_filter: Optional[str] = None,
    ) -> List[Tuple[AbstractSemanticEmbedding, float]]:
        """
        Find analogical concepts in shared space.
        """
        results = self.query(source_embedding.vector, top_k=10)

        # Filter by domain if specified
        if target_domain_filter:
            results = [
                (e, s) for e, s in results if e.metadata.get("domain") == target_domain_filter
            ]

        return results

    def get_active_embeddings(self) -> List[AbstractSemanticEmbedding]:
        """Get all active embeddings."""
        return self._active.copy()

    def clear_active(self) -> None:
        """Clear active embeddings."""
        self._active = []

    def statistics(self) -> Dict[str, Any]:
        """Get integration statistics."""
        modality_counts = {}
        for e in self._active:
            modality_counts[e.modality.value] = modality_counts.get(e.modality.value, 0) + 1

        return {
            "n_active_embeddings": len(self._active),
            "n_memory_concepts": len(self._memory),
            "n_projections": self._n_projections,
            "n_queries": self._n_queries,
            "n_bindings": self._n_bindings,
            "modality_distribution": modality_counts,
            "binder_stats": self.binder.statistics(),
            "abstraction_stats": self.abstraction_engine.statistics(),
        }
