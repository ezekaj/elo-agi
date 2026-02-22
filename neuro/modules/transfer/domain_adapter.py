"""
Domain Adapter: Adapt representations across domains.

Implements domain alignment, transfer mapping, and representation adaptation.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
import numpy as np


@dataclass
class DomainEmbedding:
    """Embedding space for a domain."""

    domain_name: str
    embedding_dim: int
    examples: np.ndarray  # (n_examples, embedding_dim)
    labels: Optional[List[str]] = None
    centroid: Optional[np.ndarray] = None


@dataclass
class DomainAlignment:
    """Alignment between two domains."""

    source_domain: str
    target_domain: str
    transformation: np.ndarray  # (embedding_dim, embedding_dim)
    inverse_transformation: np.ndarray
    alignment_score: float
    method: str


@dataclass
class TransferMapping:
    """A learned mapping for transfer."""

    id: str
    source_domain: str
    target_domain: str
    transformation: np.ndarray
    bias: np.ndarray
    quality_score: float


@dataclass
class AdaptedRepresentation:
    """A representation adapted to a new domain."""

    original: np.ndarray
    adapted: np.ndarray
    source_domain: str
    target_domain: str
    confidence: float


class SubspaceAlignment:
    """
    Align source and target domain subspaces.
    """

    def __init__(
        self,
        n_components: int = 50,
    ):
        self.n_components = n_components

    def align(
        self,
        source: DomainEmbedding,
        target: DomainEmbedding,
    ) -> DomainAlignment:
        """
        Compute subspace alignment transformation.
        """
        # Center the data
        source_centered = source.examples - source.examples.mean(axis=0)
        target_centered = target.examples - target.examples.mean(axis=0)

        # Compute principal components
        source_pcs = self._compute_pcs(source_centered)
        target_pcs = self._compute_pcs(target_centered)

        # Alignment transformation: project source to source PCs, then to target PCs
        transformation = source_pcs.T @ target_pcs

        # Inverse transformation
        inverse = target_pcs.T @ source_pcs

        # Compute alignment score (based on transformation orthogonality)
        identity = np.eye(self.n_components)
        score = 1.0 - np.linalg.norm(transformation @ inverse - identity) / self.n_components

        return DomainAlignment(
            source_domain=source.domain_name,
            target_domain=target.domain_name,
            transformation=transformation,
            inverse_transformation=inverse,
            alignment_score=float(score),
            method="subspace",
        )

    def _compute_pcs(self, data: np.ndarray) -> np.ndarray:
        """Compute principal components."""
        # SVD
        n = min(self.n_components, data.shape[1], data.shape[0])
        try:
            u, s, vt = np.linalg.svd(data, full_matrices=False)
            return vt[:n].T
        except Exception:
            # Fallback to identity
            return np.eye(data.shape[1])[:, :n]


class CorrelationAlignment:
    """
    CORAL: Correlation Alignment for domain adaptation.
    """

    def __init__(
        self,
        regularization: float = 1e-5,
    ):
        self.regularization = regularization

    def align(
        self,
        source: DomainEmbedding,
        target: DomainEmbedding,
    ) -> DomainAlignment:
        """
        Compute CORAL transformation.
        """
        d = source.examples.shape[1]

        # Compute covariance matrices
        source_cov = np.cov(source.examples.T) + self.regularization * np.eye(d)
        target_cov = np.cov(target.examples.T) + self.regularization * np.eye(d)

        # Whitening transformation for source
        source_sqrt_inv = self._matrix_sqrt_inv(source_cov)

        # Coloring transformation for target
        target_sqrt = self._matrix_sqrt(target_cov)

        # Combined transformation
        transformation = source_sqrt_inv @ target_sqrt

        # Inverse
        inverse = self._matrix_sqrt_inv(target_cov) @ self._matrix_sqrt(source_cov)

        # Score (correlation distance)
        aligned_source = (source.examples - source.examples.mean(axis=0)) @ transformation
        aligned_cov = np.cov(aligned_source.T) + self.regularization * np.eye(d)
        score = 1.0 - np.linalg.norm(aligned_cov - target_cov) / np.linalg.norm(target_cov)

        return DomainAlignment(
            source_domain=source.domain_name,
            target_domain=target.domain_name,
            transformation=transformation,
            inverse_transformation=inverse,
            alignment_score=float(max(0, score)),
            method="coral",
        )

    def _matrix_sqrt(self, A: np.ndarray) -> np.ndarray:
        """Compute matrix square root."""
        try:
            eigvals, eigvecs = np.linalg.eigh(A)
            eigvals = np.maximum(eigvals, 1e-10)
            return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
        except Exception:
            return np.eye(A.shape[0])

    def _matrix_sqrt_inv(self, A: np.ndarray) -> np.ndarray:
        """Compute inverse matrix square root."""
        try:
            eigvals, eigvecs = np.linalg.eigh(A)
            eigvals = np.maximum(eigvals, 1e-10)
            return eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        except Exception:
            return np.eye(A.shape[0])


class OptimalTransport:
    """
    Optimal transport for domain alignment.
    """

    def __init__(
        self,
        regularization: float = 0.1,
        n_iter: int = 100,
    ):
        self.regularization = regularization
        self.n_iter = n_iter

    def compute_transport(
        self,
        source: DomainEmbedding,
        target: DomainEmbedding,
    ) -> np.ndarray:
        """
        Compute optimal transport plan (Sinkhorn algorithm).

        Returns:
            Transport matrix (n_source, n_target)
        """
        n_source = source.examples.shape[0]
        n_target = target.examples.shape[0]

        # Cost matrix (pairwise distances)
        cost = np.zeros((n_source, n_target))
        for i in range(n_source):
            for j in range(n_target):
                cost[i, j] = np.linalg.norm(source.examples[i] - target.examples[j])

        # Sinkhorn iteration
        K = np.exp(-cost / self.regularization)

        u = np.ones(n_source) / n_source
        v = np.ones(n_target) / n_target

        for _ in range(self.n_iter):
            u = 1.0 / (K @ v + 1e-10)
            v = 1.0 / (K.T @ u + 1e-10)

        transport = np.diag(u) @ K @ np.diag(v)
        return transport

    def transport_samples(
        self,
        source: DomainEmbedding,
        target: DomainEmbedding,
    ) -> np.ndarray:
        """Transport source samples to target domain."""
        transport = self.compute_transport(source, target)

        # Barycentric mapping
        transported = transport @ target.examples
        transported = transported / (transport.sum(axis=1, keepdims=True) + 1e-10)

        return transported


class DomainAdapter:
    """
    Complete domain adaptation system.

    Combines multiple alignment methods for robust transfer.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        default_method: str = "coral",
    ):
        self.embedding_dim = embedding_dim
        self.default_method = default_method

        # Alignment methods
        self.subspace = SubspaceAlignment()
        self.coral = CorrelationAlignment()
        self.ot = OptimalTransport()

        # Domain embeddings
        self._domains: Dict[str, DomainEmbedding] = {}

        # Learned mappings
        self._mappings: Dict[str, TransferMapping] = {}
        self._mapping_counter = 0

    def register_domain(
        self,
        name: str,
        examples: np.ndarray,
        labels: Optional[List[str]] = None,
    ) -> DomainEmbedding:
        """Register a domain with examples."""
        centroid = examples.mean(axis=0)

        embedding = DomainEmbedding(
            domain_name=name,
            embedding_dim=examples.shape[1],
            examples=examples,
            labels=labels,
            centroid=centroid,
        )

        self._domains[name] = embedding
        return embedding

    def align_domains(
        self,
        source_name: str,
        target_name: str,
        method: Optional[str] = None,
    ) -> DomainAlignment:
        """
        Align two domains.
        """
        source = self._domains.get(source_name)
        target = self._domains.get(target_name)

        if source is None:
            raise ValueError(f"Unknown source domain: {source_name}")
        if target is None:
            raise ValueError(f"Unknown target domain: {target_name}")

        method = method or self.default_method

        if method == "subspace":
            return self.subspace.align(source, target)
        elif method == "coral":
            return self.coral.align(source, target)
        else:
            raise ValueError(f"Unknown method: {method}")

    def create_mapping(
        self,
        source_name: str,
        target_name: str,
        method: Optional[str] = None,
    ) -> TransferMapping:
        """Create a transfer mapping between domains."""
        alignment = self.align_domains(source_name, target_name, method)

        source = self._domains[source_name]
        target = self._domains[target_name]

        self._mapping_counter += 1

        mapping = TransferMapping(
            id=f"mapping_{self._mapping_counter}",
            source_domain=source_name,
            target_domain=target_name,
            transformation=alignment.transformation,
            bias=target.centroid - source.centroid @ alignment.transformation,
            quality_score=alignment.alignment_score,
        )

        key = f"{source_name}->{target_name}"
        self._mappings[key] = mapping

        return mapping

    def adapt(
        self,
        representation: np.ndarray,
        source_domain: str,
        target_domain: str,
    ) -> AdaptedRepresentation:
        """
        Adapt a representation from source to target domain.
        """
        key = f"{source_domain}->{target_domain}"
        mapping = self._mappings.get(key)

        if mapping is None:
            # Create mapping on the fly
            mapping = self.create_mapping(source_domain, target_domain)

        # Apply transformation
        adapted = representation @ mapping.transformation + mapping.bias

        return AdaptedRepresentation(
            original=representation,
            adapted=adapted,
            source_domain=source_domain,
            target_domain=target_domain,
            confidence=mapping.quality_score,
        )

    def adapt_batch(
        self,
        representations: np.ndarray,
        source_domain: str,
        target_domain: str,
    ) -> List[AdaptedRepresentation]:
        """Adapt multiple representations."""
        return [self.adapt(r, source_domain, target_domain) for r in representations]

    def compute_domain_distance(
        self,
        domain1: str,
        domain2: str,
    ) -> float:
        """Compute distance between two domains."""
        d1 = self._domains.get(domain1)
        d2 = self._domains.get(domain2)

        if d1 is None or d2 is None:
            return float("inf")

        # Maximum Mean Discrepancy (simplified)
        mean1 = d1.examples.mean(axis=0)
        mean2 = d2.examples.mean(axis=0)

        return float(np.linalg.norm(mean1 - mean2))

    def find_nearest_domain(
        self,
        query_examples: np.ndarray,
    ) -> Tuple[str, float]:
        """Find the nearest registered domain to query examples."""
        query_mean = query_examples.mean(axis=0)

        best_domain = None
        best_distance = float("inf")

        for name, domain in self._domains.items():
            dist = np.linalg.norm(query_mean - domain.centroid)
            if dist < best_distance:
                best_distance = dist
                best_domain = name

        return best_domain, best_distance

    def get_mapping(
        self,
        source: str,
        target: str,
    ) -> Optional[TransferMapping]:
        """Get an existing mapping."""
        return self._mappings.get(f"{source}->{target}")

    def statistics(self) -> Dict[str, Any]:
        """Get adapter statistics."""
        return {
            "n_domains": len(self._domains),
            "n_mappings": len(self._mappings),
            "domains": list(self._domains.keys()),
            "mappings": [
                {
                    "id": m.id,
                    "source": m.source_domain,
                    "target": m.target_domain,
                    "quality": m.quality_score,
                }
                for m in self._mappings.values()
            ],
        }
