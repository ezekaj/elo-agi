"""
Schema Refinement for Dynamic Knowledge Structure Updates.

Implements mechanisms for updating, generalizing, and refining
schemas based on new experiences during consolidation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum
import numpy as np


class SchemaUpdateType(Enum):
    """Types of schema updates."""
    ASSIMILATE = "assimilate"       # New instance fits schema
    ACCOMMODATE = "accommodate"     # Schema adjusted for new instance
    GENERALIZE = "generalize"       # Schema broadened
    SPECIALIZE = "specialize"       # Schema narrowed
    MERGE = "merge"                 # Schemas combined
    SPLIT = "split"                 # Schema divided


@dataclass
class Schema:
    """
    Knowledge schema representation.

    Schemas are abstract knowledge structures that organize
    related memories and enable generalization.
    """
    name: str
    prototype: np.ndarray           # Centroid/average representation
    variance: np.ndarray            # Variance along each dimension
    coverage: float                 # Proportion of domain covered
    instances: List[str]            # Memory IDs that instantiate this schema
    sub_schemas: List[str] = field(default_factory=list)
    parent_schema: Optional[str] = None
    constraints: Dict[str, Any] = field(default_factory=dict)
    last_updated: float = 0.0
    creation_time: float = 0.0
    n_updates: int = 0
    confidence: float = 1.0

    def __post_init__(self):
        if self.variance is None:
            self.variance = np.ones_like(self.prototype)

    def contains(self, vector: np.ndarray, threshold: float = 2.0) -> bool:
        """Check if a vector falls within this schema (Mahalanobis-like)."""
        if len(vector) != len(self.prototype):
            return False

        diff = vector - self.prototype
        # Simplified Mahalanobis using variance
        safe_var = np.maximum(self.variance, 1e-8)
        distance = np.sqrt(np.sum(diff ** 2 / safe_var))

        return distance <= threshold

    def similarity_to(self, vector: np.ndarray) -> float:
        """Compute similarity to a vector."""
        norm_p = np.linalg.norm(self.prototype)
        norm_v = np.linalg.norm(vector)
        if norm_p < 1e-8 or norm_v < 1e-8:
            return 0.0
        return float(np.dot(self.prototype, vector) / (norm_p * norm_v))


@dataclass
class SchemaUpdate:
    """Record of a schema update."""
    schema_name: str
    update_type: SchemaUpdateType
    timestamp: float
    old_prototype: np.ndarray
    new_prototype: np.ndarray
    triggering_memory: Optional[str] = None
    delta_coverage: float = 0.0
    delta_confidence: float = 0.0


class SchemaRefiner:
    """
    Refines schemas based on consolidation and new experiences.

    Implements:
    - Schema updating (assimilation/accommodation)
    - Schema generalization and specialization
    - Schema merging and splitting
    - Coverage and confidence tracking
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        similarity_threshold: float = 0.7,
        coverage_threshold: float = 0.8,
        min_instances_for_schema: int = 3,
        random_seed: Optional[int] = None,
    ):
        self.embedding_dim = embedding_dim
        self.similarity_threshold = similarity_threshold
        self.coverage_threshold = coverage_threshold
        self.min_instances_for_schema = min_instances_for_schema
        self._rng = np.random.default_rng(random_seed)

        # Schema storage
        self._schemas: Dict[str, Schema] = {}

        # Instance vectors (for recomputation)
        self._instance_vectors: Dict[str, np.ndarray] = {}

        # Update history
        self._updates: List[SchemaUpdate] = []

        # Statistics
        self._n_updates = 0
        self._n_generalizations = 0
        self._n_specializations = 0
        self._n_merges = 0

    def create_schema(
        self,
        name: str,
        prototype: np.ndarray,
        instances: Optional[List[str]] = None,
        timestamp: float = 0.0,
    ) -> Schema:
        """Create a new schema."""
        instances = instances or []

        # Compute variance from instances if available
        if instances and all(i in self._instance_vectors for i in instances):
            vectors = [self._instance_vectors[i] for i in instances]
            variance = np.var(vectors, axis=0) if len(vectors) > 1 else np.ones(self.embedding_dim)
        else:
            variance = np.ones(self.embedding_dim)

        schema = Schema(
            name=name,
            prototype=prototype.copy(),
            variance=variance,
            coverage=len(instances) / max(1, len(self._instance_vectors)),
            instances=instances.copy(),
            creation_time=timestamp,
            last_updated=timestamp,
        )

        self._schemas[name] = schema
        return schema

    def register_instance(
        self,
        instance_id: str,
        vector: np.ndarray,
    ) -> None:
        """Register an instance vector."""
        self._instance_vectors[instance_id] = vector.copy()

    def update_schema(
        self,
        schema_name: str,
        new_instance: str,
        instance_vector: np.ndarray,
        timestamp: float = 0.0,
    ) -> Tuple[Schema, SchemaUpdateType]:
        """
        Update a schema with a new instance.

        Returns updated schema and type of update performed.
        """
        self._n_updates += 1
        self.register_instance(new_instance, instance_vector)

        schema = self._schemas.get(schema_name)
        if schema is None:
            # Create new schema
            schema = self.create_schema(
                name=schema_name,
                prototype=instance_vector,
                instances=[new_instance],
                timestamp=timestamp,
            )
            return schema, SchemaUpdateType.ASSIMILATE

        old_prototype = schema.prototype.copy()

        # Check if instance fits current schema
        if schema.contains(instance_vector):
            # Assimilate - slight prototype adjustment
            update_type = SchemaUpdateType.ASSIMILATE
            alpha = 1.0 / (len(schema.instances) + 1)
            schema.prototype = (1 - alpha) * schema.prototype + alpha * instance_vector
        else:
            # Accommodate - larger adjustment
            update_type = SchemaUpdateType.ACCOMMODATE
            # Stronger update for outliers
            alpha = 0.3
            schema.prototype = (1 - alpha) * schema.prototype + alpha * instance_vector

        # Update variance
        if schema.instances:
            vectors = [
                self._instance_vectors.get(i, schema.prototype)
                for i in schema.instances
            ]
            vectors.append(instance_vector)
            schema.variance = np.var(vectors, axis=0)

        # Update metadata
        schema.instances.append(new_instance)
        schema.last_updated = timestamp
        schema.n_updates += 1

        # Recompute coverage
        schema.coverage = self.compute_coverage(schema_name)

        # Record update
        self._updates.append(SchemaUpdate(
            schema_name=schema_name,
            update_type=update_type,
            timestamp=timestamp,
            old_prototype=old_prototype,
            new_prototype=schema.prototype.copy(),
            triggering_memory=new_instance,
        ))

        return schema, update_type

    def compute_coverage(
        self,
        schema_name: str,
        instance_ids: Optional[List[str]] = None,
    ) -> float:
        """Compute what fraction of instances a schema covers."""
        schema = self._schemas.get(schema_name)
        if schema is None:
            return 0.0

        instances = instance_ids or list(self._instance_vectors.keys())
        if not instances:
            return 0.0

        covered = 0
        for inst_id in instances:
            vector = self._instance_vectors.get(inst_id)
            if vector is not None and schema.contains(vector):
                covered += 1

        return covered / len(instances)

    def generalize_schema(
        self,
        schema_name: str,
        instances: List[str],
        timestamp: float = 0.0,
    ) -> Schema:
        """
        Generalize a schema to cover more instances.

        Expands variance and adjusts prototype.
        """
        self._n_generalizations += 1

        schema = self._schemas.get(schema_name)
        if schema is None:
            return None

        old_prototype = schema.prototype.copy()

        # Get all instance vectors
        vectors = [
            self._instance_vectors[i]
            for i in instances
            if i in self._instance_vectors
        ]

        if not vectors:
            return schema

        # Update prototype to centroid
        new_prototype = np.mean(vectors, axis=0)
        schema.prototype = new_prototype

        # Increase variance to cover all
        new_variance = np.var(vectors, axis=0)
        schema.variance = np.maximum(schema.variance, new_variance * 1.5)

        # Update instances
        for inst in instances:
            if inst not in schema.instances:
                schema.instances.append(inst)

        schema.last_updated = timestamp
        schema.n_updates += 1
        schema.coverage = self.compute_coverage(schema_name)

        self._updates.append(SchemaUpdate(
            schema_name=schema_name,
            update_type=SchemaUpdateType.GENERALIZE,
            timestamp=timestamp,
            old_prototype=old_prototype,
            new_prototype=new_prototype,
        ))

        return schema

    def specialize_schema(
        self,
        schema_name: str,
        exceptions: List[str],
        timestamp: float = 0.0,
    ) -> List[Schema]:
        """
        Specialize a schema by creating sub-schemas for exceptions.

        Returns list of new specialized schemas.
        """
        self._n_specializations += 1

        schema = self._schemas.get(schema_name)
        if schema is None:
            return []

        # Separate exception vectors
        exception_vectors = [
            self._instance_vectors[e]
            for e in exceptions
            if e in self._instance_vectors
        ]

        if not exception_vectors:
            return []

        # Cluster exceptions (simple k-means with k=2)
        exception_centroid = np.mean(exception_vectors, axis=0)

        # Create specialized sub-schema
        sub_name = f"{schema_name}_specialized_{len(schema.sub_schemas)}"
        sub_schema = self.create_schema(
            name=sub_name,
            prototype=exception_centroid,
            instances=exceptions,
            timestamp=timestamp,
        )
        sub_schema.parent_schema = schema_name
        sub_schema.variance = np.var(exception_vectors, axis=0) if len(exception_vectors) > 1 else schema.variance * 0.5

        # Update parent
        schema.sub_schemas.append(sub_name)

        # Remove exceptions from parent instances
        schema.instances = [i for i in schema.instances if i not in exceptions]

        # Recompute parent prototype
        remaining_vectors = [
            self._instance_vectors[i]
            for i in schema.instances
            if i in self._instance_vectors
        ]
        if remaining_vectors:
            schema.prototype = np.mean(remaining_vectors, axis=0)
            schema.variance = np.var(remaining_vectors, axis=0) if len(remaining_vectors) > 1 else schema.variance

        return [sub_schema]

    def merge_schemas(
        self,
        schema_a: str,
        schema_b: str,
        new_name: Optional[str] = None,
        timestamp: float = 0.0,
    ) -> Schema:
        """
        Merge two schemas into one.

        Creates a more general schema covering both.
        """
        self._n_merges += 1

        s_a = self._schemas.get(schema_a)
        s_b = self._schemas.get(schema_b)

        if s_a is None or s_b is None:
            return None

        new_name = new_name or f"{schema_a}_{schema_b}_merged"

        # Combine instances
        all_instances = list(set(s_a.instances + s_b.instances))

        # Get all vectors
        all_vectors = [
            self._instance_vectors[i]
            for i in all_instances
            if i in self._instance_vectors
        ]

        if not all_vectors:
            # Fallback to prototype average
            new_prototype = (s_a.prototype + s_b.prototype) / 2
            new_variance = (s_a.variance + s_b.variance) / 2
        else:
            new_prototype = np.mean(all_vectors, axis=0)
            new_variance = np.var(all_vectors, axis=0)

        merged = Schema(
            name=new_name,
            prototype=new_prototype,
            variance=new_variance,
            coverage=len(all_instances) / max(1, len(self._instance_vectors)),
            instances=all_instances,
            creation_time=timestamp,
            last_updated=timestamp,
            confidence=(s_a.confidence + s_b.confidence) / 2,
        )

        self._schemas[new_name] = merged

        self._updates.append(SchemaUpdate(
            schema_name=new_name,
            update_type=SchemaUpdateType.MERGE,
            timestamp=timestamp,
            old_prototype=s_a.prototype,  # Arbitrary choice
            new_prototype=new_prototype,
        ))

        return merged

    def find_best_schema(
        self,
        vector: np.ndarray,
    ) -> Optional[Tuple[str, float]]:
        """Find schema that best matches a vector."""
        best_schema = None
        best_similarity = -1.0

        for name, schema in self._schemas.items():
            similarity = schema.similarity_to(vector)
            if similarity > best_similarity:
                best_similarity = similarity
                best_schema = name

        if best_similarity < self.similarity_threshold:
            return None

        return (best_schema, best_similarity)

    def find_schemas_containing(
        self,
        vector: np.ndarray,
    ) -> List[str]:
        """Find all schemas that contain a vector."""
        containing = []
        for name, schema in self._schemas.items():
            if schema.contains(vector):
                containing.append(name)
        return containing

    def get_schema(self, name: str) -> Optional[Schema]:
        """Get a schema by name."""
        return self._schemas.get(name)

    def get_all_schemas(self) -> Dict[str, Schema]:
        """Get all schemas."""
        return self._schemas.copy()

    def prune_low_coverage_schemas(
        self,
        min_coverage: float = 0.1,
        min_instances: int = 2,
    ) -> List[str]:
        """Remove schemas with low coverage or few instances."""
        to_remove = []

        for name, schema in self._schemas.items():
            if schema.coverage < min_coverage and len(schema.instances) < min_instances:
                to_remove.append(name)

        for name in to_remove:
            del self._schemas[name]

        return to_remove

    def auto_discover_schemas(
        self,
        similarity_threshold: float = 0.6,
        timestamp: float = 0.0,
    ) -> List[Schema]:
        """
        Automatically discover schemas from instances.

        Uses simple clustering based on similarity.
        """
        if len(self._instance_vectors) < self.min_instances_for_schema:
            return []

        vectors = list(self._instance_vectors.values())
        ids = list(self._instance_vectors.keys())

        # Simple hierarchical clustering
        assigned = set()
        new_schemas = []

        for i, (inst_id, vector) in enumerate(zip(ids, vectors)):
            if inst_id in assigned:
                continue

            # Find similar instances
            cluster = [inst_id]
            cluster_vectors = [vector]

            for j, (other_id, other_vector) in enumerate(zip(ids[i+1:], vectors[i+1:]), i+1):
                if other_id in assigned:
                    continue

                sim = np.dot(vector, other_vector) / (
                    np.linalg.norm(vector) * np.linalg.norm(other_vector) + 1e-8
                )

                if sim >= similarity_threshold:
                    cluster.append(other_id)
                    cluster_vectors.append(other_vector)

            if len(cluster) >= self.min_instances_for_schema:
                # Create schema
                prototype = np.mean(cluster_vectors, axis=0)
                schema_name = f"auto_schema_{len(new_schemas)}"

                schema = self.create_schema(
                    name=schema_name,
                    prototype=prototype,
                    instances=cluster,
                    timestamp=timestamp,
                )
                new_schemas.append(schema)

                assigned.update(cluster)

        return new_schemas

    def statistics(self) -> Dict[str, Any]:
        """Get schema refinement statistics."""
        if not self._schemas:
            return {
                "total_schemas": 0,
                "total_instances": len(self._instance_vectors),
            }

        coverages = [s.coverage for s in self._schemas.values()]
        confidences = [s.confidence for s in self._schemas.values()]
        instance_counts = [len(s.instances) for s in self._schemas.values()]

        # Update type distribution
        update_types = {}
        for update in self._updates:
            t = update.update_type.value
            update_types[t] = update_types.get(t, 0) + 1

        return {
            "total_schemas": len(self._schemas),
            "total_instances": len(self._instance_vectors),
            "average_coverage": float(np.mean(coverages)),
            "average_confidence": float(np.mean(confidences)),
            "average_instances_per_schema": float(np.mean(instance_counts)),
            "max_instances_per_schema": int(np.max(instance_counts)),
            "n_updates": self._n_updates,
            "n_generalizations": self._n_generalizations,
            "n_specializations": self._n_specializations,
            "n_merges": self._n_merges,
            "update_type_distribution": update_types,
        }


__all__ = [
    'SchemaUpdateType',
    'Schema',
    'SchemaUpdate',
    'SchemaRefiner',
]
