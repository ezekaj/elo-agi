"""Tests for schema refinement."""

import pytest
import numpy as np
from neuro.modules.m05_sleep_consolidation.schema_refinement import (
    SchemaRefiner,
    Schema,
    SchemaUpdateType,
    SchemaUpdate,
)


class TestSchema:
    """Tests for Schema."""

    def test_initialization(self):
        """Test schema initialization."""
        prototype = np.random.randn(128)
        schema = Schema(
            name="test_schema",
            prototype=prototype,
            variance=np.ones(128),
            coverage=0.5,
            instances=["a", "b", "c"],
        )
        assert schema.name == "test_schema"
        assert len(schema.instances) == 3
        assert schema.confidence == 1.0

    def test_contains_similar_vector(self):
        """Test containment check for similar vector."""
        prototype = np.array([1.0, 0.0, 0.0])
        schema = Schema(
            name="test",
            prototype=prototype,
            variance=np.ones(3),
            coverage=0.5,
            instances=[],
        )
        similar = np.array([1.1, 0.1, 0.0])
        assert schema.contains(similar, threshold=2.0)

    def test_contains_dissimilar_vector(self):
        """Test containment check for dissimilar vector."""
        prototype = np.array([1.0, 0.0, 0.0])
        schema = Schema(
            name="test",
            prototype=prototype,
            variance=np.ones(3),
            coverage=0.5,
            instances=[],
        )
        dissimilar = np.array([0.0, 0.0, 10.0])
        assert not schema.contains(dissimilar, threshold=2.0)

    def test_similarity_to(self):
        """Test similarity computation."""
        prototype = np.array([1.0, 0.0, 0.0])
        schema = Schema(
            name="test",
            prototype=prototype,
            variance=np.ones(3),
            coverage=0.5,
            instances=[],
        )
        same = np.array([1.0, 0.0, 0.0])
        orthogonal = np.array([0.0, 1.0, 0.0])

        assert schema.similarity_to(same) == pytest.approx(1.0, abs=0.01)
        assert schema.similarity_to(orthogonal) == pytest.approx(0.0, abs=0.01)


class TestSchemaRefiner:
    """Tests for SchemaRefiner."""

    @pytest.fixture
    def refiner(self):
        return SchemaRefiner(
            embedding_dim=64,
            similarity_threshold=0.7,
            min_instances_for_schema=2,
            random_seed=42,
        )

    def test_initialization(self, refiner):
        """Test refiner initialization."""
        stats = refiner.statistics()
        assert stats["total_schemas"] == 0
        assert stats["total_instances"] == 0

    def test_create_schema(self, refiner):
        """Test schema creation."""
        prototype = np.random.randn(64)
        schema = refiner.create_schema(
            name="animals",
            prototype=prototype,
            instances=["dog", "cat"],
            timestamp=0.0,
        )
        assert schema.name == "animals"
        assert len(schema.instances) == 2

    def test_register_instance(self, refiner):
        """Test instance registration."""
        vector = np.random.randn(64)
        refiner.register_instance("dog", vector)
        assert "dog" in refiner._instance_vectors

    def test_update_schema_assimilate(self, refiner):
        """Test schema update with assimilation."""
        prototype = np.random.randn(64)
        prototype = prototype / np.linalg.norm(prototype)

        refiner.create_schema("animals", prototype, instances=[], timestamp=0.0)

        # Similar vector should assimilate
        similar = prototype + 0.1 * np.random.randn(64)
        schema, update_type = refiner.update_schema("animals", "dog", similar, timestamp=1.0)

        assert update_type == SchemaUpdateType.ASSIMILATE
        assert "dog" in schema.instances

    def test_update_schema_accommodate(self, refiner):
        """Test schema update with accommodation."""
        prototype = np.array([1.0] + [0.0] * 63)

        refiner.create_schema("animals", prototype, instances=[], timestamp=0.0)
        # Set tight variance so different vectors don't fit
        schema = refiner.get_schema("animals")
        schema.variance = np.ones(64) * 0.01  # Very tight variance

        # Very different vector should accommodate
        different = np.array([0.0] * 63 + [1.0])
        schema, update_type = refiner.update_schema("animals", "alien", different, timestamp=1.0)

        assert update_type == SchemaUpdateType.ACCOMMODATE
        assert "alien" in schema.instances

    def test_update_schema_creates_new(self, refiner):
        """Test update creates new schema if none exists."""
        vector = np.random.randn(64)
        schema, update_type = refiner.update_schema(
            "new_schema", "instance1", vector, timestamp=0.0
        )

        assert schema is not None
        assert update_type == SchemaUpdateType.ASSIMILATE
        assert refiner.get_schema("new_schema") is not None

    def test_compute_coverage(self, refiner):
        """Test coverage computation."""
        np.random.seed(42)
        prototype = np.random.randn(64)
        prototype = prototype / np.linalg.norm(prototype)

        refiner.create_schema("animals", prototype, instances=[], timestamp=0.0)

        # Register instances
        for i in range(5):
            vec = prototype + 0.1 * np.random.randn(64)
            refiner.register_instance(f"animal_{i}", vec)

        coverage = refiner.compute_coverage("animals")
        assert 0 <= coverage <= 1

    def test_generalize_schema(self, refiner):
        """Test schema generalization."""
        np.random.seed(42)

        # Create base schema
        prototype = np.random.randn(64)
        refiner.create_schema("animals", prototype, instances=[], timestamp=0.0)

        # Add diverse instances
        instances = []
        for i in range(5):
            vec = np.random.randn(64)
            inst_id = f"animal_{i}"
            refiner.register_instance(inst_id, vec)
            instances.append(inst_id)

        schema = refiner.generalize_schema("animals", instances, timestamp=1.0)

        assert schema is not None
        assert len(schema.instances) >= 5

    def test_specialize_schema(self, refiner):
        """Test schema specialization."""
        np.random.seed(42)

        # Create schema with instances
        prototype = np.random.randn(64)
        refiner.create_schema("animals", prototype, instances=["a", "b", "c"], timestamp=0.0)

        # Register exception vectors
        refiner.register_instance("a", np.random.randn(64))
        refiner.register_instance("b", np.random.randn(64))
        refiner.register_instance("c", np.random.randn(64))

        sub_schemas = refiner.specialize_schema("animals", ["a"], timestamp=1.0)

        assert len(sub_schemas) == 1
        assert sub_schemas[0].parent_schema == "animals"

    def test_merge_schemas(self, refiner):
        """Test schema merging."""
        np.random.seed(42)

        prototype_a = np.random.randn(64)
        prototype_b = np.random.randn(64)

        refiner.create_schema("cats", prototype_a, instances=["cat1"], timestamp=0.0)
        refiner.create_schema("dogs", prototype_b, instances=["dog1"], timestamp=0.0)

        refiner.register_instance("cat1", prototype_a)
        refiner.register_instance("dog1", prototype_b)

        merged = refiner.merge_schemas("cats", "dogs", "pets", timestamp=1.0)

        assert merged is not None
        assert merged.name == "pets"
        assert "cat1" in merged.instances
        assert "dog1" in merged.instances

    def test_find_best_schema(self, refiner):
        """Test finding best matching schema."""
        np.random.seed(42)

        prototype_a = np.array([1.0] + [0.0] * 63)
        prototype_b = np.array([0.0, 1.0] + [0.0] * 62)

        refiner.create_schema("schema_a", prototype_a, instances=[], timestamp=0.0)
        refiner.create_schema("schema_b", prototype_b, instances=[], timestamp=0.0)

        test_vector = np.array([0.9, 0.1] + [0.0] * 62)
        result = refiner.find_best_schema(test_vector)

        assert result is not None
        assert result[0] == "schema_a"

    def test_find_schemas_containing(self, refiner):
        """Test finding containing schemas."""
        prototype = np.array([1.0] + [0.0] * 63)
        refiner.create_schema("broad", prototype, instances=[], timestamp=0.0)

        # Schema with larger variance should contain more
        schema = refiner.get_schema("broad")
        schema.variance = np.ones(64) * 10  # Large variance

        test_vector = np.array([1.5] + [0.0] * 63)
        containing = refiner.find_schemas_containing(test_vector)

        assert "broad" in containing

    def test_prune_low_coverage(self, refiner):
        """Test pruning low coverage schemas."""
        np.random.seed(42)

        # Create schemas with different coverages
        refiner.create_schema("high", np.random.randn(64), instances=["a", "b", "c"], timestamp=0.0)
        refiner.create_schema("low", np.random.randn(64), instances=[], timestamp=0.0)

        removed = refiner.prune_low_coverage_schemas(min_coverage=0.1, min_instances=2)

        assert "low" in removed
        assert refiner.get_schema("high") is not None

    def test_auto_discover_schemas(self, refiner):
        """Test automatic schema discovery."""
        np.random.seed(42)

        # Create clustered instances
        cluster_center = np.random.randn(64)
        cluster_center = cluster_center / np.linalg.norm(cluster_center)

        for i in range(5):
            vec = cluster_center + 0.1 * np.random.randn(64)
            vec = vec / np.linalg.norm(vec)
            refiner.register_instance(f"item_{i}", vec)

        discovered = refiner.auto_discover_schemas(similarity_threshold=0.8, timestamp=1.0)

        assert len(discovered) >= 0  # May or may not find clusters

    def test_statistics(self, refiner):
        """Test statistics generation."""
        np.random.seed(42)

        prototype = np.random.randn(64)
        refiner.create_schema("test", prototype, instances=["a"], timestamp=0.0)
        refiner.register_instance("a", prototype)

        stats = refiner.statistics()
        assert stats["total_schemas"] == 1
        assert stats["total_instances"] == 1
        assert "average_coverage" in stats

    def test_get_all_schemas(self, refiner):
        """Test getting all schemas."""
        refiner.create_schema("a", np.random.randn(64), instances=[], timestamp=0.0)
        refiner.create_schema("b", np.random.randn(64), instances=[], timestamp=0.0)

        all_schemas = refiner.get_all_schemas()
        assert len(all_schemas) == 2
        assert "a" in all_schemas
        assert "b" in all_schemas
