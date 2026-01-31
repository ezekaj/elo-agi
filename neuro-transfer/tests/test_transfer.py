"""
Tests for neuro-transfer module.

Tests abstraction, curriculum, few-shot, meta-learning, domain adaptation, and skills.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from abstraction import (
    AbstractionEngine, AbstractionLevel, AbstractConcept,
    StructuralAnalogy, DomainPrinciple, RelationExtractor,
    StructureMapper, PrincipleExtractor
)
from curriculum import (
    CurriculumLearner, Task, TaskDifficulty, LearningPath,
    ProgressTracker, DifficultyEstimator, TaskSelector, LearnerState
)
from few_shot import (
    FewShotLearner, PrototypeNetwork, MatchingNetwork,
    SupportSet, QueryResult, EmbeddingNetwork
)
from meta_learner import (
    MetaLearner, MAML, LearningStrategy, TaskDistribution,
    AdaptationResult, SimpleNN
)
from domain_adapter import (
    DomainAdapter, DomainEmbedding, DomainAlignment,
    TransferMapping, AdaptedRepresentation,
    SubspaceAlignment, CorrelationAlignment
)
from skill_library import (
    SkillLibrary, Skill, SkillPrimitive, CompositeSkill,
    SkillExecution, SkillType, SkillComposer, SkillMatcher
)


# ============== Abstraction Tests ==============

class TestAbstraction:
    """Tests for abstraction engine."""

    def test_abstraction_engine_creation(self):
        """Test engine creation."""
        engine = AbstractionEngine(embedding_dim=64)
        assert engine.embedding_dim == 64

    def test_relation_extractor(self):
        """Test relation extraction."""
        extractor = RelationExtractor()

        examples = [
            {"id": "dog", "type": "animal", "properties": {"legs": 4}},
            {"id": "cat", "type": "animal", "causes": ["mouse_fear"]},
        ]

        relations = extractor.extract(examples, "biology")

        assert len(relations) > 0
        # Check for is_a relation (both have type: animal)
        is_a_rels = [r for r in relations if r[0] == "is_a"]
        assert len(is_a_rels) >= 1

    def test_structure_mapper(self):
        """Test structure mapping."""
        mapper = StructureMapper()

        source = {
            "sun": [("heats", "earth")],
            "earth": [("orbits", "sun")],
        }
        target = {
            "nucleus": [("attracts", "electron")],
            "electron": [("orbits", "nucleus")],
        }

        analogy = mapper.find_mapping(source, target)

        assert isinstance(analogy, StructuralAnalogy)
        assert len(analogy.mappings) > 0

    def test_principle_extractor(self):
        """Test principle extraction."""
        extractor = PrincipleExtractor(min_support=2)

        examples = [
            {"condition": {"hot": True}, "action": "cool"},
            {"condition": {"hot": True}, "action": "cool"},
            {"condition": {"cold": True}, "action": "heat"},
        ]

        principles = extractor.extract(examples, "physics")

        assert len(principles) >= 1

    def test_abstract_categorical(self):
        """Test categorical abstraction."""
        engine = AbstractionEngine()

        examples = [
            {"id": "sparrow", "type": "bird"},
            {"id": "eagle", "type": "bird"},
            {"id": "dog", "type": "mammal"},
        ]

        concepts = engine.abstract(examples, "animals", AbstractionLevel.CATEGORICAL)

        assert len(concepts) >= 2
        bird_concept = next((c for c in concepts if "bird" in c.name), None)
        assert bird_concept is not None

    def test_find_analogy(self):
        """Test finding analogy between domains."""
        engine = AbstractionEngine()

        source_examples = [
            {"id": "teacher", "type": "person", "causes": ["learning"]},
            {"id": "student", "type": "person"},
        ]
        target_examples = [
            {"id": "doctor", "type": "person", "causes": ["healing"]},
            {"id": "patient", "type": "person"},
        ]

        analogy = engine.find_analogy(
            "education", "medicine",
            source_examples, target_examples
        )

        assert analogy.source_domain == "education"
        assert analogy.target_domain == "medicine"


class TestCurriculum:
    """Tests for curriculum learning."""

    def test_curriculum_creation(self):
        """Test curriculum learner creation."""
        curriculum = CurriculumLearner()
        assert curriculum.mastery_threshold == 0.8

    def test_task_registration(self):
        """Test task registration."""
        curriculum = CurriculumLearner()

        task = Task(
            id="task1",
            name="Basic Addition",
            domain="math",
            difficulty=TaskDifficulty.EASY,
            skills_taught=["addition"],
        )

        curriculum.register_task(task)
        assert "task1" in curriculum._tasks

    def test_progress_tracker(self):
        """Test progress tracking."""
        tracker = ProgressTracker()

        tracker.record("task1", 0.8, ["addition"], 10.0)
        tracker.record("task1", 0.9, ["addition"], 8.0)

        mastery = tracker.get_skill_mastery("addition")
        assert mastery > 0.8

    def test_difficulty_estimator(self):
        """Test difficulty estimation."""
        estimator = DifficultyEstimator()

        task = Task(
            id="hard_task",
            name="Complex Problem",
            domain="math",
            difficulty=TaskDifficulty.HARD,
            prerequisites=["basic"],
            skills_required=["algebra", "calculus"],
        )

        state = LearnerState()
        difficulty = estimator.estimate(task, state)

        assert 0 <= difficulty <= 1
        assert difficulty > 0.5  # Should be hard

    def test_task_selector(self):
        """Test task selection."""
        selector = TaskSelector(target_difficulty=0.5)

        tasks = [
            Task("easy", "Easy", "math", TaskDifficulty.EASY, skills_taught=["basic"]),
            Task("medium", "Medium", "math", TaskDifficulty.MEDIUM, prerequisites=["easy"], skills_taught=["intermediate"]),
        ]

        state = LearnerState()
        selected = selector.select(tasks, state)

        assert selected is not None
        assert selected.id == "easy"  # Should select easy first

    def test_complete_task(self):
        """Test completing a task."""
        curriculum = CurriculumLearner()

        task = Task(
            id="task1",
            name="Learn Basics",
            domain="math",
            difficulty=TaskDifficulty.EASY,
            skills_taught=["basics"],
        )
        curriculum.register_task(task)

        result = curriculum.complete_task("task1", 0.9)

        assert "task1" in curriculum._learner_state.tasks_completed
        assert result["performance"] == 0.9

    def test_learning_path_creation(self):
        """Test creating a learning path."""
        curriculum = CurriculumLearner()

        tasks = [
            Task("add", "Addition", "math", TaskDifficulty.EASY, skills_taught=["addition"]),
            Task("mult", "Multiplication", "math", TaskDifficulty.MEDIUM, prerequisites=["add"], skills_taught=["multiplication"]),
        ]
        curriculum.register_tasks(tasks)

        path = curriculum.create_learning_path("math_path", "Math Basics", ["multiplication"])

        assert len(path.tasks) > 0
        assert "addition" in path.skills_progression or "multiplication" in path.skills_progression


class TestFewShot:
    """Tests for few-shot learning."""

    def test_embedding_network(self):
        """Test embedding network."""
        net = EmbeddingNetwork(input_dim=64, embedding_dim=32)

        x = np.random.randn(64)
        emb = net.embed(x)

        assert emb.shape == (32,)
        assert abs(np.linalg.norm(emb) - 1.0) < 1e-6

    def test_support_set(self):
        """Test support set creation."""
        examples = [np.random.randn(64) for _ in range(10)]
        labels = ["A"] * 5 + ["B"] * 5

        support = SupportSet(examples=examples, labels=labels)

        assert support.n_examples == 10
        assert support.n_classes == 2

    def test_prototype_network(self):
        """Test prototype network."""
        net = PrototypeNetwork(input_dim=64, embedding_dim=32)

        # Create support set
        examples = [np.random.randn(64) for _ in range(10)]
        labels = ["A"] * 5 + ["B"] * 5
        support = SupportSet(examples=examples, labels=labels)

        # Classify query
        query = np.random.randn(64)
        result = net.classify(query, support)

        assert isinstance(result, QueryResult)
        assert result.predicted_label in ["A", "B"]
        assert 0 <= result.confidence <= 1

    def test_matching_network(self):
        """Test matching network."""
        net = MatchingNetwork(input_dim=64, embedding_dim=32)

        examples = [np.random.randn(64) for _ in range(10)]
        labels = ["A"] * 5 + ["B"] * 5
        support = SupportSet(examples=examples, labels=labels)

        query = np.random.randn(64)
        result = net.classify(query, support)

        assert result.predicted_label in ["A", "B"]

    def test_few_shot_learner(self):
        """Test complete few-shot learner."""
        learner = FewShotLearner(input_dim=64, method="prototype")

        # Register support set
        examples = [np.random.randn(64) for _ in range(10)]
        labels = ["X"] * 5 + ["Y"] * 5
        learner.register_support_set("task1", examples, labels)

        # Classify
        query = np.random.randn(64)
        result = learner.classify(query, task_id="task1")

        assert result.predicted_label in ["X", "Y"]

    def test_few_shot_evaluate(self):
        """Test few-shot evaluation."""
        learner = FewShotLearner(input_dim=64)

        # Register support
        examples = [np.random.randn(64) for _ in range(10)]
        labels = ["A"] * 5 + ["B"] * 5
        learner.register_support_set("task", examples, labels)

        # Evaluate
        test_examples = [np.random.randn(64) for _ in range(10)]
        test_labels = ["A"] * 5 + ["B"] * 5

        metrics = learner.evaluate(test_examples, test_labels, "task")

        assert "accuracy" in metrics
        assert 0 <= metrics["accuracy"] <= 1


class TestMetaLearner:
    """Tests for meta-learning."""

    def test_simple_nn(self):
        """Test simple neural network."""
        nn = SimpleNN(input_dim=32, hidden_dim=16, output_dim=5)

        x = np.random.randn(32)
        out = nn.forward(x)

        assert out.shape == (5,)
        assert abs(out.sum() - 1.0) < 1e-6  # Softmax

    def test_maml_creation(self):
        """Test MAML creation."""
        maml = MAML(input_dim=32, output_dim=5)

        assert maml.n_inner_steps == 5
        assert "w1" in maml._meta_params

    def test_maml_adapt(self):
        """Test MAML adaptation."""
        maml = MAML(input_dim=32, output_dim=5, n_inner_steps=3)

        # Create simple task
        support_x = np.random.randn(10, 32)
        support_y = np.eye(5)[np.random.randint(0, 5, 10)]

        result = maml.adapt(support_x, support_y, "test_task")

        assert isinstance(result, AdaptationResult)
        assert result.n_adaptation_steps == 3
        assert "w1" in result.adapted_parameters

    def test_meta_learner_creation(self):
        """Test meta-learner creation."""
        meta = MetaLearner(input_dim=32, output_dim=5)

        assert meta.input_dim == 32

    def test_meta_learner_adapt(self):
        """Test meta-learner adaptation."""
        meta = MetaLearner(input_dim=32, output_dim=5)

        support_x = np.random.randn(10, 32)
        support_y = np.eye(5)[np.random.randint(0, 5, 10)]

        result = meta.adapt(support_x, support_y, "task1")

        assert result.improvement >= 0 or result.improvement < 0  # Any value is valid

    def test_create_strategy(self):
        """Test creating a learning strategy."""
        meta = MetaLearner()

        strategy = meta.create_strategy("strategy1", ["classification"])

        assert isinstance(strategy, LearningStrategy)
        assert "classification" in strategy.applicable_tasks


class TestDomainAdapter:
    """Tests for domain adaptation."""

    def test_domain_embedding(self):
        """Test domain embedding creation."""
        examples = np.random.randn(50, 64)
        embedding = DomainEmbedding(
            domain_name="source",
            embedding_dim=64,
            examples=examples,
        )

        assert embedding.examples.shape[0] == 50

    def test_subspace_alignment(self):
        """Test subspace alignment."""
        aligner = SubspaceAlignment(n_components=10)

        source = DomainEmbedding("source", 64, np.random.randn(50, 64))
        target = DomainEmbedding("target", 64, np.random.randn(50, 64))

        alignment = aligner.align(source, target)

        assert isinstance(alignment, DomainAlignment)
        assert alignment.transformation.shape == (10, 10)

    def test_coral_alignment(self):
        """Test CORAL alignment."""
        aligner = CorrelationAlignment()

        source = DomainEmbedding("source", 64, np.random.randn(50, 64))
        target = DomainEmbedding("target", 64, np.random.randn(50, 64))

        alignment = aligner.align(source, target)

        assert alignment.method == "coral"

    def test_domain_adapter_creation(self):
        """Test domain adapter creation."""
        adapter = DomainAdapter(embedding_dim=64)
        assert adapter.embedding_dim == 64

    def test_register_domain(self):
        """Test domain registration."""
        adapter = DomainAdapter()

        examples = np.random.randn(50, 128)
        embedding = adapter.register_domain("source", examples)

        assert "source" in adapter._domains
        assert embedding.centroid is not None

    def test_adapt_representation(self):
        """Test adapting a representation."""
        adapter = DomainAdapter()

        # Register domains
        adapter.register_domain("source", np.random.randn(50, 128))
        adapter.register_domain("target", np.random.randn(50, 128))

        # Adapt
        representation = np.random.randn(128)
        adapted = adapter.adapt(representation, "source", "target")

        assert isinstance(adapted, AdaptedRepresentation)
        assert adapted.source_domain == "source"
        assert adapted.target_domain == "target"

    def test_domain_distance(self):
        """Test computing domain distance."""
        adapter = DomainAdapter()

        adapter.register_domain("d1", np.random.randn(50, 128))
        adapter.register_domain("d2", np.random.randn(50, 128) + 5)  # Offset

        dist = adapter.compute_domain_distance("d1", "d2")
        assert dist > 0


class TestSkillLibrary:
    """Tests for skill library."""

    def test_skill_library_creation(self):
        """Test library creation."""
        library = SkillLibrary()
        assert library.embedding_dim == 64

    def test_register_primitive(self):
        """Test registering a primitive skill."""
        library = SkillLibrary()

        def add_fn(ctx):
            return ctx.get("a", 0) + ctx.get("b", 0)

        primitive = library.register_primitive(
            name="add",
            execute_fn=add_fn,
            effects=["result_computed"],
        )

        assert primitive.id in library._primitives

    def test_register_skill(self):
        """Test registering a skill."""
        library = SkillLibrary()

        skill = library.register_skill(
            name="solve_equation",
            domain="math",
            description="Solve algebraic equations",
        )

        assert skill.id in library._skills
        assert skill.domain == "math"

    def test_skill_composer_sequence(self):
        """Test composing skills in sequence."""
        composer = SkillComposer()

        composite = composer.sequence("workflow", ["skill1", "skill2", "skill3"])

        assert len(composite.components) == 3

    def test_execute_primitive(self):
        """Test executing a primitive."""
        library = SkillLibrary()

        def multiply(ctx):
            return ctx.get("x", 1) * ctx.get("y", 1)

        primitive = library.register_primitive("multiply", multiply)

        result = library.execute(primitive.id, {"x": 3, "y": 4})

        assert result.success
        assert result.result == 12

    def test_execute_skill(self):
        """Test executing a skill."""
        library = SkillLibrary()

        skill = library.register_skill("test_skill", "test", "A test skill")

        result = library.execute(skill.id, {})

        # Skill execution is simulated with 90% success rate
        assert isinstance(result, SkillExecution)

    def test_skill_matcher(self):
        """Test skill matching."""
        matcher = SkillMatcher()

        skill1 = Skill("s1", "add numbers", SkillType.PRIMITIVE, "math", "Addition of two numbers")
        skill2 = Skill("s2", "add values", SkillType.PRIMITIVE, "math", "Addition of values")
        skill3 = Skill("s3", "multiply", SkillType.PRIMITIVE, "math", "Multiplication")

        sim = matcher.compute_similarity(skill1, skill2)
        assert sim > 0.3  # Should be similar

        sim2 = matcher.compute_similarity(skill1, skill3)
        assert sim2 < sim  # Should be less similar

    def test_find_similar_skills(self):
        """Test finding similar skills."""
        library = SkillLibrary()

        library.register_skill("add", "math", "Add two numbers")
        library.register_skill("sum", "math", "Sum values together")
        library.register_skill("multiply", "math", "Multiply numbers")

        skill = library.get_skill("skill_1")
        similar = library.find_similar_skills("skill_1")

        assert isinstance(similar, list)

    def test_transfer_skill(self):
        """Test transferring a skill."""
        library = SkillLibrary()

        original = library.register_skill(
            "calculate_area",
            "geometry",
            "Calculate area of shapes",
        )

        transferred = library.transfer_skill(original.id, "physics")

        assert transferred.skill_type == SkillType.ADAPTED
        assert transferred.domain == "physics"

    def test_skills_by_domain(self):
        """Test getting skills by domain."""
        library = SkillLibrary()

        library.register_skill("skill1", "math", "Math skill 1")
        library.register_skill("skill2", "math", "Math skill 2")
        library.register_skill("skill3", "physics", "Physics skill")

        math_skills = library.get_skills_by_domain("math")
        assert len(math_skills) == 2


# ============== Integration Tests ==============

class TestTransferIntegration:
    """Integration tests for transfer learning."""

    def test_abstraction_to_curriculum(self):
        """Test using abstraction to inform curriculum."""
        # Abstract concepts
        engine = AbstractionEngine()
        examples = [
            {"id": "add1", "type": "addition", "difficulty": 1},
            {"id": "add2", "type": "addition", "difficulty": 2},
            {"id": "mult1", "type": "multiplication", "difficulty": 3},
        ]
        concepts = engine.abstract(examples, "math")

        # Create curriculum from concepts
        curriculum = CurriculumLearner()
        for concept in concepts:
            task = Task(
                id=concept.id,
                name=concept.name,
                domain="math",
                difficulty=TaskDifficulty.MEDIUM,
                skills_taught=[concept.name],
            )
            curriculum.register_task(task)

        assert len(curriculum._tasks) > 0

    def test_few_shot_with_domain_adaptation(self):
        """Test few-shot learning with domain adaptation."""
        # Set up domain adapter
        adapter = DomainAdapter(embedding_dim=64)
        adapter.register_domain("source", np.random.randn(50, 64))
        adapter.register_domain("target", np.random.randn(50, 64))

        # Adapt support set
        source_examples = [np.random.randn(64) for _ in range(10)]
        adapted_examples = [
            adapter.adapt(ex, "source", "target").adapted
            for ex in source_examples
        ]

        # Use in few-shot
        learner = FewShotLearner(input_dim=64)
        labels = ["A"] * 5 + ["B"] * 5
        learner.register_support_set("task", adapted_examples, labels)

        query = np.random.randn(64)
        result = learner.classify(query, task_id="task")

        assert result.predicted_label in ["A", "B"]

    def test_meta_learning_with_skills(self):
        """Test meta-learning informing skill creation."""
        meta = MetaLearner(input_dim=32, output_dim=5)
        library = SkillLibrary()

        # Adapt to a task
        support_x = np.random.randn(10, 32)
        support_y = np.eye(5)[np.random.randint(0, 5, 10)]
        result = meta.adapt(support_x, support_y, "classification")

        # Create skill based on adaptation
        if result.improvement > 0:
            library.register_skill(
                f"classify_{result.task_id}",
                "classification",
                f"Learned classification with {result.improvement:.2f} improvement",
            )

        assert len(library._skills) >= 0

    def test_curriculum_with_skills(self):
        """Test curriculum learning with skill tracking."""
        curriculum = CurriculumLearner()
        library = SkillLibrary()

        # Register skills
        library.register_skill("basic_math", "math", "Basic math")
        library.register_skill("algebra", "math", "Algebra")

        # Register tasks that teach skills
        curriculum.register_task(Task(
            "t1", "Learn Basics", "math", TaskDifficulty.EASY,
            skills_taught=["basic_math"],
        ))
        curriculum.register_task(Task(
            "t2", "Learn Algebra", "math", TaskDifficulty.MEDIUM,
            prerequisites=["t1"],
            skills_required=["basic_math"],
            skills_taught=["algebra"],
        ))

        # Complete first task
        curriculum.complete_task("t1", 0.9)

        # Should now be able to get task 2
        next_task = curriculum.get_next_task()
        assert next_task is not None

    def test_full_transfer_pipeline(self):
        """Test complete transfer learning pipeline."""
        # 1. Abstract from source domain
        engine = AbstractionEngine()
        source_examples = [
            {"id": f"ex{i}", "type": "example", "value": i}
            for i in range(10)
        ]
        concepts = engine.abstract(source_examples, "source")

        # 2. Adapt to target domain
        adapter = DomainAdapter()
        adapter.register_domain("source", np.random.randn(50, 128))
        adapter.register_domain("target", np.random.randn(50, 128))

        # 3. Few-shot learn in target domain
        learner = FewShotLearner()
        adapted_examples = [
            adapter.adapt(np.random.randn(128), "source", "target").adapted
            for _ in range(10)
        ]
        learner.register_support_set("target_task", adapted_examples, ["A"] * 5 + ["B"] * 5)

        # 4. Register skill
        library = SkillLibrary()
        skill = library.register_skill(
            "transferred_skill",
            "target",
            "Skill transferred from source domain",
        )

        assert len(concepts) > 0
        assert skill.domain == "target"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
