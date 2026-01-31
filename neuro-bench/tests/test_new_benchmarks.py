"""Tests for new AGI capability benchmarks."""

import pytest
import numpy as np

from src.causal_bench import (
    CausalGraph,
    CounterfactualBenchmark,
    InterventionBenchmark,
    CausalDiscoveryBenchmark,
    NestedCounterfactualBenchmark,
    create_causal_benchmark_suite,
)
from src.abstraction_bench import (
    SymbolBindingBenchmark,
    CompositionalBenchmark,
    ProgramSynthesisBenchmark,
    AnalogyBenchmark,
    AbstractionLevelBenchmark,
    create_abstraction_benchmark_suite,
)
from src.robustness_bench import (
    OODDetectionBenchmark,
    CalibrationBenchmark,
    AdversarialBenchmark,
    UncertaintyBenchmark,
    SelectivePredictionBenchmark,
    create_robustness_benchmark_suite,
)
from src.consolidation_bench import (
    RetentionBenchmark,
    InterferenceBenchmark,
    LearningEfficiencyBenchmark,
    SchemaFormationBenchmark,
    SpacedRepetitionBenchmark,
    create_consolidation_benchmark_suite,
)
from src.base_benchmark import BenchmarkConfig


class TestCausalBenchmarks:
    """Tests for causal reasoning benchmarks."""

    def test_causal_graph_parents_children(self):
        """Test causal graph parent/child queries."""
        graph = CausalGraph(
            variables=["A", "B", "C"],
            edges=[("A", "B"), ("B", "C")],
        )
        assert graph.parents("B") == ["A"]
        assert graph.parents("A") == []
        assert graph.children("A") == ["B"]
        assert graph.children("C") == []

    def test_counterfactual_generate_trial(self):
        """Test counterfactual trial generation."""
        config = BenchmarkConfig(random_seed=42)
        bench = CounterfactualBenchmark(config)

        query, expected = bench.generate_trial(0)

        assert query.query_type == "counterfactual"
        assert query.target_variable == "Z"
        assert "X" in query.observation
        assert "X" in query.intervention
        assert isinstance(expected, float)

    def test_counterfactual_evaluate_exact(self):
        """Test counterfactual evaluation with exact match."""
        bench = CounterfactualBenchmark()

        success, score = bench.evaluate(5.0, 5.0)
        assert success
        assert score == pytest.approx(1.0)

    def test_counterfactual_evaluate_close(self):
        """Test counterfactual evaluation with close value."""
        bench = CounterfactualBenchmark()

        success, score = bench.evaluate(5.0, 5.4)  # 8% error
        assert success
        assert score > 0.9

    def test_counterfactual_evaluate_far(self):
        """Test counterfactual evaluation with far value."""
        bench = CounterfactualBenchmark()

        success, score = bench.evaluate(5.0, 10.0)  # 100% error
        assert not success
        assert score < 0.5

    def test_intervention_benchmark(self):
        """Test intervention prediction benchmark."""
        config = BenchmarkConfig(random_seed=42)
        bench = InterventionBenchmark(config)

        query, expected = bench.generate_trial(0)

        assert query.query_type == "intervention"
        assert query.target_variable == "Y"
        assert isinstance(expected, float)

    def test_causal_discovery_benchmark(self):
        """Test causal discovery benchmark."""
        config = BenchmarkConfig(random_seed=42)
        bench = CausalDiscoveryBenchmark(config)

        trial_input, true_graph = bench.generate_trial(0)

        assert "data" in trial_input
        assert "variables" in trial_input
        assert isinstance(true_graph, CausalGraph)

    def test_causal_discovery_evaluate_perfect(self):
        """Test causal discovery with perfect prediction."""
        bench = CausalDiscoveryBenchmark()

        graph = CausalGraph(
            variables=["A", "B"],
            edges=[("A", "B")],
        )

        success, score = bench.evaluate(graph, graph)
        assert success
        assert score == pytest.approx(1.0)

    def test_causal_discovery_evaluate_partial(self):
        """Test causal discovery with partial prediction."""
        bench = CausalDiscoveryBenchmark()

        true_graph = CausalGraph(
            variables=["A", "B", "C"],
            edges=[("A", "B"), ("B", "C")],
        )
        # Predict only one edge
        predicted = [("A", "B")]

        success, score = bench.evaluate(true_graph, predicted)
        # F1 = 2 * (0.5 * 1.0) / (0.5 + 1.0) = 0.667
        assert 0.5 < score < 0.8

    def test_nested_counterfactual_benchmark(self):
        """Test nested counterfactual benchmark."""
        config = BenchmarkConfig(random_seed=42)
        bench = NestedCounterfactualBenchmark(config)

        trial_input, expected = bench.generate_trial(0)

        assert "model" in trial_input
        assert "observation" in trial_input
        assert "primary_intervention" in trial_input
        assert "secondary_intervention" in trial_input
        assert isinstance(expected, float)

    def test_create_causal_suite(self):
        """Test suite creation."""
        suite = create_causal_benchmark_suite()
        assert len(suite) == 4


class TestAbstractionBenchmarks:
    """Tests for abstraction benchmarks."""

    def test_symbol_binding_generate_trial(self):
        """Test symbol binding trial generation."""
        config = BenchmarkConfig(random_seed=42)
        bench = SymbolBindingBenchmark(config)

        task, expected = bench.generate_trial(0)

        assert len(task.symbols) == 4
        assert task.query_symbol in task.symbols
        assert expected.shape == (64,)

    def test_symbol_binding_evaluate_exact(self):
        """Test symbol binding with exact match."""
        bench = SymbolBindingBenchmark()

        vec = np.array([1.0, 0.0, 0.0])
        vec = vec / np.linalg.norm(vec)

        success, score = bench.evaluate(vec, vec)
        assert success
        assert score > 0.95

    def test_compositional_generate_trial(self):
        """Test compositional trial generation."""
        config = BenchmarkConfig(random_seed=42)
        bench = CompositionalBenchmark(config)

        task, expected = bench.generate_trial(0)

        assert "primitives" in task
        assert "test_composition" in task
        assert "input_value" in task
        assert isinstance(expected, (int, float, np.integer))

    def test_compositional_evaluate(self):
        """Test compositional evaluation."""
        bench = CompositionalBenchmark()

        success, score = bench.evaluate(42, 42)
        assert success
        assert score == 1.0

        success, score = bench.evaluate(42, 43)
        assert not success
        assert score == 0.0

    def test_program_synthesis_benchmark(self):
        """Test program synthesis benchmark."""
        config = BenchmarkConfig(random_seed=42)
        bench = ProgramSynthesisBenchmark(config)

        task, target_fn = bench.generate_trial(0)

        assert "examples" in task
        assert "test_inputs" in task
        assert callable(target_fn)

    def test_program_synthesis_evaluate(self):
        """Test program synthesis evaluation."""
        bench = ProgramSynthesisBenchmark()

        target_fn = lambda x: x + 1
        correct_fn = lambda x: x + 1
        wrong_fn = lambda x: x * 2

        success, score = bench.evaluate(target_fn, correct_fn)
        assert success
        assert score > 0.9

        success, score = bench.evaluate(target_fn, wrong_fn)
        assert not success

    def test_analogy_benchmark(self):
        """Test analogy completion benchmark."""
        config = BenchmarkConfig(random_seed=42)
        bench = AnalogyBenchmark(config)

        task, expected = bench.generate_trial(0)

        assert task.a is not None
        assert task.b is not None
        assert task.c is not None
        assert isinstance(expected, str)

    def test_analogy_evaluate(self):
        """Test analogy evaluation."""
        bench = AnalogyBenchmark()

        success, score = bench.evaluate("cold", "cold")
        assert success
        assert score == 1.0

        success, score = bench.evaluate("cold", "hot")
        assert not success
        assert score == 0.0

    def test_abstraction_level_benchmark(self):
        """Test abstraction level benchmark."""
        config = BenchmarkConfig(random_seed=42)
        bench = AbstractionLevelBenchmark(config)

        task, expected = bench.generate_trial(0)

        assert "item" in task
        assert "current_level" in task
        assert "target_level" in task
        assert isinstance(expected, str)

    def test_create_abstraction_suite(self):
        """Test suite creation."""
        suite = create_abstraction_benchmark_suite()
        assert len(suite) == 5


class TestRobustnessBenchmarks:
    """Tests for robustness benchmarks."""

    def test_ood_detection_generate_trial(self):
        """Test OOD detection trial generation."""
        config = BenchmarkConfig(random_seed=42)
        bench = OODDetectionBenchmark(config)

        sample, is_ood = bench.generate_trial(0)

        assert sample.features.shape == (32,)
        assert isinstance(is_ood, bool)
        assert sample.domain in ["in_distribution", "near_ood", "far_ood"]

    def test_ood_evaluate_bool(self):
        """Test OOD evaluation with boolean."""
        bench = OODDetectionBenchmark()

        success, score = bench.evaluate(True, True)
        assert success
        assert score == 1.0

        success, score = bench.evaluate(True, False)
        assert not success

    def test_ood_evaluate_score(self):
        """Test OOD evaluation with score."""
        bench = OODDetectionBenchmark()

        # High score for true OOD
        success, score = bench.evaluate(True, 0.9)
        assert success
        assert score == pytest.approx(0.9)

        # Low score for not OOD
        success, score = bench.evaluate(False, 0.1)
        assert success
        assert score == pytest.approx(0.9)

    def test_calibration_benchmark(self):
        """Test calibration benchmark."""
        config = BenchmarkConfig(random_seed=42)
        bench = CalibrationBenchmark(config)

        sample, true_class = bench.generate_trial(0)

        assert sample.features.shape == (32,)
        assert 0 <= true_class < 5

    def test_calibration_evaluate(self):
        """Test calibration evaluation."""
        bench = CalibrationBenchmark()

        # Correct with high confidence - good
        success, score = bench.evaluate(2, (2, 0.9))
        assert success
        assert score == pytest.approx(0.9)

        # Wrong with low confidence - partial credit
        success, score = bench.evaluate(2, (3, 0.1))
        assert not success
        assert score == pytest.approx(0.9)

    def test_adversarial_benchmark(self):
        """Test adversarial benchmark."""
        config = BenchmarkConfig(random_seed=42)
        bench = AdversarialBenchmark(config)

        sample, true_class = bench.generate_trial(0)

        assert sample.original.shape == (64,)
        assert sample.perturbation.shape == (64,)
        assert 0 <= true_class < 10

    def test_uncertainty_benchmark(self):
        """Test uncertainty benchmark."""
        config = BenchmarkConfig(random_seed=42)
        bench = UncertaintyBenchmark(config)

        task, expected_uncertainty = bench.generate_trial(0)

        assert "features" in task
        assert "difficulty" in task
        assert 0 <= expected_uncertainty <= 1

    def test_uncertainty_evaluate(self):
        """Test uncertainty evaluation."""
        bench = UncertaintyBenchmark()

        # Close to expected
        success, score = bench.evaluate(0.5, 0.55)
        assert success
        assert score > 0.8

        # Far from expected
        success, score = bench.evaluate(0.1, 0.9)
        assert not success

    def test_selective_prediction_benchmark(self):
        """Test selective prediction benchmark."""
        config = BenchmarkConfig(random_seed=42)
        bench = SelectivePredictionBenchmark(config)

        task, expected = bench.generate_trial(0)

        assert "features" in task
        assert "true_class" in expected
        assert "should_abstain" in expected

    def test_selective_evaluate_abstain(self):
        """Test selective prediction with abstention."""
        bench = SelectivePredictionBenchmark()

        expected = {"true_class": 2, "should_abstain": True}

        # Correctly abstained
        success, score = bench.evaluate(expected, "abstain")
        assert success
        assert score == 1.0

        # Should abstain but predicted
        success, score = bench.evaluate(expected, 2)
        assert success  # Correct prediction
        assert score == 0.8  # But risky

    def test_create_robustness_suite(self):
        """Test suite creation."""
        suite = create_robustness_benchmark_suite()
        assert len(suite) == 5


class TestConsolidationBenchmarks:
    """Tests for consolidation benchmarks."""

    def test_retention_benchmark(self):
        """Test retention benchmark."""
        config = BenchmarkConfig(random_seed=42)
        bench = RetentionBenchmark(config)

        task, expected = bench.generate_trial(0)

        assert "memory" in task
        assert "delay_days" in task
        assert 0 < expected <= 1

    def test_retention_evaluate(self):
        """Test retention evaluation."""
        bench = RetentionBenchmark()

        success, score = bench.evaluate(0.5, 0.52)
        assert success
        assert score > 0.9

        success, score = bench.evaluate(0.5, 0.1)
        assert not success

    def test_interference_benchmark(self):
        """Test interference benchmark."""
        config = BenchmarkConfig(random_seed=42)
        bench = InterferenceBenchmark(config)

        scenario, expected = bench.generate_trial(0)

        assert scenario.original_memory is not None
        assert scenario.interfering_memory is not None
        assert 0 <= scenario.similarity <= 1
        assert expected in ["none", "differentiate", "interleave"]

    def test_interference_evaluate(self):
        """Test interference evaluation."""
        bench = InterferenceBenchmark()

        success, score = bench.evaluate("differentiate", "differentiate")
        assert success
        assert score == 1.0

        # Alternative word
        success, score = bench.evaluate("differentiate", "separate")
        assert success
        assert score == 0.8

    def test_learning_efficiency_benchmark(self):
        """Test learning efficiency benchmark."""
        config = BenchmarkConfig(random_seed=42)
        bench = LearningEfficiencyBenchmark(config)

        scenario, expected = bench.generate_trial(0)

        assert len(scenario.items) >= 3
        assert scenario.target_strength == 0.8
        assert expected > 0

    def test_learning_efficiency_evaluate(self):
        """Test learning efficiency evaluation."""
        bench = LearningEfficiencyBenchmark()

        # At expected
        success, score = bench.evaluate(10, 10)
        assert success
        assert score == 1.0

        # Better than expected
        success, score = bench.evaluate(10, 8)
        assert success
        assert score == 1.0

        # Worse but acceptable
        success, score = bench.evaluate(10, 14)
        assert success
        assert score > 0.5

    def test_schema_formation_benchmark(self):
        """Test schema formation benchmark."""
        config = BenchmarkConfig(random_seed=42)
        bench = SchemaFormationBenchmark(config)

        task, prototype = bench.generate_trial(0)

        assert "instances" in task
        assert len(task["instances"]) >= 5
        assert prototype.shape == (64,)

    def test_schema_formation_evaluate(self):
        """Test schema formation evaluation."""
        bench = SchemaFormationBenchmark()

        np.random.seed(42)
        prototype = np.random.randn(64)
        prototype = prototype / np.linalg.norm(prototype)

        # Exact match
        success, score = bench.evaluate(prototype, prototype)
        assert success
        assert score > 0.95

        # Similar
        similar = prototype + 0.1 * np.random.randn(64)
        similar = similar / np.linalg.norm(similar)
        success, score = bench.evaluate(prototype, similar)
        assert score > 0.8

    def test_spaced_repetition_benchmark(self):
        """Test spaced repetition benchmark."""
        config = BenchmarkConfig(random_seed=42)
        bench = SpacedRepetitionBenchmark(config)

        task, expected = bench.generate_trial(0)

        assert "review_history" in task
        assert "current_interval" in task
        assert "easiness_factor" in task
        assert expected > 0

    def test_spaced_repetition_evaluate(self):
        """Test spaced repetition evaluation."""
        bench = SpacedRepetitionBenchmark()

        success, score = bench.evaluate(6.0, 6.0)
        assert success
        assert score == pytest.approx(1.0)

        success, score = bench.evaluate(6.0, 7.0)  # ~17% error
        assert success
        assert score > 0.8

    def test_create_consolidation_suite(self):
        """Test suite creation."""
        suite = create_consolidation_benchmark_suite()
        assert len(suite) == 5


class TestBenchmarkExecution:
    """Tests for running benchmarks."""

    def test_run_counterfactual_benchmark(self):
        """Test running counterfactual benchmark."""
        config = BenchmarkConfig(random_seed=42, n_trials=5, save_results=False)
        bench = CounterfactualBenchmark(config)

        # Simple agent that returns 0
        def agent(query):
            return 0.0

        result = bench.run(agent)

        assert result.n_trials == 5
        assert 0 <= result.mean_score <= 1

    def test_run_ood_benchmark(self):
        """Test running OOD benchmark."""
        config = BenchmarkConfig(random_seed=42, n_trials=10, save_results=False)
        bench = OODDetectionBenchmark(config)

        # Random agent
        def agent(sample):
            return np.random.random() > 0.5

        result = bench.run(agent)

        assert result.n_trials == 10
        # Random should get ~50%
        assert 0.2 < result.success_rate < 0.8

    def test_run_retention_benchmark(self):
        """Test running retention benchmark."""
        config = BenchmarkConfig(random_seed=42, n_trials=5, save_results=False)
        bench = RetentionBenchmark(config)

        # Agent that predicts 0.5 always
        def agent(task):
            return 0.5

        result = bench.run(agent)

        assert result.n_trials == 5
        assert 0 <= result.mean_score <= 1
