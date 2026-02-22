"""
Tests for the Neuro Benchmark Suite.

Covers:
- Base benchmark infrastructure
- Reasoning benchmarks
- Memory benchmarks
- Language benchmarks
- Planning benchmarks
- Benchmark runner
"""

import pytest
import numpy as np

from neuro.modules.bench.base_benchmark import (
    BenchmarkConfig,
    BenchmarkResult,
    TrialResult,
    BenchmarkSuite,
)
from neuro.modules.bench.reasoning_bench import PatternCompletion, AnalogySolving, LogicalInference
from neuro.modules.bench.memory_bench import (
    WorkingMemoryTest,
    EpisodicRecall,
    SequenceMemory,
    AssociativeMemory,
)
from neuro.modules.bench.language_bench import (
    TextCompletion,
    InstructionFollowing,
    SemanticSimilarity,
    QuestionAnswering,
)
from neuro.modules.bench.planning_bench import (
    GoalAchievement,
    MultiStepPlanning,
    ResourcePlanning,
    ConstraintSatisfaction,
)
from neuro.modules.bench.runner import (
    BenchmarkRunner,
    RunConfig,
    RunResult,
    create_random_agent,
    quick_benchmark,
)

# =============================================================================
# Helper Functions
# =============================================================================


def create_perfect_agent():
    """Create an agent that returns expected outputs."""

    def agent(input_data):
        # Return expected for testing
        return input_data.get("_expected", None)

    return agent


def create_simple_agent():
    """Create a simple rule-based agent."""

    def agent(input_data):
        if isinstance(input_data, dict):
            if "sequence" in input_data:
                return input_data["sequence"]
            elif "premises" in input_data:
                return True
            elif "text1" in input_data:
                return True
            elif "prompt" in input_data:
                return "unknown"
            elif "instruction" in input_data:
                return "result"
            elif "start" in input_data:
                return ["right", "right", "down", "down"]
            elif "task" in input_data:
                return ["step1", "step2", "step3"]
        return np.zeros(64)

    return agent


# =============================================================================
# Tests: Base Benchmark
# =============================================================================


class TestBenchmarkConfig:
    """Tests for BenchmarkConfig."""

    def test_default_config(self):
        config = BenchmarkConfig()
        assert config.n_trials == 100
        assert config.difficulty == "medium"

    def test_custom_config(self):
        config = BenchmarkConfig(name="test", n_trials=50, difficulty="hard")
        assert config.name == "test"
        assert config.n_trials == 50


class TestTrialResult:
    """Tests for TrialResult."""

    def test_creation(self):
        result = TrialResult(
            trial_id=0,
            success=True,
            score=0.9,
            latency=0.1,
        )
        assert result.success
        assert result.score == 0.9


class TestBenchmarkResult:
    """Tests for BenchmarkResult."""

    def test_properties(self):
        trials = [
            TrialResult(trial_id=0, success=True, score=1.0, latency=0.1),
            TrialResult(trial_id=1, success=True, score=0.8, latency=0.2),
            TrialResult(trial_id=2, success=False, score=0.5, latency=0.15),
        ]
        result = BenchmarkResult(
            benchmark_name="test",
            config=BenchmarkConfig(),
            trials=trials,
            total_time=1.0,
        )

        assert result.n_trials == 3
        assert result.success_rate == pytest.approx(2 / 3)
        assert result.mean_score == pytest.approx((1.0 + 0.8 + 0.5) / 3)

    def test_to_dict(self):
        result = BenchmarkResult(
            benchmark_name="test",
            config=BenchmarkConfig(),
            trials=[],
            total_time=1.0,
        )
        d = result.to_dict()
        assert "benchmark_name" in d
        assert "success_rate" in d

    def test_summary(self):
        result = BenchmarkResult(
            benchmark_name="test",
            config=BenchmarkConfig(),
            trials=[],
            total_time=1.0,
        )
        summary = result.summary()
        assert "test" in summary


class TestBenchmarkSuite:
    """Tests for BenchmarkSuite."""

    def test_creation(self):
        suite = BenchmarkSuite("test-suite")
        assert suite.name == "test-suite"

    def test_add_benchmark(self):
        suite = BenchmarkSuite("test-suite")
        suite.add(PatternCompletion())
        assert "pattern_completion" in suite.list_benchmarks()

    def test_remove_benchmark(self):
        suite = BenchmarkSuite("test-suite")
        suite.add(PatternCompletion())
        assert suite.remove("pattern_completion")
        assert "pattern_completion" not in suite.list_benchmarks()


# =============================================================================
# Tests: Reasoning Benchmarks
# =============================================================================


class TestPatternCompletion:
    """Tests for PatternCompletion benchmark."""

    def test_creation(self):
        bench = PatternCompletion()
        assert bench.name == "pattern_completion"

    def test_generate_trial(self):
        bench = PatternCompletion()
        trial_data, expected = bench.generate_trial(0)
        assert "examples" in trial_data
        assert "test_input" in trial_data
        assert isinstance(expected, np.ndarray)

    def test_evaluate_exact(self):
        bench = PatternCompletion()
        expected = np.array([[1, 0], [0, 1]])
        actual = np.array([[1, 0], [0, 1]])
        success, score = bench.evaluate(expected, actual)
        assert success
        assert score == 1.0

    def test_evaluate_partial(self):
        bench = PatternCompletion()
        expected = np.array([[1, 0], [0, 1]])
        actual = np.array([[1, 0], [0, 0]])  # One mismatch
        success, score = bench.evaluate(expected, actual)
        assert score == 0.75


class TestAnalogySolving:
    """Tests for AnalogySolving benchmark."""

    def test_creation(self):
        bench = AnalogySolving()
        assert bench.name == "analogy_solving"

    def test_generate_trial(self):
        bench = AnalogySolving()
        trial_data, expected = bench.generate_trial(0)
        assert "A" in trial_data
        assert "B" in trial_data
        assert "C" in trial_data


class TestLogicalInference:
    """Tests for LogicalInference benchmark."""

    def test_creation(self):
        bench = LogicalInference()
        assert bench.name == "logical_inference"

    def test_generate_trial(self):
        bench = LogicalInference()
        trial_data, expected = bench.generate_trial(0)
        assert "premises" in trial_data
        assert "conclusion" in trial_data
        assert isinstance(expected, bool)

    def test_evaluate(self):
        bench = LogicalInference()
        success, score = bench.evaluate(True, True)
        assert success
        assert score == 1.0

        success, score = bench.evaluate(True, False)
        assert not success
        assert score == 0.0


# =============================================================================
# Tests: Memory Benchmarks
# =============================================================================


class TestWorkingMemoryTest:
    """Tests for WorkingMemoryTest benchmark."""

    def test_creation(self):
        bench = WorkingMemoryTest()
        assert bench.name == "working_memory"

    def test_generate_trial(self):
        bench = WorkingMemoryTest()
        trial_data, expected = bench.generate_trial(0)
        assert "sequence" in trial_data
        assert "distractors" in trial_data
        assert isinstance(expected, list)

    def test_evaluate_exact(self):
        bench = WorkingMemoryTest()
        expected = [1, 2, 3, 4, 5]
        success, score = bench.evaluate(expected, [1, 2, 3, 4, 5])
        assert success
        assert score == 1.0

    def test_evaluate_partial(self):
        bench = WorkingMemoryTest()
        expected = [1, 2, 3, 4, 5]
        success, score = bench.evaluate(expected, [1, 2, 3, 0, 0])
        assert score == 0.6


class TestEpisodicRecall:
    """Tests for EpisodicRecall benchmark."""

    def test_creation(self):
        bench = EpisodicRecall()
        assert bench.name == "episodic_recall"

    def test_generate_trial(self):
        bench = EpisodicRecall()
        trial_data, expected = bench.generate_trial(0)
        assert "episodes" in trial_data
        assert "query" in trial_data
        assert isinstance(expected, dict)


class TestSequenceMemory:
    """Tests for SequenceMemory benchmark."""

    def test_creation(self):
        bench = SequenceMemory()
        assert bench.name == "sequence_memory"

    def test_generate_trial(self):
        bench = SequenceMemory()
        trial_data, expected = bench.generate_trial(0)
        assert "sequence" in trial_data
        assert isinstance(expected, list)


class TestAssociativeMemory:
    """Tests for AssociativeMemory benchmark."""

    def test_creation(self):
        bench = AssociativeMemory()
        assert bench.name == "associative_memory"

    def test_generate_trial(self):
        bench = AssociativeMemory()
        trial_data, expected = bench.generate_trial(0)
        assert "pairs" in trial_data
        assert "cue" in trial_data


# =============================================================================
# Tests: Language Benchmarks
# =============================================================================


class TestTextCompletion:
    """Tests for TextCompletion benchmark."""

    def test_creation(self):
        bench = TextCompletion()
        assert bench.name == "text_completion"

    def test_generate_trial(self):
        bench = TextCompletion()
        trial_data, expected = bench.generate_trial(0)
        assert "prompt" in trial_data
        assert isinstance(expected, str)

    def test_evaluate(self):
        bench = TextCompletion()
        success, score = bench.evaluate("Paris", "Paris")
        assert success
        assert score == 1.0


class TestInstructionFollowing:
    """Tests for InstructionFollowing benchmark."""

    def test_creation(self):
        bench = InstructionFollowing()
        assert bench.name == "instruction_following"

    def test_generate_trial(self):
        bench = InstructionFollowing()
        trial_data, expected = bench.generate_trial(0)
        assert "instruction" in trial_data


class TestSemanticSimilarity:
    """Tests for SemanticSimilarity benchmark."""

    def test_creation(self):
        bench = SemanticSimilarity()
        assert bench.name == "semantic_similarity"

    def test_generate_trial(self):
        bench = SemanticSimilarity()
        trial_data, expected = bench.generate_trial(0)
        assert "text1" in trial_data
        assert "text2" in trial_data
        assert isinstance(expected, bool)


class TestQuestionAnswering:
    """Tests for QuestionAnswering benchmark."""

    def test_creation(self):
        bench = QuestionAnswering()
        assert bench.name == "question_answering"

    def test_generate_trial(self):
        bench = QuestionAnswering()
        trial_data, expected = bench.generate_trial(0)
        assert "context" in trial_data
        assert "question" in trial_data


# =============================================================================
# Tests: Planning Benchmarks
# =============================================================================


class TestGoalAchievement:
    """Tests for GoalAchievement benchmark."""

    def test_creation(self):
        bench = GoalAchievement()
        assert bench.name == "goal_achievement"

    def test_generate_trial(self):
        bench = GoalAchievement()
        trial_data, expected = bench.generate_trial(0)
        assert "grid_size" in trial_data
        assert "start" in trial_data
        assert "goal" in trial_data

    def test_evaluate_valid_path(self):
        bench = GoalAchievement()
        expected = ["right", "right", "down", "down"]
        success, score = bench.evaluate(expected, ["right", "right", "down", "down"])
        assert success


class TestMultiStepPlanning:
    """Tests for MultiStepPlanning benchmark."""

    def test_creation(self):
        bench = MultiStepPlanning()
        assert bench.name == "multi_step_planning"

    def test_generate_trial(self):
        bench = MultiStepPlanning()
        trial_data, expected = bench.generate_trial(0)
        assert "task" in trial_data
        assert isinstance(expected, list)


class TestResourcePlanning:
    """Tests for ResourcePlanning benchmark."""

    def test_creation(self):
        bench = ResourcePlanning()
        assert bench.name == "resource_planning"

    def test_generate_trial(self):
        bench = ResourcePlanning()
        trial_data, expected = bench.generate_trial(0)
        assert "resources" in trial_data
        assert "recipes" in trial_data


class TestConstraintSatisfaction:
    """Tests for ConstraintSatisfaction benchmark."""

    def test_creation(self):
        bench = ConstraintSatisfaction()
        assert bench.name == "constraint_satisfaction"

    def test_generate_trial(self):
        bench = ConstraintSatisfaction()
        trial_data, expected = bench.generate_trial(0)
        assert "tasks" in trial_data
        assert "constraints" in trial_data


# =============================================================================
# Tests: Runner
# =============================================================================


class TestRunConfig:
    """Tests for RunConfig."""

    def test_defaults(self):
        config = RunConfig()
        assert config.verbose is True
        assert config.save_results is True


class TestBenchmarkRunner:
    """Tests for BenchmarkRunner."""

    def test_creation(self):
        runner = BenchmarkRunner()
        assert runner.config is not None

    def test_create_default_suite(self):
        runner = BenchmarkRunner()
        suite = runner.create_default_suite()
        benchmarks = suite.list_benchmarks()
        assert len(benchmarks) > 0

    def test_run_single(self):
        runner = BenchmarkRunner(RunConfig(verbose=False, save_results=False))
        bench = PatternCompletion()
        agent = create_random_agent()
        result = runner.run_single(agent, bench, n_trials=3)
        assert result.n_trials == 3

    def test_generate_report(self):
        runner = BenchmarkRunner(RunConfig(verbose=False, save_results=False))
        runner.create_default_suite()
        agent = create_random_agent()
        run_result = runner.run(agent, n_trials=2)
        report = runner.generate_report(run_result)
        assert "NEURO BENCHMARK REPORT" in report


class TestRunResult:
    """Tests for RunResult."""

    def test_properties(self):
        results = {
            "test1": BenchmarkResult(
                "test1", BenchmarkConfig(), [TrialResult(0, True, 0.8, 0.1)], 1.0
            ),
            "test2": BenchmarkResult(
                "test2", BenchmarkConfig(), [TrialResult(0, True, 0.6, 0.1)], 1.0
            ),
        }
        run_result = RunResult(
            suite_name="test",
            results=results,
            total_time=2.0,
        )
        assert run_result.n_benchmarks == 2
        assert run_result.overall_score == pytest.approx(0.7)

    def test_to_dict(self):
        run_result = RunResult(
            suite_name="test",
            results={},
            total_time=1.0,
        )
        d = run_result.to_dict()
        assert "suite_name" in d
        assert "overall_score" in d


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_create_random_agent(self):
        agent = create_random_agent()
        result = agent({"sequence": [1, 2, 3]})
        assert result is not None

    def test_quick_benchmark(self):
        agent = create_random_agent()
        # Just test it runs without error
        result = quick_benchmark(agent, n_trials=2)
        assert result.n_benchmarks > 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for the benchmark suite."""

    def test_full_suite_run(self):
        """Test running a complete suite."""
        runner = BenchmarkRunner(RunConfig(verbose=False, save_results=False))
        runner.create_default_suite()

        agent = create_random_agent()
        result = runner.run(agent, n_trials=2)

        assert result.n_benchmarks > 0
        assert result.total_time > 0

    def test_benchmark_reproducibility(self):
        """Test that benchmarks are reproducible with same seed."""
        config = BenchmarkConfig(random_seed=42, n_trials=5)
        bench1 = PatternCompletion(config)
        bench2 = PatternCompletion(config)

        data1, expected1 = bench1.generate_trial(0)
        bench2._rng = np.random.default_rng(42)  # Reset
        data2, expected2 = bench2.generate_trial(0)

        # Should generate same trial
        assert np.array_equal(expected1, expected2)

    def test_all_benchmarks_run(self):
        """Test that all benchmark types can generate trials."""
        benchmarks = [
            PatternCompletion(),
            AnalogySolving(),
            LogicalInference(),
            WorkingMemoryTest(),
            EpisodicRecall(),
            SequenceMemory(),
            AssociativeMemory(),
            TextCompletion(),
            InstructionFollowing(),
            SemanticSimilarity(),
            QuestionAnswering(),
            GoalAchievement(),
            MultiStepPlanning(),
            ResourcePlanning(),
            ConstraintSatisfaction(),
        ]

        for bench in benchmarks:
            trial_data, expected = bench.generate_trial(0)
            assert trial_data is not None
            assert expected is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
