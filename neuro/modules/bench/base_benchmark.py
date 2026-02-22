"""
Base Benchmark: Abstract interface for all benchmarks.

Defines the contract that all benchmarks must implement
for consistent evaluation of cognitive capabilities.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
import numpy as np
import time
import json
from pathlib import Path


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark."""

    name: str = "benchmark"
    description: str = ""
    n_trials: int = 100
    timeout_per_trial: float = 10.0
    random_seed: Optional[int] = None
    difficulty: str = "medium"  # "easy", "medium", "hard"
    save_results: bool = True
    output_dir: Path = field(default_factory=lambda: Path("results"))


@dataclass
class TrialResult:
    """Result of a single benchmark trial."""

    trial_id: int
    success: bool
    score: float  # 0.0 to 1.0
    latency: float  # seconds
    input_data: Any = None
    expected_output: Any = None
    actual_output: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Aggregate result of a benchmark run."""

    benchmark_name: str
    config: BenchmarkConfig
    trials: List[TrialResult]
    total_time: float
    timestamp: float = field(default_factory=time.time)

    @property
    def n_trials(self) -> int:
        return len(self.trials)

    @property
    def success_rate(self) -> float:
        if not self.trials:
            return 0.0
        return sum(1 for t in self.trials if t.success) / len(self.trials)

    @property
    def mean_score(self) -> float:
        if not self.trials:
            return 0.0
        return float(np.mean([t.score for t in self.trials]))

    @property
    def std_score(self) -> float:
        if not self.trials:
            return 0.0
        return float(np.std([t.score for t in self.trials]))

    @property
    def mean_latency(self) -> float:
        if not self.trials:
            return 0.0
        return float(np.mean([t.latency for t in self.trials]))

    @property
    def error_count(self) -> int:
        return sum(1 for t in self.trials if t.error is not None)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "benchmark_name": self.benchmark_name,
            "n_trials": self.n_trials,
            "success_rate": self.success_rate,
            "mean_score": self.mean_score,
            "std_score": self.std_score,
            "mean_latency": self.mean_latency,
            "error_count": self.error_count,
            "total_time": self.total_time,
            "timestamp": self.timestamp,
            "config": {
                "name": self.config.name,
                "difficulty": self.config.difficulty,
                "n_trials": self.config.n_trials,
            },
        }

    def summary(self) -> str:
        """Get human-readable summary."""
        return (
            f"Benchmark: {self.benchmark_name}\n"
            f"  Trials: {self.n_trials}\n"
            f"  Success Rate: {self.success_rate:.1%}\n"
            f"  Mean Score: {self.mean_score:.3f} (±{self.std_score:.3f})\n"
            f"  Mean Latency: {self.mean_latency:.3f}s\n"
            f"  Errors: {self.error_count}\n"
            f"  Total Time: {self.total_time:.1f}s"
        )


class Benchmark(ABC):
    """
    Abstract base class for all benchmarks.

    A benchmark:
    1. Generates test instances
    2. Evaluates agent responses
    3. Tracks performance metrics
    """

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or BenchmarkConfig()
        self._rng = np.random.default_rng(self.config.random_seed)

    @property
    @abstractmethod
    def name(self) -> str:
        """Benchmark name."""
        pass

    @property
    def description(self) -> str:
        """Benchmark description."""
        return self.config.description

    @abstractmethod
    def generate_trial(self, trial_id: int) -> Tuple[Any, Any]:
        """
        Generate a single trial.

        Args:
            trial_id: Trial identifier

        Returns:
            Tuple of (input_data, expected_output)
        """
        pass

    @abstractmethod
    def evaluate(self, expected: Any, actual: Any) -> Tuple[bool, float]:
        """
        Evaluate agent response.

        Args:
            expected: Expected output
            actual: Actual output from agent

        Returns:
            Tuple of (success, score)
        """
        pass

    def run(
        self,
        agent_fn: Callable[[Any], Any],
        n_trials: Optional[int] = None,
    ) -> BenchmarkResult:
        """
        Run the benchmark.

        Args:
            agent_fn: Function that takes input and returns output
            n_trials: Number of trials (uses config if None)

        Returns:
            BenchmarkResult with all trial results
        """
        n_trials = n_trials or self.config.n_trials
        trials = []
        start_time = time.time()

        for trial_id in range(n_trials):
            trial_result = self._run_trial(agent_fn, trial_id)
            trials.append(trial_result)

        total_time = time.time() - start_time

        result = BenchmarkResult(
            benchmark_name=self.name,
            config=self.config,
            trials=trials,
            total_time=total_time,
        )

        if self.config.save_results:
            self._save_result(result)

        return result

    def _run_trial(self, agent_fn: Callable[[Any], Any], trial_id: int) -> TrialResult:
        """Run a single trial."""
        # Generate trial
        input_data, expected_output = self.generate_trial(trial_id)

        # Run agent with timeout
        start_time = time.time()
        error = None
        actual_output = None

        try:
            actual_output = agent_fn(input_data)
            latency = time.time() - start_time

            # Check timeout
            if latency > self.config.timeout_per_trial:
                error = f"Timeout: {latency:.2f}s > {self.config.timeout_per_trial}s"
                success, score = False, 0.0
            else:
                success, score = self.evaluate(expected_output, actual_output)

        except Exception as e:
            latency = time.time() - start_time
            error = str(e)
            success, score = False, 0.0

        return TrialResult(
            trial_id=trial_id,
            success=success,
            score=score,
            latency=latency,
            input_data=input_data,
            expected_output=expected_output,
            actual_output=actual_output,
            error=error,
        )

    def _save_result(self, result: BenchmarkResult) -> None:
        """Save result to file."""
        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{self.name}_{int(result.timestamp)}.json"
        filepath = output_dir / filename

        with open(filepath, "w") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)


class BenchmarkSuite:
    """
    Collection of benchmarks for comprehensive evaluation.
    """

    def __init__(self, name: str = "neuro-suite"):
        self.name = name
        self._benchmarks: Dict[str, Benchmark] = {}
        self._results: Dict[str, BenchmarkResult] = {}

    def add(self, benchmark: Benchmark) -> None:
        """Add a benchmark to the suite."""
        self._benchmarks[benchmark.name] = benchmark

    def remove(self, name: str) -> bool:
        """Remove a benchmark by name."""
        if name in self._benchmarks:
            del self._benchmarks[name]
            return True
        return False

    def list_benchmarks(self) -> List[str]:
        """List all benchmark names."""
        return list(self._benchmarks.keys())

    def run_all(
        self,
        agent_fn: Callable[[Any], Any],
        n_trials: Optional[int] = None,
    ) -> Dict[str, BenchmarkResult]:
        """
        Run all benchmarks in the suite.

        Args:
            agent_fn: Agent function to evaluate
            n_trials: Trials per benchmark (uses individual configs if None)

        Returns:
            Dictionary of benchmark_name -> result
        """
        results = {}

        for name, benchmark in self._benchmarks.items():
            result = benchmark.run(agent_fn, n_trials)
            results[name] = result
            self._results[name] = result

        return results

    def run_one(
        self,
        name: str,
        agent_fn: Callable[[Any], Any],
        n_trials: Optional[int] = None,
    ) -> Optional[BenchmarkResult]:
        """Run a single benchmark by name."""
        if name not in self._benchmarks:
            return None

        result = self._benchmarks[name].run(agent_fn, n_trials)
        self._results[name] = result
        return result

    def get_result(self, name: str) -> Optional[BenchmarkResult]:
        """Get result for a benchmark."""
        return self._results.get(name)

    def get_all_results(self) -> Dict[str, BenchmarkResult]:
        """Get all results."""
        return self._results.copy()

    def summary(self) -> str:
        """Get summary of all results."""
        lines = [f"=== {self.name} Results ==="]

        for name, result in self._results.items():
            lines.append(f"\n{result.summary()}")

        # Overall statistics
        if self._results:
            all_scores = [r.mean_score for r in self._results.values()]
            all_success = [r.success_rate for r in self._results.values()]
            lines.append("\n=== Overall ===")
            lines.append(f"  Benchmarks: {len(self._results)}")
            lines.append(f"  Mean Score: {np.mean(all_scores):.3f}")
            lines.append(f"  Mean Success: {np.mean(all_success):.1%}")

        return "\n".join(lines)

    def save_all(self, output_dir: Optional[Path] = None) -> None:
        """Save all results to files."""
        output_dir = output_dir or Path("results")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save summary
        summary_path = output_dir / f"{self.name}_summary.txt"
        with open(summary_path, "w") as f:
            f.write(self.summary())

        # Save detailed results
        results_path = output_dir / f"{self.name}_results.json"
        results_dict = {name: result.to_dict() for name, result in self._results.items()}
        with open(results_path, "w") as f:
            json.dump(results_dict, f, indent=2, default=str)
