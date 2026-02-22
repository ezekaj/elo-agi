"""
Benchmark Runner: Executes and reports on benchmark suites.

Provides:
- Unified benchmark execution
- Progress tracking
- Result aggregation
- Report generation
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
import numpy as np
import time
import json
from pathlib import Path

try:
    from .base_benchmark import Benchmark, BenchmarkResult, BenchmarkSuite
except ImportError:
    from base_benchmark import Benchmark, BenchmarkResult, BenchmarkSuite


@dataclass
class RunConfig:
    """Configuration for benchmark runner."""

    output_dir: Path = field(default_factory=lambda: Path("results"))
    n_trials: Optional[int] = None  # Override per-benchmark settings
    verbose: bool = True
    save_results: bool = True
    parallel: bool = False  # Future: parallel execution


@dataclass
class RunResult:
    """Result of a complete benchmark run."""

    suite_name: str
    results: Dict[str, BenchmarkResult]
    total_time: float
    timestamp: float = field(default_factory=time.time)
    config: RunConfig = field(default_factory=RunConfig)

    @property
    def n_benchmarks(self) -> int:
        return len(self.results)

    @property
    def overall_score(self) -> float:
        if not self.results:
            return 0.0
        scores = [r.mean_score for r in self.results.values()]
        return float(np.mean(scores))

    @property
    def overall_success_rate(self) -> float:
        if not self.results:
            return 0.0
        rates = [r.success_rate for r in self.results.values()]
        return float(np.mean(rates))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "suite_name": self.suite_name,
            "n_benchmarks": self.n_benchmarks,
            "overall_score": self.overall_score,
            "overall_success_rate": self.overall_success_rate,
            "total_time": self.total_time,
            "timestamp": self.timestamp,
            "benchmarks": {name: result.to_dict() for name, result in self.results.items()},
        }


class BenchmarkRunner:
    """
    Runs benchmark suites and aggregates results.
    """

    def __init__(self, config: Optional[RunConfig] = None):
        self.config = config or RunConfig()
        self._suites: Dict[str, BenchmarkSuite] = {}
        self._run_history: List[RunResult] = []

    def add_suite(self, suite: BenchmarkSuite) -> None:
        """Add a benchmark suite."""
        self._suites[suite.name] = suite

    def create_default_suite(self) -> BenchmarkSuite:
        """Create a default comprehensive suite."""
        try:
            from .reasoning_bench import PatternCompletion, AnalogySolving, LogicalInference
            from .memory_bench import WorkingMemoryTest, EpisodicRecall, SequenceMemory
            from .language_bench import TextCompletion, InstructionFollowing, QuestionAnswering
            from .planning_bench import GoalAchievement, MultiStepPlanning
        except ImportError:
            from reasoning_bench import PatternCompletion, AnalogySolving, LogicalInference
            from memory_bench import WorkingMemoryTest, EpisodicRecall, SequenceMemory
            from language_bench import TextCompletion, InstructionFollowing, QuestionAnswering
            from planning_bench import GoalAchievement, MultiStepPlanning

        suite = BenchmarkSuite("neuro-comprehensive")

        # Add reasoning benchmarks
        suite.add(PatternCompletion())
        suite.add(AnalogySolving())
        suite.add(LogicalInference())

        # Add memory benchmarks
        suite.add(WorkingMemoryTest())
        suite.add(EpisodicRecall())
        suite.add(SequenceMemory())

        # Add language benchmarks
        suite.add(TextCompletion())
        suite.add(InstructionFollowing())
        suite.add(QuestionAnswering())

        # Add planning benchmarks
        suite.add(GoalAchievement())
        suite.add(MultiStepPlanning())

        self._suites[suite.name] = suite
        return suite

    def run(
        self,
        agent_fn: Callable[[Any], Any],
        suite_name: Optional[str] = None,
        n_trials: Optional[int] = None,
    ) -> RunResult:
        """
        Run benchmarks on an agent.

        Args:
            agent_fn: Function that takes input and returns output
            suite_name: Specific suite to run (runs all if None)
            n_trials: Override number of trials (uses config value if None)

        Returns:
            RunResult with all benchmark results
        """
        # Override config n_trials if provided
        if n_trials is not None:
            self.config.n_trials = n_trials
        if not self._suites:
            self.create_default_suite()

        if suite_name:
            suites_to_run = {suite_name: self._suites[suite_name]}
        else:
            suites_to_run = self._suites

        all_results = {}
        start_time = time.time()

        for name, suite in suites_to_run.items():
            if self.config.verbose:
                print(f"\n=== Running Suite: {name} ===")

            results = suite.run_all(agent_fn, self.config.n_trials)

            for bench_name, result in results.items():
                all_results[f"{name}/{bench_name}"] = result

                if self.config.verbose:
                    print(
                        f"  {bench_name}: score={result.mean_score:.3f}, "
                        f"success={result.success_rate:.1%}"
                    )

        total_time = time.time() - start_time

        run_result = RunResult(
            suite_name=suite_name or "all",
            results=all_results,
            total_time=total_time,
            config=self.config,
        )

        self._run_history.append(run_result)

        if self.config.save_results:
            self._save_run(run_result)

        if self.config.verbose:
            print("\n=== Overall ===")
            print(f"  Score: {run_result.overall_score:.3f}")
            print(f"  Success Rate: {run_result.overall_success_rate:.1%}")
            print(f"  Time: {total_time:.1f}s")

        return run_result

    def run_single(
        self,
        agent_fn: Callable[[Any], Any],
        benchmark: Benchmark,
        n_trials: Optional[int] = None,
    ) -> BenchmarkResult:
        """Run a single benchmark."""
        n_trials = n_trials or self.config.n_trials
        result = benchmark.run(agent_fn, n_trials)

        if self.config.verbose:
            print(result.summary())

        return result

    def _save_run(self, run_result: RunResult) -> None:
        """Save run result to file."""
        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"run_{int(run_result.timestamp)}.json"
        filepath = output_dir / filename

        with open(filepath, "w") as f:
            json.dump(run_result.to_dict(), f, indent=2, default=str)

    def get_history(self) -> List[RunResult]:
        """Get run history."""
        return self._run_history.copy()

    def compare_runs(self, run1_idx: int, run2_idx: int) -> Dict[str, Any]:
        """Compare two runs."""
        if run1_idx >= len(self._run_history) or run2_idx >= len(self._run_history):
            return {}

        run1 = self._run_history[run1_idx]
        run2 = self._run_history[run2_idx]

        comparison = {
            "run1_score": run1.overall_score,
            "run2_score": run2.overall_score,
            "score_diff": run2.overall_score - run1.overall_score,
            "run1_success": run1.overall_success_rate,
            "run2_success": run2.overall_success_rate,
            "success_diff": run2.overall_success_rate - run1.overall_success_rate,
            "per_benchmark": {},
        }

        # Per-benchmark comparison
        all_benchmarks = set(run1.results.keys()) | set(run2.results.keys())
        for bench in all_benchmarks:
            score1 = run1.results.get(bench, BenchmarkResult(bench, None, [], 0)).mean_score
            score2 = run2.results.get(bench, BenchmarkResult(bench, None, [], 0)).mean_score
            comparison["per_benchmark"][bench] = {
                "run1": score1,
                "run2": score2,
                "diff": score2 - score1,
            }

        return comparison

    def generate_report(self, run_result: Optional[RunResult] = None) -> str:
        """Generate a detailed report."""
        if run_result is None:
            if not self._run_history:
                return "No runs to report."
            run_result = self._run_history[-1]

        lines = [
            "=" * 60,
            "NEURO BENCHMARK REPORT",
            "=" * 60,
            "",
            f"Suite: {run_result.suite_name}",
            f"Timestamp: {time.ctime(run_result.timestamp)}",
            f"Total Time: {run_result.total_time:.1f}s",
            "",
            "-" * 60,
            "OVERALL RESULTS",
            "-" * 60,
            f"  Benchmarks Run: {run_result.n_benchmarks}",
            f"  Overall Score: {run_result.overall_score:.3f}",
            f"  Success Rate: {run_result.overall_success_rate:.1%}",
            "",
            "-" * 60,
            "PER-BENCHMARK RESULTS",
            "-" * 60,
        ]

        # Group by category
        categories = {}
        for name, result in run_result.results.items():
            parts = name.split("/")
            category = parts[0] if len(parts) > 1 else "default"
            bench_name = parts[-1]

            if category not in categories:
                categories[category] = []
            categories[category].append((bench_name, result))

        for category, benchmarks in categories.items():
            lines.append(f"\n  {category.upper()}")
            for bench_name, result in benchmarks:
                lines.append(
                    f"    {bench_name:25s} "
                    f"score={result.mean_score:.3f} "
                    f"success={result.success_rate:.1%} "
                    f"latency={result.mean_latency:.3f}s"
                )

        lines.extend(
            [
                "",
                "=" * 60,
                "END OF REPORT",
                "=" * 60,
            ]
        )

        return "\n".join(lines)

    def generate_html_report(
        self,
        run_result: Optional[RunResult] = None,
        output_file: Optional[Path] = None,
    ) -> str:
        """
        Generate an HTML report for benchmark results.

        Args:
            run_result: Result to report (uses latest if None)
            output_file: Path to save HTML file (optional)

        Returns:
            HTML content as string
        """
        if run_result is None:
            if not self._run_history:
                return "<html><body>No runs to report.</body></html>"
            run_result = self._run_history[-1]

        # Group benchmarks by category
        categories = {}
        for name, result in run_result.results.items():
            parts = name.split("/")
            category = parts[0] if len(parts) > 1 else "default"
            bench_name = parts[-1]

            if category not in categories:
                categories[category] = []
            categories[category].append((bench_name, result))

        # Build HTML
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Neuro AGI Benchmark Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
        }}
        .summary {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .metric {{
            display: inline-block;
            margin: 10px 20px 10px 0;
            padding: 15px 25px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 8px;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            display: block;
        }}
        .metric-label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        th, td {{
            padding: 12px 15px;
            text-align: left;
        }}
        th {{
            background: #3498db;
            color: white;
        }}
        tr:nth-child(even) {{
            background: #f8f9fa;
        }}
        tr:hover {{
            background: #e8f4fd;
        }}
        .score-bar {{
            background: #e0e0e0;
            border-radius: 4px;
            height: 20px;
            position: relative;
            overflow: hidden;
        }}
        .score-fill {{
            height: 100%;
            background: linear-gradient(90deg, #4CAF50 0%, #8BC34A 100%);
            border-radius: 4px;
        }}
        .category {{
            margin-bottom: 30px;
        }}
        .category-title {{
            background: #2c3e50;
            color: white;
            padding: 10px 15px;
            border-radius: 8px 8px 0 0;
            margin-bottom: 0;
        }}
        .timestamp {{
            color: #7f8c8d;
            font-size: 0.9em;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #7f8c8d;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <h1>🧠 Neuro AGI Benchmark Report</h1>

    <div class="summary">
        <p class="timestamp">Generated: {time.ctime(run_result.timestamp)}</p>
        <p class="timestamp">Suite: {run_result.suite_name} | Total Time: {run_result.total_time:.1f}s</p>

        <div class="metric">
            <span class="metric-value">{run_result.overall_score:.3f}</span>
            <span class="metric-label">Overall Score</span>
        </div>
        <div class="metric">
            <span class="metric-value">{run_result.overall_success_rate:.1%}</span>
            <span class="metric-label">Success Rate</span>
        </div>
        <div class="metric">
            <span class="metric-value">{run_result.n_benchmarks}</span>
            <span class="metric-label">Benchmarks</span>
        </div>
    </div>
"""

        # Add benchmark tables by category
        for category, benchmarks in sorted(categories.items()):
            html += f"""
    <div class="category">
        <h3 class="category-title">{category.replace("-", " ").title()}</h3>
        <table>
            <thead>
                <tr>
                    <th>Benchmark</th>
                    <th>Score</th>
                    <th>Success Rate</th>
                    <th>Latency</th>
                    <th>Progress</th>
                </tr>
            </thead>
            <tbody>
"""
            for bench_name, result in benchmarks:
                score_pct = min(100, result.mean_score * 100)
                html += f"""
                <tr>
                    <td>{bench_name}</td>
                    <td><strong>{result.mean_score:.3f}</strong></td>
                    <td>{result.success_rate:.1%}</td>
                    <td>{result.mean_latency:.3f}s</td>
                    <td>
                        <div class="score-bar">
                            <div class="score-fill" style="width: {score_pct}%"></div>
                        </div>
                    </td>
                </tr>
"""
            html += """
            </tbody>
        </table>
    </div>
"""

        html += """
    <div class="footer">
        <p>Generated by <strong>Neuro AGI Benchmark Suite v0.9.0</strong></p>
        <p>38 cognitive modules | Neuroscience-inspired architecture</p>
    </div>
</body>
</html>
"""

        # Save to file if requested
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(html)

        return html


def create_random_agent() -> Callable[[Any], Any]:
    """Create a random agent for baseline testing."""
    rng = np.random.default_rng()

    def random_agent(input_data: Any) -> Any:
        """Agent that returns random outputs."""
        if isinstance(input_data, dict):
            # Try to determine expected output type
            if "sequence" in input_data:
                return rng.integers(0, 10, size=7).tolist()
            elif "grid_size" in input_data:
                return ["right"] * 5
            elif "premises" in input_data:
                return rng.choice([True, False])
            else:
                return rng.random(64)
        else:
            return rng.random(64)

    return random_agent


def quick_benchmark(agent_fn: Callable[[Any], Any], n_trials: int = 10) -> RunResult:
    """Quickly benchmark an agent with reduced trials."""
    runner = BenchmarkRunner(
        RunConfig(
            n_trials=n_trials,
            verbose=True,
            save_results=False,
        )
    )
    runner.create_default_suite()
    return runner.run(agent_fn)
