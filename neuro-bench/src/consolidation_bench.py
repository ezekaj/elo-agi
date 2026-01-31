"""
Consolidation Benchmark: Tests for memory retention and learning efficiency.

Evaluates:
- Memory retention over time (forgetting curves)
- Interference resolution success
- Learning efficiency (replays needed)
- Schema formation quality
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

from .base_benchmark import Benchmark, BenchmarkConfig


@dataclass
class MemoryItem:
    """A memory item for consolidation testing."""
    memory_id: str
    content: np.ndarray
    initial_strength: float
    timestamp: float
    category: str


@dataclass
class InterferenceScenario:
    """An interference resolution scenario."""
    original_memory: MemoryItem
    interfering_memory: MemoryItem
    similarity: float  # 0-1, higher = more similar
    expected_resolution: str  # "differentiate", "merge", "interleave"


@dataclass
class LearningScenario:
    """A learning efficiency scenario."""
    items: List[MemoryItem]
    target_strength: float
    max_replays: int


class RetentionBenchmark(Benchmark):
    """
    Benchmark for memory retention over time.

    Tests agent's ability to maintain memory strength
    with realistic forgetting curves.
    """

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        config = config or BenchmarkConfig(
            name="retention",
            description="Memory retention over time benchmark",
            n_trials=50,
        )
        super().__init__(config)
        self.embedding_dim = 64

    @property
    def name(self) -> str:
        return "retention"

    def generate_trial(self, trial_id: int) -> Tuple[Dict, float]:
        """Generate a retention trial."""
        # Create memory with known initial strength
        initial_strength = self._rng.uniform(0.3, 0.9)

        # Generate time delay (in "days")
        delay = self._rng.choice([1, 3, 7, 14, 30])

        # Expected retention based on Ebbinghaus forgetting curve
        # R = e^(-t/S) where S is stability
        stability = 5.0 + 10.0 * initial_strength  # Stronger = more stable
        expected_retention = np.exp(-delay / stability)
        expected_retention = max(0.1, expected_retention)  # Floor at 10%

        content = self._rng.standard_normal(self.embedding_dim)
        content = content / np.linalg.norm(content)

        task = {
            "memory": MemoryItem(
                memory_id=f"mem_{trial_id}",
                content=content,
                initial_strength=initial_strength,
                timestamp=0.0,
                category="episodic",
            ),
            "delay_days": delay,
            "query_content": content + self._rng.normal(0, 0.1, self.embedding_dim),
        }

        return task, expected_retention

    def evaluate(self, expected: float, actual: Any) -> Tuple[bool, float]:
        """Evaluate retention prediction."""
        if actual is None:
            return False, 0.0

        try:
            retention = float(actual)
        except (ValueError, TypeError):
            return False, 0.0

        # Should be in [0, 1]
        retention = np.clip(retention, 0, 1)

        # Score based on relative error
        if expected < 0.1:
            error = abs(retention - expected)
        else:
            error = abs(retention - expected) / expected

        score = max(0.0, 1.0 - error)
        success = error < 0.15  # Within 15% of expected

        return success, score


class InterferenceBenchmark(Benchmark):
    """
    Benchmark for interference resolution.

    Tests agent's ability to handle similar/conflicting memories.
    """

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        config = config or BenchmarkConfig(
            name="interference",
            description="Interference resolution benchmark",
            n_trials=50,
        )
        super().__init__(config)
        self.embedding_dim = 64

    @property
    def name(self) -> str:
        return "interference"

    def generate_trial(self, trial_id: int) -> Tuple[InterferenceScenario, str]:
        """Generate an interference trial."""
        # Create original memory
        original_content = self._rng.standard_normal(self.embedding_dim)
        original_content = original_content / np.linalg.norm(original_content)

        original = MemoryItem(
            memory_id=f"original_{trial_id}",
            content=original_content,
            initial_strength=0.7,
            timestamp=0.0,
            category="episodic",
        )

        # Create interfering memory with controlled similarity
        similarity_level = ["low", "medium", "high"][trial_id % 3]

        if similarity_level == "low":
            # Independent memory
            similarity = self._rng.uniform(0.0, 0.3)
            interf_content = self._rng.standard_normal(self.embedding_dim)
            expected_resolution = "none"  # No interference
        elif similarity_level == "medium":
            # Similar but distinct
            similarity = self._rng.uniform(0.5, 0.7)
            noise = self._rng.standard_normal(self.embedding_dim) * 0.5
            interf_content = original_content * similarity + noise * (1 - similarity)
            expected_resolution = "differentiate"
        else:
            # Very similar (confusable)
            similarity = self._rng.uniform(0.85, 0.95)
            noise = self._rng.standard_normal(self.embedding_dim) * 0.15
            interf_content = original_content * similarity + noise * (1 - similarity)
            expected_resolution = "interleave"

        interf_content = interf_content / np.linalg.norm(interf_content)

        interfering = MemoryItem(
            memory_id=f"interfering_{trial_id}",
            content=interf_content,
            initial_strength=0.6,
            timestamp=1.0,
            category="episodic",
        )

        scenario = InterferenceScenario(
            original_memory=original,
            interfering_memory=interfering,
            similarity=similarity,
            expected_resolution=expected_resolution,
        )

        return scenario, expected_resolution

    def evaluate(self, expected: str, actual: Any) -> Tuple[bool, float]:
        """Evaluate interference resolution."""
        if actual is None:
            return False, 0.0

        actual_str = str(actual).strip().lower()
        expected_str = expected.strip().lower()

        # Exact match
        if actual_str == expected_str:
            return True, 1.0

        # Partial credit for reasonable alternatives
        alternatives = {
            "differentiate": ["separate", "distinguish", "disambiguate"],
            "interleave": ["alternate", "mix", "replay"],
            "none": ["ignore", "independent", "no_action"],
        }

        for alt in alternatives.get(expected_str, []):
            if alt in actual_str:
                return True, 0.8

        return False, 0.0


class LearningEfficiencyBenchmark(Benchmark):
    """
    Benchmark for learning efficiency.

    Tests how many replays are needed to reach target strength.
    """

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        config = config or BenchmarkConfig(
            name="learning_efficiency",
            description="Learning efficiency benchmark",
            n_trials=40,
        )
        super().__init__(config)
        self.embedding_dim = 64

    @property
    def name(self) -> str:
        return "learning_efficiency"

    def generate_trial(self, trial_id: int) -> Tuple[LearningScenario, int]:
        """Generate a learning efficiency trial."""
        # Number of items to learn
        n_items = 3 + (trial_id % 5)

        items = []
        for i in range(n_items):
            content = self._rng.standard_normal(self.embedding_dim)
            content = content / np.linalg.norm(content)

            items.append(MemoryItem(
                memory_id=f"item_{trial_id}_{i}",
                content=content,
                initial_strength=0.2,
                timestamp=float(i),
                category="semantic" if i % 2 == 0 else "episodic",
            ))

        target_strength = 0.8

        # Expected replays based on learning theory
        # More items = more replays needed
        # Base: ~3 replays per item to reach 0.8 from 0.2
        base_replays_per_item = 3
        expected_replays = n_items * base_replays_per_item

        scenario = LearningScenario(
            items=items,
            target_strength=target_strength,
            max_replays=expected_replays * 2,  # Allow some buffer
        )

        return scenario, expected_replays

    def evaluate(self, expected: int, actual: Any) -> Tuple[bool, float]:
        """Evaluate learning efficiency."""
        if actual is None:
            return False, 0.0

        try:
            replays_used = int(actual)
        except (ValueError, TypeError):
            return False, 0.0

        # Efficiency: fewer replays is better
        if replays_used <= 0:
            return False, 0.0

        # Score based on how close to optimal
        if replays_used <= expected:
            # At or better than expected - full credit
            score = 1.0
        else:
            # More replays needed - partial credit
            overshoot = (replays_used - expected) / expected
            score = max(0.0, 1.0 - overshoot)

        # Success if within 50% of expected
        success = replays_used <= expected * 1.5

        return success, score


class SchemaFormationBenchmark(Benchmark):
    """
    Benchmark for schema/prototype formation.

    Tests agent's ability to abstract common structure from examples.
    """

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        config = config or BenchmarkConfig(
            name="schema_formation",
            description="Schema formation quality benchmark",
            n_trials=40,
        )
        super().__init__(config)
        self.embedding_dim = 64

    @property
    def name(self) -> str:
        return "schema_formation"

    def generate_trial(self, trial_id: int) -> Tuple[Dict, np.ndarray]:
        """Generate a schema formation trial."""
        # Create true prototype
        prototype = self._rng.standard_normal(self.embedding_dim)
        prototype = prototype / np.linalg.norm(prototype)

        # Generate instances around prototype
        n_instances = 5 + self._rng.integers(0, 6)
        variance = 0.1 + 0.3 * (trial_id % 4) / 4  # Varying difficulty

        instances = []
        for i in range(n_instances):
            noise = self._rng.normal(0, variance, self.embedding_dim)
            instance = prototype + noise
            instance = instance / np.linalg.norm(instance)
            instances.append(instance)

        task = {
            "instances": instances,
            "n_instances": n_instances,
            "category": f"category_{trial_id % 5}",
        }

        return task, prototype

    def evaluate(self, expected: np.ndarray, actual: Any) -> Tuple[bool, float]:
        """Evaluate schema quality."""
        if actual is None:
            return False, 0.0

        try:
            schema = np.array(actual)
        except (ValueError, TypeError):
            return False, 0.0

        if schema.shape != expected.shape:
            return False, 0.0

        # Compute similarity to true prototype
        norm_e = np.linalg.norm(expected)
        norm_a = np.linalg.norm(schema)

        if norm_e < 1e-8 or norm_a < 1e-8:
            return False, 0.0

        similarity = np.dot(expected, schema) / (norm_e * norm_a)
        score = (similarity + 1) / 2  # Map [-1, 1] to [0, 1]

        # Success if similarity > 0.85
        success = similarity > 0.85

        return success, float(score)


class SpacedRepetitionBenchmark(Benchmark):
    """
    Benchmark for spaced repetition scheduling.

    Tests agent's ability to optimize review schedules.
    """

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        config = config or BenchmarkConfig(
            name="spaced_repetition",
            description="Spaced repetition scheduling benchmark",
            n_trials=50,
        )
        super().__init__(config)

    @property
    def name(self) -> str:
        return "spaced_repetition"

    def generate_trial(self, trial_id: int) -> Tuple[Dict, float]:
        """Generate a spaced repetition trial."""
        # Memory history
        n_reviews = self._rng.integers(1, 10)
        review_qualities = self._rng.uniform(2, 5, n_reviews).tolist()

        # Current easiness factor (SM-2 style)
        ef = 2.5
        for q in review_qualities:
            ef = max(1.3, ef + 0.1 - (5 - q) * (0.08 + (5 - q) * 0.02))

        # Current interval
        if n_reviews == 1:
            current_interval = 1.0
        elif n_reviews == 2:
            current_interval = 6.0
        else:
            current_interval = 6.0 * (ef ** (n_reviews - 2))
            current_interval = min(365, current_interval)

        # Last review quality
        last_quality = review_qualities[-1]

        # Expected next interval
        if last_quality >= 3:
            expected_interval = current_interval * ef
        else:
            expected_interval = 1.0

        expected_interval = min(365, expected_interval)

        task = {
            "review_history": review_qualities,
            "current_interval": current_interval,
            "easiness_factor": ef,
            "last_quality": last_quality,
        }

        return task, expected_interval

    def evaluate(self, expected: float, actual: Any) -> Tuple[bool, float]:
        """Evaluate interval prediction."""
        if actual is None:
            return False, 0.0

        try:
            interval = float(actual)
        except (ValueError, TypeError):
            return False, 0.0

        if interval <= 0:
            return False, 0.0

        # Score based on relative error
        if expected < 1:
            error = abs(interval - expected)
        else:
            error = abs(interval - expected) / expected

        score = max(0.0, 1.0 - error)
        success = error < 0.2  # Within 20%

        return success, score


def create_consolidation_benchmark_suite() -> List[Benchmark]:
    """Create all consolidation benchmarks."""
    return [
        RetentionBenchmark(),
        InterferenceBenchmark(),
        LearningEfficiencyBenchmark(),
        SchemaFormationBenchmark(),
        SpacedRepetitionBenchmark(),
    ]


__all__ = [
    'MemoryItem',
    'InterferenceScenario',
    'LearningScenario',
    'RetentionBenchmark',
    'InterferenceBenchmark',
    'LearningEfficiencyBenchmark',
    'SchemaFormationBenchmark',
    'SpacedRepetitionBenchmark',
    'create_consolidation_benchmark_suite',
]
