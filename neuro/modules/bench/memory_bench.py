"""
Memory Benchmarks: Tests for memory systems.

Includes:
- Working memory capacity
- Episodic recall
- Semantic association
- Sequence memory
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

from .base_benchmark import Benchmark, BenchmarkConfig


@dataclass
class MemoryConfig(BenchmarkConfig):
    """Configuration for memory benchmarks."""
    sequence_length: int = 7  # Miller's magic number
    retention_delay: int = 5  # Steps between encoding and recall
    n_items: int = 10
    similarity_threshold: float = 0.8


class MemoryBenchmark(Benchmark):
    """Base class for memory benchmarks."""

    @property
    def name(self) -> str:
        return "memory"


class WorkingMemoryTest(MemoryBenchmark):
    """
    Working memory capacity test.

    Present items, then test recall after delay.
    Tests Miller's Law (7±2 items).
    """

    def __init__(self, config: Optional[MemoryConfig] = None):
        self.memory_config = config or MemoryConfig(name="working_memory")
        super().__init__(self.memory_config)

    @property
    def name(self) -> str:
        return "working_memory"

    def generate_trial(self, trial_id: int) -> Tuple[Dict[str, Any], List[int]]:
        """Generate a working memory trial."""
        # Vary sequence length based on difficulty
        if self.config.difficulty == "easy":
            length = self._rng.integers(3, 5)
        elif self.config.difficulty == "hard":
            length = self._rng.integers(7, 10)
        else:
            length = self._rng.integers(5, 8)

        # Generate sequence
        sequence = self._rng.integers(0, 10, size=length).tolist()

        # Generate distractor task (simple math)
        distractors = []
        for _ in range(self.memory_config.retention_delay):
            a = self._rng.integers(1, 10)
            b = self._rng.integers(1, 10)
            distractors.append({'question': f"{a} + {b} = ?", 'answer': a + b})

        trial_data = {
            'sequence': sequence,
            'distractors': distractors,
            'delay': self.memory_config.retention_delay,
        }

        return trial_data, sequence

    def evaluate(self, expected: List[int], actual: Any) -> Tuple[bool, float]:
        """Evaluate working memory recall."""
        if actual is None:
            return False, 0.0

        try:
            if isinstance(actual, np.ndarray):
                actual = actual.tolist()

            if not isinstance(actual, list):
                actual = list(actual)

            # Exact match
            if actual == expected:
                return True, 1.0

            # Partial credit for correct items in correct positions
            min_len = min(len(actual), len(expected))
            matches = sum(1 for i in range(min_len) if actual[i] == expected[i])
            score = matches / len(expected)

            return score >= 0.8, score

        except Exception:
            return False, 0.0


class EpisodicRecall(MemoryBenchmark):
    """
    Episodic memory recall test.

    Store events with context, then query by context.
    Tests context-dependent memory retrieval.
    """

    def __init__(self, config: Optional[MemoryConfig] = None):
        self.memory_config = config or MemoryConfig(name="episodic_recall")
        super().__init__(self.memory_config)

    @property
    def name(self) -> str:
        return "episodic_recall"

    def generate_trial(self, trial_id: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Generate an episodic recall trial."""
        # Create episodes (event + context)
        locations = ['kitchen', 'office', 'garden', 'bedroom', 'garage']
        objects = ['apple', 'book', 'key', 'phone', 'cup']
        actions = ['found', 'placed', 'moved', 'saw', 'picked up']

        n_episodes = self._rng.integers(3, 6)
        episodes = []

        for i in range(n_episodes):
            episode = {
                'id': i,
                'location': self._rng.choice(locations),
                'object': self._rng.choice(objects),
                'action': self._rng.choice(actions),
                'time': i,
            }
            episodes.append(episode)

        # Choose query type
        query_type = self._rng.choice(['location', 'object', 'action'])

        # Find target episode
        target_idx = self._rng.integers(0, len(episodes))
        target = episodes[target_idx]

        if query_type == 'location':
            query = f"What happened in the {target['location']}?"
            expected = {
                'object': target['object'],
                'action': target['action'],
            }
        elif query_type == 'object':
            query = f"Where was the {target['object']}?"
            expected = {
                'location': target['location'],
                'action': target['action'],
            }
        else:
            query = f"What did you {target['action']}?"
            expected = {
                'location': target['location'],
                'object': target['object'],
            }

        trial_data = {
            'episodes': episodes,
            'query': query,
            'query_type': query_type,
        }

        return trial_data, expected

    def evaluate(self, expected: Dict[str, Any], actual: Any) -> Tuple[bool, float]:
        """Evaluate episodic recall."""
        if actual is None:
            return False, 0.0

        try:
            if not isinstance(actual, dict):
                return False, 0.0

            # Check each expected field
            matches = 0
            total = len(expected)

            for key, value in expected.items():
                if key in actual and actual[key] == value:
                    matches += 1

            score = matches / total
            return score == 1.0, score

        except Exception:
            return False, 0.0


class SequenceMemory(MemoryBenchmark):
    """
    Sequence memory test.

    Remember and reproduce sequences of varying complexity.
    """

    def __init__(self, config: Optional[MemoryConfig] = None):
        self.memory_config = config or MemoryConfig(name="sequence_memory")
        super().__init__(self.memory_config)

    @property
    def name(self) -> str:
        return "sequence_memory"

    def generate_trial(self, trial_id: int) -> Tuple[Dict[str, Any], List[str]]:
        """Generate a sequence memory trial."""
        # Different sequence types
        seq_type = self._rng.choice(['numeric', 'letter', 'pattern', 'mixed'])

        length = self._rng.integers(4, 9)

        if seq_type == 'numeric':
            sequence = [str(self._rng.integers(0, 10)) for _ in range(length)]
        elif seq_type == 'letter':
            letters = 'ABCDEFGHIJ'
            sequence = [self._rng.choice(list(letters)) for _ in range(length)]
        elif seq_type == 'pattern':
            # Repeating pattern
            pattern_len = self._rng.integers(2, 4)
            pattern = [str(self._rng.integers(0, 5)) for _ in range(pattern_len)]
            sequence = (pattern * ((length // pattern_len) + 1))[:length]
        else:
            # Mixed
            chars = list('ABCD0123')
            sequence = [self._rng.choice(chars) for _ in range(length)]

        trial_data = {
            'sequence': sequence,
            'type': seq_type,
            'length': length,
        }

        return trial_data, sequence

    def evaluate(self, expected: List[str], actual: Any) -> Tuple[bool, float]:
        """Evaluate sequence memory."""
        if actual is None:
            return False, 0.0

        try:
            if isinstance(actual, str):
                actual = list(actual)
            elif isinstance(actual, np.ndarray):
                actual = [str(x) for x in actual.flatten()]
            else:
                actual = [str(x) for x in actual]

            expected_str = [str(x) for x in expected]

            if actual == expected_str:
                return True, 1.0

            # Partial credit
            min_len = min(len(actual), len(expected_str))
            matches = sum(1 for i in range(min_len) if actual[i] == expected_str[i])
            score = matches / len(expected_str)

            return score >= 0.9, score

        except Exception:
            return False, 0.0


class AssociativeMemory(MemoryBenchmark):
    """
    Associative memory test.

    Learn associations and retrieve by cue.
    """

    def __init__(self, config: Optional[MemoryConfig] = None):
        self.memory_config = config or MemoryConfig(name="associative_memory")
        super().__init__(self.memory_config)

    @property
    def name(self) -> str:
        return "associative_memory"

    def generate_trial(self, trial_id: int) -> Tuple[Dict[str, Any], str]:
        """Generate an associative memory trial."""
        # Word pairs to learn
        word_pairs = [
            ('cat', 'dog'), ('sun', 'moon'), ('book', 'page'),
            ('tree', 'leaf'), ('car', 'road'), ('pen', 'paper'),
            ('fish', 'water'), ('bird', 'sky'), ('fire', 'smoke'),
            ('door', 'key'),
        ]

        n_pairs = self._rng.integers(3, 7)
        selected = self._rng.choice(len(word_pairs), size=n_pairs, replace=False)
        pairs = [word_pairs[i] for i in selected]

        # Choose cue
        cue_idx = self._rng.integers(0, len(pairs))
        cue_word, target_word = pairs[cue_idx]

        # Randomly cue from first or second word
        if self._rng.random() > 0.5:
            cue = cue_word
            expected = target_word
        else:
            cue = target_word
            expected = cue_word

        trial_data = {
            'pairs': pairs,
            'cue': cue,
        }

        return trial_data, expected

    def evaluate(self, expected: str, actual: Any) -> Tuple[bool, float]:
        """Evaluate associative memory."""
        if actual is None:
            return False, 0.0

        try:
            actual_str = str(actual).lower().strip()
            expected_str = expected.lower().strip()

            if actual_str == expected_str:
                return True, 1.0

            # Partial credit for close matches (edit distance)
            from difflib import SequenceMatcher
            ratio = SequenceMatcher(None, actual_str, expected_str).ratio()

            return ratio > 0.8, ratio

        except Exception:
            return False, 0.0
