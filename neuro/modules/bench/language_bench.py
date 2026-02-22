"""
Language Benchmarks: Tests for language understanding and generation.

Includes:
- Text completion
- Instruction following
- Semantic similarity
- Question answering
"""

from dataclasses import dataclass
from typing import Dict, Optional, Any, Tuple
import numpy as np
import re

from .base_benchmark import Benchmark, BenchmarkConfig


@dataclass
class LanguageConfig(BenchmarkConfig):
    """Configuration for language benchmarks."""

    max_length: int = 100
    vocabulary_size: int = 1000
    semantic_threshold: float = 0.7


class LanguageBenchmark(Benchmark):
    """Base class for language benchmarks."""

    @property
    def name(self) -> str:
        return "language"


class TextCompletion(LanguageBenchmark):
    """
    Text completion benchmark.

    Complete sentences or phrases appropriately.
    """

    def __init__(self, config: Optional[LanguageConfig] = None):
        self.lang_config = config or LanguageConfig(name="text_completion")
        super().__init__(self.lang_config)

    @property
    def name(self) -> str:
        return "text_completion"

    def generate_trial(self, trial_id: int) -> Tuple[Dict[str, Any], str]:
        """Generate a text completion trial."""
        templates = [
            {
                "prompt": "The capital of France is",
                "answer": "Paris",
                "category": "factual",
            },
            {
                "prompt": "Water boils at 100 degrees",
                "answer": "Celsius",
                "category": "factual",
            },
            {
                "prompt": "The opposite of hot is",
                "answer": "cold",
                "category": "semantic",
            },
            {
                "prompt": "The sun rises in the",
                "answer": "east",
                "category": "factual",
            },
            {
                "prompt": "A group of fish is called a",
                "answer": "school",
                "category": "vocabulary",
            },
            {
                "prompt": "The color you get by mixing red and blue is",
                "answer": "purple",
                "category": "reasoning",
            },
            {
                "prompt": "An apple a day keeps the doctor",
                "answer": "away",
                "category": "idiom",
            },
            {
                "prompt": "Two plus two equals",
                "answer": "four",
                "category": "math",
            },
        ]

        idx = trial_id % len(templates)
        template = templates[idx]

        trial_data = {
            "prompt": template["prompt"],
            "category": template["category"],
        }

        return trial_data, template["answer"]

    def evaluate(self, expected: str, actual: Any) -> Tuple[bool, float]:
        """Evaluate text completion."""
        if actual is None:
            return False, 0.0

        try:
            actual_str = str(actual).lower().strip()
            expected_str = expected.lower().strip()

            # Remove punctuation for comparison
            actual_clean = re.sub(r"[^\w\s]", "", actual_str)
            expected_clean = re.sub(r"[^\w\s]", "", expected_str)

            # Exact match
            if expected_clean in actual_clean:
                return True, 1.0

            # Check if first word matches
            actual_words = actual_clean.split()
            if actual_words and actual_words[0] == expected_clean:
                return True, 1.0

            # Partial credit
            from difflib import SequenceMatcher

            ratio = SequenceMatcher(None, actual_clean, expected_clean).ratio()

            return ratio > 0.8, ratio

        except Exception:
            return False, 0.0


class InstructionFollowing(LanguageBenchmark):
    """
    Instruction following benchmark.

    Follow simple instructions accurately.
    """

    def __init__(self, config: Optional[LanguageConfig] = None):
        self.lang_config = config or LanguageConfig(name="instruction_following")
        super().__init__(self.lang_config)

    @property
    def name(self) -> str:
        return "instruction_following"

    def generate_trial(self, trial_id: int) -> Tuple[Dict[str, Any], Any]:
        """Generate an instruction following trial."""
        instruction_types = ["count", "list", "transform", "select", "order"]

        inst_type = self._rng.choice(instruction_types)

        if inst_type == "count":
            # Count items
            items = self._rng.choice(["apple", "book", "cat"], size=self._rng.integers(2, 6))
            items_list = list(items)
            instruction = f"Count how many items are in this list: {items_list}"
            expected = len(items_list)

        elif inst_type == "list":
            # List items matching criteria
            numbers = self._rng.integers(1, 20, size=8).tolist()
            instruction = f"List all even numbers from: {numbers}"
            expected = [n for n in numbers if n % 2 == 0]

        elif inst_type == "transform":
            # Transform text
            word = self._rng.choice(["hello", "world", "python", "test"])
            instruction = f"Convert to uppercase: {word}"
            expected = word.upper()

        elif inst_type == "select":
            # Select based on criteria
            words = ["cat", "dog", "elephant", "ant", "bee"]
            instruction = f"Select the longest word from: {words}"
            expected = max(words, key=len)

        else:  # order
            # Sort items
            numbers = self._rng.integers(1, 50, size=5).tolist()
            instruction = f"Sort these numbers in ascending order: {numbers}"
            expected = sorted(numbers)

        trial_data = {
            "instruction": instruction,
            "type": inst_type,
        }

        return trial_data, expected

    def evaluate(self, expected: Any, actual: Any) -> Tuple[bool, float]:
        """Evaluate instruction following."""
        if actual is None:
            return False, 0.0

        try:
            # Handle different types
            if isinstance(expected, int):
                if isinstance(actual, str):
                    # Try to extract number
                    numbers = re.findall(r"\d+", actual)
                    if numbers:
                        actual = int(numbers[0])
                    else:
                        return False, 0.0
                actual = int(actual)
                return actual == expected, 1.0 if actual == expected else 0.0

            elif isinstance(expected, list):
                if isinstance(actual, str):
                    # Try to parse list from string
                    actual = re.findall(r"\w+", actual)

                if isinstance(actual, np.ndarray):
                    actual = actual.tolist()

                # Convert elements
                if expected and isinstance(expected[0], int):
                    actual = [int(x) if str(x).isdigit() else x for x in actual]

                if actual == expected:
                    return True, 1.0

                # Partial credit for set overlap
                set_expected = set(map(str, expected))
                set_actual = set(map(str, actual))
                overlap = len(set_expected & set_actual)
                score = overlap / len(set_expected) if set_expected else 0

                return score >= 0.8, score

            elif isinstance(expected, str):
                actual_str = str(actual).strip()
                if actual_str.lower() == expected.lower():
                    return True, 1.0

                # Partial match
                if expected.lower() in actual_str.lower():
                    return True, 0.9

                return False, 0.0

            else:
                return str(actual) == str(expected), 1.0 if str(actual) == str(expected) else 0.0

        except Exception:
            return False, 0.0


class SemanticSimilarity(LanguageBenchmark):
    """
    Semantic similarity benchmark.

    Determine if two texts have similar meaning.
    """

    def __init__(self, config: Optional[LanguageConfig] = None):
        self.lang_config = config or LanguageConfig(name="semantic_similarity")
        super().__init__(self.lang_config)

    @property
    def name(self) -> str:
        return "semantic_similarity"

    def generate_trial(self, trial_id: int) -> Tuple[Dict[str, Any], bool]:
        """Generate a semantic similarity trial."""
        pairs = [
            # Similar pairs
            ("The cat sat on the mat", "A feline rested on the rug", True),
            ("It is raining outside", "There is precipitation outdoors", True),
            ("She is very happy", "She feels joyful", True),
            ("The car is fast", "The vehicle moves quickly", True),
            # Dissimilar pairs
            ("The sun is hot", "Ice cream is cold", False),
            ("Dogs are animals", "Mathematics is hard", False),
            ("I like pizza", "The weather is nice", False),
            ("Books are informative", "Cars need fuel", False),
        ]

        idx = trial_id % len(pairs)
        text1, text2, similar = pairs[idx]

        trial_data = {
            "text1": text1,
            "text2": text2,
        }

        return trial_data, similar

    def evaluate(self, expected: bool, actual: Any) -> Tuple[bool, float]:
        """Evaluate semantic similarity judgment."""
        if actual is None:
            return False, 0.0

        try:
            if isinstance(actual, (bool, np.bool_)):
                actual_bool = bool(actual)
            elif isinstance(actual, str):
                actual_bool = actual.lower() in ["true", "yes", "similar", "1"]
            elif isinstance(actual, (int, float)):
                actual_bool = actual > 0.5
            else:
                actual_bool = bool(actual)

            correct = actual_bool == expected
            return correct, 1.0 if correct else 0.0

        except Exception:
            return False, 0.0


class QuestionAnswering(LanguageBenchmark):
    """
    Question answering benchmark.

    Answer questions based on provided context.
    """

    def __init__(self, config: Optional[LanguageConfig] = None):
        self.lang_config = config or LanguageConfig(name="question_answering")
        super().__init__(self.lang_config)

    @property
    def name(self) -> str:
        return "question_answering"

    def generate_trial(self, trial_id: int) -> Tuple[Dict[str, Any], str]:
        """Generate a question answering trial."""
        qa_pairs = [
            {
                "context": "John went to the store to buy apples. He bought five apples.",
                "question": "How many apples did John buy?",
                "answer": "five",
            },
            {
                "context": "The Eiffel Tower is located in Paris, France. It was built in 1889.",
                "question": "Where is the Eiffel Tower located?",
                "answer": "Paris",
            },
            {
                "context": "Water freezes at 0 degrees Celsius and boils at 100 degrees Celsius.",
                "question": "At what temperature does water freeze?",
                "answer": "0",
            },
            {
                "context": "The dog named Max loves to play fetch in the park every morning.",
                "question": "What is the dog's name?",
                "answer": "Max",
            },
            {
                "context": "Python is a programming language created by Guido van Rossum.",
                "question": "Who created Python?",
                "answer": "Guido van Rossum",
            },
        ]

        idx = trial_id % len(qa_pairs)
        qa = qa_pairs[idx]

        trial_data = {
            "context": qa["context"],
            "question": qa["question"],
        }

        return trial_data, qa["answer"]

    def evaluate(self, expected: str, actual: Any) -> Tuple[bool, float]:
        """Evaluate question answering."""
        if actual is None:
            return False, 0.0

        try:
            actual_str = str(actual).lower().strip()
            expected_str = expected.lower().strip()

            # Check if expected is contained in actual
            if expected_str in actual_str:
                return True, 1.0

            # Check word overlap
            expected_words = set(expected_str.split())
            actual_words = set(actual_str.split())
            overlap = len(expected_words & actual_words)

            if overlap == len(expected_words):
                return True, 1.0

            score = overlap / len(expected_words) if expected_words else 0
            return score >= 0.8, score

        except Exception:
            return False, 0.0
