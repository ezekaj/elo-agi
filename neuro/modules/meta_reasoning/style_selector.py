"""
Reasoning Style Selector

Selects appropriate reasoning styles for problems:
- Style fitness computation
- Adaptive selection based on feedback
- Multi-style combination
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import numpy as np

try:
    from .problem_classifier import ProblemAnalysis, ProblemType, ProblemDifficulty
except ImportError:
    from problem_classifier import ProblemAnalysis, ProblemType, ProblemDifficulty


class ReasoningStyle(Enum):
    """Available reasoning styles."""

    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"
    ABDUCTIVE = "abductive"
    ANALOGICAL = "analogical"
    CAUSAL = "causal"
    SPATIAL = "spatial"
    TEMPORAL = "temporal"
    HEURISTIC = "heuristic"
    SYSTEMATIC = "systematic"
    CREATIVE = "creative"


@dataclass
class StyleSelectorConfig:
    """Configuration for style selection."""

    exploration_rate: float = 0.1
    learning_rate: float = 0.1
    min_fitness_threshold: float = 0.3
    allow_style_combination: bool = True
    max_combined_styles: int = 3


@dataclass
class StyleSelection:
    """Result of style selection."""

    primary_style: ReasoningStyle
    primary_fitness: float
    secondary_styles: List[Tuple[ReasoningStyle, float]]
    confidence: float
    rationale: str


@dataclass
class StyleFeedback:
    """Feedback on style effectiveness."""

    style: ReasoningStyle
    problem_type: ProblemType
    success: bool
    efficiency: float
    quality: float


class StyleSelector:
    """
    Selects reasoning styles for problems.

    Capabilities:
    - Compute style fitness for problems
    - Learn from feedback
    - Combine multiple styles
    - Adapt selection over time
    """

    def __init__(
        self,
        config: Optional[StyleSelectorConfig] = None,
        random_seed: Optional[int] = None,
    ):
        self.config = config or StyleSelectorConfig()
        self._rng = np.random.default_rng(random_seed)

        self._style_problem_fitness: Dict[Tuple[ReasoningStyle, ProblemType], float] = {}
        self._initialize_fitness_matrix()

        self._feedback_history: List[StyleFeedback] = []
        self._total_selections = 0

    def _initialize_fitness_matrix(self) -> None:
        """Initialize style-problem fitness matrix with priors."""
        priors = {
            (ReasoningStyle.DEDUCTIVE, ProblemType.LOGICAL): 0.9,
            (ReasoningStyle.DEDUCTIVE, ProblemType.MATHEMATICAL): 0.8,
            (ReasoningStyle.INDUCTIVE, ProblemType.UNKNOWN): 0.7,
            (ReasoningStyle.ABDUCTIVE, ProblemType.CAUSAL): 0.8,
            (ReasoningStyle.ANALOGICAL, ProblemType.ANALOGICAL): 0.9,
            (ReasoningStyle.ANALOGICAL, ProblemType.CREATIVE): 0.7,
            (ReasoningStyle.CAUSAL, ProblemType.CAUSAL): 0.9,
            (ReasoningStyle.CAUSAL, ProblemType.TEMPORAL): 0.6,
            (ReasoningStyle.SPATIAL, ProblemType.SPATIAL): 0.9,
            (ReasoningStyle.TEMPORAL, ProblemType.TEMPORAL): 0.9,
            (ReasoningStyle.TEMPORAL, ProblemType.PLANNING): 0.7,
            (ReasoningStyle.HEURISTIC, ProblemType.UNKNOWN): 0.6,
            (ReasoningStyle.SYSTEMATIC, ProblemType.LOGICAL): 0.8,
            (ReasoningStyle.SYSTEMATIC, ProblemType.MATHEMATICAL): 0.8,
            (ReasoningStyle.CREATIVE, ProblemType.CREATIVE): 0.9,
        }

        for style in ReasoningStyle:
            for ptype in ProblemType:
                key = (style, ptype)
                if key in priors:
                    self._style_problem_fitness[key] = priors[key]
                else:
                    self._style_problem_fitness[key] = 0.5

    def select_style(
        self,
        analysis: ProblemAnalysis,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> StyleSelection:
        """
        Select reasoning style for a problem.

        Args:
            analysis: Problem analysis
            constraints: Optional constraints on selection

        Returns:
            StyleSelection with selected styles
        """
        fitness_scores = {}
        for style in ReasoningStyle:
            fitness = self.compute_style_fitness(style, analysis)

            if constraints:
                fitness = self._apply_constraints(style, fitness, constraints)

            fitness_scores[style] = fitness

        if self._rng.random() < self.config.exploration_rate:
            primary_style = self._rng.choice(list(ReasoningStyle))
            primary_fitness = fitness_scores[primary_style]
        else:
            primary_style = max(fitness_scores, key=fitness_scores.get)
            primary_fitness = fitness_scores[primary_style]

        secondary_styles = []
        if self.config.allow_style_combination:
            for style, fitness in sorted(fitness_scores.items(), key=lambda x: x[1], reverse=True):
                if style != primary_style and fitness >= self.config.min_fitness_threshold:
                    secondary_styles.append((style, fitness))
                    if len(secondary_styles) >= self.config.max_combined_styles - 1:
                        break

        confidence = self._compute_selection_confidence(
            primary_fitness, [f for _, f in secondary_styles]
        )

        rationale = self._generate_rationale(primary_style, analysis.problem_type, primary_fitness)

        self._total_selections += 1

        return StyleSelection(
            primary_style=primary_style,
            primary_fitness=primary_fitness,
            secondary_styles=secondary_styles,
            confidence=confidence,
            rationale=rationale,
        )

    def compute_style_fitness(
        self,
        style: ReasoningStyle,
        problem: ProblemAnalysis,
    ) -> float:
        """
        Compute fitness of a style for a problem.

        Args:
            style: Reasoning style
            problem: Problem analysis

        Returns:
            Fitness score [0, 1]
        """
        key = (style, problem.problem_type)
        base_fitness = self._style_problem_fitness.get(key, 0.5)

        complexity_factor = 1.0
        if problem.difficulty == ProblemDifficulty.EXPERT:
            if style in [ReasoningStyle.SYSTEMATIC, ReasoningStyle.DEDUCTIVE]:
                complexity_factor = 1.1
            else:
                complexity_factor = 0.9
        elif problem.difficulty == ProblemDifficulty.EASY:
            if style == ReasoningStyle.HEURISTIC:
                complexity_factor = 1.1

        fitness = base_fitness * complexity_factor

        return float(np.clip(fitness, 0, 1))

    def _apply_constraints(
        self,
        style: ReasoningStyle,
        fitness: float,
        constraints: Dict[str, Any],
    ) -> float:
        """Apply constraints to fitness score."""
        if "excluded_styles" in constraints:
            if style in constraints["excluded_styles"]:
                return 0.0

        if "required_styles" in constraints:
            if style in constraints["required_styles"]:
                fitness *= 1.5

        if "time_limit" in constraints:
            if constraints["time_limit"] < 1.0:
                if style == ReasoningStyle.SYSTEMATIC:
                    fitness *= 0.5
                elif style == ReasoningStyle.HEURISTIC:
                    fitness *= 1.2

        return float(np.clip(fitness, 0, 1))

    def _compute_selection_confidence(
        self,
        primary_fitness: float,
        secondary_fitnesses: List[float],
    ) -> float:
        """Compute confidence in selection."""
        if not secondary_fitnesses:
            return primary_fitness

        gap = primary_fitness - max(secondary_fitnesses) if secondary_fitnesses else primary_fitness
        confidence = primary_fitness * (1 + gap)

        return float(np.clip(confidence, 0, 1))

    def _generate_rationale(
        self,
        style: ReasoningStyle,
        problem_type: ProblemType,
        fitness: float,
    ) -> str:
        """Generate rationale for style selection."""
        rationales = {
            ReasoningStyle.DEDUCTIVE: "Uses logical deduction from premises",
            ReasoningStyle.INDUCTIVE: "Generalizes from specific observations",
            ReasoningStyle.ABDUCTIVE: "Infers best explanation for observations",
            ReasoningStyle.ANALOGICAL: "Maps from similar known problems",
            ReasoningStyle.CAUSAL: "Traces cause-effect relationships",
            ReasoningStyle.SPATIAL: "Reasons about spatial relationships",
            ReasoningStyle.TEMPORAL: "Reasons about temporal sequences",
            ReasoningStyle.HEURISTIC: "Uses quick heuristics for efficiency",
            ReasoningStyle.SYSTEMATIC: "Methodically explores solution space",
            ReasoningStyle.CREATIVE: "Generates novel solutions",
        }

        base = rationales.get(style, "Selected based on fitness")
        return f"{base} (fitness: {fitness:.2f} for {problem_type.value} problems)"

    def adaptive_selection(
        self,
        problem: ProblemAnalysis,
        feedback_history: Optional[List[StyleFeedback]] = None,
    ) -> StyleSelection:
        """
        Adaptively select style based on feedback history.

        Args:
            problem: Problem analysis
            feedback_history: Optional history to consider

        Returns:
            StyleSelection
        """
        if feedback_history:
            for feedback in feedback_history:
                self.update_from_feedback(feedback)

        return self.select_style(problem)

    def update_from_feedback(self, feedback: StyleFeedback) -> None:
        """
        Update style fitness from feedback.

        Args:
            feedback: Feedback on style effectiveness
        """
        key = (feedback.style, feedback.problem_type)

        current = self._style_problem_fitness.get(key, 0.5)

        target = 0.0
        if feedback.success:
            target = (feedback.efficiency + feedback.quality) / 2
        else:
            target = 0.2

        new_value = current + self.config.learning_rate * (target - current)
        self._style_problem_fitness[key] = float(np.clip(new_value, 0.1, 0.95))

        self._feedback_history.append(feedback)

    def record_feedback(
        self,
        style: ReasoningStyle,
        problem_type: ProblemType,
        success: bool,
        efficiency: float = 0.5,
        quality: float = 0.5,
    ) -> None:
        """Record feedback on style effectiveness."""
        feedback = StyleFeedback(
            style=style,
            problem_type=problem_type,
            success=success,
            efficiency=efficiency,
            quality=quality,
        )
        self.update_from_feedback(feedback)

    def get_style_rankings(
        self,
        problem_type: ProblemType,
    ) -> List[Tuple[ReasoningStyle, float]]:
        """Get style rankings for a problem type."""
        rankings = []
        for style in ReasoningStyle:
            key = (style, problem_type)
            fitness = self._style_problem_fitness.get(key, 0.5)
            rankings.append((style, fitness))

        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings

    def get_fitness_matrix(self) -> Dict[str, Dict[str, float]]:
        """Get the full fitness matrix."""
        matrix: Dict[str, Dict[str, float]] = {}

        for style in ReasoningStyle:
            matrix[style.value] = {}
            for ptype in ProblemType:
                key = (style, ptype)
                matrix[style.value][ptype.value] = self._style_problem_fitness.get(key, 0.5)

        return matrix

    def statistics(self) -> Dict[str, Any]:
        """Get selector statistics."""
        style_counts: Dict[str, int] = {}
        for feedback in self._feedback_history:
            s = feedback.style.value
            style_counts[s] = style_counts.get(s, 0) + 1

        success_rate = 0.0
        if self._feedback_history:
            successes = sum(1 for f in self._feedback_history if f.success)
            success_rate = successes / len(self._feedback_history)

        return {
            "total_selections": self._total_selections,
            "total_feedbacks": len(self._feedback_history),
            "style_usage": style_counts,
            "success_rate": success_rate,
            "exploration_rate": self.config.exploration_rate,
        }
