"""
Meta Learner: Learns how to learn and improve.

The meta learner optimizes the self-improvement process itself,
learning which modifications work best and when to apply them.

Based on:
- Meta-learning (learning to learn)
- AutoML and neural architecture search
- arXiv:2503.00735 - LADDER self-improvement
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from enum import Enum
import numpy as np
import time

from .generator import Modification, ModificationType
from .verifier import VerificationResult
from .updater import UpdateResult, UpdateStatus


class StrategyType(Enum):
    """Types of meta-learning strategies."""

    GRADIENT = "gradient"  # Follow improvement gradients
    EVOLUTIONARY = "evolutionary"  # Evolve modification patterns
    BAYESIAN = "bayesian"  # Bayesian optimization
    BANDIT = "bandit"  # Multi-armed bandit
    CURRICULUM = "curriculum"  # Ordered learning schedule


@dataclass
class MetaParams:
    """Parameters for meta-learning."""

    history_window: int = 100  # Window for computing statistics
    exploration_rate: float = 0.1  # Exploration vs exploitation
    learning_rate: float = 0.01  # Meta-learning rate
    discount: float = 0.95  # Discount for older experiences
    strategy_adaptation: bool = True  # Adapt strategies over time


@dataclass
class LearningStrategy:
    """A learned strategy for self-improvement."""

    strategy_id: str
    strategy_type: StrategyType
    parameters: Dict[str, float]
    success_rate: float
    avg_improvement: float
    n_applications: int
    last_used: float = field(default_factory=time.time)

    def score(self) -> float:
        """Compute strategy score."""
        recency = 1.0 / (1.0 + (time.time() - self.last_used) / 3600)
        experience = min(1.0, self.n_applications / 100)
        return (
            self.success_rate * 0.4
            + self.avg_improvement * 10 * 0.4
            + recency * 0.1
            + experience * 0.1
        )


class MetaLearner:
    """
    Meta-learner that learns how to improve.

    The meta-learner operates at a higher level than individual
    modifications, learning patterns about:

    1. **When to modify**: Timing of self-improvement
    2. **What to modify**: Which components to focus on
    3. **How to modify**: Best modification strategies
    4. **How much**: Magnitude of changes

    This enables the system to get better at getting better.
    """

    def __init__(self, params: Optional[MetaParams] = None):
        self.params = params or MetaParams()

        # Learned strategies
        self._strategies: Dict[str, LearningStrategy] = {}
        self._strategy_weights: Dict[
            str, float
        ] = {}  # Must be initialized before _initialize_strategies
        self._initialize_strategies()

        # Experience history
        self._experience: List[Tuple[Modification, UpdateResult]] = []

        # Performance tracking
        self._performance_history: List[Tuple[float, float]] = []  # (time, performance)

        # Meta-state
        self._current_strategy: Optional[str] = None
        self._improvement_velocity: float = 0.0

    def _initialize_strategies(self) -> None:
        """Initialize default strategies."""
        for strategy_type in StrategyType:
            strategy = LearningStrategy(
                strategy_id=strategy_type.value,
                strategy_type=strategy_type,
                parameters=self._default_params(strategy_type),
                success_rate=0.5,
                avg_improvement=0.0,
                n_applications=0,
            )
            self._strategies[strategy.strategy_id] = strategy
            self._strategy_weights[strategy.strategy_id] = 1.0 / len(StrategyType)

    def _default_params(self, strategy_type: StrategyType) -> Dict[str, float]:
        """Get default parameters for a strategy type."""
        defaults = {
            StrategyType.GRADIENT: {
                "step_size": 0.1,
                "momentum": 0.9,
            },
            StrategyType.EVOLUTIONARY: {
                "mutation_rate": 0.1,
                "population_size": 10,
            },
            StrategyType.BAYESIAN: {
                "acquisition": 0.5,  # Balance exploration/exploitation
                "kernel_scale": 1.0,
            },
            StrategyType.BANDIT: {
                "epsilon": 0.1,
                "temperature": 1.0,
            },
            StrategyType.CURRICULUM: {
                "difficulty_scale": 0.1,
                "progression_rate": 0.05,
            },
        }
        return defaults.get(strategy_type, {})

    def select_strategy(self) -> LearningStrategy:
        """Select a strategy for the next improvement cycle."""
        if np.random.rand() < self.params.exploration_rate:
            # Explore: random strategy
            strategy_id = np.random.choice(list(self._strategies.keys()))
        else:
            # Exploit: best strategy (weighted)
            weights = np.array([self._strategy_weights[s] for s in self._strategies])
            weights = weights / weights.sum()
            strategy_id = np.random.choice(list(self._strategies.keys()), p=weights)

        self._current_strategy = strategy_id
        strategy = self._strategies[strategy_id]
        strategy.last_used = time.time()

        return strategy

    def record_experience(
        self,
        modification: Modification,
        result: UpdateResult,
    ) -> None:
        """Record an improvement experience."""
        self._experience.append((modification, result))

        if len(self._experience) > self.params.history_window * 10:
            self._experience.pop(0)

        # Update current strategy statistics
        if self._current_strategy and self._current_strategy in self._strategies:
            strategy = self._strategies[self._current_strategy]

            if result.status == UpdateStatus.APPLIED:
                # Update success rate
                n = strategy.n_applications
                strategy.success_rate = (strategy.success_rate * n + 1.0) / (n + 1)

                # Update average improvement
                strategy.avg_improvement = (
                    strategy.avg_improvement * n + result.performance_delta
                ) / (n + 1)
            else:
                # Failed application
                n = strategy.n_applications
                strategy.success_rate = (strategy.success_rate * n + 0.0) / (n + 1)

            strategy.n_applications += 1

        # Update strategy weights
        self._update_strategy_weights()

    def _update_strategy_weights(self) -> None:
        """Update strategy weights based on performance."""
        if not self.params.strategy_adaptation:
            return

        for strategy_id, strategy in self._strategies.items():
            score = strategy.score()
            # Soft update of weights
            old_weight = self._strategy_weights[strategy_id]
            self._strategy_weights[strategy_id] = (
                1 - self.params.learning_rate
            ) * old_weight + self.params.learning_rate * score

        # Normalize weights
        total = sum(self._strategy_weights.values())
        if total > 0:
            for s in self._strategy_weights:
                self._strategy_weights[s] /= total

    def record_performance(self, performance: float) -> None:
        """Record current system performance."""
        current_time = time.time()
        self._performance_history.append((current_time, performance))

        if len(self._performance_history) > self.params.history_window:
            self._performance_history.pop(0)

        # Compute improvement velocity
        if len(self._performance_history) >= 2:
            times = [t for t, _ in self._performance_history]
            perfs = [p for _, p in self._performance_history]
            if times[-1] != times[0]:
                self._improvement_velocity = (perfs[-1] - perfs[0]) / (times[-1] - times[0])

    def should_improve(self) -> Tuple[bool, str]:
        """
        Decide whether to attempt improvement.

        Returns:
            (should_improve, reason)
        """
        if len(self._performance_history) < 2:
            return True, "insufficient_data"

        # Check if performance is plateauing
        recent_perfs = [p for _, p in self._performance_history[-10:]]
        if len(recent_perfs) >= 5:
            variance = np.var(recent_perfs)
            if variance < 0.001:
                return True, "plateau_detected"

        # Check improvement velocity
        if self._improvement_velocity < 0:
            return True, "declining_performance"

        if self._improvement_velocity < 0.001:
            return True, "slow_improvement"

        # Regular improvement interval
        recent_mods = [m for m, r in self._experience[-10:] if r.status == UpdateStatus.APPLIED]
        if len(recent_mods) < 3:
            return True, "regular_interval"

        return False, "performing_well"

    def recommend_target(self) -> Tuple[str, float]:
        """
        Recommend which component to target for modification.

        Returns:
            (component_name, priority)
        """
        # Analyze which components have been most fruitful
        component_stats: Dict[str, Dict[str, float]] = {}

        for mod, result in self._experience:
            comp = mod.target_component
            if comp not in component_stats:
                component_stats[comp] = {"success": 0, "total": 0, "improvement": 0}

            component_stats[comp]["total"] += 1
            if result.status == UpdateStatus.APPLIED:
                component_stats[comp]["success"] += 1
                component_stats[comp]["improvement"] += result.performance_delta

        if not component_stats:
            return "system", 0.5

        # Score components
        best_component = None
        best_score = -float("inf")

        for comp, stats in component_stats.items():
            if stats["total"] == 0:
                continue

            success_rate = stats["success"] / stats["total"]
            avg_improvement = stats["improvement"] / max(1, stats["success"])
            score = success_rate * 0.5 + avg_improvement * 10 * 0.5

            if score > best_score:
                best_score = score
                best_component = comp

        return best_component or "system", float(best_score)

    def recommend_magnitude(self) -> float:
        """Recommend modification magnitude."""
        # Analyze recent successful modifications
        recent_successful = [
            m
            for m, r in self._experience[-50:]
            if r.status == UpdateStatus.APPLIED and r.performance_delta > 0
        ]

        if not recent_successful:
            return 0.1  # Default moderate magnitude

        magnitudes = [
            m.changes.get("adjustment_scale", 0.1)
            for m in recent_successful
            if "adjustment_scale" in m.changes
        ]

        if magnitudes:
            return float(np.mean(magnitudes))

        return 0.1

    def get_curriculum(self) -> List[Dict[str, Any]]:
        """
        Get a curriculum of improvements to attempt.

        Returns ordered list of improvement focuses.
        """
        curriculum = []

        # Start with high-success strategies
        sorted_strategies = sorted(
            self._strategies.values(), key=lambda s: s.success_rate, reverse=True
        )

        for strategy in sorted_strategies[:3]:
            curriculum.append(
                {
                    "strategy": strategy.strategy_id,
                    "priority": strategy.success_rate,
                    "parameters": strategy.parameters,
                }
            )

        # Add exploration items
        low_experience = [s for s in self._strategies.values() if s.n_applications < 10]
        for strategy in low_experience[:2]:
            curriculum.append(
                {
                    "strategy": strategy.strategy_id,
                    "priority": 0.3,
                    "parameters": strategy.parameters,
                    "explore": True,
                }
            )

        return curriculum

    def adapt_parameters(
        self,
        strategy: LearningStrategy,
        success: bool,
        improvement: float,
    ) -> None:
        """Adapt strategy parameters based on outcome."""
        params = strategy.parameters

        for key in params:
            if isinstance(params[key], (int, float)):
                if success and improvement > 0:
                    # Reinforce current parameters
                    params[key] *= 1 + self.params.learning_rate * 0.1
                elif not success:
                    # Perturb parameters
                    params[key] *= 1 + self.params.learning_rate * np.random.randn()

                # Keep in reasonable bounds
                if isinstance(params[key], float):
                    params[key] = np.clip(params[key], 0.001, 10.0)

    def get_statistics(self) -> Dict[str, Any]:
        """Get meta-learner statistics."""
        if not self._experience:
            return {
                "n_experiences": 0,
                "improvement_velocity": 0.0,
                "best_strategy": None,
            }

        successful = [(m, r) for m, r in self._experience if r.status == UpdateStatus.APPLIED]

        best_strategy = max(self._strategies.values(), key=lambda s: s.score())

        return {
            "n_experiences": len(self._experience),
            "n_successful": len(successful),
            "improvement_velocity": self._improvement_velocity,
            "best_strategy": best_strategy.strategy_id,
            "best_strategy_score": best_strategy.score(),
            "strategy_weights": self._strategy_weights.copy(),
            "current_performance": self._performance_history[-1][1]
            if self._performance_history
            else 0.0,
        }

    def reset(self) -> None:
        """Reset meta-learner state."""
        self._experience = []
        self._performance_history = []
        self._current_strategy = None
        self._improvement_velocity = 0.0
        self._initialize_strategies()
