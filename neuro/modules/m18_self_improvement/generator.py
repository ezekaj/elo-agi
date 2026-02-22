"""
Modification Generator: Proposes changes to improve the system.

The generator explores the space of possible modifications to the
cognitive architecture, proposing changes that might improve performance.

Based on:
- Darwin Gödel Machine (arXiv:2505.22954)
- Neural Architecture Search
- Evolutionary computation
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import numpy as np
import time
import copy


class ModificationType(Enum):
    """Types of modifications that can be proposed."""

    WEIGHT_ADJUSTMENT = "weight_adjustment"  # Modify connection weights
    ARCHITECTURE = "architecture"  # Add/remove components
    HYPERPARAMETER = "hyperparameter"  # Adjust hyperparameters
    ATTENTION = "attention"  # Modify attention patterns
    MEMORY = "memory"  # Modify memory systems
    LEARNING_RATE = "learning_rate"  # Adjust learning rates
    ACTIVATION = "activation"  # Modify activation functions
    CONNECTIVITY = "connectivity"  # Modify module connections


@dataclass
class GeneratorParams:
    """Parameters for the modification generator."""

    n_candidates: int = 10  # Candidates per generation
    mutation_rate: float = 0.1  # Probability of random mutation
    crossover_rate: float = 0.3  # Probability of combining strategies
    elite_fraction: float = 0.2  # Fraction of best to keep
    exploration_weight: float = 0.3  # Balance exploration vs exploitation
    novelty_bonus: float = 0.1  # Bonus for novel modifications
    history_size: int = 1000  # History of past modifications


@dataclass
class Modification:
    """A proposed modification to the system."""

    mod_id: str
    mod_type: ModificationType
    target_component: str  # Which component to modify
    changes: Dict[str, Any]  # Specific changes to make
    expected_improvement: float  # Predicted improvement
    confidence: float  # Confidence in prediction
    complexity_cost: float  # Complexity added
    reversible: bool  # Can be undone
    parent_mods: List[str] = field(default_factory=list)  # Parent modifications
    timestamp: float = field(default_factory=time.time)

    @property
    def net_expected_value(self) -> float:
        """Expected value minus complexity cost."""
        return self.expected_improvement * self.confidence - self.complexity_cost


class ModificationGenerator:
    """
    Generator that proposes modifications to improve the system.

    The generator uses multiple strategies to explore the space of
    possible improvements:

    1. **Gradient-based**: Follow performance gradients
    2. **Evolutionary**: Mutate and combine successful modifications
    3. **Curiosity-driven**: Explore underexplored regions
    4. **Meta-learned**: Use learned heuristics for modification

    The generator maintains a history of past modifications and their
    effects, using this to guide future proposals.
    """

    def __init__(self, params: Optional[GeneratorParams] = None):
        self.params = params or GeneratorParams()

        # History of modifications
        self._modification_history: List[Tuple[Modification, float]] = []  # (mod, outcome)
        self._successful_patterns: List[Dict[str, Any]] = []

        # Current generation of candidates
        self._current_generation: List[Modification] = []
        self._generation_count = 0

        # Component registry
        self._components: Dict[str, Dict[str, Any]] = {}

        # Strategy weights (learned over time)
        self._strategy_weights = {
            "gradient": 0.25,
            "evolutionary": 0.25,
            "curiosity": 0.25,
            "meta": 0.25,
        }

        # Counter for unique IDs
        self._mod_counter = 0

    def register_component(
        self,
        name: str,
        component_info: Dict[str, Any],
    ) -> None:
        """Register a component that can be modified."""
        self._components[name] = component_info

    def generate_candidates(
        self,
        current_performance: float,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Modification]:
        """
        Generate candidate modifications.

        Args:
            current_performance: Current system performance
            context: Additional context for generation

        Returns:
            List of candidate modifications
        """
        candidates = []

        # Allocate candidates across strategies
        for strategy, weight in self._strategy_weights.items():
            n_candidates = max(1, int(self.params.n_candidates * weight))

            if strategy == "gradient":
                candidates.extend(self._gradient_candidates(n_candidates, context))
            elif strategy == "evolutionary":
                candidates.extend(self._evolutionary_candidates(n_candidates))
            elif strategy == "curiosity":
                candidates.extend(self._curiosity_candidates(n_candidates))
            elif strategy == "meta":
                candidates.extend(self._meta_candidates(n_candidates, context))

        # Score and rank candidates
        for candidate in candidates:
            candidate.expected_improvement = self._predict_improvement(
                candidate, current_performance
            )

        # Sort by net expected value
        candidates.sort(key=lambda m: m.net_expected_value, reverse=True)

        self._current_generation = candidates[: self.params.n_candidates]
        self._generation_count += 1

        return self._current_generation

    def _gradient_candidates(
        self,
        n: int,
        context: Optional[Dict[str, Any]],
    ) -> List[Modification]:
        """Generate candidates by following performance gradients."""
        candidates = []

        for _ in range(n):
            # Pick random component
            if not self._components:
                continue

            component = np.random.choice(list(self._components.keys()))
            comp_info = self._components[component]

            # Propose weight adjustment
            mod = Modification(
                mod_id=self._next_id(),
                mod_type=ModificationType.WEIGHT_ADJUSTMENT,
                target_component=component,
                changes={
                    "adjustment_scale": np.random.randn() * 0.1,
                    "target_params": comp_info.get("adjustable_params", ["weights"]),
                    "direction": "gradient" if context else "random",
                },
                expected_improvement=0.0,  # Will be computed
                confidence=0.6,
                complexity_cost=0.01,
                reversible=True,
            )
            candidates.append(mod)

        return candidates

    def _evolutionary_candidates(self, n: int) -> List[Modification]:
        """Generate candidates via evolutionary operators."""
        candidates = []

        # Get successful past modifications
        successful = [(mod, outcome) for mod, outcome in self._modification_history if outcome > 0]

        for _ in range(n):
            if successful and np.random.rand() < self.params.crossover_rate:
                # Crossover: combine two successful modifications
                parent1, _ = successful[np.random.randint(len(successful))]
                parent2, _ = successful[np.random.randint(len(successful))]
                mod = self._crossover(parent1, parent2)
            elif successful and np.random.rand() < 1 - self.params.mutation_rate:
                # Mutate: modify a successful modification
                parent, _ = successful[np.random.randint(len(successful))]
                mod = self._mutate(parent)
            else:
                # Random: generate new random modification
                mod = self._random_modification()

            candidates.append(mod)

        return candidates

    def _curiosity_candidates(self, n: int) -> List[Modification]:
        """Generate candidates for underexplored regions."""
        candidates = []

        # Find underexplored components
        component_counts = {}
        for mod, _ in self._modification_history:
            component_counts[mod.target_component] = (
                component_counts.get(mod.target_component, 0) + 1
            )

        # Components with fewest modifications
        unexplored = sorted(self._components.keys(), key=lambda c: component_counts.get(c, 0))

        for i in range(n):
            if unexplored:
                component = unexplored[i % len(unexplored)]
            elif self._components:
                component = np.random.choice(list(self._components.keys()))
            else:
                continue

            mod = Modification(
                mod_id=self._next_id(),
                mod_type=np.random.choice(list(ModificationType)),
                target_component=component,
                changes={
                    "exploration": True,
                    "novelty_seed": np.random.randint(10000),
                },
                expected_improvement=0.0,
                confidence=0.3,  # Lower confidence for exploration
                complexity_cost=0.05,
                reversible=True,
            )
            # Add novelty bonus
            mod.expected_improvement = self.params.novelty_bonus
            candidates.append(mod)

        return candidates

    def _meta_candidates(
        self,
        n: int,
        context: Optional[Dict[str, Any]],
    ) -> List[Modification]:
        """Generate candidates using learned meta-heuristics."""
        candidates = []

        # Use successful patterns from history
        for pattern in self._successful_patterns[:n]:
            mod = Modification(
                mod_id=self._next_id(),
                mod_type=ModificationType(pattern.get("type", "weight_adjustment")),
                target_component=pattern.get("component", "unknown"),
                changes=pattern.get("changes", {}),
                expected_improvement=pattern.get("avg_improvement", 0.1),
                confidence=0.7,
                complexity_cost=pattern.get("complexity", 0.02),
                reversible=True,
            )
            candidates.append(mod)

        # Fill remaining with variations
        while len(candidates) < n:
            if self._successful_patterns:
                pattern = np.random.choice(self._successful_patterns)
                mod = self._vary_pattern(pattern)
            else:
                mod = self._random_modification()
            candidates.append(mod)

        return candidates

    def _crossover(self, parent1: Modification, parent2: Modification) -> Modification:
        """Combine two modifications."""
        # Mix changes from both parents
        changes = {}
        all_keys = set(parent1.changes.keys()) | set(parent2.changes.keys())

        for key in all_keys:
            if key in parent1.changes and key in parent2.changes:
                # Average numeric values, random choice for others
                v1, v2 = parent1.changes[key], parent2.changes[key]
                if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                    changes[key] = (v1 + v2) / 2
                else:
                    changes[key] = v1 if np.random.rand() < 0.5 else v2
            elif key in parent1.changes:
                changes[key] = parent1.changes[key]
            else:
                changes[key] = parent2.changes[key]

        return Modification(
            mod_id=self._next_id(),
            mod_type=parent1.mod_type if np.random.rand() < 0.5 else parent2.mod_type,
            target_component=parent1.target_component
            if np.random.rand() < 0.5
            else parent2.target_component,
            changes=changes,
            expected_improvement=0.0,
            confidence=(parent1.confidence + parent2.confidence) / 2,
            complexity_cost=(parent1.complexity_cost + parent2.complexity_cost) / 2,
            reversible=parent1.reversible and parent2.reversible,
            parent_mods=[parent1.mod_id, parent2.mod_id],
        )

    def _mutate(self, parent: Modification) -> Modification:
        """Mutate a modification."""
        changes = copy.deepcopy(parent.changes)

        # Randomly modify one change
        if changes:
            key = np.random.choice(list(changes.keys()))
            if isinstance(changes[key], (int, float)):
                changes[key] *= 1 + np.random.randn() * 0.2
            elif isinstance(changes[key], bool):
                changes[key] = not changes[key]

        return Modification(
            mod_id=self._next_id(),
            mod_type=parent.mod_type,
            target_component=parent.target_component,
            changes=changes,
            expected_improvement=0.0,
            confidence=parent.confidence * 0.9,
            complexity_cost=parent.complexity_cost,
            reversible=parent.reversible,
            parent_mods=[parent.mod_id],
        )

    def _random_modification(self) -> Modification:
        """Generate a random modification."""
        component = (
            np.random.choice(list(self._components.keys())) if self._components else "system"
        )
        mod_type = np.random.choice(list(ModificationType))

        return Modification(
            mod_id=self._next_id(),
            mod_type=mod_type,
            target_component=component,
            changes={
                "random_seed": np.random.randint(10000),
                "magnitude": np.random.rand(),
            },
            expected_improvement=0.0,
            confidence=0.3,
            complexity_cost=0.05,
            reversible=True,
        )

    def _vary_pattern(self, pattern: Dict[str, Any]) -> Modification:
        """Create variation of a learned pattern."""
        changes = copy.deepcopy(pattern.get("changes", {}))

        # Add small variation
        for key in changes:
            if isinstance(changes[key], (int, float)):
                changes[key] *= 1 + np.random.randn() * 0.1

        return Modification(
            mod_id=self._next_id(),
            mod_type=ModificationType(pattern.get("type", "weight_adjustment")),
            target_component=pattern.get("component", "unknown"),
            changes=changes,
            expected_improvement=pattern.get("avg_improvement", 0.1) * 0.9,
            confidence=0.6,
            complexity_cost=pattern.get("complexity", 0.02),
            reversible=True,
        )

    def _predict_improvement(
        self,
        mod: Modification,
        current_performance: float,
    ) -> float:
        """Predict expected improvement from a modification."""
        # Base prediction from similar past modifications
        similar_mods = [
            (m, o)
            for m, o in self._modification_history
            if m.mod_type == mod.mod_type and m.target_component == mod.target_component
        ]

        if similar_mods:
            avg_outcome = np.mean([o for _, o in similar_mods])
            return float(avg_outcome)

        # Default: small positive expectation
        return 0.05 * (1 - current_performance)  # More room for improvement if performance is low

    def record_outcome(
        self,
        mod: Modification,
        outcome: float,
    ) -> None:
        """Record the outcome of a modification."""
        self._modification_history.append((mod, outcome))

        if len(self._modification_history) > self.params.history_size:
            self._modification_history.pop(0)

        # Update successful patterns
        if outcome > 0:
            pattern = {
                "type": mod.mod_type.value,
                "component": mod.target_component,
                "changes": mod.changes,
                "avg_improvement": outcome,
                "complexity": mod.complexity_cost,
            }

            # Update or add pattern
            updated = False
            for i, p in enumerate(self._successful_patterns):
                if p["type"] == pattern["type"] and p["component"] == pattern["component"]:
                    # Running average
                    p["avg_improvement"] = 0.7 * p["avg_improvement"] + 0.3 * outcome
                    updated = True
                    break

            if not updated:
                self._successful_patterns.append(pattern)

        # Update strategy weights based on outcomes
        self._update_strategy_weights(mod, outcome)

    def _update_strategy_weights(self, mod: Modification, outcome: float) -> None:
        """Update strategy weights based on modification outcomes."""
        # Determine which strategy produced this modification
        if "gradient" in mod.changes.get("direction", ""):
            strategy = "gradient"
        elif mod.parent_mods:
            strategy = "evolutionary"
        elif mod.changes.get("exploration"):
            strategy = "curiosity"
        else:
            strategy = "meta"

        # Update weight based on outcome
        learning_rate = 0.01
        if outcome > 0:
            self._strategy_weights[strategy] += learning_rate
        else:
            self._strategy_weights[strategy] -= learning_rate * 0.5

        # Normalize weights
        total = sum(self._strategy_weights.values())
        for s in self._strategy_weights:
            self._strategy_weights[s] = max(0.05, self._strategy_weights[s] / total)

    def _next_id(self) -> str:
        """Generate unique modification ID."""
        self._mod_counter += 1
        return f"mod_{self._generation_count}_{self._mod_counter}"

    def get_statistics(self) -> Dict[str, Any]:
        """Get generator statistics."""
        if not self._modification_history:
            return {
                "n_modifications": 0,
                "success_rate": 0.0,
                "strategy_weights": self._strategy_weights,
            }

        successful = sum(1 for _, o in self._modification_history if o > 0)

        return {
            "n_modifications": len(self._modification_history),
            "success_rate": successful / len(self._modification_history),
            "avg_improvement": float(np.mean([o for _, o in self._modification_history])),
            "n_patterns": len(self._successful_patterns),
            "strategy_weights": self._strategy_weights.copy(),
            "generation_count": self._generation_count,
        }

    def reset(self) -> None:
        """Reset generator state."""
        self._modification_history = []
        self._successful_patterns = []
        self._current_generation = []
        self._generation_count = 0
        self._mod_counter = 0
