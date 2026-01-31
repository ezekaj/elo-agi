"""
Contribution Accounting with Shapley Values

Implements fair credit assignment using game-theoretic methods:
- Shapley value computation for module contributions
- Reward distribution across modules
- Performance tracking
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable, Set
from itertools import permutations, combinations
import math
import numpy as np


@dataclass
class ShapleyConfig:
    """Configuration for Shapley value computation."""
    use_approximation: bool = True
    num_samples: int = 100
    min_contribution: float = 1e-6
    track_history: bool = True


@dataclass
class Contribution:
    """A module's contribution record."""
    module_id: str
    shapley_value: float
    marginal_contributions: List[float]
    coalition_values: Dict[str, float]
    timestamp: int


class ContributionAccountant:
    """
    Tracks and computes module contributions using Shapley values.

    Shapley values provide a fair way to distribute credit:
    - Efficiency: Total credit equals total outcome
    - Symmetry: Equal contributors get equal credit
    - Null player: Non-contributors get zero credit
    - Additivity: Credit is additive across outcomes
    """

    def __init__(
        self,
        config: Optional[ShapleyConfig] = None,
        random_seed: Optional[int] = None,
    ):
        self.config = config or ShapleyConfig()
        self._rng = np.random.default_rng(random_seed)

        self._module_values: Dict[str, List[float]] = {}
        self._contribution_history: List[Contribution] = []
        self._cumulative_contributions: Dict[str, float] = {}

        self._timestep = 0
        self._total_computations = 0

    def compute_shapley_values(
        self,
        outcome: float,
        active_modules: List[str],
        value_function: Callable[[Set[str]], float],
    ) -> Dict[str, float]:
        """
        Compute Shapley values for each module.

        Args:
            outcome: Total outcome value
            active_modules: List of active module IDs
            value_function: Function mapping coalition to value

        Returns:
            Dict mapping module_id to Shapley value
        """
        if not active_modules:
            return {}

        n = len(active_modules)

        if self.config.use_approximation and n > 5:
            shapley_values = self._approximate_shapley(
                active_modules, value_function
            )
        else:
            shapley_values = self._exact_shapley(
                active_modules, value_function
            )

        total_shapley = sum(shapley_values.values())
        if abs(total_shapley) > 1e-8 and abs(total_shapley - outcome) > 1e-4:
            scale = outcome / total_shapley
            shapley_values = {k: v * scale for k, v in shapley_values.items()}

        for module_id, value in shapley_values.items():
            if module_id not in self._cumulative_contributions:
                self._cumulative_contributions[module_id] = 0.0
            self._cumulative_contributions[module_id] += value

        self._total_computations += 1

        return shapley_values

    def _exact_shapley(
        self,
        modules: List[str],
        value_function: Callable[[Set[str]], float],
    ) -> Dict[str, float]:
        """Compute exact Shapley values (exponential in number of modules)."""
        n = len(modules)
        shapley_values = {m: 0.0 for m in modules}

        for perm in permutations(modules):
            coalition = set()
            for module in perm:
                v_with = value_function(coalition | {module})
                v_without = value_function(coalition)
                marginal = v_with - v_without
                shapley_values[module] += marginal
                coalition.add(module)

        num_perms = math.factorial(n)
        shapley_values = {k: v / num_perms for k, v in shapley_values.items()}

        return shapley_values

    def _approximate_shapley(
        self,
        modules: List[str],
        value_function: Callable[[Set[str]], float],
    ) -> Dict[str, float]:
        """Approximate Shapley values using sampling."""
        shapley_values = {m: 0.0 for m in modules}

        for _ in range(self.config.num_samples):
            perm = list(modules)
            self._rng.shuffle(perm)

            coalition = set()
            for module in perm:
                v_with = value_function(coalition | {module})
                v_without = value_function(coalition)
                marginal = v_with - v_without
                shapley_values[module] += marginal
                coalition.add(module)

        shapley_values = {
            k: v / self.config.num_samples
            for k, v in shapley_values.items()
        }

        return shapley_values

    def distribute_reward(
        self,
        reward: float,
        active_modules: List[str],
        module_outputs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        Distribute reward across modules using simple proportional method.

        For fast distribution without full Shapley computation.

        Args:
            reward: Reward to distribute
            active_modules: List of active modules
            module_outputs: Optional dict of module outputs for weighting

        Returns:
            Dict mapping module_id to reward share
        """
        if not active_modules:
            return {}

        if module_outputs is None:
            share = reward / len(active_modules)
            distribution = {m: share for m in active_modules}

            for module_id, value in distribution.items():
                if module_id not in self._cumulative_contributions:
                    self._cumulative_contributions[module_id] = 0.0
                self._cumulative_contributions[module_id] += value

            return distribution

        weights = {}
        for module_id in active_modules:
            output = module_outputs.get(module_id)
            if output is None:
                weights[module_id] = 1.0
            elif isinstance(output, (int, float)):
                weights[module_id] = abs(output) + 1.0
            elif isinstance(output, np.ndarray):
                weights[module_id] = np.linalg.norm(output) + 1.0
            else:
                weights[module_id] = 1.0

        total_weight = sum(weights.values())
        if total_weight < 1e-8:
            share = reward / len(active_modules)
            return {m: share for m in active_modules}

        distribution = {
            m: reward * w / total_weight
            for m, w in weights.items()
        }

        for module_id, value in distribution.items():
            if module_id not in self._cumulative_contributions:
                self._cumulative_contributions[module_id] = 0.0
            self._cumulative_contributions[module_id] += value

        return distribution

    def record_contribution(
        self,
        module_id: str,
        shapley_value: float,
        marginal_contributions: Optional[List[float]] = None,
        coalition_values: Optional[Dict[str, float]] = None,
    ) -> Contribution:
        """Record a contribution for history tracking."""
        contribution = Contribution(
            module_id=module_id,
            shapley_value=shapley_value,
            marginal_contributions=marginal_contributions or [],
            coalition_values=coalition_values or {},
            timestamp=self._timestep,
        )

        if self.config.track_history:
            self._contribution_history.append(contribution)

        if module_id not in self._module_values:
            self._module_values[module_id] = []
        self._module_values[module_id].append(shapley_value)

        self._timestep += 1

        return contribution

    def identify_underperforming_modules(
        self,
        threshold: float = 0.0,
        window: int = 100,
    ) -> List[str]:
        """
        Identify modules with consistently low contributions.

        Args:
            threshold: Minimum acceptable average contribution
            window: Number of recent contributions to consider

        Returns:
            List of underperforming module IDs
        """
        underperforming = []

        for module_id, values in self._module_values.items():
            recent = values[-window:] if len(values) > window else values
            if not recent:
                continue

            avg_contribution = np.mean(recent)
            if avg_contribution < threshold:
                underperforming.append(module_id)

        return underperforming

    def identify_top_contributors(
        self,
        n: int = 5,
        window: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """
        Identify top contributing modules.

        Args:
            n: Number of top contributors to return
            window: Optional window of recent contributions

        Returns:
            List of (module_id, average_contribution) tuples
        """
        averages = []

        for module_id, values in self._module_values.items():
            if window:
                recent = values[-window:] if len(values) > window else values
            else:
                recent = values

            if recent:
                avg = np.mean(recent)
                averages.append((module_id, avg))

        averages.sort(key=lambda x: x[1], reverse=True)
        return averages[:n]

    def get_cumulative_contributions(self) -> Dict[str, float]:
        """Get cumulative contributions per module."""
        return dict(self._cumulative_contributions)

    def get_contribution_history(
        self,
        module_id: Optional[str] = None,
        n: Optional[int] = None,
    ) -> List[Contribution]:
        """Get contribution history, optionally filtered."""
        history = self._contribution_history

        if module_id:
            history = [c for c in history if c.module_id == module_id]

        if n:
            history = history[-n:]

        return history

    def compute_contribution_variance(
        self,
        module_id: str,
    ) -> float:
        """Compute variance in module's contributions."""
        values = self._module_values.get(module_id, [])
        if len(values) < 2:
            return 0.0
        return float(np.var(values))

    def reset(self) -> None:
        """Reset all tracking."""
        self._module_values.clear()
        self._contribution_history.clear()
        self._cumulative_contributions.clear()
        self._timestep = 0

    def statistics(self) -> Dict[str, Any]:
        """Get accounting statistics."""
        total_modules = len(self._module_values)
        total_contributions = sum(len(v) for v in self._module_values.values())

        return {
            "total_modules": total_modules,
            "total_contributions": total_contributions,
            "total_computations": self._total_computations,
            "history_size": len(self._contribution_history),
            "cumulative_contributions": self._cumulative_contributions,
            "top_contributors": self.identify_top_contributors(3),
            "timestep": self._timestep,
        }
