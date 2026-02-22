"""
Aggregator: Combine results from distributed workers.

Implements various aggregation strategies for distributed results.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Union
from enum import Enum
from abc import ABC, abstractmethod
import numpy as np


class AggregationStrategy(Enum):
    """Aggregation strategies."""

    FIRST = "first"  # Take first result
    LAST = "last"  # Take last result
    MEAN = "mean"  # Average numeric results
    WEIGHTED_MEAN = "weighted_mean"  # Weighted average
    VOTE = "vote"  # Majority voting
    CONCAT = "concat"  # Concatenate results
    CUSTOM = "custom"  # Custom function


@dataclass
class AggregatedResult:
    """Result of aggregation."""

    value: Any
    strategy: AggregationStrategy
    n_inputs: int
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class Aggregator(ABC):
    """Base class for aggregators."""

    @abstractmethod
    def aggregate(self, results: List[Any]) -> Any:
        """Aggregate results."""
        pass


class WeightedAverage(Aggregator):
    """
    Weighted average aggregation.
    """

    def __init__(
        self,
        weights: Optional[List[float]] = None,
        normalize: bool = True,
    ):
        self.weights = weights
        self.normalize = normalize

    def aggregate(
        self,
        results: List[Any],
        weights: Optional[List[float]] = None,
    ) -> AggregatedResult:
        """
        Compute weighted average of results.

        Args:
            results: List of numeric results or arrays
            weights: Optional weights for each result

        Returns:
            AggregatedResult with averaged value
        """
        if not results:
            return AggregatedResult(
                value=None,
                strategy=AggregationStrategy.WEIGHTED_MEAN,
                n_inputs=0,
                confidence=0.0,
            )

        weights = weights or self.weights
        if weights is None:
            weights = [1.0] * len(results)

        if len(weights) != len(results):
            weights = [1.0] * len(results)

        # Normalize weights
        if self.normalize:
            total = sum(weights)
            if total > 0:
                weights = [w / total for w in weights]

        # Convert to arrays if needed
        try:
            arrays = [np.asarray(r) for r in results]
            weighted_sum = sum(w * arr for w, arr in zip(weights, arrays))

            # Compute confidence based on agreement
            if len(arrays) > 1:
                variance = np.var([arr for arr in arrays], axis=0)
                confidence = 1.0 / (1.0 + float(np.mean(variance)))
            else:
                confidence = 1.0

            return AggregatedResult(
                value=weighted_sum,
                strategy=AggregationStrategy.WEIGHTED_MEAN,
                n_inputs=len(results),
                confidence=confidence,
            )
        except:
            # Fall back to first result
            return AggregatedResult(
                value=results[0],
                strategy=AggregationStrategy.FIRST,
                n_inputs=len(results),
                confidence=0.5,
            )


class Voting(Aggregator):
    """
    Majority voting aggregation.
    """

    def __init__(
        self,
        min_agreement: float = 0.5,
    ):
        self.min_agreement = min_agreement

    def aggregate(
        self,
        results: List[Any],
        weights: Optional[List[float]] = None,
    ) -> AggregatedResult:
        """
        Aggregate by majority vote.

        Args:
            results: List of results (hashable)
            weights: Optional weights for votes

        Returns:
            AggregatedResult with majority value
        """
        if not results:
            return AggregatedResult(
                value=None,
                strategy=AggregationStrategy.VOTE,
                n_inputs=0,
                confidence=0.0,
            )

        weights = weights or [1.0] * len(results)

        # Count weighted votes
        votes: Dict[Any, float] = {}
        for result, weight in zip(results, weights):
            # Handle unhashable types
            key = str(result) if not isinstance(result, (str, int, float, bool, tuple)) else result
            votes[key] = votes.get(key, 0.0) + weight

        # Find winner
        total_votes = sum(votes.values())
        winner = max(votes, key=votes.get)
        winner_votes = votes[winner]

        # Compute confidence
        confidence = winner_votes / total_votes if total_votes > 0 else 0.0

        # Map back to original value
        original_winner = None
        for result in results:
            key = str(result) if not isinstance(result, (str, int, float, bool, tuple)) else result
            if key == winner:
                original_winner = result
                break

        return AggregatedResult(
            value=original_winner,
            strategy=AggregationStrategy.VOTE,
            n_inputs=len(results),
            confidence=confidence,
            metadata={"votes": dict(votes), "agreement": confidence},
        )


class ConcatAggregator(Aggregator):
    """
    Concatenation aggregation.
    """

    def __init__(
        self,
        axis: int = 0,
    ):
        self.axis = axis

    def aggregate(self, results: List[Any]) -> AggregatedResult:
        """
        Concatenate results.

        Args:
            results: List of array-like results

        Returns:
            AggregatedResult with concatenated value
        """
        if not results:
            return AggregatedResult(
                value=None,
                strategy=AggregationStrategy.CONCAT,
                n_inputs=0,
                confidence=0.0,
            )

        try:
            # Try numpy concatenation
            arrays = [np.asarray(r) for r in results]
            concatenated = np.concatenate(arrays, axis=self.axis)

            return AggregatedResult(
                value=concatenated,
                strategy=AggregationStrategy.CONCAT,
                n_inputs=len(results),
                confidence=1.0,
            )
        except:
            # Fall back to list concatenation
            if all(isinstance(r, list) for r in results):
                concatenated = []
                for r in results:
                    concatenated.extend(r)
            else:
                concatenated = list(results)

            return AggregatedResult(
                value=concatenated,
                strategy=AggregationStrategy.CONCAT,
                n_inputs=len(results),
                confidence=1.0,
            )


class ResultAggregator:
    """
    Complete result aggregation system.

    Supports multiple strategies and custom aggregation functions.
    """

    def __init__(
        self,
        default_strategy: AggregationStrategy = AggregationStrategy.MEAN,
    ):
        self.default_strategy = default_strategy

        # Built-in aggregators
        self._aggregators: Dict[AggregationStrategy, Aggregator] = {
            AggregationStrategy.WEIGHTED_MEAN: WeightedAverage(),
            AggregationStrategy.VOTE: Voting(),
            AggregationStrategy.CONCAT: ConcatAggregator(),
        }

        # Custom aggregators
        self._custom_aggregators: Dict[str, Callable] = {}

    def register_custom(
        self,
        name: str,
        aggregator_fn: Callable[[List[Any]], Any],
    ) -> None:
        """Register a custom aggregation function."""
        self._custom_aggregators[name] = aggregator_fn

    def aggregate(
        self,
        results: List[Any],
        strategy: Optional[AggregationStrategy] = None,
        weights: Optional[List[float]] = None,
        custom_name: Optional[str] = None,
    ) -> AggregatedResult:
        """
        Aggregate results using specified strategy.

        Args:
            results: List of results to aggregate
            strategy: Aggregation strategy
            weights: Optional weights for weighted strategies
            custom_name: Name of custom aggregator

        Returns:
            AggregatedResult
        """
        strategy = strategy or self.default_strategy

        if not results:
            return AggregatedResult(
                value=None,
                strategy=strategy,
                n_inputs=0,
                confidence=0.0,
            )

        # Simple strategies
        if strategy == AggregationStrategy.FIRST:
            return AggregatedResult(
                value=results[0],
                strategy=strategy,
                n_inputs=len(results),
                confidence=1.0 if len(results) == 1 else 0.5,
            )

        if strategy == AggregationStrategy.LAST:
            return AggregatedResult(
                value=results[-1],
                strategy=strategy,
                n_inputs=len(results),
                confidence=1.0 if len(results) == 1 else 0.5,
            )

        if strategy == AggregationStrategy.MEAN:
            try:
                arrays = [np.asarray(r) for r in results]
                mean_value = np.mean(arrays, axis=0)
                variance = np.var(arrays, axis=0)
                confidence = 1.0 / (1.0 + float(np.mean(variance)))

                return AggregatedResult(
                    value=mean_value,
                    strategy=strategy,
                    n_inputs=len(results),
                    confidence=confidence,
                )
            except:
                return self.aggregate(results, AggregationStrategy.FIRST)

        # Aggregator-based strategies
        if strategy in self._aggregators:
            aggregator = self._aggregators[strategy]
            if hasattr(aggregator, "aggregate"):
                if strategy in [AggregationStrategy.WEIGHTED_MEAN, AggregationStrategy.VOTE]:
                    return aggregator.aggregate(results, weights)
                return aggregator.aggregate(results)

        # Custom aggregator
        if strategy == AggregationStrategy.CUSTOM and custom_name:
            if custom_name in self._custom_aggregators:
                custom_fn = self._custom_aggregators[custom_name]
                try:
                    value = custom_fn(results)
                    return AggregatedResult(
                        value=value,
                        strategy=strategy,
                        n_inputs=len(results),
                        confidence=1.0,
                        metadata={"custom_name": custom_name},
                    )
                except Exception as e:
                    return AggregatedResult(
                        value=None,
                        strategy=strategy,
                        n_inputs=len(results),
                        confidence=0.0,
                        metadata={"error": str(e)},
                    )

        # Fallback
        return self.aggregate(results, AggregationStrategy.FIRST)

    def aggregate_dict(
        self,
        results: List[Dict[str, Any]],
        key_strategies: Optional[Dict[str, AggregationStrategy]] = None,
    ) -> Dict[str, AggregatedResult]:
        """
        Aggregate dictionary results with per-key strategies.

        Args:
            results: List of dictionaries
            key_strategies: Strategy for each key

        Returns:
            Dictionary of aggregated results by key
        """
        if not results:
            return {}

        key_strategies = key_strategies or {}
        aggregated = {}

        # Get all keys
        all_keys = set()
        for r in results:
            all_keys.update(r.keys())

        # Aggregate each key
        for key in all_keys:
            values = [r.get(key) for r in results if key in r]
            strategy = key_strategies.get(key, self.default_strategy)
            aggregated[key] = self.aggregate(values, strategy)

        return aggregated

    def statistics(self) -> Dict[str, Any]:
        """Get aggregator statistics."""
        return {
            "default_strategy": self.default_strategy.value,
            "n_custom_aggregators": len(self._custom_aggregators),
            "available_strategies": [s.value for s in AggregationStrategy],
        }
