"""
Pruning: Remove unnecessary network connections.

Implements various pruning strategies for model compression.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from abc import ABC, abstractmethod
import numpy as np


class PruningStrategy(Enum):
    """Pruning strategies."""

    MAGNITUDE = "magnitude"  # Prune smallest weights
    RANDOM = "random"  # Random pruning
    STRUCTURED = "structured"  # Prune entire structures
    GRADIENT = "gradient"  # Gradient-based importance
    SENSITIVITY = "sensitivity"  # Layer sensitivity


@dataclass
class PruningResult:
    """Result of pruning operation."""

    original_params: int
    pruned_params: int
    sparsity: float
    layer_sparsities: Dict[str, float]
    accuracy_change: Optional[float] = None


@dataclass
class PruningConfig:
    """Configuration for pruning."""

    target_sparsity: float = 0.5
    strategy: PruningStrategy = PruningStrategy.MAGNITUDE
    granularity: str = "unstructured"  # unstructured, structured, channel
    iterative: bool = False
    n_iterations: int = 1


class PruningMask:
    """
    Mask for pruned weights.
    """

    def __init__(self, shape: Tuple[int, ...]):
        self.shape = shape
        self._mask = np.ones(shape, dtype=bool)

    def apply(self, weights: np.ndarray) -> np.ndarray:
        """Apply mask to weights."""
        return weights * self._mask

    def update(self, new_mask: np.ndarray) -> None:
        """Update the mask."""
        self._mask = new_mask.astype(bool)

    @property
    def sparsity(self) -> float:
        """Get sparsity (fraction of zeros)."""
        return 1.0 - np.mean(self._mask)

    @property
    def mask(self) -> np.ndarray:
        return self._mask.copy()


class Pruner(ABC):
    """Base class for pruning methods."""

    @abstractmethod
    def compute_mask(
        self,
        weights: np.ndarray,
        sparsity: float,
    ) -> np.ndarray:
        """Compute pruning mask."""
        pass


class MagnitudePruning(Pruner):
    """
    Magnitude-based pruning.

    Prunes weights with smallest absolute values.
    """

    def compute_mask(
        self,
        weights: np.ndarray,
        sparsity: float,
    ) -> np.ndarray:
        """
        Compute mask based on weight magnitudes.

        Args:
            weights: Weight matrix
            sparsity: Target sparsity (0-1)

        Returns:
            Boolean mask (True = keep, False = prune)
        """
        if sparsity <= 0:
            return np.ones_like(weights, dtype=bool)
        if sparsity >= 1:
            return np.zeros_like(weights, dtype=bool)

        # Compute threshold
        magnitudes = np.abs(weights)
        threshold = np.percentile(magnitudes, sparsity * 100)

        # Create mask
        mask = magnitudes >= threshold

        return mask


class RandomPruning(Pruner):
    """
    Random pruning.

    Randomly prunes weights.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)

    def compute_mask(
        self,
        weights: np.ndarray,
        sparsity: float,
    ) -> np.ndarray:
        """Compute random pruning mask."""
        if sparsity <= 0:
            return np.ones_like(weights, dtype=bool)
        if sparsity >= 1:
            return np.zeros_like(weights, dtype=bool)

        mask = self.rng.random(weights.shape) >= sparsity
        return mask


class StructuredPruning(Pruner):
    """
    Structured pruning.

    Prunes entire channels or filters.
    """

    def __init__(self, axis: int = 0):
        self.axis = axis

    def compute_mask(
        self,
        weights: np.ndarray,
        sparsity: float,
    ) -> np.ndarray:
        """
        Compute structured pruning mask.

        Args:
            weights: Weight tensor (typically 4D for conv)
            sparsity: Target sparsity

        Returns:
            Boolean mask
        """
        if sparsity <= 0:
            return np.ones_like(weights, dtype=bool)
        if sparsity >= 1:
            return np.zeros_like(weights, dtype=bool)

        # Compute importance per channel
        axes = tuple(i for i in range(weights.ndim) if i != self.axis)
        importances = np.sum(np.abs(weights), axis=axes)

        # Threshold
        n_prune = int(len(importances) * sparsity)
        if n_prune == 0:
            return np.ones_like(weights, dtype=bool)

        threshold_idx = np.argsort(importances)[n_prune]
        threshold = importances[threshold_idx]

        # Create mask
        channel_mask = importances >= threshold

        # Expand to full shape
        shape = [1] * weights.ndim
        shape[self.axis] = len(channel_mask)
        expanded_mask = channel_mask.reshape(shape)

        mask = np.broadcast_to(expanded_mask, weights.shape).copy()

        return mask


class NetworkPruner:
    """
    Complete network pruning system.

    Implements:
    - Layer-wise pruning
    - Iterative pruning
    - Pruning schedules
    """

    def __init__(self, config: Optional[PruningConfig] = None):
        self.config = config or PruningConfig()

        # Pruning methods
        self._pruners: Dict[PruningStrategy, Pruner] = {
            PruningStrategy.MAGNITUDE: MagnitudePruning(),
            PruningStrategy.RANDOM: RandomPruning(),
            PruningStrategy.STRUCTURED: StructuredPruning(),
        }

        # Layer masks
        self._masks: Dict[str, PruningMask] = {}

    def prune(
        self,
        weights: Dict[str, np.ndarray],
        sparsity: Optional[float] = None,
        layer_sparsities: Optional[Dict[str, float]] = None,
    ) -> Tuple[Dict[str, np.ndarray], PruningResult]:
        """
        Prune a network.

        Args:
            weights: Dictionary of layer_name -> weight matrix
            sparsity: Global target sparsity
            layer_sparsities: Per-layer sparsities (overrides global)

        Returns:
            Tuple of (pruned_weights, result)
        """
        sparsity = sparsity or self.config.target_sparsity
        layer_sparsities = layer_sparsities or {}

        pruner = self._pruners.get(self.config.strategy, MagnitudePruning())

        original_params = sum(w.size for w in weights.values())
        pruned_weights = {}
        layer_sparsity_results = {}

        for name, weight in weights.items():
            target_sparsity = layer_sparsities.get(name, sparsity)

            # Compute mask
            mask = pruner.compute_mask(weight, target_sparsity)

            # Store mask
            if name not in self._masks:
                self._masks[name] = PruningMask(weight.shape)
            self._masks[name].update(mask)

            # Apply mask
            pruned_weights[name] = weight * mask
            layer_sparsity_results[name] = 1.0 - np.mean(mask)

        # Compute overall stats
        pruned_params = sum(np.sum(self._masks[name].mask) for name in weights)
        overall_sparsity = 1.0 - pruned_params / original_params

        result = PruningResult(
            original_params=original_params,
            pruned_params=int(pruned_params),
            sparsity=overall_sparsity,
            layer_sparsities=layer_sparsity_results,
        )

        return pruned_weights, result

    def prune_iterative(
        self,
        weights: Dict[str, np.ndarray],
        target_sparsity: float,
        n_iterations: int = 10,
    ) -> Tuple[Dict[str, np.ndarray], List[PruningResult]]:
        """
        Iteratively prune to target sparsity.

        Args:
            weights: Network weights
            target_sparsity: Final target sparsity
            n_iterations: Number of pruning iterations

        Returns:
            Final pruned weights and list of results per iteration
        """
        results = []
        current_weights = {k: v.copy() for k, v in weights.items()}

        for i in range(n_iterations):
            # Linear sparsity schedule
            iter_sparsity = target_sparsity * (i + 1) / n_iterations

            current_weights, result = self.prune(
                current_weights,
                sparsity=iter_sparsity,
            )
            results.append(result)

        return current_weights, results

    def get_mask(self, layer_name: str) -> Optional[PruningMask]:
        """Get mask for a layer."""
        return self._masks.get(layer_name)

    def apply_masks(
        self,
        weights: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Apply stored masks to weights."""
        result = {}
        for name, weight in weights.items():
            if name in self._masks:
                result[name] = self._masks[name].apply(weight)
            else:
                result[name] = weight.copy()
        return result

    def reset_masks(self) -> None:
        """Reset all masks."""
        self._masks.clear()

    def statistics(self) -> Dict[str, Any]:
        """Get pruner statistics."""
        return {
            "strategy": self.config.strategy.value,
            "target_sparsity": self.config.target_sparsity,
            "n_layers_masked": len(self._masks),
            "layer_sparsities": {name: mask.sparsity for name, mask in self._masks.items()},
        }
