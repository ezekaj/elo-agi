"""
Structural Plasticity

- Synaptic: Weight changes (seconds-hours)
- Structural: New synapses/dendrites (days-weeks)
- Functional: Region takes over function (weeks-months)
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class StructuralParams:
    """Structural plasticity parameters"""

    synapse_formation_rate: float = 0.01
    synapse_elimination_rate: float = 0.005
    activity_threshold: float = 0.1
    weight_threshold: float = 0.1
    max_synapses_per_neuron: int = 100


class StructuralPlasticity:
    """Structural plasticity - formation and elimination of synapses"""

    def __init__(self, n_neurons: int, params: Optional[StructuralParams] = None):
        self.n_neurons = n_neurons
        self.params = params or StructuralParams()

        # Connectivity matrix (binary - synapse exists or not)
        self.connectivity = np.random.rand(n_neurons, n_neurons) < 0.1
        np.fill_diagonal(self.connectivity, False)  # No self-connections

        # Weight matrix (only where connections exist)
        self.weights = np.random.rand(n_neurons, n_neurons) * 0.5
        self.weights = self.weights * self.connectivity

        # Activity history
        self.activity_history = np.zeros((n_neurons, 100))
        self.history_idx = 0

        # Synapse age (for pruning decisions)
        self.synapse_age = np.zeros((n_neurons, n_neurons))

    def record_activity(self, activity: np.ndarray) -> None:
        """Record neural activity for plasticity decisions"""
        self.activity_history[:, self.history_idx] = activity
        self.history_idx = (self.history_idx + 1) % 100

    def get_mean_activity(self) -> np.ndarray:
        """Get mean activity per neuron"""
        return np.mean(self.activity_history, axis=1)

    def form_synapses(self) -> int:
        """Form new synapses based on activity correlation

        Returns:
            Number of new synapses formed
        """
        activity = self.get_mean_activity()
        formed = 0

        for i in range(self.n_neurons):
            # Count current synapses
            n_synapses = np.sum(self.connectivity[i, :])
            if n_synapses >= self.params.max_synapses_per_neuron:
                continue

            for j in range(self.n_neurons):
                if i == j or self.connectivity[i, j]:
                    continue

                # Form synapse if both neurons are active
                if (
                    activity[i] > self.params.activity_threshold
                    and activity[j] > self.params.activity_threshold
                ):
                    if np.random.rand() < self.params.synapse_formation_rate:
                        self.connectivity[i, j] = True
                        self.weights[i, j] = 0.1  # Start weak
                        self.synapse_age[i, j] = 0
                        formed += 1

        return formed

    def eliminate_synapses(self) -> int:
        """Eliminate weak/inactive synapses

        Returns:
            Number of synapses eliminated
        """
        eliminated = 0

        for i in range(self.n_neurons):
            for j in range(self.n_neurons):
                if not self.connectivity[i, j]:
                    continue

                # Eliminate weak synapses
                if self.weights[i, j] < self.params.weight_threshold:
                    if np.random.rand() < self.params.synapse_elimination_rate:
                        self.connectivity[i, j] = False
                        self.weights[i, j] = 0
                        eliminated += 1

        return eliminated

    def update(self, activity: np.ndarray) -> Tuple[int, int]:
        """Full structural plasticity update

        Args:
            activity: Current neural activity

        Returns:
            (synapses formed, synapses eliminated)
        """
        self.record_activity(activity)

        # Age existing synapses
        self.synapse_age += self.connectivity.astype(float)

        # Form new synapses
        formed = self.form_synapses()

        # Eliminate weak synapses
        eliminated = self.eliminate_synapses()

        return formed, eliminated

    def get_connectivity_density(self) -> float:
        """Get fraction of possible connections that exist"""
        n_possible = self.n_neurons * (self.n_neurons - 1)
        n_actual = np.sum(self.connectivity)
        return n_actual / n_possible


class SynapticPruning:
    """Synaptic pruning - selective elimination of synapses

    "Use it or lose it" - inactive connections are removed.
    """

    def __init__(
        self, weights: np.ndarray, pruning_rate: float = 0.01, activity_threshold: float = 0.1
    ):
        self.weights = weights.copy()
        self.pruning_rate = pruning_rate
        self.activity_threshold = activity_threshold

        # Track synapse usage
        self.usage_count = np.zeros_like(weights)
        self.total_updates = 0

    def record_usage(self, pre_activity: np.ndarray, post_activity: np.ndarray) -> None:
        """Record which synapses were used"""
        usage = np.outer(
            post_activity > self.activity_threshold, pre_activity > self.activity_threshold
        )
        self.usage_count += usage.astype(float)
        self.total_updates += 1

    def prune(self) -> int:
        """Prune unused synapses

        Returns:
            Number of synapses pruned
        """
        if self.total_updates < 10:
            return 0

        # Calculate usage rate
        usage_rate = self.usage_count / self.total_updates

        # Prune low-usage synapses
        prune_mask = (usage_rate < 0.01) & (self.weights > 0)
        prune_mask = prune_mask & (np.random.rand(*self.weights.shape) < self.pruning_rate)

        n_pruned = np.sum(prune_mask)
        self.weights[prune_mask] = 0

        return n_pruned

    def get_weights(self) -> np.ndarray:
        """Get current weights"""
        return self.weights


class DendriticGrowth:
    """Dendritic growth - extension of dendritic tree

    Models activity-dependent dendritic growth.
    """

    def __init__(self, n_neurons: int, growth_rate: float = 0.01, retraction_rate: float = 0.005):
        self.n_neurons = n_neurons
        self.growth_rate = growth_rate
        self.retraction_rate = retraction_rate

        # Dendritic tree size per neuron (arbitrary units)
        self.dendritic_size = np.ones(n_neurons)

        # Maximum receptive field
        self.max_size = 10.0

    def update(self, activity: np.ndarray) -> np.ndarray:
        """Update dendritic sizes based on activity

        Active neurons grow dendrites, inactive retract.

        Args:
            activity: Neural activity levels

        Returns:
            Updated dendritic sizes
        """
        # Growth for active neurons
        growth = self.growth_rate * activity * (self.max_size - self.dendritic_size)

        # Retraction for inactive neurons
        retraction = self.retraction_rate * (1 - activity) * self.dendritic_size

        # Update
        self.dendritic_size = self.dendritic_size + growth - retraction
        self.dendritic_size = np.clip(self.dendritic_size, 0.1, self.max_size)

        return self.dendritic_size

    def get_receptive_field_sizes(self) -> np.ndarray:
        """Get effective receptive field sizes"""
        return self.dendritic_size
