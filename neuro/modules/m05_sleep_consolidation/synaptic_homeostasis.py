"""
Synaptic Homeostasis Hypothesis Implementation

During wake: Net synaptic potentiation (learning increases weights)
During sleep (SWS): Global synaptic downscaling

Key insight: Sleep downscales ALL synapses proportionally,
preserving relative differences (signal-to-noise ratio maintained).

Some synapses can be "tagged" for protection from downscaling.
"""

import numpy as np
from typing import Set, List, Dict, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class Synapse:
    """Single synaptic connection"""

    id: int
    weight: float
    source: int  # Pre-synaptic neuron
    target: int  # Post-synaptic neuron
    is_tagged: bool = False  # Protected from downscaling
    tag_strength: float = 0.0
    last_potentiation: float = 0.0  # Time of last strengthening
    activity_count: int = 0  # Number of activations


class SynapticHomeostasis:
    """Sleep-dependent synaptic downscaling.

    Implements the Synaptic Homeostasis Hypothesis (SHY):
    - Wake: Learning causes net potentiation (weights increase)
    - Sleep: Global downscaling restores baseline
    - Result: Maintains signal-to-noise ratio
    """

    def __init__(
        self,
        n_neurons: int = 100,
        connectivity: float = 0.1,
        baseline_weight: float = 0.5,
        downscale_rate: float = 0.1,
    ):
        """Initialize synaptic homeostasis system.

        Args:
            n_neurons: Number of neurons
            connectivity: Fraction of possible connections
            baseline_weight: Target baseline synaptic weight
            downscale_rate: Rate of downscaling during sleep
        """
        self.n_neurons = n_neurons
        self.baseline_weight = baseline_weight
        self.downscale_rate = downscale_rate

        # Create synaptic connections
        self.synapses: Dict[int, Synapse] = {}
        self._create_network(connectivity)

        # For matrix operations
        self.weight_matrix = np.zeros((n_neurons, n_neurons))
        self._update_weight_matrix()

        # Energy tracking
        self.baseline_energy = self._compute_total_energy()

        # Statistics
        self.total_potentiation = 0.0
        self.total_downscaling = 0.0

    def _create_network(self, connectivity: float) -> None:
        """Create random synaptic network."""
        synapse_id = 0
        for i in range(self.n_neurons):
            for j in range(self.n_neurons):
                if i != j and np.random.random() < connectivity:
                    self.synapses[synapse_id] = Synapse(
                        id=synapse_id, weight=self.baseline_weight, source=i, target=j
                    )
                    synapse_id += 1

    def _update_weight_matrix(self) -> None:
        """Update weight matrix from synapses."""
        self.weight_matrix = np.zeros((self.n_neurons, self.n_neurons))
        for synapse in self.synapses.values():
            self.weight_matrix[synapse.source, synapse.target] = synapse.weight

    def measure_total_strength(self) -> float:
        """Measure total synaptic strength (sum of all weights)."""
        return sum(s.weight for s in self.synapses.values())

    def _compute_total_energy(self) -> float:
        """Compute metabolic energy cost of synapses.

        Energy ~ sum of squared weights (quadratic cost)
        """
        return sum(s.weight**2 for s in self.synapses.values())

    def compute_energy_cost(self) -> float:
        """Get current metabolic energy cost."""
        return self._compute_total_energy()

    def potentiate(self, synapse_id: int, amount: float, timestamp: float = 0.0) -> None:
        """Potentiate (strengthen) a synapse.

        This happens during wake/learning.

        Args:
            synapse_id: Synapse to potentiate
            amount: Amount to increase weight
            timestamp: Current time
        """
        if synapse_id not in self.synapses:
            return

        synapse = self.synapses[synapse_id]
        synapse.weight += amount
        synapse.last_potentiation = timestamp
        synapse.activity_count += 1
        self.total_potentiation += amount

        self._update_weight_matrix()

    def hebbian_potentiation(
        self,
        pre_activity: np.ndarray,
        post_activity: np.ndarray,
        learning_rate: float = 0.01,
        timestamp: float = 0.0,
    ) -> float:
        """Apply Hebbian potentiation based on neural activity.

        "Neurons that fire together, wire together"

        Args:
            pre_activity: Pre-synaptic activity vector
            post_activity: Post-synaptic activity vector
            learning_rate: Learning rate
            timestamp: Current time

        Returns:
            Total potentiation applied
        """
        total_pot = 0.0

        for synapse in self.synapses.values():
            pre = pre_activity[synapse.source]
            post = post_activity[synapse.target]

            # Hebbian: weight change = pre * post
            delta = learning_rate * pre * post

            if delta > 0:
                synapse.weight += delta
                synapse.last_potentiation = timestamp
                synapse.activity_count += 1
                total_pot += delta

        self.total_potentiation += total_pot
        self._update_weight_matrix()

        return total_pot

    def downscale(self, factor: float) -> float:
        """Globally downscale all (non-tagged) synapses.

        Args:
            factor: Downscaling factor (0-1, lower = more downscaling)

        Returns:
            Total weight reduction
        """
        total_reduction = 0.0

        for synapse in self.synapses.values():
            if not synapse.is_tagged:
                old_weight = synapse.weight
                synapse.weight *= factor
                total_reduction += old_weight - synapse.weight

        self.total_downscaling += total_reduction
        self._update_weight_matrix()

        return total_reduction

    def downscale_to_baseline(self) -> float:
        """Downscale to restore baseline energy level.

        Returns:
            Total weight reduction
        """
        current_energy = self._compute_total_energy()

        if current_energy <= self.baseline_energy:
            return 0.0

        # Calculate required downscaling factor
        factor = np.sqrt(self.baseline_energy / current_energy)

        return self.downscale(factor)

    def sleep_downscaling_step(self, dt: float = 1.0) -> float:
        """Perform one step of sleep-dependent downscaling.

        Args:
            dt: Time step

        Returns:
            Total weight reduction
        """
        factor = 1.0 - self.downscale_rate * dt
        factor = max(0.1, factor)  # Don't downscale too aggressively

        return self.downscale(factor)

    def preserve_ratios(self) -> Dict[int, float]:
        """Verify that downscaling preserves relative weight ratios.

        Returns:
            Dictionary of synapse_id -> relative_weight
        """
        total = self.measure_total_strength()
        if total < 1e-8:
            return {}

        return {sid: s.weight / total for sid, s in self.synapses.items()}

    def get_statistics(self) -> Dict:
        """Get homeostasis statistics."""
        weights = [s.weight for s in self.synapses.values()]

        return {
            "n_synapses": len(self.synapses),
            "total_strength": self.measure_total_strength(),
            "mean_weight": np.mean(weights) if weights else 0,
            "std_weight": np.std(weights) if weights else 0,
            "energy_cost": self.compute_energy_cost(),
            "baseline_energy": self.baseline_energy,
            "total_potentiation": self.total_potentiation,
            "total_downscaling": self.total_downscaling,
            "n_tagged": sum(1 for s in self.synapses.values() if s.is_tagged),
        }

    def reset_statistics(self) -> None:
        """Reset potentiation/downscaling counters."""
        self.total_potentiation = 0.0
        self.total_downscaling = 0.0


class SelectiveConsolidation:
    """Selective protection of important synapses from downscaling.

    Some synapses can be "tagged" to prevent downscaling.
    Tags are based on:
    - Recent strong activity
    - High learning (large recent potentiation)
    - Importance markers (e.g., emotional significance)
    """

    def __init__(
        self,
        homeostasis: SynapticHomeostasis,
        tag_threshold: float = 0.7,
        tag_decay_rate: float = 0.1,
    ):
        """Initialize selective consolidation.

        Args:
            homeostasis: The synaptic homeostasis system
            tag_threshold: Threshold for tagging (relative to max)
            tag_decay_rate: Rate at which tags decay
        """
        self.homeostasis = homeostasis
        self.tag_threshold = tag_threshold
        self.tag_decay_rate = tag_decay_rate

        # Track tagged synapses
        self.tagged_synapses: Set[int] = set()

    def tag_for_protection(self, synapse_id: int, strength: float = 1.0) -> bool:
        """Tag a synapse for protection from downscaling.

        Args:
            synapse_id: Synapse to tag
            strength: Tag strength (0-1)

        Returns:
            True if tagging successful
        """
        if synapse_id not in self.homeostasis.synapses:
            return False

        synapse = self.homeostasis.synapses[synapse_id]
        synapse.is_tagged = True
        synapse.tag_strength = strength
        self.tagged_synapses.add(synapse_id)

        return True

    def tag_by_activity(
        self, activity_threshold: int = 5, recency_window: float = 100.0, current_time: float = 0.0
    ) -> int:
        """Tag synapses based on recent activity.

        Args:
            activity_threshold: Minimum activity count for tagging
            recency_window: Time window for "recent" activity
            current_time: Current timestamp

        Returns:
            Number of synapses tagged
        """
        tagged_count = 0

        for synapse_id, synapse in self.homeostasis.synapses.items():
            is_recent = (current_time - synapse.last_potentiation) < recency_window
            is_active = synapse.activity_count >= activity_threshold

            if is_recent and is_active:
                self.tag_for_protection(synapse_id)
                tagged_count += 1

        return tagged_count

    def tag_by_weight(self, percentile: float = 90) -> int:
        """Tag synapses above a weight percentile.

        Args:
            percentile: Percentile threshold (0-100)

        Returns:
            Number of synapses tagged
        """
        weights = [s.weight for s in self.homeostasis.synapses.values()]
        if not weights:
            return 0

        threshold = np.percentile(weights, percentile)
        tagged_count = 0

        for synapse_id, synapse in self.homeostasis.synapses.items():
            if synapse.weight >= threshold:
                self.tag_for_protection(synapse_id)
                tagged_count += 1

        return tagged_count

    def downscale_untagged(self, factor: float) -> float:
        """Downscale only untagged synapses.

        Args:
            factor: Downscaling factor

        Returns:
            Total weight reduction
        """
        # Tags are already respected in homeostasis.downscale()
        return self.homeostasis.downscale(factor)

    def decay_tags(self, dt: float = 1.0) -> int:
        """Decay tag strengths over time.

        Args:
            dt: Time step

        Returns:
            Number of tags removed
        """
        removed_count = 0
        to_remove = []

        for synapse_id in self.tagged_synapses:
            if synapse_id not in self.homeostasis.synapses:
                to_remove.append(synapse_id)
                continue

            synapse = self.homeostasis.synapses[synapse_id]
            synapse.tag_strength -= self.tag_decay_rate * dt

            if synapse.tag_strength <= 0:
                synapse.is_tagged = False
                synapse.tag_strength = 0
                to_remove.append(synapse_id)
                removed_count += 1

        for sid in to_remove:
            self.tagged_synapses.discard(sid)

        return removed_count

    def clear_tags(self) -> None:
        """Remove all tags."""
        for synapse_id in self.tagged_synapses:
            if synapse_id in self.homeostasis.synapses:
                synapse = self.homeostasis.synapses[synapse_id]
                synapse.is_tagged = False
                synapse.tag_strength = 0

        self.tagged_synapses.clear()

    def get_tagged_count(self) -> int:
        """Get number of tagged synapses."""
        return len(self.tagged_synapses)

    def get_protected_weight(self) -> float:
        """Get total weight of protected synapses."""
        return sum(
            self.homeostasis.synapses[sid].weight
            for sid in self.tagged_synapses
            if sid in self.homeostasis.synapses
        )


class SleepWakeCycle:
    """Manages synaptic changes across sleep-wake cycles.

    Wake: Net potentiation from learning
    Sleep: Downscaling with selective protection
    """

    def __init__(self, n_neurons: int = 100, connectivity: float = 0.1):
        self.homeostasis = SynapticHomeostasis(n_neurons=n_neurons, connectivity=connectivity)
        self.consolidation = SelectiveConsolidation(self.homeostasis)

        # Cycle tracking
        self.is_awake = True
        self.cycle_count = 0

        # History
        self.weight_history: List[float] = []
        self.energy_history: List[float] = []

    def wake_learning(self, duration: float, learning_intensity: float = 0.1) -> Dict:
        """Simulate wake period with learning.

        Args:
            duration: Wake duration
            learning_intensity: How much learning occurs

        Returns:
            Statistics about the wake period
        """
        self.is_awake = True

        # Simulate learning through random activity
        n_events = int(duration * 10)
        total_pot = 0.0

        for _ in range(n_events):
            # Random neural activity
            pre = np.random.random(self.homeostasis.n_neurons) > 0.5
            post = np.random.random(self.homeostasis.n_neurons) > 0.5

            pot = self.homeostasis.hebbian_potentiation(
                pre.astype(float), post.astype(float), learning_rate=learning_intensity
            )
            total_pot += pot

        self.weight_history.append(self.homeostasis.measure_total_strength())
        self.energy_history.append(self.homeostasis.compute_energy_cost())

        return {
            "duration": duration,
            "total_potentiation": total_pot,
            "final_strength": self.homeostasis.measure_total_strength(),
        }

    def sleep_consolidation(self, duration: float, tag_percentile: float = 90) -> Dict:
        """Simulate sleep period with downscaling.

        Args:
            duration: Sleep duration
            tag_percentile: Percentile for tagging protection

        Returns:
            Statistics about the sleep period
        """
        self.is_awake = False

        # Tag important synapses for protection
        n_tagged = self.consolidation.tag_by_weight(tag_percentile)

        # Gradual downscaling
        n_steps = int(duration * 10)
        total_downscaling = 0.0

        for _ in range(n_steps):
            reduction = self.homeostasis.sleep_downscaling_step(dt=0.1)
            total_downscaling += reduction

            # Decay tags over time
            self.consolidation.decay_tags(dt=0.1)

        # Record history
        self.weight_history.append(self.homeostasis.measure_total_strength())
        self.energy_history.append(self.homeostasis.compute_energy_cost())

        # Clear remaining tags
        self.consolidation.clear_tags()
        self.cycle_count += 1

        return {
            "duration": duration,
            "n_tagged": n_tagged,
            "total_downscaling": total_downscaling,
            "final_strength": self.homeostasis.measure_total_strength(),
        }

    def run_full_cycle(
        self,
        wake_duration: float = 16.0,
        sleep_duration: float = 8.0,
        learning_intensity: float = 0.1,
    ) -> Dict:
        """Run a complete wake-sleep cycle.

        Args:
            wake_duration: Hours of wake
            sleep_duration: Hours of sleep
            learning_intensity: Learning rate during wake

        Returns:
            Combined statistics
        """
        wake_stats = self.wake_learning(wake_duration, learning_intensity)
        sleep_stats = self.sleep_consolidation(sleep_duration)

        return {
            "cycle": self.cycle_count,
            "wake": wake_stats,
            "sleep": sleep_stats,
            "net_change": sleep_stats["final_strength"] - wake_stats["final_strength"],
        }

    def get_cycle_history(self) -> Dict:
        """Get history across cycles."""
        return {
            "weight_history": self.weight_history,
            "energy_history": self.energy_history,
            "n_cycles": self.cycle_count,
        }
