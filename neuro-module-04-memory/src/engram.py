"""
Engram: Physical memory trace

Based on research showing memories are stored as patterns of
synaptic connections that can be reactivated.

Key principle: Hebbian learning - "neurons that fire together, wire together"
"""

import numpy as np
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field
from enum import Enum


class EngramState(Enum):
    """State of engram consolidation"""
    LABILE = "labile"           # Recently formed, easily modified
    CONSOLIDATED = "consolidated"  # Stabilized, resistant to change
    REACTIVATED = "reactivated"   # Retrieved, temporarily labile again


@dataclass
class Neuron:
    """Single unit in engram network"""
    id: int
    activation: float = 0.0
    threshold: float = 0.5
    connections: Dict[int, float] = field(default_factory=dict)

    def is_active(self) -> bool:
        """Check if neuron is firing"""
        return self.activation >= self.threshold

    def activate(self, input_strength: float) -> float:
        """Apply input and compute activation"""
        self.activation = np.tanh(input_strength)  # Bounded activation
        return self.activation

    def reset(self) -> None:
        """Reset activation to zero"""
        self.activation = 0.0


class Engram:
    """
    Physical trace of a memory - pattern of synaptic connections

    Implements Hebbian learning and pattern completion.
    """

    def __init__(
        self,
        n_neurons: int = 100,
        connectivity: float = 0.3,
        learning_rate: float = 0.1,
        consolidation_rate: float = 0.05
    ):
        """
        Initialize engram network.

        Args:
            n_neurons: Number of neurons in the engram
            connectivity: Probability of connection between neurons
            learning_rate: Hebbian learning rate
            consolidation_rate: Rate of consolidation strengthening
        """
        self.n_neurons = n_neurons
        self.learning_rate = learning_rate
        self.consolidation_rate = consolidation_rate

        # Create neurons
        self.neurons: List[Neuron] = [Neuron(id=i) for i in range(n_neurons)]

        # Initialize random sparse connectivity
        self._init_connections(connectivity)

        # Engram state
        self.pattern: Optional[np.ndarray] = None
        self.strength: float = 0.0
        self.age: float = 0.0
        self.state: EngramState = EngramState.LABILE

    def _init_connections(self, connectivity: float) -> None:
        """Initialize sparse random connections"""
        for neuron in self.neurons:
            for other in self.neurons:
                if neuron.id != other.id and np.random.random() < connectivity:
                    # Small random initial weight
                    neuron.connections[other.id] = np.random.normal(0, 0.1)

    def encode(self, input_pattern: np.ndarray) -> None:
        """
        Create engram by co-activating neurons.

        Args:
            input_pattern: Activation pattern to encode (should be n_neurons length)
        """
        # Normalize input
        if len(input_pattern) != self.n_neurons:
            # Resize pattern to match neuron count
            input_pattern = np.interp(
                np.linspace(0, 1, self.n_neurons),
                np.linspace(0, 1, len(input_pattern)),
                input_pattern
            )

        self.pattern = input_pattern.copy()

        # Activate neurons according to pattern
        for i, neuron in enumerate(self.neurons):
            neuron.activate(input_pattern[i])

        # Apply Hebbian learning
        self.hebbian_update(self.learning_rate)

        # Set state
        self.state = EngramState.LABILE
        self.strength = 0.3  # Initial strength

    def hebbian_update(self, learning_rate: float) -> None:
        """
        Neurons that fire together, wire together.

        Updates connection weights based on co-activation.
        """
        for pre in self.neurons:
            if not pre.is_active():
                continue

            for post_id, weight in list(pre.connections.items()):
                post = self.neurons[post_id]
                if post.is_active():
                    # Strengthen connection between co-active neurons
                    delta = learning_rate * pre.activation * post.activation
                    pre.connections[post_id] = np.clip(weight + delta, -1.0, 1.0)
                else:
                    # Slight weakening for non-co-active (anti-Hebbian)
                    delta = -learning_rate * 0.1 * pre.activation
                    pre.connections[post_id] = np.clip(weight + delta, -1.0, 1.0)

    def reactivate(self, cue: np.ndarray, iterations: int = 10) -> np.ndarray:
        """
        Trigger recall by partial pattern activation.

        Uses pattern completion via spreading activation.

        Args:
            cue: Partial activation pattern
            iterations: Number of settling iterations

        Returns:
            Reconstructed pattern
        """
        # Normalize cue
        if len(cue) != self.n_neurons:
            cue = np.interp(
                np.linspace(0, 1, self.n_neurons),
                np.linspace(0, 1, len(cue)),
                cue
            )

        # Initialize activations from cue
        for i, neuron in enumerate(self.neurons):
            neuron.activate(cue[i])

        # Iterate to settle into attractor
        for _ in range(iterations):
            new_activations = np.zeros(self.n_neurons)

            for i, neuron in enumerate(self.neurons):
                # Sum weighted inputs from connected neurons
                total_input = 0.0
                for pre in self.neurons:
                    if i in pre.connections:
                        total_input += pre.activation * pre.connections[i]

                # Mix with external cue
                external = cue[i] * 0.3
                new_activations[i] = np.tanh(total_input + external)

            # Update activations
            for i, neuron in enumerate(self.neurons):
                neuron.activation = new_activations[i]

        # Mark as reactivated (labile)
        self.state = EngramState.REACTIVATED

        # Return reconstructed pattern
        return np.array([n.activation for n in self.neurons])

    def consolidate(self) -> None:
        """
        Strengthen connections, change state to consolidated.

        Simulates offline consolidation during sleep.
        """
        if self.pattern is not None:
            # Replay the pattern
            for i, neuron in enumerate(self.neurons):
                neuron.activate(self.pattern[i])

            # Strengthen with consolidation rate
            self.hebbian_update(self.consolidation_rate)

        # Update state
        self.strength = min(1.0, self.strength + 0.1)
        self.state = EngramState.CONSOLIDATED

    def destabilize(self) -> None:
        """
        Prepare for reconsolidation (memory becomes labile).

        Called when memory is retrieved.
        """
        self.state = EngramState.REACTIVATED

    def restabilize(self, modified_pattern: Optional[np.ndarray] = None) -> None:
        """
        Update engram with modifications and re-consolidate.

        Args:
            modified_pattern: Optional new pattern to blend with original
        """
        if modified_pattern is not None and self.pattern is not None:
            # Blend original and modified patterns
            blend_factor = 0.3  # How much modification to incorporate
            self.pattern = (1 - blend_factor) * self.pattern + blend_factor * modified_pattern

            # Re-encode the blended pattern
            for i, neuron in enumerate(self.neurons):
                neuron.activate(self.pattern[i])

            self.hebbian_update(self.learning_rate)

        self.consolidate()

    def prune(self, threshold: float = 0.05) -> int:
        """
        Remove weak connections (forgetting).

        Args:
            threshold: Minimum connection weight to keep

        Returns:
            Number of connections pruned
        """
        pruned_count = 0

        for neuron in self.neurons:
            original_count = len(neuron.connections)
            neuron.connections = {
                k: v for k, v in neuron.connections.items()
                if abs(v) > threshold
            }
            pruned_count += original_count - len(neuron.connections)

        # Weaken overall engram
        self.strength = max(0.0, self.strength - 0.1)

        return pruned_count

    def get_pattern(self) -> Optional[np.ndarray]:
        """Get the stored pattern"""
        return self.pattern.copy() if self.pattern is not None else None

    def get_current_activation(self) -> np.ndarray:
        """Get current activation state of all neurons"""
        return np.array([n.activation for n in self.neurons])

    def similarity(self, other_pattern: np.ndarray) -> float:
        """
        Compute similarity between stored pattern and another pattern.

        Args:
            other_pattern: Pattern to compare

        Returns:
            Cosine similarity (-1 to 1)
        """
        if self.pattern is None:
            return 0.0

        # Resize if needed
        if len(other_pattern) != self.n_neurons:
            other_pattern = np.interp(
                np.linspace(0, 1, self.n_neurons),
                np.linspace(0, 1, len(other_pattern)),
                other_pattern
            )

        # Cosine similarity
        dot = np.dot(self.pattern, other_pattern)
        norm = np.linalg.norm(self.pattern) * np.linalg.norm(other_pattern)

        if norm == 0:
            return 0.0

        return dot / norm

    def reset_activations(self) -> None:
        """Reset all neuron activations to zero"""
        for neuron in self.neurons:
            neuron.reset()

    def age_step(self, dt: float) -> None:
        """Advance engram age"""
        self.age += dt

    def is_consolidated(self) -> bool:
        """Check if engram is consolidated"""
        return self.state == EngramState.CONSOLIDATED

    def is_labile(self) -> bool:
        """Check if engram is labile (can be modified)"""
        return self.state in (EngramState.LABILE, EngramState.REACTIVATED)

    def get_connection_stats(self) -> Dict[str, float]:
        """Get statistics about connections"""
        all_weights = []
        for neuron in self.neurons:
            all_weights.extend(neuron.connections.values())

        if not all_weights:
            return {"mean": 0, "std": 0, "count": 0}

        return {
            "mean": np.mean(all_weights),
            "std": np.std(all_weights),
            "count": len(all_weights),
            "positive": sum(1 for w in all_weights if w > 0),
            "negative": sum(1 for w in all_weights if w < 0)
        }
