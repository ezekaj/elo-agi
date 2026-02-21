"""
Multi-Timescale Temporal Dynamics

Implements the temporal hierarchy of predictive processing:
- Lower layers: fast timescales (~10ms sensory processing)
- Higher layers: slow timescales (seconds to minutes for abstract concepts)

This temporal separation emerges from different time constants at each level.
"""

import numpy as np
from typing import List, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum


class DecayType(Enum):
    """Types of temporal decay"""
    EXPONENTIAL = "exponential"
    POWER_LAW = "power_law"
    OSCILLATORY = "oscillatory"


@dataclass
class TemporalState:
    """State with temporal dynamics"""
    value: np.ndarray
    timescale: float
    age: float = 0.0
    decay_rate: float = 0.1


class TemporalLayer:
    """Single layer with characteristic timescale.

    Implements leaky integration with time constant τ:
    τ(dx/dt) = -x + input
    """

    def __init__(
        self,
        dim: int,
        timescale: float = 1.0,
        decay_rate: float = 0.1,
        nonlinearity: Callable = np.tanh
    ):
        """Initialize temporal layer.

        Args:
            dim: Dimensionality of the layer
            timescale: Time constant τ (higher = slower)
            decay_rate: Rate of decay toward baseline
            nonlinearity: Activation function
        """
        self.dim = dim
        self.timescale = timescale
        self.decay_rate = decay_rate
        self.nonlinearity = nonlinearity

        # Layer state
        self.state = np.zeros(dim)
        self.baseline = np.zeros(dim)

        # Temporal buffer for integration
        self.buffer: List[np.ndarray] = []
        self.buffer_size = int(timescale * 10)  # Store ~10 time constants

        # Time tracking
        self.current_time = 0.0

    def update(self, input_signal: np.ndarray, dt: float = 0.1) -> np.ndarray:
        """Update state with new input.

        Implements leaky integration:
        τ(dx/dt) = -decay_rate * (x - baseline) + input

        Args:
            input_signal: Input to integrate
            dt: Time step

        Returns:
            Updated state
        """
        effective_dt = dt / self.timescale

        # Leaky integration toward baseline
        decay = self.decay_rate * (self.state - self.baseline)

        # State update
        self.state += effective_dt * (-decay + input_signal)
        self.state = self.nonlinearity(self.state)

        # Update buffer
        self.buffer.append(self.state.copy())
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

        self.current_time += dt
        return self.state

    def get_integrated_state(self, window: Optional[float] = None) -> np.ndarray:
        """Get temporally integrated state.

        Args:
            window: Time window for integration (default: one timescale)

        Returns:
            Integrated state over window
        """
        if window is None:
            window = self.timescale

        # Number of buffer entries to use
        n_entries = min(len(self.buffer), max(1, int(window * 10 / self.timescale)))

        if n_entries == 0:
            return self.state

        recent = np.array(self.buffer[-n_entries:])
        return np.mean(recent, axis=0)

    def get_temporal_derivative(self) -> np.ndarray:
        """Estimate temporal derivative of state"""
        if len(self.buffer) < 2:
            return np.zeros(self.dim)

        return self.buffer[-1] - self.buffer[-2]

    def reset(self) -> None:
        """Reset layer state"""
        self.state = np.zeros(self.dim)
        self.buffer = []
        self.current_time = 0.0


class TemporalHierarchy:
    """Hierarchy with multiple timescales.

    Each level operates at a different timescale:
    - Level 0: fastest (sensory, ~10ms)
    - Level 1: medium (~100ms)
    - Level 2: slow (~1s)
    - Level N: slowest (abstract concepts, minutes+)
    """

    def __init__(
        self,
        layer_dims: List[int],
        base_timescale: float = 0.01,
        timescale_factor: float = 10.0,
        decay_rate: float = 0.1
    ):
        """Initialize temporal hierarchy.

        Args:
            layer_dims: Dimensions of each layer
            base_timescale: Timescale of fastest layer (seconds)
            timescale_factor: Ratio between adjacent timescales
            decay_rate: Base decay rate
        """
        self.n_layers = len(layer_dims)
        self.layer_dims = layer_dims

        # Create layers with increasing timescales
        self.layers: List[TemporalLayer] = []
        for i, dim in enumerate(layer_dims):
            timescale = base_timescale * (timescale_factor ** i)
            layer = TemporalLayer(
                dim=dim,
                timescale=timescale,
                decay_rate=decay_rate
            )
            self.layers.append(layer)

        # Inter-layer connections (for feeding forward/backward)
        self._init_connections()

    def _init_connections(self) -> None:
        """Initialize connection weights between layers"""
        self.W_up: List[np.ndarray] = []    # Bottom-up weights
        self.W_down: List[np.ndarray] = []  # Top-down weights

        for i in range(self.n_layers - 1):
            dim_low = self.layer_dims[i]
            dim_high = self.layer_dims[i + 1]

            # Bottom-up: compress fast to slow
            self.W_up.append(np.random.randn(dim_high, dim_low) * 0.1)

            # Top-down: expand slow to fast
            self.W_down.append(np.random.randn(dim_low, dim_high) * 0.1)

    def step(self, input_signal: np.ndarray, dt: float = 0.1) -> List[np.ndarray]:
        """Process one time step through the hierarchy.

        Args:
            input_signal: Input to lowest (fastest) layer
            dt: Time step

        Returns:
            List of states at each layer
        """
        states = []

        # Bottom-up processing
        current_signal = input_signal
        for i, layer in enumerate(self.layers):
            # Add top-down prediction from layer above (if exists and has state)
            if i < self.n_layers - 1:
                # Higher layer (i+1) predicts this layer (i)
                # W_down[i] maps from dim[i+1] to dim[i]
                higher_state = self.layers[i + 1].state
                if np.any(higher_state != 0):
                    top_down = self.W_down[i] @ higher_state
                    current_signal = current_signal + 0.5 * top_down

            # Update this layer
            state = layer.update(current_signal, dt)
            states.append(state)

            # Transform for next layer
            if i < self.n_layers - 1:
                current_signal = self.W_up[i] @ state

        return states

    def get_timescales(self) -> List[float]:
        """Get timescales of all layers"""
        return [layer.timescale for layer in self.layers]

    def get_layer_activities(self) -> List[float]:
        """Get activity levels (magnitude) at each layer"""
        return [float(np.linalg.norm(layer.state)) for layer in self.layers]

    def get_temporal_derivatives(self) -> List[np.ndarray]:
        """Get temporal derivatives at each layer"""
        return [layer.get_temporal_derivative() for layer in self.layers]

    def reset(self) -> None:
        """Reset all layers"""
        for layer in self.layers:
            layer.reset()


class TemporalPrediction:
    """Temporal prediction across multiple timescales.

    Uses the hierarchy to predict at different temporal horizons.
    """

    def __init__(self, hierarchy: TemporalHierarchy):
        self.hierarchy = hierarchy

        # Prediction weights for each timescale
        self.prediction_weights: List[np.ndarray] = []
        for i, layer in enumerate(hierarchy.layers[:-1]):
            W = np.random.randn(
                hierarchy.layer_dims[0],
                layer.dim
            ) * 0.1
            self.prediction_weights.append(W)

    def predict(self, horizon: float) -> np.ndarray:
        """Generate prediction for given temporal horizon.

        Args:
            horizon: How far ahead to predict (in seconds)

        Returns:
            Predicted state
        """
        # Find the layer whose timescale best matches the horizon
        timescales = self.hierarchy.get_timescales()

        # Weight predictions from each layer based on timescale match
        prediction = np.zeros(self.hierarchy.layer_dims[0])
        total_weight = 0.0

        for i, (layer, timescale) in enumerate(zip(
            self.hierarchy.layers[:-1], timescales[:-1]
        )):
            # Gaussian weight based on timescale-horizon match
            weight = np.exp(-((np.log(timescale) - np.log(horizon)) ** 2) / 2.0)
            state = layer.get_integrated_state()
            prediction += weight * (self.prediction_weights[i] @ state)
            total_weight += weight

        if total_weight > 0:
            prediction /= total_weight

        return prediction

    def predict_sequence(
        self,
        horizons: List[float]
    ) -> List[np.ndarray]:
        """Generate predictions for multiple horizons"""
        return [self.predict(h) for h in horizons]


class TemporalBuffer:
    """Circular buffer for temporal sequences with decay"""

    def __init__(
        self,
        dim: int,
        max_duration: float = 10.0,
        resolution: float = 0.01,
        decay_type: DecayType = DecayType.EXPONENTIAL,
        decay_rate: float = 0.1
    ):
        """Initialize temporal buffer.

        Args:
            dim: Dimensionality of stored values
            max_duration: Maximum duration to store (seconds)
            resolution: Time resolution (seconds per entry)
            decay_type: Type of temporal decay
            decay_rate: Rate of decay
        """
        self.dim = dim
        self.max_duration = max_duration
        self.resolution = resolution
        self.decay_type = decay_type
        self.decay_rate = decay_rate

        # Buffer size
        self.buffer_size = int(max_duration / resolution)
        self.buffer = np.zeros((self.buffer_size, dim))
        self.timestamps = np.zeros(self.buffer_size)

        # Current write position
        self.write_pos = 0
        self.current_time = 0.0

    def add(self, value: np.ndarray, timestamp: Optional[float] = None) -> None:
        """Add value to buffer"""
        if timestamp is None:
            timestamp = self.current_time

        self.buffer[self.write_pos] = value
        self.timestamps[self.write_pos] = timestamp

        self.write_pos = (self.write_pos + 1) % self.buffer_size
        self.current_time = timestamp

    def get_weighted_history(self, window: float) -> np.ndarray:
        """Get decay-weighted history over window.

        Args:
            window: Time window to retrieve

        Returns:
            Weighted average of history
        """
        # Calculate weights based on age
        ages = self.current_time - self.timestamps
        ages = np.clip(ages, 0, window)

        if self.decay_type == DecayType.EXPONENTIAL:
            weights = np.exp(-self.decay_rate * ages)
        elif self.decay_type == DecayType.POWER_LAW:
            weights = 1.0 / (1.0 + ages) ** self.decay_rate
        else:  # OSCILLATORY
            weights = np.exp(-self.decay_rate * ages) * np.cos(2 * np.pi * ages)
            weights = np.abs(weights)

        # Mask for valid entries within window
        valid = ages < window
        weights = weights * valid

        # Weighted average
        if np.sum(weights) > 0:
            return np.average(self.buffer, axis=0, weights=weights)
        return np.zeros(self.dim)

    def get_recent(self, n: int) -> np.ndarray:
        """Get n most recent entries"""
        indices = [(self.write_pos - 1 - i) % self.buffer_size for i in range(n)]
        return self.buffer[indices]

    def clear(self) -> None:
        """Clear the buffer"""
        self.buffer = np.zeros((self.buffer_size, self.dim))
        self.timestamps = np.zeros(self.buffer_size)
        self.write_pos = 0


class MultiTimescaleIntegrator:
    """Integrates information across multiple timescales simultaneously.

    Maintains parallel representations at different temporal resolutions.
    """

    def __init__(
        self,
        dim: int,
        timescales: List[float] = [0.01, 0.1, 1.0, 10.0]
    ):
        """Initialize multi-timescale integrator.

        Args:
            dim: Dimensionality
            timescales: List of timescales to track
        """
        self.dim = dim
        self.timescales = timescales
        self.n_scales = len(timescales)

        # One buffer per timescale
        self.buffers = [
            TemporalBuffer(dim, max_duration=ts * 10, resolution=ts / 10)
            for ts in timescales
        ]

        # Integrated states at each timescale
        self.integrated_states = [np.zeros(dim) for _ in timescales]

    def update(self, value: np.ndarray, timestamp: float) -> List[np.ndarray]:
        """Update all timescales with new value.

        Args:
            value: New value to integrate
            timestamp: Current time

        Returns:
            Integrated states at each timescale
        """
        for i, (buffer, ts) in enumerate(zip(self.buffers, self.timescales)):
            buffer.current_time = timestamp
            buffer.add(value, timestamp)
            self.integrated_states[i] = buffer.get_weighted_history(ts)

        return self.integrated_states

    def get_scale_comparison(self) -> np.ndarray:
        """Compare states across timescales.

        Returns matrix showing agreement between timescales.
        """
        comparison = np.zeros((self.n_scales, self.n_scales))

        for i in range(self.n_scales):
            for j in range(self.n_scales):
                # Cosine similarity between states
                s1 = self.integrated_states[i]
                s2 = self.integrated_states[j]
                norm = np.linalg.norm(s1) * np.linalg.norm(s2) + 1e-8
                comparison[i, j] = np.dot(s1, s2) / norm

        return comparison

    def detect_timescale_conflict(self, threshold: float = 0.5) -> List[Tuple[int, int]]:
        """Detect conflicts between timescales.

        Conflicts occur when fast and slow timescales disagree.

        Returns:
            List of (scale_i, scale_j) pairs that are in conflict
        """
        comparison = self.get_scale_comparison()
        conflicts = []

        for i in range(self.n_scales):
            for j in range(i + 1, self.n_scales):
                if comparison[i, j] < threshold:
                    conflicts.append((i, j))

        return conflicts

    def reset(self) -> None:
        """Reset all buffers"""
        for buffer in self.buffers:
            buffer.clear()
        self.integrated_states = [np.zeros(self.dim) for _ in self.timescales]
