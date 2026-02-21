"""
Geometric Cognitive Manifold

Represents cognitive states as points on a Riemannian manifold.
Thinking = gradient flow minimizing a cognitive potential function.

The potential combines:
- Accuracy: prediction error minimization
- Parsimony: complexity penalty (Occam's razor)
- Utility: goal-directed value
"""

import numpy as np
from typing import Callable, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum


class FlowType(Enum):
    """Types of gradient flow on the manifold"""
    GRADIENT_DESCENT = "gradient_descent"
    NATURAL_GRADIENT = "natural_gradient"
    GEODESIC = "geodesic"


@dataclass
class ManifoldPoint:
    """A point on the cognitive manifold with local geometry"""
    position: np.ndarray
    tangent: Optional[np.ndarray] = None
    curvature: Optional[float] = None


class CognitiveState:
    """Point on the cognitive manifold with local metric structure.

    The metric tensor determines distances and gradients in state space.
    Anisotropic metrics lead to different "ease" of movement in different
    directions - this is key for dual-process emergence.
    """

    def __init__(
        self,
        position: np.ndarray,
        metric: Optional[np.ndarray] = None
    ):
        """Initialize cognitive state.

        Args:
            position: State vector (point on manifold)
            metric: Local Riemannian metric tensor (default: identity)
        """
        self.position = position.astype(float)
        self.dim = len(position)

        # Metric tensor G_ij - determines distances and gradients
        if metric is None:
            self.metric = np.eye(self.dim)
        else:
            self.metric = metric

        # Inverse metric for raising indices
        self._update_inverse_metric()

        # Velocity (for momentum-based dynamics)
        self.velocity = np.zeros(self.dim)

        # History for trajectory analysis
        self.trajectory: List[np.ndarray] = [position.copy()]

    def _update_inverse_metric(self) -> None:
        """Update inverse metric tensor"""
        self.metric_inv = np.linalg.inv(self.metric + 1e-8 * np.eye(self.dim))

    def distance_to(self, other: 'CognitiveState') -> float:
        """Compute Riemannian distance to another state.

        Uses the metric tensor: d² = (x-y)ᵀ G (x-y)
        """
        diff = self.position - other.position
        return float(np.sqrt(diff @ self.metric @ diff))

    def inner_product(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute inner product of tangent vectors using the metric"""
        return float(v1 @ self.metric @ v2)

    def norm(self, v: np.ndarray) -> float:
        """Compute norm of tangent vector"""
        return np.sqrt(self.inner_product(v, v))

    def raise_index(self, covector: np.ndarray) -> np.ndarray:
        """Convert covector to vector using inverse metric"""
        return self.metric_inv @ covector

    def lower_index(self, vector: np.ndarray) -> np.ndarray:
        """Convert vector to covector using metric"""
        return self.metric @ vector

    def update_metric(self, new_metric: np.ndarray) -> None:
        """Update the local metric tensor"""
        self.metric = new_metric
        self._update_inverse_metric()

    def move(self, direction: np.ndarray, dt: float = 1.0) -> None:
        """Move state in given direction"""
        self.position += dt * direction
        self.trajectory.append(self.position.copy())

    def copy(self) -> 'CognitiveState':
        """Create a copy of this state"""
        new_state = CognitiveState(self.position.copy(), self.metric.copy())
        new_state.velocity = self.velocity.copy()
        return new_state


class CognitiveManifold:
    """The cognitive manifold with potential function and gradient flow.

    States are points on this manifold. Thinking is gradient descent
    on the cognitive potential, which combines multiple objectives.
    """

    def __init__(
        self,
        dim: int,
        accuracy_weight: float = 1.0,
        parsimony_weight: float = 0.1,
        utility_weight: float = 0.5
    ):
        """Initialize the cognitive manifold.

        Args:
            dim: Dimensionality of the state space
            accuracy_weight: Weight for prediction error term
            parsimony_weight: Weight for complexity penalty
            utility_weight: Weight for goal-directed value
        """
        self.dim = dim
        self.accuracy_weight = accuracy_weight
        self.parsimony_weight = parsimony_weight
        self.utility_weight = utility_weight

        # Current cognitive state
        self.state = CognitiveState(np.zeros(dim))

        # Target/goal state for utility computation
        self.goal: Optional[np.ndarray] = None

        # External prediction error signal
        self.prediction_error: Optional[np.ndarray] = None

        # Custom potential components (can be set externally)
        self._custom_accuracy: Optional[Callable] = None
        self._custom_parsimony: Optional[Callable] = None
        self._custom_utility: Optional[Callable] = None

    def set_prediction_error(self, error: np.ndarray) -> None:
        """Set current prediction error for accuracy computation"""
        self.prediction_error = error

    def set_goal(self, goal: np.ndarray) -> None:
        """Set goal state for utility computation"""
        self.goal = goal

    def accuracy_potential(self, state: np.ndarray) -> float:
        """Accuracy component: prediction error magnitude.

        L_accuracy = ||ε||² = ||y - ŷ||²
        """
        if self._custom_accuracy is not None:
            return self._custom_accuracy(state)

        if self.prediction_error is None:
            return 0.0

        return float(np.sum(self.prediction_error ** 2))

    def parsimony_potential(self, state: np.ndarray) -> float:
        """Parsimony component: complexity penalty.

        L_parsimony = ||x||² (L2 regularization)
        Encourages simpler representations.
        """
        if self._custom_parsimony is not None:
            return self._custom_parsimony(state)

        return float(np.sum(state ** 2))

    def utility_potential(self, state: np.ndarray) -> float:
        """Utility component: distance to goal.

        L_utility = ||x - x_goal||²
        """
        if self._custom_utility is not None:
            return self._custom_utility(state)

        if self.goal is None:
            return 0.0

        diff = state - self.goal
        return float(np.sum(diff ** 2))

    def potential(self, state: Optional[np.ndarray] = None) -> float:
        """Total cognitive potential to be minimized.

        F(x) = w_a * L_accuracy + w_p * L_parsimony + w_u * L_utility
        """
        if state is None:
            state = self.state.position

        F = (
            self.accuracy_weight * self.accuracy_potential(state) +
            self.parsimony_weight * self.parsimony_potential(state) +
            self.utility_weight * self.utility_potential(state)
        )
        return F

    def gradient(
        self,
        state: Optional[np.ndarray] = None,
        eps: float = 1e-5
    ) -> np.ndarray:
        """Compute gradient of potential (numerical).

        ∇F = direction of steepest ascent
        """
        if state is None:
            state = self.state.position

        grad = np.zeros(self.dim)
        for i in range(self.dim):
            state_plus = state.copy()
            state_minus = state.copy()
            state_plus[i] += eps
            state_minus[i] -= eps
            grad[i] = (self.potential(state_plus) - self.potential(state_minus)) / (2 * eps)

        return grad

    def natural_gradient(
        self,
        state: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute natural gradient using the metric.

        ∇̃F = G⁻¹ ∇F (metric-aware gradient)
        """
        euclidean_grad = self.gradient(state)
        return self.state.raise_index(euclidean_grad)

    def flow(
        self,
        dt: float = 0.1,
        flow_type: FlowType = FlowType.GRADIENT_DESCENT,
        momentum: float = 0.0
    ) -> np.ndarray:
        """Move state along gradient flow = "thinking".

        ẋ = -∇F (gradient descent)
        or ẋ = -G⁻¹∇F (natural gradient)

        Args:
            dt: Time step
            flow_type: Type of gradient to use
            momentum: Momentum coefficient (0 = no momentum)

        Returns:
            New position after flow
        """
        if flow_type == FlowType.GRADIENT_DESCENT:
            grad = self.gradient()
        elif flow_type == FlowType.NATURAL_GRADIENT:
            grad = self.natural_gradient()
        else:
            grad = self.gradient()  # Default to standard gradient

        # Update with optional momentum
        if momentum > 0:
            self.state.velocity = momentum * self.state.velocity - dt * grad
            direction = self.state.velocity
        else:
            direction = -dt * grad

        self.state.move(direction)
        return self.state.position

    def flow_until_convergence(
        self,
        dt: float = 0.1,
        max_steps: int = 1000,
        tolerance: float = 1e-6,
        flow_type: FlowType = FlowType.GRADIENT_DESCENT
    ) -> Tuple[np.ndarray, int]:
        """Flow until potential converges.

        Args:
            dt: Time step
            max_steps: Maximum number of steps
            tolerance: Convergence tolerance
            flow_type: Type of gradient flow

        Returns:
            Tuple of (final position, number of steps taken)
        """
        prev_potential = self.potential()

        for step in range(max_steps):
            self.flow(dt, flow_type)
            current_potential = self.potential()

            if abs(current_potential - prev_potential) < tolerance:
                return self.state.position, step + 1

            prev_potential = current_potential

        return self.state.position, max_steps

    def get_trajectory(self) -> np.ndarray:
        """Get the trajectory of states visited"""
        return np.array(self.state.trajectory)

    def reset(self, position: Optional[np.ndarray] = None) -> None:
        """Reset to initial or specified state"""
        if position is None:
            position = np.zeros(self.dim)
        self.state = CognitiveState(position)
        self.prediction_error = None


class DualProcess:
    """Dual-process cognition emerging from manifold geometry.

    System 1 (fast): steep gradients, automatic
    System 2 (slow): shallow gradients, deliberate

    Key insight: NOT separate modules, but different regions of
    the same metric landscape.
    """

    def __init__(
        self,
        manifold: CognitiveManifold,
        fast_threshold: float = 1.0,
        slow_threshold: float = 0.1
    ):
        """Initialize dual-process system.

        Args:
            manifold: The cognitive manifold
            fast_threshold: Gradient magnitude for System 1
            slow_threshold: Gradient magnitude for System 2
        """
        self.manifold = manifold
        self.fast_threshold = fast_threshold
        self.slow_threshold = slow_threshold

        # Track which system is currently active
        self.current_system = 1  # Start with System 1

        # Anisotropic metric regions
        # Steep metric = fast responses (System 1)
        # Flat metric = slow deliberation (System 2)
        self.fast_metric = np.eye(manifold.dim) * 10.0  # Amplifies gradients
        self.slow_metric = np.eye(manifold.dim) * 0.1   # Dampens gradients

    def get_gradient_magnitude(self) -> float:
        """Get current gradient magnitude (steepness of landscape)"""
        grad = self.manifold.gradient()
        return float(np.linalg.norm(grad))

    def determine_system(self) -> int:
        """Determine which cognitive system should be active.

        Based on local gradient magnitude:
        - Steep gradient → System 1 (fast, automatic)
        - Shallow gradient → System 2 (slow, deliberate)

        Returns:
            1 for System 1, 2 for System 2
        """
        grad_mag = self.get_gradient_magnitude()

        if grad_mag > self.fast_threshold:
            return 1
        elif grad_mag < self.slow_threshold:
            return 2
        else:
            # Interpolate based on gradient
            return self.current_system  # Hysteresis

    def fast_path(self, dt: float = 0.1) -> np.ndarray:
        """System 1: Follow steep gradients quickly.

        Fast, automatic, heuristic-based.
        """
        self.manifold.state.update_metric(self.fast_metric)
        return self.manifold.flow(dt * 2.0, FlowType.GRADIENT_DESCENT)

    def slow_path(self, dt: float = 0.1) -> np.ndarray:
        """System 2: Careful exploration with natural gradient.

        Slow, deliberate, analytical.
        """
        self.manifold.state.update_metric(self.slow_metric)
        return self.manifold.flow(dt * 0.5, FlowType.NATURAL_GRADIENT)

    def step(self, dt: float = 0.1) -> Tuple[np.ndarray, int]:
        """Take one cognitive step, automatically choosing system.

        Returns:
            Tuple of (new position, system used)
        """
        self.current_system = self.determine_system()

        if self.current_system == 1:
            position = self.fast_path(dt)
        else:
            position = self.slow_path(dt)

        return position, self.current_system

    def think(
        self,
        max_steps: int = 100,
        dt: float = 0.1,
        tolerance: float = 1e-6
    ) -> Tuple[np.ndarray, List[int]]:
        """Complete a thinking process, switching systems as needed.

        Returns:
            Tuple of (final position, list of systems used at each step)
        """
        systems_used = []
        prev_potential = self.manifold.potential()

        for _ in range(max_steps):
            _, system = self.step(dt)
            systems_used.append(system)

            current_potential = self.manifold.potential()
            if abs(current_potential - prev_potential) < tolerance:
                break
            prev_potential = current_potential

        return self.manifold.state.position, systems_used

    def get_system_balance(self, systems: List[int]) -> Tuple[float, float]:
        """Get proportion of System 1 vs System 2 usage.

        Args:
            systems: List of systems used

        Returns:
            Tuple of (System 1 proportion, System 2 proportion)
        """
        if not systems:
            return 0.5, 0.5

        s1 = sum(1 for s in systems if s == 1)
        s2 = len(systems) - s1
        total = len(systems)

        return s1 / total, s2 / total


class AttractorLandscape:
    """Attractor dynamics on the cognitive manifold.

    Attractors represent stable cognitive states (beliefs, concepts).
    The potential landscape has local minima that act as attractors.
    """

    def __init__(self, manifold: CognitiveManifold):
        self.manifold = manifold
        self.attractors: List[np.ndarray] = []
        self.attractor_strengths: List[float] = []

    def add_attractor(
        self,
        position: np.ndarray,
        strength: float = 1.0
    ) -> None:
        """Add an attractor (stable state) to the landscape"""
        self.attractors.append(position.copy())
        self.attractor_strengths.append(strength)

    def attractor_potential(self, state: np.ndarray) -> float:
        """Compute potential contribution from all attractors.

        Each attractor creates a "well" in the potential landscape.
        """
        if not self.attractors:
            return 0.0

        potential = 0.0
        for attractor, strength in zip(self.attractors, self.attractor_strengths):
            dist_sq = np.sum((state - attractor) ** 2)
            # Gaussian well
            potential -= strength * np.exp(-dist_sq / 2.0)

        return potential

    def find_nearest_attractor(self, state: np.ndarray) -> Optional[int]:
        """Find the index of the nearest attractor"""
        if not self.attractors:
            return None

        distances = [np.linalg.norm(state - a) for a in self.attractors]
        return int(np.argmin(distances))

    def in_basin(self, state: np.ndarray, attractor_idx: int, radius: float = 1.0) -> bool:
        """Check if state is in the basin of attraction"""
        if attractor_idx >= len(self.attractors):
            return False
        dist = np.linalg.norm(state - self.attractors[attractor_idx])
        return dist < radius
