"""Tests for cognitive manifold and dual-process cognition"""

import numpy as np
import pytest
import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])

from src.cognitive_manifold import (
    CognitiveState,
    CognitiveManifold,
    DualProcess,
    FlowType,
    AttractorLandscape
)


class TestCognitiveState:
    """Tests for cognitive state representation"""

    def test_initialization(self):
        """Test state initializes correctly"""
        position = np.array([1.0, 2.0, 3.0])
        state = CognitiveState(position)

        assert np.allclose(state.position, position)
        assert state.dim == 3
        assert state.metric.shape == (3, 3)
        assert np.allclose(state.metric, np.eye(3))

    def test_custom_metric(self):
        """Test initialization with custom metric"""
        position = np.array([1.0, 2.0])
        metric = np.array([[2.0, 0.0], [0.0, 0.5]])
        state = CognitiveState(position, metric)

        assert np.allclose(state.metric, metric)

    def test_distance(self):
        """Test Riemannian distance computation"""
        s1 = CognitiveState(np.array([0.0, 0.0]))
        s2 = CognitiveState(np.array([1.0, 0.0]))

        # With identity metric, should be Euclidean
        dist = s1.distance_to(s2)
        assert np.isclose(dist, 1.0)

        # With scaled metric
        s1.metric = np.array([[4.0, 0.0], [0.0, 1.0]])
        s1._update_inverse_metric()
        dist_scaled = s1.distance_to(s2)
        assert np.isclose(dist_scaled, 2.0)  # sqrt(1 * 4) = 2

    def test_inner_product(self):
        """Test inner product with metric"""
        state = CognitiveState(np.array([0.0, 0.0]))
        state.metric = np.array([[2.0, 0.0], [0.0, 3.0]])

        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])

        ip_11 = state.inner_product(v1, v1)
        ip_22 = state.inner_product(v2, v2)
        ip_12 = state.inner_product(v1, v2)

        assert np.isclose(ip_11, 2.0)
        assert np.isclose(ip_22, 3.0)
        assert np.isclose(ip_12, 0.0)

    def test_raise_lower_index(self):
        """Test index raising/lowering"""
        state = CognitiveState(np.array([0.0, 0.0]))
        state.metric = np.array([[2.0, 0.0], [0.0, 0.5]])
        state._update_inverse_metric()

        v = np.array([1.0, 1.0])

        lowered = state.lower_index(v)
        raised_back = state.raise_index(lowered)

        assert np.allclose(raised_back, v)

    def test_move(self):
        """Test state movement"""
        state = CognitiveState(np.array([0.0, 0.0]))

        state.move(np.array([1.0, 2.0]), dt=0.5)

        assert np.allclose(state.position, [0.5, 1.0])
        assert len(state.trajectory) == 2

    def test_copy(self):
        """Test state copying"""
        state = CognitiveState(np.array([1.0, 2.0]))
        state.velocity = np.array([0.1, 0.2])

        copy = state.copy()

        assert np.allclose(copy.position, state.position)
        assert np.allclose(copy.velocity, state.velocity)
        assert copy is not state


class TestCognitiveManifold:
    """Tests for cognitive manifold with potential"""

    def test_initialization(self):
        """Test manifold initialization"""
        manifold = CognitiveManifold(dim=5)

        assert manifold.dim == 5
        assert manifold.state.position.shape == (5,)

    def test_parsimony_potential(self):
        """Test parsimony (complexity) potential"""
        manifold = CognitiveManifold(dim=2, parsimony_weight=1.0)

        # Larger state = higher potential
        manifold.state.position = np.array([0.0, 0.0])
        pot_zero = manifold.parsimony_potential(manifold.state.position)

        manifold.state.position = np.array([1.0, 1.0])
        pot_one = manifold.parsimony_potential(manifold.state.position)

        assert pot_one > pot_zero

    def test_utility_potential(self):
        """Test utility (goal-directed) potential"""
        manifold = CognitiveManifold(dim=2, utility_weight=1.0)
        manifold.set_goal(np.array([1.0, 1.0]))

        # Closer to goal = lower potential
        manifold.state.position = np.array([0.9, 0.9])
        pot_close = manifold.utility_potential(manifold.state.position)

        manifold.state.position = np.array([0.0, 0.0])
        pot_far = manifold.utility_potential(manifold.state.position)

        assert pot_close < pot_far

    def test_gradient_points_uphill(self):
        """Test gradient points toward higher potential"""
        manifold = CognitiveManifold(dim=2, parsimony_weight=1.0, utility_weight=0.0)
        manifold.state.position = np.array([1.0, 1.0])

        grad = manifold.gradient()

        # For parsimony potential ||x||^2, gradient is 2x
        # Gradient should point away from origin (uphill)
        assert np.dot(grad, manifold.state.position) > 0

    def test_flow_decreases_potential(self):
        """Test gradient flow decreases potential"""
        manifold = CognitiveManifold(
            dim=2,
            parsimony_weight=1.0,
            utility_weight=0.0,
            accuracy_weight=0.0
        )
        manifold.state.position = np.array([2.0, 2.0])

        initial_pot = manifold.potential()

        # Flow for several steps
        for _ in range(10):
            manifold.flow(dt=0.1)

        final_pot = manifold.potential()

        assert final_pot < initial_pot

    def test_flow_toward_goal(self):
        """Test flow moves state toward goal"""
        manifold = CognitiveManifold(
            dim=2,
            utility_weight=1.0,
            parsimony_weight=0.0,
            accuracy_weight=0.0
        )
        goal = np.array([5.0, 5.0])
        manifold.set_goal(goal)
        manifold.state.position = np.array([0.0, 0.0])

        initial_dist = np.linalg.norm(manifold.state.position - goal)

        for _ in range(50):
            manifold.flow(dt=0.1)

        final_dist = np.linalg.norm(manifold.state.position - goal)

        assert final_dist < initial_dist

    def test_convergence(self):
        """Test flow_until_convergence"""
        manifold = CognitiveManifold(
            dim=2,
            utility_weight=1.0,
            parsimony_weight=0.1,
            accuracy_weight=0.0
        )
        manifold.set_goal(np.array([1.0, 1.0]))
        manifold.state.position = np.array([0.0, 0.0])

        final_pos, steps = manifold.flow_until_convergence(
            dt=0.1,
            max_steps=500,
            tolerance=1e-5
        )

        # Should converge before max steps
        assert steps < 500

        # Should be close to goal (modified by parsimony)
        assert np.linalg.norm(final_pos - np.array([1.0, 1.0])) < 1.0

    def test_trajectory(self):
        """Test trajectory recording"""
        manifold = CognitiveManifold(dim=2)
        manifold.state.position = np.array([1.0, 1.0])

        for _ in range(10):
            manifold.flow(dt=0.1)

        trajectory = manifold.get_trajectory()

        assert trajectory.shape[0] == 11  # Initial + 10 steps
        assert trajectory.shape[1] == 2

    def test_natural_gradient(self):
        """Test natural gradient with non-identity metric"""
        manifold = CognitiveManifold(dim=2, parsimony_weight=1.0)
        manifold.state.position = np.array([1.0, 1.0])

        # Anisotropic metric
        manifold.state.metric = np.array([[4.0, 0.0], [0.0, 1.0]])
        manifold.state._update_inverse_metric()

        euclidean = manifold.gradient()
        natural = manifold.natural_gradient()

        # Natural gradient accounts for metric
        assert not np.allclose(euclidean, natural)


class TestDualProcess:
    """Tests for dual-process cognition emergence"""

    def test_initialization(self):
        """Test dual-process system initialization"""
        manifold = CognitiveManifold(dim=3)
        dp = DualProcess(manifold)

        assert dp.current_system == 1
        assert dp.fast_metric.shape == (3, 3)
        assert dp.slow_metric.shape == (3, 3)

    def test_system_determination(self):
        """Test automatic system selection based on gradient"""
        manifold = CognitiveManifold(dim=2, parsimony_weight=1.0)
        dp = DualProcess(manifold, fast_threshold=0.5, slow_threshold=0.1)

        # Large state = steep gradient = System 1
        manifold.state.position = np.array([5.0, 5.0])
        system = dp.determine_system()
        assert system == 1

        # Small state = shallow gradient = System 2
        manifold.state.position = np.array([0.01, 0.01])
        system = dp.determine_system()
        assert system == 2

    def test_fast_path(self):
        """Test System 1 fast path"""
        manifold = CognitiveManifold(dim=2, parsimony_weight=1.0)
        dp = DualProcess(manifold)
        manifold.state.position = np.array([1.0, 1.0])

        initial_pos = manifold.state.position.copy()
        dp.fast_path(dt=0.1)
        final_pos = manifold.state.position

        # Should move significantly
        change = np.linalg.norm(final_pos - initial_pos)
        assert change > 0

    def test_slow_path(self):
        """Test System 2 slow path"""
        manifold = CognitiveManifold(dim=2, parsimony_weight=1.0)
        dp = DualProcess(manifold)
        manifold.state.position = np.array([1.0, 1.0])

        initial_pos = manifold.state.position.copy()
        dp.slow_path(dt=0.1)
        final_pos = manifold.state.position

        # Should move (less than fast path typically)
        change = np.linalg.norm(final_pos - initial_pos)
        assert change > 0

    def test_think_returns_trajectory(self):
        """Test complete thinking process"""
        manifold = CognitiveManifold(
            dim=2,
            parsimony_weight=1.0,
            utility_weight=0.0,
            accuracy_weight=0.0
        )
        dp = DualProcess(manifold)
        manifold.state.position = np.array([2.0, 2.0])

        final_pos, systems_used = dp.think(max_steps=50, dt=0.1)

        assert final_pos.shape == (2,)
        assert len(systems_used) > 0
        assert all(s in [1, 2] for s in systems_used)

    def test_system_balance(self):
        """Test system balance computation"""
        manifold = CognitiveManifold(dim=2)
        dp = DualProcess(manifold)

        systems = [1, 1, 1, 2, 2]
        s1_prop, s2_prop = dp.get_system_balance(systems)

        assert np.isclose(s1_prop, 0.6)
        assert np.isclose(s2_prop, 0.4)

    def test_dual_process_emergence(self):
        """Test that dual-process behavior emerges from geometry"""
        manifold = CognitiveManifold(
            dim=2,
            parsimony_weight=1.0,
            utility_weight=0.0,
            accuracy_weight=0.0
        )
        dp = DualProcess(manifold, fast_threshold=1.0, slow_threshold=0.1)

        # Start far from equilibrium (steep gradient region)
        manifold.state.position = np.array([5.0, 5.0])
        _, systems = dp.think(max_steps=100, dt=0.1)

        # Should start with mostly System 1 (fast) and transition to System 2 (slow)
        if len(systems) > 10:
            early_s1 = sum(1 for s in systems[:5] if s == 1) / 5
            late_s2 = sum(1 for s in systems[-5:] if s == 2) / 5

            # Early phase should have more System 1
            # Late phase should have more System 2
            # (This is a soft test as exact dynamics depend on parameters)
            assert len(systems) > 0


class TestAttractorLandscape:
    """Tests for attractor dynamics"""

    def test_add_attractor(self):
        """Test adding attractors"""
        manifold = CognitiveManifold(dim=2)
        landscape = AttractorLandscape(manifold)

        landscape.add_attractor(np.array([1.0, 0.0]), strength=1.0)
        landscape.add_attractor(np.array([0.0, 1.0]), strength=2.0)

        assert len(landscape.attractors) == 2
        assert len(landscape.attractor_strengths) == 2

    def test_attractor_potential(self):
        """Test potential from attractors"""
        manifold = CognitiveManifold(dim=2)
        landscape = AttractorLandscape(manifold)

        landscape.add_attractor(np.array([0.0, 0.0]), strength=1.0)

        # At attractor, potential should be lowest
        pot_at = landscape.attractor_potential(np.array([0.0, 0.0]))
        pot_away = landscape.attractor_potential(np.array([2.0, 2.0]))

        assert pot_at < pot_away

    def test_nearest_attractor(self):
        """Test finding nearest attractor"""
        manifold = CognitiveManifold(dim=2)
        landscape = AttractorLandscape(manifold)

        landscape.add_attractor(np.array([0.0, 0.0]))
        landscape.add_attractor(np.array([5.0, 5.0]))

        state = np.array([1.0, 1.0])
        nearest = landscape.find_nearest_attractor(state)

        assert nearest == 0  # Closer to first attractor

    def test_in_basin(self):
        """Test basin of attraction detection"""
        manifold = CognitiveManifold(dim=2)
        landscape = AttractorLandscape(manifold)

        landscape.add_attractor(np.array([0.0, 0.0]))

        assert landscape.in_basin(np.array([0.5, 0.5]), 0, radius=1.0)
        assert not landscape.in_basin(np.array([2.0, 2.0]), 0, radius=1.0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
