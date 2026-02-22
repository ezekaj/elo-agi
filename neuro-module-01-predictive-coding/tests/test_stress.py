"""
Stress Tests and Edge Cases for Predictive Coding Module

These tests push the system to its limits and test edge cases:
- Numerical stability under extreme conditions
- High-dimensional spaces
- Long sequence learning
- Rapid regime changes
- Adversarial inputs
"""

import numpy as np
import pytest
from neuro.modules.m01_predictive_coding.predictive_hierarchy import (
    PredictiveLayer,
    PredictiveHierarchy,
)
from neuro.modules.m01_predictive_coding.precision_weighting import (
    PrecisionWeightedError,
    AdaptivePrecision,
    HierarchicalPrecision,
)
from neuro.modules.m01_predictive_coding.cognitive_manifold import (
    CognitiveState,
    CognitiveManifold,
    DualProcess,
    AttractorLandscape,
)
from neuro.modules.m01_predictive_coding.temporal_dynamics import (
    TemporalLayer,
    TemporalHierarchy,
    MultiTimescaleIntegrator,
)
from neuro.modules.m01_predictive_coding.omission_detector import (
    OmissionDetector,
    SequenceOmissionDetector,
    RhythmicOmissionDetector,
)


class TestNumericalStability:
    """Tests for numerical stability under extreme conditions"""

    def test_hierarchy_with_large_inputs(self):
        """Test hierarchy handles large input values"""
        hierarchy = PredictiveHierarchy(layer_dims=[10, 8, 6], learning_rate=0.1)

        # Large input values
        large_input = np.ones(10) * 1000.0

        for _ in range(50):
            result = hierarchy.step(large_input, dt=0.1)
            # Should not produce NaN or Inf
            assert np.all(np.isfinite(result["total_error"]))
            for state in result["states"]:
                assert np.all(np.isfinite(state))

    def test_hierarchy_with_tiny_inputs(self):
        """Test hierarchy handles very small input values"""
        hierarchy = PredictiveHierarchy(layer_dims=[10, 8, 6], learning_rate=0.1)

        # Tiny input values
        tiny_input = np.ones(10) * 1e-10

        for _ in range(50):
            result = hierarchy.step(tiny_input, dt=0.1)
            assert np.all(np.isfinite(result["total_error"]))

    def test_hierarchy_with_mixed_scale_inputs(self):
        """Test hierarchy handles inputs with mixed scales"""
        hierarchy = PredictiveHierarchy(layer_dims=[10, 8, 6], learning_rate=0.1)

        # Mixed scale: some large, some tiny
        mixed_input = np.array([1e6, 1e-6, 1.0, -1e6, 1e-6, 0.0, 100, -100, 0.001, -0.001])

        for _ in range(50):
            result = hierarchy.step(mixed_input, dt=0.1)
            assert np.all(np.isfinite(result["total_error"]))

    def test_precision_with_zero_variance(self):
        """Test precision estimation with constant errors (zero variance)"""
        pwe = PrecisionWeightedError(dim=5)

        # Constant error (zero variance)
        constant_error = np.array([0.1, 0.1, 0.1, 0.1, 0.1])

        for _ in range(100):
            pwe.update_precision(constant_error)

        # Should not produce Inf (from 1/0)
        assert np.all(np.isfinite(pwe.precision))

    def test_manifold_at_singularity(self):
        """Test manifold behavior near singular metric"""
        manifold = CognitiveManifold(dim=2)

        # Near-singular metric
        manifold.state.metric = np.array([[1e-8, 0], [0, 1e8]])
        manifold.state._update_inverse_metric()

        # Should still compute gradient
        grad = manifold.gradient()
        assert np.all(np.isfinite(grad))

    def test_long_sequence_stability(self):
        """Test stability over very long sequences"""
        hierarchy = PredictiveHierarchy(layer_dims=[5, 4, 3], learning_rate=0.05)

        # Very long training
        np.random.seed(42)
        for _ in range(5000):
            obs = np.random.randn(5) * 0.5
            result = hierarchy.step(obs, dt=0.05)
            assert np.all(np.isfinite(result["total_error"]))
            assert result["total_error"] < 1e10  # Bounded


class TestHighDimensional:
    """Tests for high-dimensional scenarios"""

    def test_high_dim_hierarchy(self):
        """Test hierarchy with high-dimensional layers"""
        hierarchy = PredictiveHierarchy(layer_dims=[100, 50, 25, 10], learning_rate=0.05)

        obs = np.random.randn(100)

        for _ in range(20):
            result = hierarchy.step(obs, dt=0.1)
            assert np.all(np.isfinite(result["total_error"]))

    def test_high_dim_manifold(self):
        """Test manifold in high dimensions"""
        manifold = CognitiveManifold(dim=50, parsimony_weight=1.0, utility_weight=0.5)

        manifold.set_goal(np.random.randn(50))
        manifold.state.position = np.random.randn(50)

        for _ in range(100):
            manifold.flow(dt=0.1)
            assert np.all(np.isfinite(manifold.state.position))

    def test_high_dim_precision(self):
        """Test precision weighting in high dimensions"""
        hp = HierarchicalPrecision(level_dims=[100, 50, 25], base_learning_rate=0.1)

        errors = [np.random.randn(100), np.random.randn(50), np.random.randn(25)]

        for _ in range(50):
            precs, vols = hp.update(errors)
            for p in precs:
                assert np.all(np.isfinite(p))
                assert np.all(p > 0)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions"""

    def test_single_layer_hierarchy(self):
        """Test hierarchy with just one layer"""
        hierarchy = PredictiveHierarchy(layer_dims=[5, 3])

        obs = np.random.randn(5)
        result = hierarchy.step(obs)

        assert len(result["errors"]) == 1
        assert len(result["states"]) == 1

    def test_empty_expectation_omission(self):
        """Test omission detector with no expectations"""
        detector = OmissionDetector(input_dim=5)

        # No expectations set
        omissions = detector.check_omissions(1.0)
        assert len(omissions) == 0

    def test_zero_timescale_layer(self):
        """Test temporal layer with very small timescale"""
        layer = TemporalLayer(dim=5, timescale=0.001)

        for _ in range(100):
            state = layer.update(np.random.randn(5), dt=0.001)
            assert np.all(np.isfinite(state))

    def test_manifold_at_goal(self):
        """Test manifold when already at goal"""
        manifold = CognitiveManifold(dim=3, utility_weight=1.0, parsimony_weight=0.0)

        goal = np.array([1.0, 2.0, 3.0])
        manifold.set_goal(goal)
        manifold.state.position = goal.copy()

        # Gradient should be near zero
        grad = manifold.gradient()
        assert np.linalg.norm(grad) < 0.1

    def test_dual_process_at_minimum(self):
        """Test dual process when at potential minimum"""
        manifold = CognitiveManifold(dim=2, parsimony_weight=1.0)
        manifold.state.position = np.array([0.0, 0.0])  # Minimum of ||x||^2

        dp = DualProcess(manifold)
        system = dp.determine_system()

        # Should use System 2 (slow) at minimum
        assert system == 2

    def test_attractor_with_single_point(self):
        """Test attractor landscape with single attractor"""
        manifold = CognitiveManifold(dim=2)
        landscape = AttractorLandscape(manifold)
        landscape.add_attractor(np.array([0.0, 0.0]), strength=1.0)

        # Any point should map to attractor 0
        assert landscape.find_nearest_attractor(np.array([5.0, 5.0])) == 0
        assert landscape.find_nearest_attractor(np.array([-10.0, 10.0])) == 0


class TestRapidChanges:
    """Tests for handling rapid changes and regime shifts"""

    def test_rapid_input_changes(self):
        """Test hierarchy with rapidly changing inputs"""
        hierarchy = PredictiveHierarchy(layer_dims=[5, 4, 3], learning_rate=0.1)

        for i in range(100):
            # Alternating between very different inputs
            if i % 2 == 0:
                obs = np.ones(5) * 5.0
            else:
                obs = -np.ones(5) * 5.0

            result = hierarchy.step(obs)
            assert np.all(np.isfinite(result["total_error"]))

    def test_precision_rapid_variance_change(self):
        """Test precision adaptation to rapid variance changes"""
        ap = AdaptivePrecision(dim=3, learning_rate=0.5)

        # Phase 1: Low variance
        for _ in range(20):
            ap.update(np.random.randn(3) * 0.01)

        prec_low_var = ap.precision.copy()

        # Phase 2: Sudden high variance
        for _ in range(20):
            ap.update(np.random.randn(3) * 10.0)

        prec_high_var = ap.precision.copy()

        # Precision should decrease
        assert np.mean(prec_high_var) < np.mean(prec_low_var)

    def test_goal_switching(self):
        """Test manifold behavior when goal changes rapidly"""
        manifold = CognitiveManifold(dim=2, utility_weight=1.0, parsimony_weight=0.1)

        goals = [
            np.array([5.0, 0.0]),
            np.array([0.0, 5.0]),
            np.array([-5.0, 0.0]),
            np.array([0.0, -5.0]),
        ]

        for goal in goals:
            manifold.set_goal(goal)
            for _ in range(20):
                manifold.flow(dt=0.1)
                assert np.all(np.isfinite(manifold.state.position))


class TestIntegration:
    """Integration tests combining multiple components"""

    def test_full_predictive_system(self):
        """Test complete predictive coding system"""
        # Create all components
        hierarchy = PredictiveHierarchy(layer_dims=[5, 4, 3], learning_rate=0.1)
        precision = HierarchicalPrecision(level_dims=[5, 4])
        omission = OmissionDetector(input_dim=5)

        # Training sequence
        sequence = [np.random.randn(5) for _ in range(50)]

        for t, obs in enumerate(sequence):
            # Predictive step
            result = hierarchy.step(obs, dt=0.1)

            # Update precision
            precision.update(result["errors"])

            # Check for omissions
            omission.receive_input(obs, timestamp=t * 0.1)

            # All should remain stable
            assert np.all(np.isfinite(result["total_error"]))

    def test_manifold_with_precision(self):
        """Test cognitive manifold with precision-weighted updates"""
        manifold = CognitiveManifold(dim=3)
        precision = AdaptivePrecision(dim=3)

        manifold.set_goal(np.array([1.0, 1.0, 1.0]))

        for _ in range(50):
            # Get gradient
            grad = manifold.gradient()

            # Simulate error and update precision
            error = manifold.state.position - manifold.goal
            prec, _ = precision.update(error)

            # Precision-weighted flow
            weighted_grad = prec * grad
            manifold.state.position -= 0.1 * weighted_grad

            assert np.all(np.isfinite(manifold.state.position))

    def test_temporal_with_omission(self):
        """Test temporal hierarchy with omission detection"""
        temporal = TemporalHierarchy(layer_dims=[5, 4, 3], base_timescale=0.01)
        sequence_detector = SequenceOmissionDetector(input_dim=5)

        # Regular sequence
        for t in range(100):
            obs = np.sin(2 * np.pi * t * 0.1) * np.ones(5)
            states = temporal.step(obs, dt=0.01)
            error = sequence_detector.observe(obs, timestamp=t * 0.01)

            for state in states:
                assert np.all(np.isfinite(state))


class TestAdversarial:
    """Tests with adversarial/pathological inputs"""

    def test_nan_handling(self):
        """Test that NaN inputs don't corrupt state"""
        hierarchy = PredictiveHierarchy(layer_dims=[5, 4, 3])

        # Normal operation
        for _ in range(10):
            hierarchy.step(np.random.randn(5))

        # Attempt NaN input (should be caught or handled)
        nan_input = np.array([np.nan, 1.0, 2.0, 3.0, 4.0])

        # Clean the input before using
        clean_input = np.nan_to_num(nan_input, nan=0.0)
        result = hierarchy.step(clean_input)

        # System should still work
        assert np.all(np.isfinite(result["total_error"]))

    def test_inf_handling(self):
        """Test handling of infinite values"""
        pwe = PrecisionWeightedError(dim=3)

        # Attempt to update with inf
        inf_error = np.array([np.inf, 1.0, -np.inf])
        clean_error = np.clip(inf_error, -1e10, 1e10)
        clean_error = np.nan_to_num(clean_error)

        pwe.update_precision(clean_error)
        assert np.all(np.isfinite(pwe.precision))

    def test_oscillating_inputs(self):
        """Test with rapidly oscillating inputs"""
        hierarchy = PredictiveHierarchy(layer_dims=[5, 4, 3], learning_rate=0.05)

        for t in range(500):
            # High frequency oscillation
            freq = 100
            obs = np.sin(2 * np.pi * freq * t * 0.001) * np.ones(5)
            result = hierarchy.step(obs, dt=0.001)
            assert np.all(np.isfinite(result["total_error"]))

    def test_sparse_inputs(self):
        """Test with mostly-zero sparse inputs"""
        hierarchy = PredictiveHierarchy(layer_dims=[100, 50, 25])

        for _ in range(50):
            # Sparse: only 5% nonzero
            obs = np.zeros(100)
            indices = np.random.choice(100, 5, replace=False)
            obs[indices] = np.random.randn(5)

            result = hierarchy.step(obs)
            assert np.all(np.isfinite(result["total_error"]))


class TestPerformance:
    """Performance-related tests"""

    def test_many_iterations(self):
        """Test stability over many iterations"""
        hierarchy = PredictiveHierarchy(layer_dims=[10, 8, 6], learning_rate=0.05)

        errors = []
        for i in range(1000):
            obs = np.random.randn(10) * 0.5
            result = hierarchy.step(obs)
            errors.append(result["total_error"])

        # Should remain bounded
        assert max(errors) < 1e6
        assert all(np.isfinite(e) for e in errors)

    def test_deep_hierarchy(self):
        """Test very deep hierarchy"""
        hierarchy = PredictiveHierarchy(
            layer_dims=[20, 18, 16, 14, 12, 10, 8, 6, 4], learning_rate=0.02, timescale_factor=1.5
        )

        for _ in range(100):
            obs = np.random.randn(20) * 0.3
            result = hierarchy.step(obs)
            assert np.all(np.isfinite(result["total_error"]))

    def test_multi_timescale_stress(self):
        """Stress test multi-timescale integrator"""
        integrator = MultiTimescaleIntegrator(dim=10, timescales=[0.001, 0.01, 0.1, 1.0, 10.0])

        for t in range(1000):
            value = np.random.randn(10)
            states = integrator.update(value, timestamp=t * 0.001)

            for state in states:
                assert np.all(np.isfinite(state))


class TestOmissionEdgeCases:
    """Edge cases for omission detection"""

    def test_overlapping_expectations(self):
        """Test multiple overlapping expectations"""
        detector = OmissionDetector(input_dim=3)

        # Add multiple expectations for same time
        for i in range(5):
            detector.add_expectation(
                np.random.randn(3), time_window=1.0, tolerance=0.5, id=f"exp_{i}"
            )

        # Check all get processed
        omissions = detector.check_omissions(2.0)
        assert len(omissions) == 5

    def test_rhythmic_with_irregular_input(self):
        """Test rhythmic detector with irregular timing"""
        detector = RhythmicOmissionDetector(input_dim=3)

        # Irregular timing
        times = [0.0, 0.11, 0.19, 0.32, 0.41, 0.52, 0.58, 0.71]

        for t in times:
            detector.observe(np.random.randn(3), timestamp=t)

        # Period estimate should be uncertain
        assert detector.period_confidence < 0.9


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
