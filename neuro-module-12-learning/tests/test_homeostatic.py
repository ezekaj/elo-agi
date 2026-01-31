"""Tests for homeostatic regulation"""

import numpy as np
import pytest
import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])

from src.homeostatic import (
    HomeostaticRegulation, SynapticScaling, MetaplasticityRegulation,
    ActivityRegulator, HomeostaticParams
)


class TestHomeostaticRegulation:
    """Tests for homeostatic regulation"""

    def test_initialization(self):
        """Test initialization"""
        hr = HomeostaticRegulation(n_neurons=10)

        assert len(hr.activity_average) == 10
        assert len(hr.excitability) == 10

    def test_activity_average_update(self):
        """Test running average updates"""
        hr = HomeostaticRegulation(n_neurons=3)

        # Initial average
        initial_avg = hr.activity_average.copy()

        # Update with new activity
        hr.update_average(np.ones(3) * 0.5)

        # Average should move toward new value
        assert not np.allclose(hr.activity_average, initial_avg)

    def test_scaling_factor_computation(self):
        """Test scaling factor computation"""
        params = HomeostaticParams(target_activity=0.1)
        hr = HomeostaticRegulation(n_neurons=3, params=params)

        # Low activity = upscale
        hr.activity_average = np.ones(3) * 0.05
        scaling_low = hr.compute_scaling_factor()

        # High activity = downscale
        hr.activity_average = np.ones(3) * 0.2
        scaling_high = hr.compute_scaling_factor()

        assert np.all(scaling_low > 1)  # Upscale
        assert np.all(scaling_high < 1)  # Downscale

    def test_weight_scaling(self):
        """Test weight matrix scaling"""
        hr = HomeostaticRegulation(n_neurons=3)
        hr.activity_average = np.array([0.05, 0.1, 0.2])

        weights = np.ones((3, 3))
        scaled = hr.scale_weights(weights)

        # Low activity neuron should have upscaled weights
        # High activity neuron should have downscaled weights
        assert np.mean(scaled[0, :]) > np.mean(scaled[2, :])

    def test_excitability_update(self):
        """Test intrinsic excitability updates"""
        hr = HomeostaticRegulation(n_neurons=3)

        # Low activity should increase excitability
        hr.activity_average = np.zeros(3)
        hr.update_excitability()

        assert np.all(hr.excitability > 1.0)

    def test_full_regulation(self):
        """Test complete regulation cycle"""
        hr = HomeostaticRegulation(n_neurons=5)

        activity = np.random.rand(5)
        weights = np.random.rand(5, 5)

        regulated_weights = hr.regulate(activity, weights)

        assert regulated_weights.shape == weights.shape
        assert np.all(np.isfinite(regulated_weights))


class TestSynapticScaling:
    """Tests for synaptic scaling"""

    def test_initialization(self):
        """Test initialization"""
        ss = SynapticScaling(target_sum=2.0)
        assert ss.target_sum == 2.0

    def test_gradual_scaling(self):
        """Test gradual weight scaling"""
        ss = SynapticScaling(target_sum=1.0, scaling_rate=0.1)

        weights = np.ones((3, 5)) * 0.5  # Sum = 2.5 per row

        # Multiple scaling steps
        for _ in range(10):
            weights = ss.scale(weights)

        # Should move toward target sum
        row_sums = np.sum(weights, axis=1)
        assert np.all(row_sums < 2.5)

    def test_immediate_normalization(self):
        """Test immediate normalization"""
        ss = SynapticScaling(target_sum=1.0)

        weights = np.random.rand(3, 5)
        normalized = ss.normalize(weights)

        row_sums = np.sum(normalized, axis=1)
        assert np.allclose(row_sums, 1.0)


class TestMetaplasticityRegulation:
    """Tests for metaplasticity"""

    def test_initialization(self):
        """Test initialization"""
        mp = MetaplasticityRegulation(n_neurons=5)

        assert len(mp.threshold) == 5

    def test_threshold_update(self):
        """Test threshold sliding with activity"""
        mp = MetaplasticityRegulation(n_neurons=3, time_constant=10.0)

        # High activity should raise threshold
        for _ in range(50):
            mp.update(np.ones(3) * 0.8)

        high_threshold = mp.threshold.copy()

        # Low activity should lower threshold
        mp2 = MetaplasticityRegulation(n_neurons=3, time_constant=10.0)
        for _ in range(50):
            mp2.update(np.ones(3) * 0.2)

        low_threshold = mp2.threshold

        assert np.all(high_threshold > low_threshold)

    def test_plasticity_direction(self):
        """Test plasticity direction computation"""
        mp = MetaplasticityRegulation(n_neurons=3)
        mp.threshold = np.ones(3) * 0.5

        activity = np.array([0.8, 0.5, 0.2])
        direction = mp.get_plasticity_direction(activity)

        assert direction[0] > 0  # Above threshold = LTP
        assert direction[1] == 0  # At threshold = neutral
        assert direction[2] < 0  # Below threshold = LTD


class TestActivityRegulator:
    """Tests for combined activity regulation"""

    def test_initialization(self):
        """Test initialization"""
        ar = ActivityRegulator(n_neurons=10)

        assert ar.homeostatic is not None
        assert ar.scaling is not None
        assert ar.metaplasticity is not None

    def test_full_regulation(self):
        """Test complete regulation"""
        ar = ActivityRegulator(n_neurons=5, target_activity=0.1)

        activity = np.random.rand(5)
        weights = np.random.rand(5, 5)

        regulated = ar.regulate(activity, weights)

        assert regulated.shape == weights.shape
        assert np.all(np.isfinite(regulated))

    def test_getters(self):
        """Test getter methods"""
        ar = ActivityRegulator(n_neurons=5)

        ar.regulate(np.random.rand(5), np.random.rand(5, 5))

        exc = ar.get_excitability()
        thresh = ar.get_plasticity_thresholds()

        assert len(exc) == 5
        assert len(thresh) == 5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
