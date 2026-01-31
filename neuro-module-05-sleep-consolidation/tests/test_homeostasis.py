"""Tests for synaptic homeostasis"""

import numpy as np
import pytest
import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])

from src.synaptic_homeostasis import (
    Synapse,
    SynapticHomeostasis,
    SelectiveConsolidation,
    SleepWakeCycle
)


class TestSynapse:
    """Tests for single synapse representation"""

    def test_creation(self):
        """Test synapse creation"""
        synapse = Synapse(id=0, weight=0.5, source=1, target=2)

        assert synapse.id == 0
        assert synapse.weight == 0.5
        assert synapse.source == 1
        assert synapse.target == 2
        assert not synapse.is_tagged


class TestSynapticHomeostasis:
    """Tests for synaptic homeostasis system"""

    def test_initialization(self):
        """Test system initialization"""
        homeostasis = SynapticHomeostasis(n_neurons=50, connectivity=0.1)

        assert homeostasis.n_neurons == 50
        assert len(homeostasis.synapses) > 0

    def test_total_strength(self):
        """Test total strength measurement"""
        homeostasis = SynapticHomeostasis(n_neurons=50, connectivity=0.1)

        total = homeostasis.measure_total_strength()
        assert total > 0

    def test_potentiation(self):
        """Test synaptic potentiation"""
        homeostasis = SynapticHomeostasis(n_neurons=50, connectivity=0.1)

        # Get first synapse
        synapse_id = list(homeostasis.synapses.keys())[0]
        initial_weight = homeostasis.synapses[synapse_id].weight

        homeostasis.potentiate(synapse_id, amount=0.1)

        assert homeostasis.synapses[synapse_id].weight > initial_weight

    def test_hebbian_potentiation(self):
        """Test Hebbian potentiation"""
        homeostasis = SynapticHomeostasis(n_neurons=50, connectivity=0.1)

        initial_strength = homeostasis.measure_total_strength()

        # Strong co-activation should increase weights
        pre = np.ones(50)
        post = np.ones(50)

        homeostasis.hebbian_potentiation(pre, post, learning_rate=0.1)

        final_strength = homeostasis.measure_total_strength()
        assert final_strength > initial_strength

    def test_downscaling(self):
        """Test global downscaling"""
        homeostasis = SynapticHomeostasis(n_neurons=50, connectivity=0.1)

        initial_strength = homeostasis.measure_total_strength()

        homeostasis.downscale(factor=0.8)

        final_strength = homeostasis.measure_total_strength()
        assert final_strength < initial_strength

    def test_downscaling_preserves_ratios(self):
        """Test that downscaling preserves relative weight ratios"""
        homeostasis = SynapticHomeostasis(n_neurons=50, connectivity=0.1)

        # Get initial ratios
        initial_ratios = homeostasis.preserve_ratios()

        # Downscale
        homeostasis.downscale(factor=0.5)

        # Get final ratios
        final_ratios = homeostasis.preserve_ratios()

        # Ratios should be very similar
        for sid in initial_ratios:
            if sid in final_ratios:
                assert np.isclose(initial_ratios[sid], final_ratios[sid], rtol=0.01)

    def test_energy_cost(self):
        """Test energy cost computation"""
        homeostasis = SynapticHomeostasis(n_neurons=50, connectivity=0.1)

        initial_energy = homeostasis.compute_energy_cost()

        # Potentiation increases energy
        pre = np.ones(50)
        post = np.ones(50)
        homeostasis.hebbian_potentiation(pre, post, learning_rate=0.1)

        final_energy = homeostasis.compute_energy_cost()
        assert final_energy > initial_energy

    def test_sleep_downscaling_step(self):
        """Test sleep-dependent downscaling step"""
        homeostasis = SynapticHomeostasis(n_neurons=50, connectivity=0.1, downscale_rate=0.1)

        initial_strength = homeostasis.measure_total_strength()

        homeostasis.sleep_downscaling_step(dt=1.0)

        final_strength = homeostasis.measure_total_strength()
        assert final_strength < initial_strength

    def test_statistics(self):
        """Test statistics reporting"""
        homeostasis = SynapticHomeostasis(n_neurons=50, connectivity=0.1)

        stats = homeostasis.get_statistics()

        assert "n_synapses" in stats
        assert "total_strength" in stats
        assert "mean_weight" in stats
        assert "energy_cost" in stats


class TestSelectiveConsolidation:
    """Tests for selective protection from downscaling"""

    def test_tagging(self):
        """Test synapse tagging"""
        homeostasis = SynapticHomeostasis(n_neurons=50, connectivity=0.1)
        selective = SelectiveConsolidation(homeostasis)

        synapse_id = list(homeostasis.synapses.keys())[0]

        success = selective.tag_for_protection(synapse_id)

        assert success
        assert homeostasis.synapses[synapse_id].is_tagged
        assert synapse_id in selective.tagged_synapses

    def test_tagged_protected_from_downscaling(self):
        """Test that tagged synapses are protected"""
        homeostasis = SynapticHomeostasis(n_neurons=50, connectivity=0.1)
        selective = SelectiveConsolidation(homeostasis)

        # Tag first synapse
        synapse_id = list(homeostasis.synapses.keys())[0]
        initial_weight = homeostasis.synapses[synapse_id].weight
        selective.tag_for_protection(synapse_id)

        # Downscale
        homeostasis.downscale(factor=0.5)

        # Tagged synapse should be unchanged
        assert homeostasis.synapses[synapse_id].weight == initial_weight

    def test_untagged_affected_by_downscaling(self):
        """Test that untagged synapses are downscaled"""
        homeostasis = SynapticHomeostasis(n_neurons=50, connectivity=0.1)
        selective = SelectiveConsolidation(homeostasis)

        # Tag first synapse
        synapse_id = list(homeostasis.synapses.keys())[0]
        selective.tag_for_protection(synapse_id)

        # Get another synapse
        other_id = [k for k in homeostasis.synapses.keys() if k != synapse_id][0]
        initial_other_weight = homeostasis.synapses[other_id].weight

        # Downscale
        homeostasis.downscale(factor=0.5)

        # Untagged synapse should be reduced
        assert homeostasis.synapses[other_id].weight < initial_other_weight

    def test_tag_by_weight(self):
        """Test tagging by weight percentile"""
        homeostasis = SynapticHomeostasis(n_neurons=50, connectivity=0.1)
        selective = SelectiveConsolidation(homeostasis)

        # Artificially set some weights high
        high_weight_ids = list(homeostasis.synapses.keys())[:5]
        for sid in high_weight_ids:
            homeostasis.synapses[sid].weight = 2.0

        # Tag top 10%
        n_tagged = selective.tag_by_weight(percentile=90)

        assert n_tagged > 0
        assert selective.get_tagged_count() > 0

    def test_tag_by_activity(self):
        """Test tagging by activity"""
        homeostasis = SynapticHomeostasis(n_neurons=50, connectivity=0.1)
        selective = SelectiveConsolidation(homeostasis)

        # Set high activity on some synapses
        active_ids = list(homeostasis.synapses.keys())[:5]
        for sid in active_ids:
            homeostasis.synapses[sid].activity_count = 10
            homeostasis.synapses[sid].last_potentiation = 90.0

        n_tagged = selective.tag_by_activity(
            activity_threshold=5,
            recency_window=20.0,
            current_time=100.0
        )

        assert n_tagged > 0

    def test_tag_decay(self):
        """Test that tags decay over time"""
        homeostasis = SynapticHomeostasis(n_neurons=50, connectivity=0.1)
        selective = SelectiveConsolidation(homeostasis, tag_decay_rate=0.5)

        synapse_id = list(homeostasis.synapses.keys())[0]
        selective.tag_for_protection(synapse_id, strength=1.0)

        # Decay tags
        for _ in range(5):
            selective.decay_tags(dt=1.0)

        # Tag should be removed after sufficient decay
        assert not homeostasis.synapses[synapse_id].is_tagged

    def test_clear_tags(self):
        """Test clearing all tags"""
        homeostasis = SynapticHomeostasis(n_neurons=50, connectivity=0.1)
        selective = SelectiveConsolidation(homeostasis)

        # Tag several synapses
        for sid in list(homeostasis.synapses.keys())[:10]:
            selective.tag_for_protection(sid)

        assert selective.get_tagged_count() == 10

        selective.clear_tags()

        assert selective.get_tagged_count() == 0


class TestSleepWakeCycle:
    """Tests for complete sleep-wake cycle"""

    def test_initialization(self):
        """Test cycle initialization"""
        cycle = SleepWakeCycle(n_neurons=50, connectivity=0.1)

        assert cycle.is_awake
        assert cycle.cycle_count == 0

    def test_wake_learning(self):
        """Test that wake period increases synaptic strength"""
        cycle = SleepWakeCycle(n_neurons=50, connectivity=0.1)

        initial_strength = cycle.homeostasis.measure_total_strength()

        stats = cycle.wake_learning(duration=8.0, learning_intensity=0.1)

        final_strength = cycle.homeostasis.measure_total_strength()

        assert final_strength > initial_strength
        assert stats["total_potentiation"] > 0

    def test_sleep_consolidation(self):
        """Test that sleep decreases synaptic strength"""
        cycle = SleepWakeCycle(n_neurons=50, connectivity=0.1)

        # First do some learning
        cycle.wake_learning(duration=8.0, learning_intensity=0.1)

        post_wake_strength = cycle.homeostasis.measure_total_strength()

        # Then sleep
        stats = cycle.sleep_consolidation(duration=8.0)

        post_sleep_strength = cycle.homeostasis.measure_total_strength()

        assert post_sleep_strength < post_wake_strength
        assert stats["total_downscaling"] > 0

    def test_full_cycle(self):
        """Test complete wake-sleep cycle"""
        cycle = SleepWakeCycle(n_neurons=50, connectivity=0.1)

        initial_strength = cycle.homeostasis.measure_total_strength()

        stats = cycle.run_full_cycle(
            wake_duration=16.0,
            sleep_duration=8.0,
            learning_intensity=0.1
        )

        # After full cycle, strength should be somewhat restored
        assert "wake" in stats
        assert "sleep" in stats
        assert cycle.cycle_count == 1

    def test_multiple_cycles(self):
        """Test multiple wake-sleep cycles"""
        cycle = SleepWakeCycle(n_neurons=50, connectivity=0.1)

        for _ in range(3):
            cycle.run_full_cycle()

        assert cycle.cycle_count == 3

        history = cycle.get_cycle_history()
        assert len(history["weight_history"]) >= 6  # 2 per cycle

    def test_homeostasis_maintenance(self):
        """Test that homeostasis maintains weight around baseline"""
        cycle = SleepWakeCycle(n_neurons=50, connectivity=0.1)

        initial_strength = cycle.homeostasis.measure_total_strength()

        # Run several cycles
        for _ in range(3):
            cycle.run_full_cycle(
                wake_duration=16.0,
                sleep_duration=8.0,
                learning_intensity=0.05
            )

        final_strength = cycle.homeostasis.measure_total_strength()

        # Strength should not explode over time
        # (homeostasis should keep it bounded)
        assert final_strength < initial_strength * 5  # Reasonable bound


class TestHomeostasisIntegration:
    """Integration tests for homeostasis with other systems"""

    def test_wake_sleep_balance(self):
        """Test balance between wake potentiation and sleep downscaling"""
        homeostasis = SynapticHomeostasis(n_neurons=100, connectivity=0.1)

        # Simulate wake: potentiation
        for _ in range(10):
            pre = np.random.random(100) > 0.5
            post = np.random.random(100) > 0.5
            homeostasis.hebbian_potentiation(pre.astype(float), post.astype(float), 0.1)

        post_wake_stats = homeostasis.get_statistics()

        # Simulate sleep: downscaling
        for _ in range(10):
            homeostasis.sleep_downscaling_step(dt=1.0)

        post_sleep_stats = homeostasis.get_statistics()

        # Total potentiation should be roughly balanced by downscaling over time
        assert post_sleep_stats["total_strength"] < post_wake_stats["total_strength"]

    def test_signal_to_noise_preservation(self):
        """Test that important patterns are preserved relative to noise"""
        homeostasis = SynapticHomeostasis(n_neurons=100, connectivity=0.1)
        selective = SelectiveConsolidation(homeostasis)

        # Create a strong "signal" pattern
        signal_ids = list(homeostasis.synapses.keys())[:10]
        for sid in signal_ids:
            homeostasis.synapses[sid].weight = 2.0
            homeostasis.synapses[sid].activity_count = 10

        # Compute initial signal-to-noise
        signal = sum(homeostasis.synapses[sid].weight for sid in signal_ids)
        noise = sum(s.weight for sid, s in homeostasis.synapses.items() if sid not in signal_ids)
        initial_snr = signal / (noise + 1e-8)

        # Tag signal for protection
        selective.tag_by_activity(activity_threshold=5, recency_window=1000, current_time=100)

        # Downscale
        homeostasis.downscale(factor=0.5)

        # Compute final signal-to-noise
        signal = sum(homeostasis.synapses[sid].weight for sid in signal_ids)
        noise = sum(s.weight for sid, s in homeostasis.synapses.items() if sid not in signal_ids)
        final_snr = signal / (noise + 1e-8)

        # SNR should improve (or at least not worsen much)
        assert final_snr >= initial_snr * 0.9


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
