"""Tests for STDP learning"""

import numpy as np
import pytest
from neuro.modules.m12_learning.stdp import STDPRule, STDPSynapse, STDPNetwork, STDPParams


class TestSTDPRule:
    """Tests for STDP rule"""

    def test_initialization(self):
        """Test STDP rule initialization"""
        rule = STDPRule()
        assert rule.params.A_plus > 0
        assert rule.params.A_minus > 0

    def test_ltp_pre_before_post(self):
        """Test LTP when pre fires before post"""
        rule = STDPRule()

        # Pre before post (delta_t > 0)
        dw = rule.compute_weight_change(10.0)  # 10ms delay

        assert dw > 0  # LTP

    def test_ltd_post_before_pre(self):
        """Test LTD when post fires before pre"""
        rule = STDPRule()

        # Post before pre (delta_t < 0)
        dw = rule.compute_weight_change(-10.0)  # -10ms delay

        assert dw < 0  # LTD

    def test_no_change_at_zero(self):
        """Test no change when simultaneous"""
        rule = STDPRule()
        dw = rule.compute_weight_change(0.0)
        assert dw == 0

    def test_exponential_decay(self):
        """Test weight change decays with time"""
        rule = STDPRule()

        dw_5ms = rule.compute_weight_change(5.0)
        dw_20ms = rule.compute_weight_change(20.0)
        dw_50ms = rule.compute_weight_change(50.0)

        # Closer timing = larger change
        assert dw_5ms > dw_20ms > dw_50ms > 0

    def test_weight_matrix_change(self):
        """Test computing weight changes for spike times"""
        rule = STDPRule()

        pre_times = np.array([10.0, 20.0, np.nan])
        post_times = np.array([15.0, 18.0])

        dW = rule.compute_weight_matrix_change(pre_times, post_times)

        assert dW.shape == (2, 3)
        # Check specific patterns
        assert dW[0, 0] > 0  # Pre at 10, post at 15 -> LTP
        assert dW[1, 1] < 0  # Pre at 20, post at 18 -> LTD


class TestSTDPSynapse:
    """Tests for single STDP synapse"""

    def test_synapse_initialization(self):
        """Test synapse initialization"""
        syn = STDPSynapse(weight=0.5)
        assert syn.weight == 0.5

    def test_pre_spike_records(self):
        """Test pre spike recording"""
        syn = STDPSynapse()
        syn.pre_spike(10.0)
        assert 10.0 in syn.pre_spike_times

    def test_post_spike_causes_ltp(self):
        """Test post spike after pre causes LTP"""
        syn = STDPSynapse(weight=0.5)

        syn.pre_spike(10.0)
        weight_before = syn.weight
        syn.post_spike(15.0)  # Post after pre

        assert syn.weight > weight_before

    def test_pre_spike_after_post_causes_ltd(self):
        """Test pre spike after post causes LTD"""
        syn = STDPSynapse(weight=0.5)

        syn.post_spike(10.0)
        weight_before = syn.weight
        syn.pre_spike(15.0)  # Pre after post

        assert syn.weight < weight_before

    def test_weight_bounds(self):
        """Test weights stay bounded"""
        params = STDPParams(A_plus=0.1, w_max=1.0, w_min=0.0)
        syn = STDPSynapse(weight=0.9, params=params)

        # Many LTP events
        for i in range(100):
            syn.pre_spike(float(i * 10))
            syn.post_spike(float(i * 10 + 5))

        assert syn.weight <= 1.0
        assert syn.weight >= 0.0


class TestSTDPNetwork:
    """Tests for STDP network"""

    def test_network_initialization(self):
        """Test network initialization"""
        net = STDPNetwork(n_pre=10, n_post=5)

        assert net.weights.shape == (5, 10)
        assert net.pre_trace.shape == (10,)
        assert net.post_trace.shape == (5,)

    def test_trace_update(self):
        """Test trace updates on spikes"""
        net = STDPNetwork(n_pre=3, n_post=2)

        pre_spikes = np.array([1, 0, 1])
        post_spikes = np.array([0, 1])

        net.update(pre_spikes, post_spikes)

        assert net.pre_trace[0] > 0
        assert net.pre_trace[1] == 0
        assert net.post_trace[1] > 0

    def test_ltp_in_network(self):
        """Test LTP occurs in network"""
        net = STDPNetwork(n_pre=3, n_post=2)

        # Pre then post pattern
        net.update(np.array([1, 0, 0]), np.array([0, 0]))  # Pre fires
        weight_before = net.weights[0, 0]
        net.update(np.array([0, 0, 0]), np.array([1, 0]))  # Post fires

        assert net.weights[0, 0] > weight_before  # LTP

    def test_training(self):
        """Test network training"""
        net = STDPNetwork(n_pre=10, n_post=5)

        pre_patterns = np.random.rand(20, 10) > 0.5
        post_patterns = np.random.rand(20, 5) > 0.5

        weight_changes = net.train(
            pre_patterns.astype(float), post_patterns.astype(float), n_epochs=3
        )

        assert len(weight_changes) == 3
        assert all(np.isfinite(wc) for wc in weight_changes)

    def test_forward_pass(self):
        """Test forward computation"""
        net = STDPNetwork(n_pre=10, n_post=5)

        pre_activity = np.random.rand(10)
        post_response = net.forward(pre_activity)

        assert post_response.shape == (5,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
