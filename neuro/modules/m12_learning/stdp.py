"""
Spike-Timing Dependent Plasticity (STDP)

Pre before post = LTP (Long-Term Potentiation)
Post before pre = LTD (Long-Term Depression)
"""

import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class STDPParams:
    """STDP parameters"""
    A_plus: float = 0.005  # LTP amplitude
    A_minus: float = 0.005  # LTD amplitude
    tau_plus: float = 20.0  # LTP time constant (ms)
    tau_minus: float = 20.0  # LTD time constant (ms)
    w_max: float = 1.0
    w_min: float = 0.0


class STDPRule:
    """Spike-Timing Dependent Plasticity rule

    If pre fires before post (Δt > 0): LTP
    If post fires before pre (Δt < 0): LTD

    Δw = A+ * exp(-Δt/τ+) for Δt > 0
    Δw = -A- * exp(Δt/τ-) for Δt < 0
    """

    def __init__(self, params: Optional[STDPParams] = None):
        self.params = params or STDPParams()

    def compute_weight_change(self, delta_t: float) -> float:
        """Compute weight change based on spike timing difference

        Args:
            delta_t: t_post - t_pre (positive = pre before post)

        Returns:
            Weight change
        """
        if delta_t > 0:
            # Pre before post: LTP
            return self.params.A_plus * np.exp(-delta_t / self.params.tau_plus)
        elif delta_t < 0:
            # Post before pre: LTD
            return -self.params.A_minus * np.exp(delta_t / self.params.tau_minus)
        else:
            return 0.0

    def compute_weight_matrix_change(
        self,
        pre_times: np.ndarray,
        post_times: np.ndarray
    ) -> np.ndarray:
        """Compute weight changes for all pre-post pairs

        Args:
            pre_times: Spike times of presynaptic neurons (n_pre,)
            post_times: Spike times of postsynaptic neurons (n_post,)

        Returns:
            Weight change matrix (n_post, n_pre)
        """
        n_pre = len(pre_times)
        n_post = len(post_times)

        dW = np.zeros((n_post, n_pre))

        for i in range(n_post):
            for j in range(n_pre):
                if not np.isnan(pre_times[j]) and not np.isnan(post_times[i]):
                    delta_t = post_times[i] - pre_times[j]
                    dW[i, j] = self.compute_weight_change(delta_t)

        return dW


class STDPSynapse:
    """Single synapse with STDP"""

    def __init__(
        self,
        weight: float = 0.5,
        params: Optional[STDPParams] = None
    ):
        self.weight = weight
        self.params = params or STDPParams()
        self.rule = STDPRule(self.params)

        # Spike history
        self.pre_spike_times: List[float] = []
        self.post_spike_times: List[float] = []
        self.current_time = 0.0

    def pre_spike(self, time: float) -> None:
        """Record presynaptic spike"""
        self.pre_spike_times.append(time)
        self.current_time = time

        # Apply LTD for any recent post spikes
        for post_t in self.post_spike_times:
            delta_t = time - post_t
            if abs(delta_t) < 100:  # Within 100ms
                dw = self.rule.compute_weight_change(-delta_t)  # Negative because post was before
                self.weight = np.clip(
                    self.weight + dw,
                    self.params.w_min,
                    self.params.w_max
                )

    def post_spike(self, time: float) -> None:
        """Record postsynaptic spike"""
        self.post_spike_times.append(time)
        self.current_time = time

        # Apply LTP for any recent pre spikes
        for pre_t in self.pre_spike_times:
            delta_t = time - pre_t
            if abs(delta_t) < 100:  # Within 100ms
                dw = self.rule.compute_weight_change(delta_t)
                self.weight = np.clip(
                    self.weight + dw,
                    self.params.w_min,
                    self.params.w_max
                )

    def reset_history(self) -> None:
        """Clear spike history"""
        self.pre_spike_times = []
        self.post_spike_times = []


class STDPNetwork:
    """Network of neurons with STDP"""

    def __init__(
        self,
        n_pre: int,
        n_post: int,
        params: Optional[STDPParams] = None
    ):
        self.n_pre = n_pre
        self.n_post = n_post
        self.params = params or STDPParams()
        self.rule = STDPRule(self.params)

        # Weight matrix
        self.weights = np.random.rand(n_post, n_pre) * 0.5

        # Spike traces (eligibility traces)
        self.pre_trace = np.zeros(n_pre)
        self.post_trace = np.zeros(n_post)

        # Trace decay
        self.trace_decay = 0.95

    def update(
        self,
        pre_spikes: np.ndarray,
        post_spikes: np.ndarray,
        dt: float = 1.0
    ) -> None:
        """Update weights based on spike patterns

        Args:
            pre_spikes: Binary spike vector (n_pre,)
            post_spikes: Binary spike vector (n_post,)
            dt: Time step
        """
        # Update traces
        self.pre_trace = self.pre_trace * self.trace_decay + pre_spikes
        self.post_trace = self.post_trace * self.trace_decay + post_spikes

        # LTP: pre before post
        # When post fires, strengthen connections from recent pre
        ltp = self.params.A_plus * np.outer(post_spikes, self.pre_trace)

        # LTD: post before pre
        # When pre fires, weaken connections to recent post
        ltd = self.params.A_minus * np.outer(self.post_trace, pre_spikes)

        # Update weights
        self.weights += ltp - ltd
        self.weights = np.clip(self.weights, self.params.w_min, self.params.w_max)

    def forward(self, pre_activity: np.ndarray) -> np.ndarray:
        """Compute postsynaptic response"""
        return self.weights @ pre_activity

    def train(
        self,
        pre_patterns: np.ndarray,
        post_patterns: np.ndarray,
        n_epochs: int = 10
    ) -> List[float]:
        """Train network with paired patterns"""
        weight_changes = []

        for epoch in range(n_epochs):
            w_before = self.weights.copy()

            for pre, post in zip(pre_patterns, post_patterns):
                self.update(pre, post)

            w_change = np.mean(np.abs(self.weights - w_before))
            weight_changes.append(w_change)

        return weight_changes

    def reset_traces(self) -> None:
        """Reset eligibility traces"""
        self.pre_trace = np.zeros(self.n_pre)
        self.post_trace = np.zeros(self.n_post)
