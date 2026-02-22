"""
Reward-Modulated Learning

Dopamine-modulated STDP: weight changes only when rewarded
"""

import numpy as np
from typing import Optional, List
from dataclasses import dataclass

from .stdp import STDPParams, STDPNetwork


@dataclass
class DopamineParams:
    """Dopamine system parameters"""

    baseline: float = 0.5  # Baseline dopamine level
    release_rate: float = 1.0  # Release per reward unit
    decay_rate: float = 0.1  # Decay rate
    prediction_rate: float = 0.1  # Reward prediction learning rate


class DopamineSystem:
    """Simplified dopamine system for reward signaling

    Dopamine signals reward prediction error:
    δ = reward - expected_reward
    """

    def __init__(self, params: Optional[DopamineParams] = None):
        self.params = params or DopamineParams()
        self.level = self.params.baseline
        self.expected_reward = 0.0
        self.prediction_error = 0.0

        # History
        self.level_history: List[float] = []
        self.error_history: List[float] = []

    def receive_reward(self, reward: float) -> float:
        """Process reward and compute prediction error

        Args:
            reward: Actual reward received

        Returns:
            Dopamine level (proportional to prediction error)
        """
        # Prediction error
        self.prediction_error = reward - self.expected_reward

        # Update expected reward
        self.expected_reward += self.params.prediction_rate * self.prediction_error

        # Dopamine level reflects prediction error
        self.level = self.params.baseline + self.params.release_rate * self.prediction_error
        self.level = max(0.0, self.level)  # Can't go negative

        # Track history
        self.level_history.append(self.level)
        self.error_history.append(self.prediction_error)

        return self.level

    def update(self, dt: float = 1.0) -> float:
        """Decay dopamine toward baseline"""
        self.level = self.level + self.params.decay_rate * (self.params.baseline - self.level) * dt
        return self.level

    def get_modulation(self) -> float:
        """Get current modulation signal for learning"""
        # Positive error = enhance learning
        # Negative error = suppress learning
        return max(0.0, self.prediction_error)

    def reset(self) -> None:
        """Reset to baseline"""
        self.level = self.params.baseline
        self.expected_reward = 0.0
        self.prediction_error = 0.0


class RewardModulatedSTDP(STDPNetwork):
    """STDP with reward modulation (three-factor rule)

    Δw = dopamine × STDP_trace

    Weight changes only consolidate when reward arrives.
    """

    def __init__(
        self,
        n_pre: int,
        n_post: int,
        stdp_params: Optional[STDPParams] = None,
        dopamine_params: Optional[DopamineParams] = None,
    ):
        super().__init__(n_pre, n_post, stdp_params)

        self.dopamine = DopamineSystem(dopamine_params)

        # Eligibility trace (potential weight change waiting for reward)
        self.eligibility_trace = np.zeros((n_post, n_pre))
        self.eligibility_decay = 0.95

    def update(self, pre_spikes: np.ndarray, post_spikes: np.ndarray, dt: float = 1.0) -> None:
        """Update eligibility traces (not weights yet)"""
        # Update pre trace first (LTP uses updated pre_trace)
        self.pre_trace = self.pre_trace * self.trace_decay + pre_spikes

        # Compute STDP-based eligibility using old post_trace for LTD
        ltp = self.params.A_plus * np.outer(post_spikes, self.pre_trace)
        ltd = self.params.A_minus * np.outer(self.post_trace, pre_spikes)

        # Now update post trace
        self.post_trace = self.post_trace * self.trace_decay + post_spikes

        # Accumulate in eligibility trace
        self.eligibility_trace = self.eligibility_trace * self.eligibility_decay + (ltp - ltd)

        # Decay dopamine
        self.dopamine.update(dt)

    def receive_reward(self, reward: float) -> None:
        """Apply reward to consolidate eligibility traces into weights

        Args:
            reward: Reward signal (positive = good, negative = bad)
        """
        # Get dopamine modulation
        dopamine_level = self.dopamine.receive_reward(reward)
        modulation = self.dopamine.get_modulation()

        # Apply modulated eligibility to weights
        self.weights += modulation * self.eligibility_trace
        self.weights = np.clip(self.weights, self.params.w_min, self.params.w_max)

        # Decay eligibility after reward
        self.eligibility_trace *= 0.5

    def train_episode(
        self, pre_sequence: np.ndarray, post_sequence: np.ndarray, reward: float
    ) -> float:
        """Train on sequence with delayed reward

        Args:
            pre_sequence: (T, n_pre) binary spikes
            post_sequence: (T, n_post) binary spikes
            reward: Reward at end of episode

        Returns:
            Total weight change
        """
        w_before = self.weights.copy()

        # Accumulate eligibility over sequence
        for pre, post in zip(pre_sequence, post_sequence):
            self.update(pre, post)

        # Apply reward at end
        self.receive_reward(reward)

        return np.mean(np.abs(self.weights - w_before))

    def reset(self) -> None:
        """Reset all traces"""
        self.reset_traces()
        self.eligibility_trace = np.zeros((self.n_post, self.n_pre))
        self.dopamine.reset()


class ErrorBasedLearning:
    """Error-based learning (supervised-like)

    Δw = η × error × input
    """

    def __init__(self, n_input: int, n_output: int, learning_rate: float = 0.01):
        self.n_input = n_input
        self.n_output = n_output
        self.learning_rate = learning_rate

        self.weights = np.random.randn(n_output, n_input) * 0.1

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute output"""
        return np.tanh(self.weights @ x)

    def update(self, x: np.ndarray, target: np.ndarray) -> float:
        """Update weights based on error

        Args:
            x: Input
            target: Target output

        Returns:
            Error magnitude
        """
        output = self.forward(x)
        error = target - output

        # Gradient descent on error
        dW = self.learning_rate * np.outer(error, x)
        self.weights += dW

        return np.mean(error**2)

    def train(self, inputs: np.ndarray, targets: np.ndarray, n_epochs: int = 100) -> List[float]:
        """Train on dataset"""
        errors = []

        for epoch in range(n_epochs):
            epoch_error = 0.0
            for x, t in zip(inputs, targets):
                epoch_error += self.update(x, t)
            errors.append(epoch_error / len(inputs))

        return errors
