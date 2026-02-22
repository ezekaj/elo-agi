"""
Homeostatic Regulation

Maintains stability in neural networks through:
- Synaptic scaling
- Intrinsic plasticity
- Activity regulation
"""

import numpy as np
from typing import Optional, List
from dataclasses import dataclass


@dataclass
class HomeostaticParams:
    """Homeostatic regulation parameters"""

    target_activity: float = 0.1  # Target firing rate
    scaling_rate: float = 0.001  # Synaptic scaling rate
    intrinsic_rate: float = 0.01  # Intrinsic plasticity rate
    time_constant: float = 100.0  # Integration time constant


class HomeostaticRegulation:
    """Homeostatic regulation of neural activity

    Maintains stable activity levels through multiple mechanisms.
    """

    def __init__(self, n_neurons: int, params: Optional[HomeostaticParams] = None):
        self.n_neurons = n_neurons
        self.params = params or HomeostaticParams()

        # Running average of activity
        self.activity_average = np.ones(n_neurons) * self.params.target_activity

        # Intrinsic excitability (threshold modulation)
        self.excitability = np.ones(n_neurons)

    def update_average(self, activity: np.ndarray) -> None:
        """Update running average of activity"""
        alpha = 1.0 / self.params.time_constant
        self.activity_average = (1 - alpha) * self.activity_average + alpha * activity

    def compute_scaling_factor(self) -> np.ndarray:
        """Compute synaptic scaling factors

        Returns:
            Scaling factors per neuron (>1 = upscale, <1 = downscale)
        """
        # Scale inversely with activity
        ratio = self.params.target_activity / (self.activity_average + 1e-8)
        return ratio**self.params.scaling_rate

    def scale_weights(self, weights: np.ndarray) -> np.ndarray:
        """Apply synaptic scaling to weight matrix

        Args:
            weights: Weight matrix (n_post, n_pre)

        Returns:
            Scaled weights
        """
        scaling = self.compute_scaling_factor()

        # Scale incoming weights per postsynaptic neuron
        scaled_weights = weights * scaling[:, np.newaxis]

        return scaled_weights

    def update_excitability(self) -> np.ndarray:
        """Update intrinsic excitability

        Low activity -> increase excitability
        High activity -> decrease excitability
        """
        error = self.params.target_activity - self.activity_average
        self.excitability += self.params.intrinsic_rate * error
        self.excitability = np.clip(self.excitability, 0.1, 10.0)
        return self.excitability

    def regulate(self, activity: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Full homeostatic regulation step

        Args:
            activity: Current activity levels
            weights: Current weight matrix

        Returns:
            Regulated weights
        """
        self.update_average(activity)
        self.update_excitability()
        return self.scale_weights(weights)


class SynapticScaling:
    """Synaptic scaling - multiplicative normalization

    Scales all synapses proportionally to maintain total drive.
    """

    def __init__(self, target_sum: float = 1.0, scaling_rate: float = 0.01):
        self.target_sum = target_sum
        self.scaling_rate = scaling_rate

    def scale(self, weights: np.ndarray) -> np.ndarray:
        """Scale weights to maintain target sum per neuron

        Args:
            weights: Weight matrix (n_post, n_pre)

        Returns:
            Scaled weights
        """
        # Current sum per postsynaptic neuron
        current_sums = np.sum(weights, axis=1, keepdims=True) + 1e-8

        # Target scaling
        target_scale = self.target_sum / current_sums

        # Gradual scaling
        scale = 1 + self.scaling_rate * (target_scale - 1)

        return weights * scale

    def normalize(self, weights: np.ndarray) -> np.ndarray:
        """Immediate normalization to target"""
        current_sums = np.sum(weights, axis=1, keepdims=True) + 1e-8
        return weights * (self.target_sum / current_sums)


class MetaplasticityRegulation:
    """Metaplasticity - plasticity of plasticity

    BCM-style sliding threshold based on activity history.
    """

    def __init__(self, n_neurons: int, time_constant: float = 1000.0):
        self.n_neurons = n_neurons
        self.time_constant = time_constant

        # Modification threshold per neuron
        self.threshold = np.ones(n_neurons) * 0.5

        # Activity history
        self.activity_squared_avg = np.ones(n_neurons) * 0.25

    def update(self, activity: np.ndarray) -> np.ndarray:
        """Update modification thresholds

        Args:
            activity: Current activity

        Returns:
            Updated thresholds
        """
        alpha = 1.0 / self.time_constant

        # Track average squared activity
        self.activity_squared_avg = (1 - alpha) * self.activity_squared_avg + alpha * activity**2

        # Threshold is related to recent activity
        self.threshold = self.activity_squared_avg

        return self.threshold

    def get_plasticity_direction(self, activity: np.ndarray) -> np.ndarray:
        """Get direction of plasticity for each neuron

        Args:
            activity: Current activity

        Returns:
            +1 for LTP, -1 for LTD
        """
        return np.sign(activity - self.threshold)


class ActivityRegulator:
    """Combined activity regulation system"""

    def __init__(self, n_neurons: int, target_activity: float = 0.1):
        self.homeostatic = HomeostaticRegulation(
            n_neurons, HomeostaticParams(target_activity=target_activity)
        )
        self.scaling = SynapticScaling()
        self.metaplasticity = MetaplasticityRegulation(n_neurons)

    def regulate(self, activity: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Apply all regulation mechanisms

        Args:
            activity: Current activity
            weights: Current weights

        Returns:
            Regulated weights
        """
        # Homeostatic regulation
        weights = self.homeostatic.regulate(activity, weights)

        # Synaptic scaling
        weights = self.scaling.scale(weights)

        # Update metaplasticity thresholds
        self.metaplasticity.update(activity)

        return weights

    def get_excitability(self) -> np.ndarray:
        """Get intrinsic excitability"""
        return self.homeostatic.excitability

    def get_plasticity_thresholds(self) -> np.ndarray:
        """Get metaplasticity thresholds"""
        return self.metaplasticity.threshold
