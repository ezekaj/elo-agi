"""
Sim2Real: Transfer from simulation to reality.

Implements methods for bridging the simulation-reality gap.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple, Callable
from enum import Enum
import numpy as np
import time


class DomainType(Enum):
    """Domain types."""

    SIMULATION = "simulation"
    REALITY = "reality"


class RandomizationType(Enum):
    """Types of domain randomization."""

    VISUAL = "visual"
    DYNAMICS = "dynamics"
    MORPHOLOGY = "morphology"
    SENSOR_NOISE = "sensor_noise"


@dataclass
class RandomizationConfig:
    """Configuration for domain randomization."""

    visual_noise_std: float = 0.1
    dynamics_friction_range: Tuple[float, float] = (0.5, 1.5)
    dynamics_mass_range: Tuple[float, float] = (0.8, 1.2)
    sensor_noise_std: float = 0.05
    morphology_scale_range: Tuple[float, float] = (0.9, 1.1)


@dataclass
class RealityGapMetrics:
    """Metrics for measuring reality gap."""

    state_distribution_distance: float
    action_effect_difference: float
    reward_difference: float
    success_rate_difference: float


class DomainRandomization:
    """
    Domain randomization for sim-to-real transfer.

    Implements:
    - Visual randomization
    - Dynamics randomization
    - Sensor noise injection
    - Morphology variations
    """

    def __init__(
        self,
        config: Optional[RandomizationConfig] = None,
        seed: Optional[int] = None,
    ):
        self.config = config or RandomizationConfig()
        self.rng = np.random.RandomState(seed)

        # History
        self._randomizations_applied: List[Dict[str, Any]] = []

    def randomize_visual(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Apply visual domain randomization.

        Args:
            image: Input image (H, W, C) or (H, W)

        Returns:
            Randomized image
        """
        randomized = image.copy().astype(float)

        # Brightness variation
        brightness = self.rng.uniform(0.8, 1.2)
        randomized = randomized * brightness

        # Contrast variation
        contrast = self.rng.uniform(0.8, 1.2)
        mean = np.mean(randomized)
        randomized = (randomized - mean) * contrast + mean

        # Color jitter (if RGB)
        if randomized.ndim == 3 and randomized.shape[2] == 3:
            color_scale = self.rng.uniform(0.9, 1.1, size=3)
            randomized = randomized * color_scale

        # Additive noise
        noise = self.rng.randn(*randomized.shape) * self.config.visual_noise_std * 255
        randomized = randomized + noise

        # Clip and convert back
        randomized = np.clip(randomized, 0, 255).astype(image.dtype)

        self._record_randomization(
            "visual",
            {
                "brightness": brightness,
                "contrast": contrast,
            },
        )

        return randomized

    def randomize_dynamics(
        self,
        mass: float,
        friction: float,
        damping: float,
    ) -> Tuple[float, float, float]:
        """
        Randomize dynamics parameters.

        Args:
            mass: Base mass
            friction: Base friction coefficient
            damping: Base damping coefficient

        Returns:
            Tuple of (randomized_mass, randomized_friction, randomized_damping)
        """
        mass_scale = self.rng.uniform(*self.config.dynamics_mass_range)
        friction_scale = self.rng.uniform(*self.config.dynamics_friction_range)
        damping_scale = self.rng.uniform(0.8, 1.2)

        new_mass = mass * mass_scale
        new_friction = friction * friction_scale
        new_damping = damping * damping_scale

        self._record_randomization(
            "dynamics",
            {
                "mass_scale": mass_scale,
                "friction_scale": friction_scale,
                "damping_scale": damping_scale,
            },
        )

        return new_mass, new_friction, new_damping

    def add_sensor_noise(
        self,
        observation: np.ndarray,
    ) -> np.ndarray:
        """
        Add sensor noise to observation.

        Args:
            observation: Clean observation

        Returns:
            Noisy observation
        """
        noise = self.rng.randn(*observation.shape) * self.config.sensor_noise_std
        noisy = observation + noise

        self._record_randomization(
            "sensor_noise",
            {
                "noise_std": self.config.sensor_noise_std,
            },
        )

        return noisy

    def randomize_morphology(
        self,
        link_lengths: np.ndarray,
        link_masses: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Randomize robot morphology.

        Args:
            link_lengths: Original link lengths
            link_masses: Original link masses

        Returns:
            Tuple of (randomized_lengths, randomized_masses)
        """
        length_scales = self.rng.uniform(
            *self.config.morphology_scale_range, size=link_lengths.shape
        )
        mass_scales = self.rng.uniform(*self.config.dynamics_mass_range, size=link_masses.shape)

        new_lengths = link_lengths * length_scales
        new_masses = link_masses * mass_scales

        self._record_randomization(
            "morphology",
            {
                "length_scales": length_scales.tolist(),
                "mass_scales": mass_scales.tolist(),
            },
        )

        return new_lengths, new_masses

    def _record_randomization(
        self,
        rand_type: str,
        params: Dict[str, Any],
    ) -> None:
        """Record applied randomization."""
        self._randomizations_applied.append(
            {
                "type": rand_type,
                "params": params,
                "timestamp": time.time(),
            }
        )

    def get_history(self) -> List[Dict[str, Any]]:
        """Get randomization history."""
        return self._randomizations_applied.copy()

    def reset(self) -> None:
        """Reset randomization history."""
        self._randomizations_applied.clear()

    def statistics(self) -> Dict[str, Any]:
        """Get randomization statistics."""
        type_counts = {}
        for r in self._randomizations_applied:
            t = r["type"]
            type_counts[t] = type_counts.get(t, 0) + 1

        return {
            "total_randomizations": len(self._randomizations_applied),
            "by_type": type_counts,
            "config": {
                "visual_noise_std": self.config.visual_noise_std,
                "sensor_noise_std": self.config.sensor_noise_std,
            },
        }


class RealityGap:
    """
    Measure and analyze the reality gap.

    Implements:
    - Distribution comparison
    - Action effect analysis
    - Performance gap estimation
    """

    def __init__(self):
        self._sim_samples: List[np.ndarray] = []
        self._real_samples: List[np.ndarray] = []
        self._sim_rewards: List[float] = []
        self._real_rewards: List[float] = []

    def add_sim_sample(
        self,
        state: np.ndarray,
        reward: Optional[float] = None,
    ) -> None:
        """Add simulation sample."""
        self._sim_samples.append(state)
        if reward is not None:
            self._sim_rewards.append(reward)

    def add_real_sample(
        self,
        state: np.ndarray,
        reward: Optional[float] = None,
    ) -> None:
        """Add real-world sample."""
        self._real_samples.append(state)
        if reward is not None:
            self._real_rewards.append(reward)

    def compute_distribution_distance(self) -> float:
        """
        Compute distance between sim and real state distributions.

        Uses Maximum Mean Discrepancy (MMD).

        Returns:
            Distribution distance
        """
        if not self._sim_samples or not self._real_samples:
            return 0.0

        sim_arr = np.array(self._sim_samples)
        real_arr = np.array(self._real_samples)

        # Compute MMD with RBF kernel
        def rbf_kernel(x, y, sigma=1.0):
            diff = x[:, np.newaxis, :] - y[np.newaxis, :, :]
            sq_dist = np.sum(diff**2, axis=2)
            return np.exp(-sq_dist / (2 * sigma**2))

        # Flatten if needed
        if sim_arr.ndim > 2:
            sim_arr = sim_arr.reshape(len(sim_arr), -1)
            real_arr = real_arr.reshape(len(real_arr), -1)

        k_ss = rbf_kernel(sim_arr, sim_arr)
        k_rr = rbf_kernel(real_arr, real_arr)
        k_sr = rbf_kernel(sim_arr, real_arr)

        mmd = np.mean(k_ss) + np.mean(k_rr) - 2 * np.mean(k_sr)
        return float(max(0, mmd))

    def compute_reward_gap(self) -> float:
        """
        Compute gap between sim and real rewards.

        Returns:
            Reward difference
        """
        if not self._sim_rewards or not self._real_rewards:
            return 0.0

        sim_mean = np.mean(self._sim_rewards)
        real_mean = np.mean(self._real_rewards)

        return float(abs(sim_mean - real_mean))

    def compute_metrics(self) -> RealityGapMetrics:
        """
        Compute comprehensive reality gap metrics.

        Returns:
            RealityGapMetrics
        """
        return RealityGapMetrics(
            state_distribution_distance=self.compute_distribution_distance(),
            action_effect_difference=0.0,  # Would need action data
            reward_difference=self.compute_reward_gap(),
            success_rate_difference=0.0,  # Would need success labels
        )

    def clear(self) -> None:
        """Clear all samples."""
        self._sim_samples.clear()
        self._real_samples.clear()
        self._sim_rewards.clear()
        self._real_rewards.clear()

    def statistics(self) -> Dict[str, Any]:
        """Get statistics."""
        return {
            "sim_samples": len(self._sim_samples),
            "real_samples": len(self._real_samples),
            "sim_rewards": len(self._sim_rewards),
            "real_rewards": len(self._real_rewards),
        }


class SimToRealTransfer:
    """
    Complete sim-to-real transfer system.

    Combines:
    - Domain randomization
    - Reality gap measurement
    - Adaptation strategies
    """

    def __init__(
        self,
        randomization_config: Optional[RandomizationConfig] = None,
    ):
        self.randomization = DomainRandomization(randomization_config)
        self.reality_gap = RealityGap()

        # Adaptation state
        self._adaptation_history: List[Dict[str, Any]] = []
        self._sim_policy: Optional[Callable] = None
        self._real_policy: Optional[Callable] = None

    def set_sim_policy(self, policy: Callable) -> None:
        """Set simulation-trained policy."""
        self._sim_policy = policy

    def set_real_policy(self, policy: Callable) -> None:
        """Set real-world adapted policy."""
        self._real_policy = policy

    def transfer_observation(
        self,
        sim_obs: np.ndarray,
        apply_randomization: bool = True,
    ) -> np.ndarray:
        """
        Transfer observation from sim format to real format.

        Args:
            sim_obs: Simulation observation
            apply_randomization: Whether to apply randomization

        Returns:
            Transferred observation
        """
        obs = sim_obs.copy()

        if apply_randomization:
            obs = self.randomization.add_sensor_noise(obs)

        return obs

    def transfer_action(
        self,
        sim_action: np.ndarray,
        action_scale: float = 1.0,
    ) -> np.ndarray:
        """
        Transfer action from sim to real.

        Args:
            sim_action: Simulation action
            action_scale: Scaling factor

        Returns:
            Real-world action
        """
        return sim_action * action_scale

    def evaluate_transfer_quality(
        self,
        n_episodes: int = 10,
    ) -> Dict[str, Any]:
        """
        Evaluate transfer quality.

        Args:
            n_episodes: Number of evaluation episodes

        Returns:
            Evaluation results
        """
        metrics = self.reality_gap.compute_metrics()

        return {
            "distribution_distance": metrics.state_distribution_distance,
            "reward_gap": metrics.reward_difference,
            "n_sim_samples": len(self.reality_gap._sim_samples),
            "n_real_samples": len(self.reality_gap._real_samples),
        }

    def adapt_policy(
        self,
        real_observations: List[np.ndarray],
        real_actions: List[np.ndarray],
        real_rewards: List[float],
    ) -> None:
        """
        Adapt policy based on real-world data.

        Args:
            real_observations: Real-world observations
            real_actions: Real-world actions
            real_rewards: Real-world rewards
        """
        # Add to reality gap tracker
        for obs, reward in zip(real_observations, real_rewards):
            self.reality_gap.add_real_sample(obs, reward)

        self._adaptation_history.append(
            {
                "timestamp": time.time(),
                "n_samples": len(real_observations),
                "mean_reward": np.mean(real_rewards) if real_rewards else 0.0,
            }
        )

    def get_adaptation_history(self) -> List[Dict[str, Any]]:
        """Get adaptation history."""
        return self._adaptation_history.copy()

    def reset(self) -> None:
        """Reset transfer system."""
        self.randomization.reset()
        self.reality_gap.clear()
        self._adaptation_history.clear()

    def statistics(self) -> Dict[str, Any]:
        """Get system statistics."""
        return {
            "randomization": self.randomization.statistics(),
            "reality_gap": self.reality_gap.statistics(),
            "adaptations": len(self._adaptation_history),
            "has_sim_policy": self._sim_policy is not None,
            "has_real_policy": self._real_policy is not None,
        }
